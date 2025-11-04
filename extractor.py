import time
import concurrent.futures
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

import config
from document import Document
from llm_service import find_values_from_layout
from text_utils import stopword_filter, calculate_stopwords

JACCARD_THRESHOLD = 0.95
FULL_SCHEMAS = dict() # armazena schemas completos para análises futuras

# Carregamos os modelos globalmente
try:
    EMBEDDING_MODEL = SentenceTransformer(
        config.EMBEDDING_MODEL_NAME, 
        device=config.DEVICE
    )
    print(f"Modelo de embedding carregado em: {config.DEVICE}")
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível carregar o SentenceTransformer: {e}")
    EMBEDDING_MODEL = None

class Extractor:
    def __init__(self):
        print("Inicializando Extractor...")
        
        try:
            self.client = OpenAI()
        except Exception as e:
            print(f"ERRO: Não foi possível inicializar o cliente OpenAI: {e}")
            self.client = None
            
        self.embedding_model = EMBEDDING_MODEL
        if not self.embedding_model:
            print("AVISO: Extractor inicializado sem modelo de embedding.")

        print("Extrator pronto.")

    def _calculate_jaccard(self, set1: frozenset[str], set2: frozenset[str]) -> float:
        """Calcula a Similaridade Jaccard entre dois conjuntos de snippets."""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 1.0  # Dois conjuntos vazios são idênticos
        
        return intersection / union

    def extract(self, label: str, extraction_schema: dict, fitz_page_dict: dict) -> dict:
        """
        Pipeline baseline: Recupera âncoras, "infla" o contexto,
        agrupa e extrai valores.
        """
        print(f"\n--- Processando Label: {label} ---")
        start_time = time.time()
        # Update FULL_SCHEMAS
        FULL_SCHEMAS[label] = {**FULL_SCHEMAS.get(label, {}), **extraction_schema}
        
        if not self.embedding_model:
            print("ERRO: Modelo de embedding não carregado. Abortando.")
            return {key: None for key in extraction_schema}
            
        try:
            doc = Document(fitz_page_dict)
            doc.embed_spans(self.embedding_model)
            doc.find_tables()

        except ValueError as e:
            print(e)
            return {key: None for key in extraction_schema}

        # --- PASSO 1: RECUPERAÇÃO DE "ÂNCORAS" ---
        print("  Iniciando Passo 1: Recuperação de Âncoras...")
        
        # retrieved_anchors_by_key armazena os OBJETOS Line, não strings
        retrieved_anchors_by_key: Dict[str, set[Line]] = {}

        # find schema stopwords
        descs = FULL_SCHEMAS[label].values()
        stopwords = calculate_stopwords(list(descs), threshold=0.5)
        
        for key, description in extraction_schema.items():
            description = stopword_filter(description, stopwords)
            # 1a. Keyword Retrieval (retorna List[Line])
            kw_lines = doc.get_keyword_matching_lines(key)
            
            # 1b. Semantic Retrieval (retorna List[Line])
            query = f"{key}: {description}"
            sem_lines = doc.get_semantic_matching_lines(self.embedding_model, query, k=config.SNIPPET_K_SEMANTIC-len(kw_lines))
            
            # 1c. Combina os objetos Line
            context_set = set(kw_lines).union(set(sem_lines))
            retrieved_anchors_by_key[key] = context_set

        # --- PASSO 2: INFLAR O CONTEXTO E AGRUPAR ---
        print("  Iniciando Passo 2: Inflar Contexto e Agrupar Chaves...")
        
        # Agora, a "assinatura" de uma chave é o seu CONTEXTO INFLADO
        retrieved_snippets_by_key: Dict[str, frozenset[str]] = {}
        
        for key, anchor_lines in retrieved_anchors_by_key.items():
            inflated_snippets = set()
            if not anchor_lines:
                # Se não encontrou âncoras, a assinatura está vazia
                retrieved_snippets_by_key[key] = frozenset()
                continue
            
            # "Infla" o contexto para cada âncora encontrada
            for line in anchor_lines:
                context_lines = doc.get_context_for_line(line, window_size=2)
                inflated_snippets.update(context_lines)

            
            # A "assinatura" da chave é o conjunto de todos os seus
            # snippets de contexto inflado
            retrieved_snippets_by_key[key] = frozenset(inflated_snippets)

        # --- PASSO 3: AGRUPAMENTO (BATCHING) ---
        # (Esta lógica não muda, mas agora opera em assinaturas de contexto
        # muito mais ricas e precisas)
        clusters = []
        keys_without_snippets = set()
        for key, snippets in retrieved_snippets_by_key.items():
            if not snippets:
                keys_without_snippets.add(key)
                continue
            clusters.append({
                'keys': {key},
                'snippets': snippets
            })

        # 2. Loop de clustering aglomerativo
        while True:
            best_score = -1
            best_pair_to_merge = None # (index_i, index_j)

            # Encontra o melhor par para fundir (O(n^2), aceitável para N < 100)
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    c1 = clusters[i]
                    c2 = clusters[j]
                    
                    score = self._calculate_jaccard(c1['snippets'], c2['snippets'])
                    
                    if score > best_score:
                        best_score = score
                        best_pair_to_merge = (i, j)
            
            # 3. Decide se funde ou para
            if best_pair_to_merge and best_score >= JACCARD_THRESHOLD:
                i, j = best_pair_to_merge
                
                # Para evitar reindexação, pegamos os índices em ordem decrescente
                c2 = clusters.pop(max(i, j))
                c1 = clusters.pop(min(i, j))
                
                # Cria o novo cluster fundido
                merged_cluster = {
                    'keys': c1['keys'].union(c2['keys']),
                    'snippets': c1['snippets'].union(c2['snippets'])
                }
                
                clusters.append(merged_cluster) # Adiciona de volta à lista
                # E continua o loop para encontrar o próximo melhor merge
            else:
                # Não há mais merges possíveis acima do limite
                break 

        # 4. Coleta os grupos finais
        final_groups: list[Set[str]] = [c['keys'] for c in clusters]
        
        # Adiciona as chaves que não tinham snippets como seu próprio grupo
        if keys_without_snippets:
            final_groups.append(keys_without_snippets)
        
        print(f"  -> Grupos de extração formados: {[list(g) for g in final_groups]}")
        # groups = [{key} for key in extraction_schema.keys()]
        # merged_groups = []
        # while groups:
        #     current_group = groups.pop(0)
        #     current_snippets = set().union(*(retrieved_snippets_by_key[key] for key in current_group))
        #     if not current_snippets: # Grupo de chaves sem contexto
        #         merged_groups.append(current_group)
        #         continue
        #
        #     merged_with_current = False
        #     remaining_groups = []
        #     for other_group in groups:
        #         other_snippets = set().union(*(retrieved_snippets_by_key[key] for key in other_group))
        #         if not other_snippets:
        #             remaining_groups.append(other_group)
        #             continue
        #
        #         if not current_snippets.isdisjoint(other_snippets):
        #             current_group.update(other_group)
        #             merged_with_current = True
        #         else:
        #             remaining_groups.append(other_group)
        #
        #     if merged_with_current:
        #         groups = [current_group] + remaining_groups
        #     else:
        #         merged_groups.append(current_group)
        #
        # print(f"  -> Grupos de extração formados: {[list(g) for g in merged_groups]}")

        # --- PASSO 4: EXTRAÇÃO (LLM EM PARALELO) ---
        print("  Iniciando Passo 4: Extração via LLM (Paralelizada)...")
        final_results = {}
        
        # Define um número máximo de chamadas simultâneas
        # para evitar ser bloqueado pela API (Rate Limit)
        MAX_CONCURRENT_CALLS = 5 
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CALLS) as executor:
            # Dicionário para mapear "futuros" de volta para seus grupos
            future_to_group = {}
            
            for key_group_set in final_groups:
                all_snippets = set().union(*(retrieved_snippets_by_key[key] for key in key_group_set))
                
                if not all_snippets:
                    print(f"  GRUPO {list(key_group_set)} não teve snippets. Retornando None.")
                    for key in key_group_set:
                        final_results[key] = None
                    continue

                lines_sorted = sorted([doc.lines[i] for i in all_snippets])
                context = "\n---\n".join([line.serialize_layout() for line in lines_sorted])
                schema_group = {key: extraction_schema[key] for key in key_group_set}
                
                print(f"  Submetendo chamada para o GRUPO: {list(key_group_set)}...")
                
                # Submete a tarefa para a pool de threads
                future = executor.submit(
                    find_values_from_layout,
                    self.client, 
                    context, 
                    schema_group
                )
                future_to_group[future] = key_group_set

            # Coleta os resultados à medida que ficam prontos
            for future in concurrent.futures.as_completed(future_to_group):
                key_group_set = future_to_group[future]
                try:
                    # Pega o resultado da thread
                    llm_values = future.result()
                    final_results.update(llm_values)
                    print(f"  -> Resultado recebido para o GRUPO: {list(key_group_set)}")
                except Exception as e:
                    print(f"ERRO: A thread para o grupo {list(key_group_set)} falhou: {e}")
                    for key in key_group_set:
                        final_results[key] = None # Garante 'None' em caso de falha

        end_time = time.time()
        print(f"--- Processamento Concluído em {end_time - start_time:.2f} segundos ---")
        
        for key in extraction_schema:
            if key not in final_results:
                final_results[key] = None
                
        return final_results

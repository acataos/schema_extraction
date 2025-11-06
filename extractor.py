import time
import re
from typing import Dict, Any, Set, List 
from collections import Counter, defaultdict 
import concurrent.futures
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from typing import Dict, Any

import config
from document import Document
from llm_service import find_values_from_layout
from thefuzz import fuzz
from text_utils import stopword_filter, calculate_stopwords, normalize_text

JACCARD_THRESHOLD = 0.6
FULL_SCHEMAS = dict() # armazena schemas completos para análises futuras

# Carregamos os modelos globalmente
try:
    EMBEDDING_MODEL = SentenceTransformer(
        config.EMBEDDING_MODEL_PATH, 
        device=config.DEVICE
    )
    print(f"Modelo de embedding carregado em: {config.DEVICE}")
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível carregar o SentenceTransformer: {e}")
    EMBEDDING_MODEL = None

class Extractor:
    def __init__(self):
        # print("Inicializando Extractor...")
        
        try:
            self.client = OpenAI()
        except Exception as e:
            print(f"ERRO: Não foi possível inicializar o cliente OpenAI: {e}")
            self.client = None
            
        self.embedding_model = EMBEDDING_MODEL
        if not self.embedding_model:
            print("AVISO: Extractor inicializado sem modelo de embedding.")

        # print("Extrator pronto.")


    def _parse_schema(self, extraction_schema: dict) -> (Dict[str, List[str]], Dict[str, List[str]], Dict[str, str]):
        """
        Analisa o schema UMA VEZ para criar três mapas:
        1. category_map: {key -> [cat1, cat2]}
        2. global_cat_to_keys_map: {cat_norm -> [key1, key2]} (para ambiguidade)
        3. pattern_map: {key -> pattern_type} (ex: {"data_nasc": "date"})
        """
        # print("  Analisando schema em busca de campos Categóricos e de Padrão...")
        category_map = {}
        global_cat_to_keys_map = defaultdict(list)
        pattern_map = {} # <-- NOSSO NOVO MAPA
        
        # Padrão RegEx para Categorias (ex: "pode ser A, B ou C")
        cat_pattern = re.compile(
            r"(?:pode ser (?:feita )?por|é efetuado por|pode ser por|pode ser)\s+(.*?)(?:\.|$)", 
            re.IGNORECASE
        )
        
        # Mapeamento de palavras-chave para tipos de padrão
        # (A ordem importa: "cnpj" deve vir antes de "id")
        PATTERN_KEYWORDS = {
            "cpf": "cpf",
            "cnpj": "cnpj",
            "cnh": "cnh",
            "habilitacao": "cnh",
            "cns": "cns",
            "pis": "pis",
            "identidade": "rg",
            "rg": "rg",
            "renavam": "renavam",
            "titulo_eleitoral": "titulo_eleitoral",
            "telefone": "phone",
            "celular": "phone",
            "data": "date",
            "valor": "money",
            "dinheiro": "money",
            "quantia": "money",
            "saldo": "money",
            # "quantidade": "quantity",
            "endereco": "cep",
            "logradouro": "cep",
            "contrato": "id",
            "numero": "id",
            "codigo": "id",
        }

        for key, description in extraction_schema.items():
            norm_desc = normalize_text(description)
            
            # 1. Tenta encontrar Categorias
            cat_match = cat_pattern.search(description)
            if cat_match:
                category_string = cat_match.group(1)
                categories = [
                    cat.strip() for cat in 
                    re.split(r',|\s+ou\s+', category_string) 
                    if cat.strip()
                ]
                
                if categories:
                    # print(f"    - Chave categórica encontrada: '{key}' -> {categories}")
                    category_map[key] = categories
                    for cat in categories:
                        norm_cat = normalize_text(cat)
                        if key not in global_cat_to_keys_map[norm_cat]:
                            global_cat_to_keys_map[norm_cat].append(key)
                
            # 2. Tenta encontrar Padrões (RegEx)
            # (Não executa se já for categórico, para evitar confusão)
            if key not in category_map:
                for keyword, pattern_type in PATTERN_KEYWORDS.items():
                    if keyword in norm_desc or keyword in normalize_text(key):
                        # print(f"    - Chave de padrão encontrada: '{key}' -> tipo '{pattern_type}'")
                        pattern_map[key] = pattern_type
                        break # Pára no primeiro padrão encontrado
        
        return category_map, dict(global_cat_to_keys_map), pattern_map

    def _calculate_jaccard(self, set1: frozenset[str], set2: frozenset[str]) -> float:
        """Calcula a Similaridade Jaccard entre dois conjuntos de snippets."""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 1.0  # Dois conjuntos vazios são idênticos
        
        return intersection / union

    def _get_cached_embedding(self, text: str) -> Any:
        """
        Gera ou recupera um embedding do cache para um determinado texto (descrição).
        """
        if text in self.description_embedding_cache:
            return self.description_embedding_cache[text]
        
        if not self.embedding_model:
            return None # Não pode gerar embeddings

        # Gera, armazena no cache e retorna
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=True)
            self.description_embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"AVISO: Falha ao gerar embedding para o texto: {text}. Erro: {e}")
            return None

    def _get_key_similarity_map(self, full_schema: dict, 
                                name_threshold: int = 80, 
                                desc_threshold: float = 0.7,
                                stopwords: Set[str] = None) -> Dict[str, Set[str]]:
        """
        Compara todas as chaves contra si mesmas usando uma abordagem HÍBRIDA:
        1. Similaridade Sintática (fuzz) nos NOMES das chaves.
        2. Similaridade Semântica (embeddings) nas DESCRIÇÕES.
        """
        print("  Construindo mapa de similaridade de chaves (Híbrido)...")
        
        key_to_similar_keys = defaultdict(set)
        
        # 1. Pré-processa todos os nomes e embeddings (para eficiência N*N)
        key_to_norm_name = {}
        key_to_embedding = {}
        
        for key, desc in full_schema.items():
            key_to_norm_name[key] = normalize_text(key.replace("_", " "))
            clean_desc = stopword_filter(desc, stopwords)
            key_to_embedding[key] = self._get_cached_embedding(clean_desc)

        keys_list = list(full_schema.keys())

        # 2. Compara todos os pares
        for i in range(len(keys_list)):
            for j in range(i + 1, len(keys_list)):
                key_a = keys_list[i]
                key_b = keys_list[j]
                
                is_similar = False # Flag de similaridade

                # --- Teste 1: Similaridade de Nome (Sintática) ---
                name_score = fuzz.ratio(key_to_norm_name[key_a], key_to_norm_name[key_b])
                if name_score >= name_threshold:
                    is_similar = True

                # --- Teste 2: Similaridade de Descrição (Semântica) ---
                if not is_similar: # Só executa se o Teste 1 falhar
                    emb_a = key_to_embedding[key_a]
                    emb_b = key_to_embedding[key_b]
                    
                    if emb_a is not None and emb_b is not None:
                        try:
                            desc_score = util.cos_sim(emb_a, emb_b).item()
                            if desc_score >= desc_threshold:
                                is_similar = True
                        except Exception as e:
                            print(f"AVISO: Falha no cos_sim entre '{key_a}' e '{key_b}': {e}")
                
                # --- Adiciona ao mapa se UMA das condições for verdadeira ---
                if is_similar:
                    key_to_similar_keys[key_a].add(key_b)
                    key_to_similar_keys[key_b].add(key_a)
                    
        return key_to_similar_keys

    def extract(self, label: str, extraction_schema: dict, fitz_page_dict: dict) -> dict:
        """
        Pipeline: Fast Path Categórico Seguro, seguido por
        Retrieval, Grouping e Extração LLM como fallback.
        """
        # print(f"\n--- Processando Label: {label} ---")
        start_time = time.time()
        
        # Mantém o cache de schema global
        FULL_SCHEMAS[label] = {**FULL_SCHEMAS.get(label, {}), **extraction_schema}
        
        if not self.embedding_model:
            print("ERRO: Modelo de embedding não carregado. Abortando.")
            return {key: None for key in extraction_schema}

        self.description_embedding_cache = {}
            
        try:
            doc = Document(fitz_page_dict)
            doc.embed_spans(self.embedding_model)
            doc.find_tables()
        except ValueError as e:
            print(e)
            return {key: None for key in extraction_schema}


        # --- PASSO 1: ANÁLISE DO SCHEMA ---
        # (Substitui sua lógica de stopwords)
        descs = list(FULL_SCHEMAS[label].values())
        all_keys = set(FULL_SCHEMAS[label].keys())
        missing_keys = all_keys - set(extraction_schema.keys())
        stopwords = calculate_stopwords(descs, threshold=0.5) # Sua função
        category_map, global_cat_to_keys_map, pattern_map = self._parse_schema(FULL_SCHEMAS[label])
        key_similarity_map = self._get_key_similarity_map(FULL_SCHEMAS[label], stopwords=stopwords)
        print("  Mapa de similaridade de chaves construído.")
        print(f"    Chaves faltantes: {missing_keys}")
        print(" key_similarity_map:", {k: list(v) for k,v in key_similarity_map.items() if v})
        breakpoint()

        final_results = {}
        keys_for_llm = [] # Chaves que falharam no fast-path

        # --- PASSO 2: FAST PATH CATEGÓRICO (COM LÓGICA DE SEGURANÇA) ---
        # print("  Iniciando Passo 2: Fast Path Categórico Seguro...")
        for key, description in extraction_schema.items():
            if key in category_map:
                categories = category_map[key]
                found_values_set = doc.search_for_categories(categories)
                
                # 1. Condição de Exclusividade Intra-Campo:
                if len(found_values_set) == 1:
                    found_value = list(found_values_set)[0]
                    norm_found_value = normalize_text(found_value)
                    
                    # 2. Condição de Exclusividade Inter-Campo:
                    keys_sharing_this_category = global_cat_to_keys_map.get(norm_found_value, [])
                    
                    if len(keys_sharing_this_category) == 1 and keys_sharing_this_category[0] == key:
                        # SUCESSO! É inequívoco.
                        final_results[key] = found_value
                        # print(f"    [FAST PATH HIT] Chave '{key}' resolvida localmente (Inequívoca): '{found_value}'")
                        continue # Pula para a próxima chave
                    else:
                        pass
                        # print(f"    [FAST PATH FAIL] Chave '{key}' ambígua. Categoria '{found_value}' é compartilhada por {keys_sharing_this_category}.")
                
                elif len(found_values_set) > 1:
                    pass
                    # print(f"    [FAST PATH FAIL] Chave '{key}' ambígua. Múltiplos valores encontrados: {found_values_set}")
                # achou nada => nao tem
                else:
                    final_results[key] = None
                    # print(f"    [FAST PATH FAIL] Chave '{key}' não encontrada. Assumimos que seu valor é null.")
                    continue

            # Se qualquer falha ocorrer, ou se não for categórica,
            # a chave é adicionada para o pipeline do LLM.
            keys_for_llm.append(key)
        
        # Se não houver chaves restantes, podemos pular tudo
        if not keys_for_llm:
            end_time = time.time()
            # print(f"--- Processamento Concluído (APENAS FAST PATH) em {end_time - start_time:.2f} segundos ---")
            # Garante que todas as chaves solicitadas estejam presentes
            for key in extraction_schema:
                if key not in final_results: final_results[key] = None
            return final_results


        # --- PASSO 3: RECUPERAÇÃO (RETRIEVAL) - Apenas para chaves do LLM ---
        # print(f"  Iniciando Passo 3: Recuperação de Âncoras (para {len(keys_for_llm)} chaves)...")
        retrieved_anchors_by_key: Dict[str, set[Line]] = {}
        
        for key in keys_for_llm:
            description = extraction_schema[key]
            clean_description = stopword_filter(description, stopwords) # Sua função
            
            # 3a. Keyword Retrieval (retorna set[Line])
            kw_lines = doc.get_keyword_matching_lines(key)
            
            # 3b. Positional Retrieval (retorna set[Line])
            pos_lines = doc.get_positional_matching_lines(description)
            
            # 3c. Categorical Retrieval (retorna set[Line])
            cat_lines = doc.get_categorical_snippets(category_map.get(key, []))

            # 3d. Pattern Retrieval (retorna set[Line])
            pattern_type = pattern_map.get(key)
            pat_lines = set()
            if pattern_type:
                pat_lines = doc.get_pattern_snippets(pattern_type)
            # print()
            # print("Chave:", key)
            # print("  Linhas por Padrão:", "\n".join([l.text for l in pat_lines]))
            # print()
            # breakpoint()
            
            # 3d. Semantic Retrieval (retorna set[Line])
            query = f"{key}: {clean_description}"
            k = max(0, config.SNIPPET_K_SEMANTIC - len(kw_lines) - len(pos_lines) - len(cat_lines) -len(pat_lines))
            sem_lines = doc.get_semantic_matching_lines(self.embedding_model, query, k=k)
            
            # 3e. Combina os objetos Line
            context_set = set(kw_lines).union(set(sem_lines)).union(set(pos_lines)).union(cat_lines).union(pat_lines)
            retrieved_anchors_by_key[key] = context_set

        # --- PASSO 4: INFLAR O CONTEXTO E AGRUPAR ---
        # print("  Iniciando Passo 4: Inflar Contexto e Agrupar Chaves...")
        retrieved_snippets_by_key: Dict[str, frozenset[int]] = {} # Armazena line INDICES
        
        for key, anchor_lines in retrieved_anchors_by_key.items():
            inflated_snippets = set()
            if not anchor_lines:
                retrieved_snippets_by_key[key] = frozenset()
                continue
            
            for line in anchor_lines:
                context_lines_indices = doc.get_context_for_line(line) # Retorna Set[int]
                inflated_snippets.update(context_lines_indices)
            
            retrieved_snippets_by_key[key] = frozenset(inflated_snippets)

        # --- PASSO 5: AGRUPAMENTO (BATCHING) ---
        # print("  Iniciando Passo 5: Agrupamento (Jaccard)...")
        clusters = []
        keys_without_snippets = set()
        for key in keys_for_llm: # Itera apenas nas chaves do LLM
            snippets = retrieved_snippets_by_key.get(key, frozenset())
            if not snippets:
                keys_without_snippets.add(key)
                continue
            clusters.append({'keys': {key}, 'snippets': snippets})

        # (Lógica de clustering Jaccard não muda)
        while True:
            best_score = -1
            best_pair_to_merge = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    c1, c2 = clusters[i], clusters[j]
                    score = self._calculate_jaccard(c1['snippets'], c2['snippets'])
                    if score > best_score:
                        best_score = score
                        best_pair_to_merge = (i, j)
            
            if best_pair_to_merge and best_score >= JACCARD_THRESHOLD:
                i, j = best_pair_to_merge
                c2 = clusters.pop(max(i, j))
                c1 = clusters.pop(min(i, j))
                merged_cluster = {
                    'keys': c1['keys'].union(c2['keys']),
                    'snippets': c1['snippets'].union(c2['snippets'])
                }
                clusters.append(merged_cluster)
            else:
                break 

        key_groups: list[Set[str]] = [c['keys'] for c in clusters]
        if keys_without_snippets:
            key_groups.append(keys_without_snippets)
        # print(f"  -> Grupos de extração formados: {[list(g) for g in final_groups]}")
        merged_groups = []
        while key_groups:
            current_group = key_groups.pop(0)
            merged_with_current = False
            remaining_groups = []
            
            # Pega TODAS as chaves faltantes similares a QUALQUER chave neste grupo
            all_similar_keys_for_group = set()
            for key in current_group:
                similar_keys = key_similarity_map.get(key, set())
                keys_to_merge = similar_keys.intersection(missing_keys)
                all_similar_keys_for_group.update(keys_to_merge)
            
            merged_groups.append(current_group)
        
        final_groups: List[Set[str]] = merged_groups
        MAX_CONCURRENT_CALLS = 5 
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CALLS) as executor:
            future_to_group = {}
            
            for key_group_set in final_groups:
                all_snippets_indices = set().union(*(retrieved_snippets_by_key.get(key, set()) for key in key_group_set))
                
                if not all_snippets_indices:
                    for key in key_group_set:
                        final_results[key] = None
                    continue
                
                # Converte os índices de volta para objetos Line e serializa
                # lines_sorted = sorted([doc.lines[i] for i in all_snippets_indices], key=lambda l: l.idx)
                context = doc.serialize_snippets_for_llm(all_snippets_indices)
                
                schema_group = {key: extraction_schema[key] for key in key_group_set}
                # print(f"  Submetendo chamada para o GRUPO: {list(key_group_set)}...")
                
                excluded_schema_group = dict([(k,v) for k,v in FULL_SCHEMAS[label].items() if k not in key_group_set])
                future = executor.submit(
                    find_values_from_layout,
                    self.client, 
                    context, 
                    schema_group,
                    excluded_schema_group,
                )
                future_to_group[future] = key_group_set

            for future in concurrent.futures.as_completed(future_to_group):
                key_group_set = future_to_group[future]
                try:
                    llm_values = future.result()
                    final_results.update(llm_values)
                    # print(f"  -> Resultado recebido para o GRUPO: {list(key_group_set)}")
                except Exception as e:
                    # print(f"ERRO: A thread para o grupo {list(key_group_set)} falhou: {e}")
                    for key in key_group_set:
                        final_results[key] = None

        end_time = time.time()
        # print(f"--- Processamento Concluído em {end_time - start_time:.2f} segundos ---")
        
        # Garante que todas as chaves solicitadas tenham pelo menos um 'None'
        for key in extraction_schema:
            if key not in final_results:
                final_results[key] = None
        print("Resultados Finais:", final_results)
                
        return final_results

import time
import random
import re
import logging
import concurrent.futures
from typing import Dict, Any, Set, List, Tuple
from collections import defaultdict
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from thefuzz import fuzz
import json

import config
from document import Document
from llm_service import find_values_from_layout
from text_utils import stopword_filter, calculate_stopwords, normalize_text, is_isolated_word_in
from pattern_utils import PATTERN_KEYWORDS, STRONG_PATTERNS

# --- Constantes ---
JACCARD_THRESHOLD = 0.95
# Define um limiar de confiança alto para o Fast Path
FUZZY_MATCH_THRESHOLD = 90 

# Configura um logger para este módulo
logger = logging.getLogger(__name__)

# --- Carregamento de Modelos Globais ---
try:
    EMBEDDING_MODEL = SentenceTransformer(
        config.EMBEDDING_MODEL_PATH, 
        device=config.DEVICE
    )
    logger.info(f"Modelo de embedding carregado com sucesso em: {config.DEVICE}")
except Exception as e:
    logger.critical(f"NÃO FOI POSSÍVEL CARREGAR O SENTENCETRANSFORMER: {e}", exc_info=True)
    EMBEDDING_MODEL = None

# --- Classe Principal do Extractor ---
class Extractor:
    """
    Coordena o pipeline de extração de ponta a ponta, gerenciando
    o Fast Path, o retrieval de snippets, o agrupamento (batching) e as
    chamadas ao LLM.
    """
    
    def __init__(self):
        """Inicializa o cliente da API e os caches de estado."""
        try:
            self.client = OpenAI()
        except Exception as e:
            logger.error(f"Não foi possível inicializar o cliente OpenAI: {e}")
            self.client = None
            
        self.embedding_model = EMBEDDING_MODEL
        if not self.embedding_model:
            logger.warning("Extractor inicializado sem modelo de embedding.")

        # Cache de schema: armazena todos os schemas vistos por label
        self.full_schemas: dict[str, dict[str, str]] = defaultdict(dict)
        
        # Cache de embedding: persiste entre chamadas de .extract()
        self.description_embedding_cache: dict[str, Any] = {}
        

    def _parse_schema(self, 
                      extraction_schema: dict[str, str]
                     ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, str], dict[str, str]]:
        """
        Analisa o schema UMA VEZ para criar mapas de acesso rápido.
        
        Retorna:
            - category_map: {key -> [cat1, cat2]}
            - global_cat_to_keys_map: {cat_norm -> [key1, key2]} (para ambiguidade)
            - pattern_map: {key -> pattern_type} (todos os padrões)
            - strong_pattern_map: {key -> pattern_type} (apenas padrões fortes para o Fast Path)
        """
        category_map = {}
        global_cat_to_keys_map = defaultdict(list)
        pattern_map = {}
        strong_pattern_map = {}
        
        # Padrão RegEx para Categorias (ex: "pode ser A, B ou C")
        cat_pattern = re.compile(
            r"(?:pode ser (?:feita )?por|é efetuado por|pode ser por|pode ser)\s+(.*?)(?:\.|$)", 
            re.IGNORECASE
        )

        for key, description in extraction_schema.items():
            norm_desc = normalize_text(description)
            norm_key = normalize_text(key)
            
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
                    category_map[key] = categories
                    for cat in categories:
                        norm_cat = normalize_text(cat)
                        if key not in global_cat_to_keys_map[norm_cat]:
                            global_cat_to_keys_map[norm_cat].append(key)
            
            # 2. Tenta encontrar Padrões (RegEx)
            if key not in category_map:
                # Itera em ordem de prioridade (definida em pattern_utils.py)
                for keyword, pattern_type in PATTERN_KEYWORDS.items():
                    # Usa 'norm_key_desc' para busca combinada
                    norm_key_desc = norm_desc + " " + norm_key
                    
                    if keyword in norm_key_desc:
                        pattern_map[key] = pattern_type
                        
                        # Se também for um padrão forte, marca para o Fast Path
                        if pattern_type in STRONG_PATTERNS:
                             strong_pattern_map[key] = pattern_type
                        
                        break # Pára no primeiro padrão (mais específico) encontrado
        
        return category_map, dict(global_cat_to_keys_map), pattern_map, strong_pattern_map

    def _calculate_jaccard(self, set1: frozenset[int], set2: frozenset[int]) -> float:
        """Calcula a Similaridade Jaccard entre dois conjuntos de índices de snippet."""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 1.0  # Dois conjuntos vazios são idênticos
        
        return intersection / union

    def _get_cached_embedding(self, text: str) -> Any:
        """Gera ou recupera um embedding do cache (Torch tensor)."""
        if text in self.description_embedding_cache:
            return self.description_embedding_cache[text]
        
        if not self.embedding_model:
            return None 

        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=True)
            self.description_embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"Falha ao gerar embedding para o texto: {text}. Erro: {e}")
            return None

    def _get_key_similarity_map(self, 
                                full_schema: dict[str, str], 
                                stopwords: set[str],
                                name_threshold: int = 85, 
                                desc_threshold: float = 0.85
                               ) -> dict[str, set[str]]:
        """
        Compara todas as chaves (do schema completo) usando uma abordagem HÍBRIDA:
        1. Similaridade Sintática (fuzz) nos NOMES das chaves.
        2. Similaridade Semântica (embeddings) nas DESCRIÇÕES.
        """
        key_to_similar_keys = defaultdict(set)
        
        # 1. Pré-processa todos os nomes e embeddings
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
                is_similar = False

                for pattern in STRONG_PATTERNS:
                # Teste 0: Padrões Fortes Mútuos (lógica do seu script)
                    if (is_isolated_word_in(key_a, pattern) and 
                        is_isolated_word_in(key_b, pattern)):
                        is_similar = True
                        break
                if is_similar:
                    key_to_similar_keys[key_a].add(key_b)
                    key_to_similar_keys[key_b].add(key_a)
                    continue # Já é similar, vai para o próximo par

                # Teste 1: Similaridade de Nome (Sintática)
                name_score = fuzz.ratio(key_to_norm_name[key_a], key_to_norm_name[key_b])
                if name_score >= name_threshold:
                    is_similar = True

                # Teste 2: Similaridade de Descrição (Semântica)
                if not is_similar: 
                    emb_a = key_to_embedding[key_a]
                    emb_b = key_to_embedding[key_b]
                    
                    if emb_a is not None and emb_b is not None:
                        try:
                            desc_score = util.cos_sim(emb_a, emb_b).item()
                            if desc_score >= desc_threshold:
                                is_similar = True
                        except Exception as e:
                            logger.warning(f"Falha no cos_sim entre '{key_a}' e '{key_b}': {e}")
                
                if is_similar:
                    key_to_similar_keys[key_a].add(key_b)
                    key_to_similar_keys[key_b].add(key_a)
                    
        return key_to_similar_keys

    def extract(self, label: str, extraction_schema: dict[str, str], fitz_page_dict: dict) -> dict[str, Any]:
        """
        Executa o pipeline de extração completo para um único documento.
        
        Args:
            label: O "tipo" de documento (usado para cache de schema).
            extraction_schema: O schema {chave: desc} para *este job*.
            fitz_page_dict: Os dados de layout brutos da página.

        Returns:
            Um dicionário com os valores extraídos.
        """
        logger.info(f"--- Processando Label: {label} ---")
        start_time = time.time()
        
        # Atualiza o cache de schema global (agora em 'self')
        self.full_schemas[label] = {**self.full_schemas.get(label, {}), **extraction_schema}
        
        if not self.embedding_model:
            logger.error("ERRO: Modelo de embedding não carregado. Abortando.")
            return {key: None for key in extraction_schema}
            
        try:
            doc = Document(fitz_page_dict)
            doc.embed_spans(self.embedding_model)
            doc.find_tables()
        except ValueError as e:
            logger.error(f"Erro ao processar o Document: {e}")
            return {key: None for key in extraction_schema}



        # --- PASSO 1: ANÁLISE DO SCHEMA ---
        # (Substitui sua lógica de stopwords)
        descs = list(self.full_schemas[label].values())
        all_keys = set(self.full_schemas[label].keys())
        missing_keys = all_keys - set(extraction_schema.keys())
        stopwords = calculate_stopwords(descs, threshold=0.5) # Sua função
        category_map, global_cat_to_keys_map, pattern_map, strong_pattern_map = self._parse_schema(self.full_schemas[label])
        key_similarity_map = self._get_key_similarity_map(self.full_schemas[label], stopwords=stopwords)

        final_results = {}
        claimed_values_mask = set()

        unsolved_categorical_keys = set()
        unsolved_pattern_keys = set()

        for key in self.full_schemas[label].keys():
            if key in category_map:
                unsolved_categorical_keys.add(key)
            elif key in strong_pattern_map:
                unsolved_pattern_keys.add(key)

        solved_keys = set()

        found_last_time = True
        i = len(unsolved_pattern_keys)
        while found_last_time and len(unsolved_pattern_keys) and i>=0:
            found_last_time = False
            for key in unsolved_pattern_keys:
                pattern_type = strong_pattern_map[key]
                candidates = doc.pattern_matches_map.get(pattern_type, [])
                available_candidates = [c for c in candidates if c not in claimed_values_mask]

                if len(available_candidates) == 0:
                    if key in extraction_schema:
                        final_results[key] = None
                    solved_keys.add(key) 

                score_threshold = 90

                for candidate in candidates:
                    if candidate in claimed_values_mask:
                        continue

                    is_match = False
                    value_box, span = candidate
                    candidate_keys = []
                    normalized_key = normalize_text(key).replace("_", " ")
                    # search inside box 
                    normalized_box_text = normalize_text(value_box.text.replace("_", " "))
                    for i in range(span[0]):
                        candidate_key = normalized_box_text[i:i+len(normalized_key)]
                        score = fuzz.ratio(normalized_key, candidate_key)

                        if score > score_threshold:
                            is_match = True
                    # search left box
                    if not is_match:
                        left_box = doc.find_close_box(value_box, 'left')
                        above_box = doc.find_close_box(value_box, 'above')
                        neighbor_boxes = [left_box, above_box]
                        for box in neighbor_boxes:
                            if box is not None:
                                box_text = normalize_text(box.text.replace("_", " "))
                                score = fuzz.ratio(normalized_key, box_text)

                                if score > score_threshold:
                                    is_match = True
                    if is_match:
                        found_last_time = True
                        text = candidate[0].text
                        span = candidate[1]
                        if key in extraction_schema:
                            final_results[key] = text[span[0]:span[1]]
                        claimed_values_mask.add(candidate)
                        solved_keys.add(key) 

            unsolved_pattern_keys -= solved_keys    
            i -= 1

        all_categories = set()
        for key, categories in category_map.items():
            all_categories.update(categories)

        cat_to_loc_map = doc.search_for_categories(all_categories)
        # --- PASSO 2: FAST PATH CATEGÓRICO (COM LÓGICA DE SEGURANÇA) ---
        found_last_time = True
        i = len(unsolved_categorical_keys)
        while found_last_time and len(unsolved_categorical_keys) and i>=0:
            found_last_time = False
            for key in unsolved_categorical_keys:
                if key in category_map:
                    categories = category_map[key]
                    normalized_categories = [normalize_text(cat) for cat in categories]
                    non_empty_categories = [cat for cat in normalized_categories if len(cat_to_loc_map[cat])]
                    
                    # 1. Condição de Exclusividade Intra-Campo:
                    if len(non_empty_categories) == 1:
                        found_cat = non_empty_categories[0]
                        found_box, span = list(cat_to_loc_map[found_cat])[0]
                        found_value = found_box.text[span[0]:span[1]]
                        norm_found_value = normalize_text(found_value)
                        
                        # 2. Condição de Exclusividade Inter-Campo:
                        keys_sharing_this_category = set(global_cat_to_keys_map.get(norm_found_value, []))
                        unsolved_keys_sharing_this_category = keys_sharing_this_category.intersection(unsolved_categorical_keys)
                        
                        if len(unsolved_keys_sharing_this_category) >= len(cat_to_loc_map[found_cat]):
                            claimed_values_mask.add(cat_to_loc_map[found_cat].pop())
                            if key in extraction_schema:
                                final_results[key] = found_value
                            solved_keys.add(key)
                            found_last_time = True
                        else:
                            pass
                    
                    elif len(non_empty_categories) > 1:
                        # mais de uma categoria encontrada => ambíguo
                        pass
                    elif key in extraction_schema:
                        # achou nada => nao tem
                        solved_keys.add(key)
                        final_results[key] = None

            unsolved_categorical_keys -= solved_keys
            i -= 1
        # --- PASSO 2c: Limpeza Final ---
        # O que sobrou em 'unsolved_categorical_keys' vai para o LLM
        keys_for_llm = [key for key in extraction_schema if key not in final_results]

        doc.mask_boxes(claimed_values_mask)
        # --- FIM DO PASSO 2 ---
        
        if not keys_for_llm:
            end_time = time.time()
            for key in extraction_schema:
                if key not in final_results: final_results[key] = None
            return final_results


        # --- PASSO 3: RECUPERAÇÃO (RETRIEVAL) - Apenas para chaves do LLM ---
        retrieved_anchors_by_key: dict[str, set[Line]] = {}
        
        for key in keys_for_llm:
            description = extraction_schema[key]
            clean_description = stopword_filter(description, stopwords)
            
            kw_lines = doc.get_keyword_matching_lines(key)
            pos_lines = doc.get_positional_matching_lines(description)
            cat_lines = doc.get_categorical_snippets(category_map.get(key, []))

            pattern_type = pattern_map.get(key)
            pat_lines = set()
            if pattern_type:
                pat_lines = doc.get_pattern_snippets(pattern_type)
            
            query = f"{key}: {clean_description}"
            # Ajusta 'k' para evitar sobreposição
            k = max(0, config.SNIPPET_K_SEMANTIC - len(kw_lines) - len(pos_lines) - len(cat_lines) -len(pat_lines))
            sem_lines = set()
            if k > 0:
                sem_lines = doc.get_semantic_matching_lines(self.embedding_model, query, k=k)
            
            context_set = kw_lines.union(sem_lines).union(pos_lines).union(cat_lines).union(pat_lines)
            retrieved_anchors_by_key[key] = context_set

        # --- PASSO 4: INFLAR O CONTEXTO E AGRUPAR (JACCARD) ---
        retrieved_snippets_by_key: dict[str, frozenset[int]] = {}
        
        for key, anchor_lines in retrieved_anchors_by_key.items():
            inflated_snippets = set()
            if not anchor_lines:
                retrieved_snippets_by_key[key] = frozenset()
                continue
            
            for line in anchor_lines:
                context_lines_indices = doc.get_context_for_line(line)
                inflated_snippets.update(context_lines_indices)
            
            retrieved_snippets_by_key[key] = frozenset(inflated_snippets)

        # --- PASSO 5: AGRUPAMENTO (BATCHING + PROMOÇÃO DE AMBIGUIDADE) ---
        clusters = []
        keys_without_snippets = set()
        for key in keys_for_llm:
            snippets = retrieved_snippets_by_key.get(key, frozenset())
            if not snippets:
                keys_without_snippets.add(key)
                continue
            clusters.append({'keys': {key}, 'snippets': snippets})

        # 1. Clustering Jaccard
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

        # 2. Promoção de Ambiguidade (Sua Lógica de "missing_keys")
        key_groups: list[set[str]] = [c['keys'] for c in clusters]
        if keys_without_snippets:
            key_groups.append(keys_without_snippets)
        merged_groups = []
        while key_groups:
            current_group = key_groups.pop(0)
            merged_with_current = False
            remaining_groups = []
            # Pega TODAS as chaves faltantes similares a QUALQUER chave neste grupo
            all_similar_keys_for_group = set()
            for key in current_group:
                similar_keys = key_similarity_map.get(key, set())
                keys_to_merge = similar_keys.intersection(missing_keys - solved_keys)
                all_similar_keys_for_group.update(keys_to_merge)
            
            merged_groups.append(all_similar_keys_for_group.union(current_group))
            # merged_groups.append(current_group)
        
        final_groups: List[Set[str]] = merged_groups

        # --- PASSO 6: EXTRAÇÃO (LLM EM PARALELO) ---
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
                
                schema_group = {key: self.full_schemas[label][key] for key in key_group_set}
                
                future = executor.submit(
                    find_values_from_layout,
                    self.client, 
                    context, 
                    schema_group,
                )
                future_to_group[future] = key_group_set

            for future in concurrent.futures.as_completed(future_to_group):
                key_group_set = future_to_group[future]
                try:
                    llm_values = future.result()
                    final_results.update(llm_values)
                except Exception as e:
                    for key in key_group_set:
                        final_results[key] = None

        end_time = time.time()
        logger.info(f"--- Processamento Concluído em {end_time - start_time:.2f} segundos ---")
        
        # Garante que todas as chaves solicitadas estejam presentes
        for key in extraction_schema:
            if key not in final_results:
                final_results[key] = None
        for key in missing_keys:
            if key in final_results:
                del final_results[key]
        print("Resultados finais para label", label, ":", final_results)

        return final_results

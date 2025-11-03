import re
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

import config
from text_utils import normalize_text
from document import Document
from llm_service import call_llm_extractor

class Extractor:
    def __init__(self):
        """
        Inicializa o Extractor, carregando os modelos e o cache em memória.
        """
        print("Carregando modelos (isso acontece uma vez)...")
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        try:
            self.client = OpenAI() # Lê OPENAI_API_KEY do ambiente
        except Exception as e:
            print(f"ERRO: Não foi possível inicializar o cliente OpenAI: {e}")
            self.client = None
            
        self.embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME, 
            device=config.DEVICE
        )
        print(f"Modelos carregados. Usando device: {config.DEVICE}")

    def _apply_heuristic(self, text: str, heuristic: dict, key: str) -> str | None:
        """
        Tenta aplicar uma heurística (simples ou grupo) ao texto completo.
        """
        heuristic_type = heuristic.get("type")
        pattern = heuristic.get("pattern")
        if not pattern:
            return None

        normalized_text = normalize_text(text)
        
        try:
            # Lógica para heurística de chave única
            if heuristic_type == "regex":
                normalized_pattern = normalize_text(pattern)
                match = re.search(normalized_pattern, normalized_text, re.DOTALL | re.IGNORECASE)
                if match and match.group(1) is not None:
                    start, end = match.start(1), match.end(1)
                    return text[start:end].strip()

            # Lógica para heurística de grupo
            elif heuristic_type == "regex_group":
                mapping = heuristic.get("group_mapping", [])
                if key not in mapping:
                    return None # Heurística não se aplica a esta chave
                
                key_index = mapping.index(key)
                capture_group_index = key_index + 1 # Grupos de RegEx são 1-based
                
                normalized_pattern = normalize_text(pattern)
                match = re.search(normalized_pattern, normalized_text, re.DOTALL | re.IGNORECASE)
                
                if match and match.group(capture_group_index) is not None:
                    start, end = match.start(capture_group_index), match.end(capture_group_index)
                    return text[start:end].strip()
                    
        except re.error as e:
            print(f"ERRO: RegEx inválido na KB: {pattern}. Erro: {e}")
            
        return None

    def extract(self, label: str, extraction_schema: dict, pdf_text: str) -> dict:
        """
        Orquestra o pipeline de extração completo.
        """
        print(f"\n--- Processando Label: {label} ---")
        start_time = time.time()
        
        heuristics = self.cache.get(label, {})
        final_results = {}
        keys_for_llm = [] 

        # --- PASSO 1: HEURISTIC PASS (Rápido / Custo Zero) ---
        print("Iniciando Passo 1: Tentativa com Heurísticas (Cache)...")
        for key, description in extraction_schema.items():
            if key in heuristics:
                value = self._apply_heuristic(pdf_text, heuristics[key], key)
                if value:
                    final_results[key] = value
                    print(f"  [CACHE HIT] Chave '{key}' encontrada com RegEx.")
                else:
                    print(f"  [CACHE MISS] RegEx falhou para '{key}'. Enviando para LLM.")
                    keys_for_llm.append(key)
            else:
                print(f"  [CACHE MISS] Sem heurística para '{key}'. Enviando para LLM.")
                keys_for_llm.append(key)

        # --- PASSO 2: LLM FALLBACK (Lento / Pago) ---
        new_heuristics_learned = {}
        if keys_for_llm:
            print(f"\nIniciando Passo 2: Fallback para LLM (Campos: {keys_for_llm})...")
            
            try:
                # 2a. Pré-processa o documento (OOP)
                doc = Document(pdf_text, self.embedding_model)
            except ValueError as e:
                print(e)
                for key in keys_for_llm: final_results[key] = None
                return final_results

            # 2b. Retrieval Pass (Híbrido)
            retrieved_snippets_by_key = {}
            for key in keys_for_llm:
                description = extraction_schema[key]
                
                # Keyword Retrieval
                kw_lines, score = doc.get_keyword_line_numbers(key)

                # Semantic Retrieval (roda sempre)
                query = f"{key}: {description}"
                sem_chunk_indices = doc.get_semantic_chunk_indices(query, kw_lines)
                
                retrieved_snippets_by_key[key] = set(doc.semantic_chunks[i] for i in sem_chunk_indices)

            # 2c. Grouping Pass (Lógica de Grafo/Interseção)
            print("  Agrupando chaves por sobreposição de snippets...")
            groups = [{key} for key in keys_for_llm]
            merged_groups = []
            while groups:
                current_group = groups.pop(0)
                current_snippets = set().union(*(retrieved_snippets_by_key[key] for key in current_group))
                merged_with_current = False
                remaining_groups = []
                for other_group in groups:
                    other_snippets = set().union(*(retrieved_snippets_by_key[key] for key in other_group))
                    if not current_snippets.isdisjoint(other_snippets):
                        current_group.update(other_group)
                        merged_with_current = True
                    else:
                        remaining_groups.append(other_group)
                if merged_with_current:
                    groups = [current_group] + remaining_groups
                else:
                    merged_groups.append(current_group)
            
            print(f"  -> Grupos de extração formados: {[list(g) for g in merged_groups]}")

            # 2d. Extract Pass
            for key_group_set in merged_groups:
                all_snippets = set().union(*(retrieved_snippets_by_key[key] for key in key_group_set))
                all_snippets.update({doc.first_chunk})
                context = "\n---\n".join(sorted(all_snippets))
                schema_group = {key: extraction_schema[key] for key in key_group_set}
                
                print(f"  Chamando LLM para o GRUPO: {list(key_group_set)}...")
                llm_response = call_llm_extractor(self.client, context, schema_group)
                
                if llm_response.get("values"):
                    final_results.update(llm_response["values"])
                
                if llm_response.get("heuristic"):
                    heuristic = llm_response["heuristic"]
                    print(f"  [APRENDIZADO EM GRUPO] LLM gerou nova heurística para {list(key_group_set)}")
                    for key in key_group_set:
                        new_heuristics_learned[key] = heuristic

        # --- PASSO 3: Salvar Conhecimento (no cache em memória) ---
        if new_heuristics_learned:
            print("\nSalvando novas heurísticas aprendidas no cache da sessão...")
            if label not in self.cache:
                self.cache[label] = {}
            self.cache[label].update(new_heuristics_learned)
            
        end_time = time.time()
        print(f"--- Processamento Concluído em {end_time - start_time:.2f} segundos ---")
        
        # Garante que todas as chaves solicitadas tenham pelo menos um 'None'
        for key in extraction_schema:
            if key not in final_results:
                final_results[key] = None
                
        return final_results

import pypdf
import argparse
import json
import re
import sys
import os
import time
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. SETUP ---

# Path to knowledge base file
KB_PATH = "knowledge_base.json"

# Load embedding model ONCE at startup
print("Carregando modelo de embedding multilíngue (só uma vez)...")
EMBEDDING_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

try:
    client = OpenAI()
except Exception as e:
    print(f"AVISO: Não foi possível carregar o cliente OpenAI. Verifique sua API key. Erro: {e}")
    client = None

# LLM model name
LLM_MODEL_NAME = "gpt-5-mini" 

print("--- Setup concluído ---")


# --- 2. KNOWLEDGE BASE HANDLING ---

def load_knowledge_base(file_path: str) -> dict:
    """Carrega o JSON da base de conhecimento do disco."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"AVISO: {file_path} está corrompido. Começando do zero.")
            return {}
    return {}

def save_knowledge_base(file_path: str, data: dict):
    """Salva a base de conhecimento atualizada no disco."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"ERRO: Não foi possível salvar a Knowledge Base em {file_path}. Erro: {e}")


# --- 3. EXTRACTION HELPERS ---

def apply_heuristic(text: str, heuristic: dict) -> str | None:
    """Tenta aplicar uma heurística (RegEx) ao texto completo."""
    if heuristic.get("type") == "regex":
        pattern = heuristic.get("pattern")
        if not pattern:
            return None
        
        try:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match and match.group(1):
                return match.group(1).strip()
        except re.error as e:
            print(f"ERRO: RegEx inválido na KB: {pattern}. Erro: {e}")
            return None
    return None

def chunk_text(text: str) -> list[str]:
    """Divide o texto em chunks lógicos usando uma estratégia robusta."""
    # Esta é a estratégia "RecursiveCharacterTextSplitter" que discutimos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,      # Tamanho do chunk em caracteres.
        chunk_overlap=20,   # Sobreposição para não quebrar o contexto.
        separators=["\n\n", "\n", ". ", " ", ""] # Ordem de prioridade
    )
    return text_splitter.split_text(text)

def find_top_k_chunks(query: str, text_chunks: list[str], chunk_embeddings: torch.Tensor, k: int = 3) -> str:
    """Encontra os K chunks de texto mais relevantes para uma query."""
    query_embedding = EMBEDDING_MODEL.encode(query, convert_to_tensor=True, show_progress_bar=False)
    
    # Cosine similarity
    cos_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    
    # Fetch top k results
    top_results = torch.topk(cos_scores, k=min(k, len(text_chunks)))
    
    # Merge top k chunks into a single context string
    relevant_context = [text_chunks[idx] for idx in top_results[1]]
    return "\n---\n".join(relevant_context)

def call_llm_extractor(context: str, key: str, description: str) -> dict:
    """Chama o LLM (gpt-5 mini) para extrair o valor E a heurística RegEx."""
    
    if not client:
        print("ERRO: Cliente OpenAI não inicializado.")
        return {"value": None, "heuristic_regex": None}

    system_prompt = """
    Você é um assistente de extração de dados em português.
    Sua tarefa é extrair um valor específico de um trecho de texto E gerar um padrão RegEx em Python para encontrar esse valor no futuro.

    INSTRUÇÕES:
    1. Encontre o valor exato para a chave fornecida.
    2. Crie um padrão RegEx (Python) robusto para capturar este valor. Foque nas palavras-chave ao redor do valor.
    3. O RegEx deve ter **apenas um grupo de captura** `(...)` contendo o valor.
    4. Se o valor não for encontrado, retorne `null` para ambos os campos.
    5. Responda APENAS com um objeto JSON. Não inclua "```json" ou qualquer explicação.
    """
    
    human_prompt = f"""
    ---
    CONTEXTO (Apenas trechos relevantes do documento):
    """
    {context}
    """
    ---
    CAMPO PARA EXTRAIR:
    Chave: "{key}"
    Descrição: "{description}"
    ---
    OBJETO JSON DE RESPOSTA:
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"} # Força saída JSON
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        print(f"ERRO: Chamada à API OpenAI falhou para a chave '{key}'. Erro: {e}")
        return {"value": None, "heuristic_regex": None}

    # Process JSON response
    try:
        data = json.loads(response_text)
        return {
            "value": data.get("value"),
            "heuristic_regex": data.get("heuristic_regex")
        }
    except json.JSONDecodeError:
        print(f"ERRO: LLM retornou um JSON inválido para a chave '{key}': {response_text}")
        return {"value": None, "heuristic_regex": None}


# --- 4. MAIN ORCHESTRATOR ---

def extract_data(label: str, extraction_schema: dict, pdf_text: str) -> dict:
    """
    Orquestrador principal que executa o pipeline "Heuristic-first, LLM-second".
    """
    print(f"\n--- Processando Label: {label} ---")
    start_time = time.time()
    
    # Load knowledge base
    kb = load_knowledge_base(KB_PATH)
    heuristics = kb.get(label, {})
    
    final_results = {}
    keys_for_llm = [] # Keys where heuristics failed
    
    # --- STEP 1: HEURISTIC ATTEMPT (Fast / Free) ---
    print("Iniciando Passo 1: Tentativa com Heurísticas (Cache)...")
    for key, description in extraction_schema.items():
        if key in heuristics:
            value = apply_heuristic(pdf_text, heuristics[key])
            if value:
                final_results[key] = value
                print(f"  [CACHE HIT] Chave '{key}' encontrada com RegEx.")
            else:
                print(f"  [CACHE MISS] RegEx falhou para '{key}'. Enviando para LLM.")
                keys_for_llm.append(key)
        else:
            print(f"  [CACHE MISS] Sem heurística para '{key}'. Enviando para LLM.")
            keys_for_llm.append(key)

    # --- STEP 2: LLM FALLBACK (Expensive) ---
    new_heuristics_learned = {}
    if keys_for_llm:
        print(f"\nIniciando Passo 2: Fallback para LLM (Campos: {keys_for_llm})...")
        
        # Prepare embeddings for document chunks
        print("  Criando chunks e embeddings do documento...")
        text_chunks = chunk_text(pdf_text)
        chunk_embeddings = EMBEDDING_MODEL.encode(text_chunks, convert_to_tensor=True, show_progress_bar=False)
        
        for key in keys_for_llm:
            description = extraction_schema[key]
            
            # 1. Retrieve (Local)
            query = f"{key}: {description}"
            context = find_top_k_chunks(query, text_chunks, chunk_embeddings, k=3)
            breakpoint()
            
            # 2. Extract (API Call)
            print(f"  Chamando LLM para '{key}'...")
            llm_response = call_llm_extractor(context, key, description)
            
            value = llm_response.get("value")
            regex = llm_response.get("heuristic_regex")
            
            final_results[key] = value # Salva o valor (mesmo que seja None)
            
            # 3. Learn (Update KB)
            if regex:
                print(f"  [APRENDIZADO] LLM gerou novo RegEx para '{key}': {regex}")
                new_heuristics_learned[key] = {"type": "regex", "pattern": regex}

    # --- STEP 3: SAVE UPDATED KNOWLEDGE BASE ---
    if new_heuristics_learned:
        print("\nSalvando novas heurísticas aprendidas na Base de Conhecimento...")
        if label not in kb:
            kb[label] = {}
        kb[label].update(new_heuristics_learned)
        save_knowledge_base(KB_PATH, kb)
        
    end_time = time.time()
    print(f"--- Processamento Concluído em {end_time - start_time:.2f} segundos ---")
    
    return final_results


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extract fields from PDFs based on a JSON schema.")
    parser.add_argument("json_file", help="Path to the JSON file containing extraction instructions.")
    args = parser.parse_args()

    # read json file
    with open(args.json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    for item in json_data:
        pdf_file = "./data/files/" + item['pdf_path']
        reader = pypdf.PdfReader(pdf_file)
        # assume pdf only has one page
        pdf_text = reader.pages[0].extract_text()
        
        result = extract_data(
            item["label"],
            item["extraction_schema"],
            pdf_text
        )





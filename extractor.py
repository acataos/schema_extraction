import argparse
import pypdf
import json
import re
import time
import torch
import unicodedata
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter
from thefuzz import fuzz, process as fuzzy_process

# --- 1. SETUP E CONFIGURAÇÃO GLOBAL ---

# Nosso cache "por sessão", que atende à restrição do projeto
KNOWLEDGE_BASE_CACHE = {}

# Limiar de confiança para a busca por keyword
KEYWORD_CONFIDENCE_THRESHOLD = 80

print("Carregando modelo de embedding multilíngue (só uma vez)...")
# 'paraphrase-multilingual-MiniLM-L12-v2' é rápido e excelente para PT/EN
EMBEDDING_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Configuração do cliente OpenAI
try:
    client = OpenAI() # Lê a key do ambiente OPENAI_API_KEY
except Exception as e:
    print(f"AVISO: Não foi possível carregar o cliente OpenAI. Verifique sua API key. Erro: {e}")
    client = None

# Modelo de LLM especificado
LLM_MODEL_NAME = "gpt-5-mini" 

print("--- Setup concluído ---")


# --- 2. FUNÇÕES HELPER DE TEXTO E BUSCA ---

def normalize_text(s: str) -> str:
    """
    Removes diacritics using Unicode normalization and converts to lowercase.
    Ex: "Canção" -> "cancao"
    """
    if not s:
        return ""
    
    # 1. Decompose characters (e.g., 'ç' -> 'c' + '¸')
    s_nfd = unicodedata.normalize('NFD', s)
    
    # 2. Filter out the diacritic marks (category 'Mn' = Mark, Nonspacing)
    s_no_diacritics = "".join(c for c in s_nfd if unicodedata.category(c) != 'Mn')
    
    # 3. Return lowercase
    return s_no_diacritics.lower()

def apply_heuristic(text: str, heuristic: dict) -> str | None:
    """
    Tenta aplicar uma heurística (RegEx) ao texto completo, 
    ignorando acentos e caixa (case).
    """
    if heuristic.get("type") == "regex":
        pattern = heuristic.get("pattern")
        if not pattern:
            return None
        
        # 1. Normaliza o padrão RegEx
        normalized_pattern = normalize_text(pattern) 
        
        # 2. Normaliza o texto onde vamos buscar
        normalized_text = normalize_text(text)
        
        try:
            # 3. Busca no texto normalizado
            # re.DOTALL faz o '.' corresponder a novas linhas (para blocos)
            # re.IGNORECASE lida com o case (embora normalize_text já faça isso)
            match = re.search(normalized_pattern, normalized_text, re.DOTALL | re.IGNORECASE)
            
            if match and match.group(1) is not None:
                # 4. Pega os índices do GRUPO DE CAPTURA (1)
                start, end = match.start(1), match.end(1)
                
                # 5. Usa os índices para fatiar o TEXTO ORIGINAL
                # Isso garante que "Canção" seja retornado, e não "cancao"
                original_value = text[start:end]
                
                return original_value.strip()
                
        except re.error as e:
            print(f"ERRO: RegEx inválido na KB (após normalização): {normalized_pattern}. Erro: {e}")
            return None
            
    return None

def chunk_text(text: str) -> list[str]:
    """
    Divide o texto em chunks lógicos usando uma estratégia robusta
    que TENTA manter a integridade estrutural (como tabelas).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,      # Maior, para capturar tabelas/blocos lógicos
        chunk_overlap=20,   # Overlap generoso para "costurar" chunks
        separators=["\n\n", "\n", ". ", " ", ""], # A prioridade por '\n\n' é a chave
        length_function=len
    )
    return text_splitter.split_text(text)

def get_keyword_snippets_and_score(text: str, key: str, k: int = 3, snippet_size: int = 3) -> (list[str], int):
    """
    Encontra os k melhores snippets de texto usando fuzzy string matching
    E RETORNA o score máximo encontrado.
    """
    normalized_text = normalize_text(text)
    # Transforma 'telefone_profissional' em 'telefone profissional'
    normalized_key = key.replace("_", " ")
    
    snippets = []
    max_score = 0

    lines = normalized_text.splitlines()
    original_lines = text.splitlines()
    # Usamos 'extractBests' para ter o score
    matches = fuzzy_process.extractBests(normalized_key, lines, limit=k, score_cutoff=50)
    
    if not matches:
        return [], 0

    # Pega o score do melhor match
    max_score = matches[0][1] 
    
    for (line, score) in matches:
        try:
            # Encontra a linha no texto normalizado para pegar o índice
            line_index = lines.index(line)
            if line_index != -1:
                # 2. Pega uma "janela" (snippet) ao redor da correspondência
                # Usando o texto ORIGINAL para o snippet
                start = max(0, line_index - 1) # Um pouco antes
                end = min(len(text), line_index + snippet_size) # E depois
                snippets.append("/n".join(original_lines[start:end]))
        except:
            continue
                
    return snippets, max_score

def find_top_k_semantic_chunks(query: str, text_chunks: list[str], chunk_embeddings: torch.Tensor, k: int = 3) -> list[str]:
    """Encontra os K chunks de texto mais relevantes para uma query."""
    query_embedding = EMBEDDING_MODEL.encode(query, convert_to_tensor=True, show_progress_bar=False)
    
    cos_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    
    top_results = torch.topk(cos_scores, k=min(k, len(text_chunks)))
    
    # Retorna uma lista de chunks
    return [text_chunks[idx] for idx in top_results[1]]


# --- 3. FUNÇÃO PRINCIPAL DO LLM ---

def call_llm_extractor(context: str, key: str, description: str) -> dict:
    """Chama o LLM (gpt-5 mini) para extrair o valor E a heurística RegEx ADAPTATIVA."""
    
    if not client:
        print("ERRO: Cliente OpenAI não inicializado.")
        return {"value": None, "regex": None}

    # --- O PROMPT OTIMIZADO ---
    system_prompt = """
    Você é um assistente de extração de dados (em português) e um especialista em RegEx.
    Sua tarefa é extrair um valor de um trecho de texto E gerar um padrão RegEx em Python adaptativo para automatizar essa extração no futuro.

    INSTRUÇÕES:
    1.  **Encontre o Valor:** Localize o valor exato para a chave solicitada.
    2.  **Identifique Âncoras:** Encontre os **rótulos ou cabeçalhos (headers) estáticos** que aparecem imediatamente *antes* e *depois* do valor no texto. (Ex: "Endereço Profissional", "SITUAÇÃO", etc.)
    3.  **Crie o RegEx:**
        * Use esses **rótulos estáticos** como âncoras para o seu RegEx.
        * **NÃO** use outros *valores* (como "JOANA D'ARC" ou "CONSELHO SECCIONAL - PARANÁ") dentro do padrão RegEx, pois eles mudam entre documentos.
        * Use `(.*?)` (não-guloso) para o grupo de captura. Isso é vital para campos multilinha (como endereços) ou campos que podem estar vazios.
        * O RegEx deve ter **apenas um grupo de captura** `(...)`.
    4.  **Valor Não Encontrado:** Se o valor não for encontrado, retorne `null` para ambos os campos.
    5.  **Formato:** Responda APENAS com um objeto JSON com as chaves `value` e `regex`. Não inclua "```json" ou qualquer explicação.

    EXEMPLO DE LÓGICA:
    -   **Texto:** "Endereço Profissional\nRua X, 123\nTelefone Profissional"
    -   **Chave:** "endereco_profissional"
    -   **RegEx CORRETO:** "Endereço Profissional\\n(.*?)\\nTelefone Profissional"
    -   **RegEx INCORRETO:** "Endereço Profissional\\n(Rua X, 123)\\nTelefone Profissional"
    """
    
    human_prompt = f"""
    ---
    CONTEXTO (Apenas trechos relevantes do documento):
    {context}
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
            response_format={"type": "json_object"} # Força saída JSON
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        print(f"ERRO: Chamada à API OpenAI falhou para a chave '{key}'. Erro: {e}")
        return {"value": None, "regex": None}

    # Processa e valida a resposta 
    try:
        data = json.loads(response_text)
        
        regex = data.get("regex")
        if regex:
            try:
                # Valida se o RegEx é compilável
                re.compile(regex, re.DOTALL | re.IGNORECASE)
            except re.error as e:
                print(f"AVISO: LLM gerou RegEx inválido '{regex}' para a chave '{key}'. Descartando. Erro: {e}")
                regex = None 

        return {
            "value": data.get("value"),
            "regex": regex
        }
    except json.JSONDecodeError:
        print(f"ERRO: LLM retornou um JSON inválido para a chave '{key}': {response_text}")
        return {"value": None, "regex": None}


# --- 4. ORQUESTRADOR PRINCIPAL ---

def extract_data(label: str, extraction_schema: dict, pdf_text: str) -> dict:
    """
    Orquestrador principal que executa o pipeline "Heuristic-first, LLM-second".
    Usa cache em memória e recuperação híbrida dinâmica.
    """
    print(f"\n--- Processando Label: {label} ---")
    start_time = time.time()
    
    # 1. Lê do cache global em memória
    heuristics = KNOWLEDGE_BASE_CACHE.get(label, {})
    
    final_results = {}
    keys_for_llm = [] 
    
    # --- PASSO 1: HEURISTIC PASS (Rápido / Custo Zero) ---
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

    # --- PASSO 2: LLM FALLBACK (Lento / Pago) ---
    new_heuristics_learned = {}
    if keys_for_llm:
        print(f"\nIniciando Passo 2: Fallback para LLM (Campos: {keys_for_llm})...")
        
        # Prepara o "Retrieve": Chunking e Embedding
        print("  Criando chunks...")
        text_chunks = chunk_text(pdf_text)
        
        if not text_chunks:
            print("ERRO: O documento está vazio ou não pôde ser dividido em chunks.")
            for key in keys_for_llm: final_results[key] = None
            return final_results
            
        # 1. Separe o primeiro chunk (Heurística Posicional)
        first_chunk = text_chunks[0]
        
        # 2. Crie embeddings APENAS para os outros chunks
        other_chunks = text_chunks[1:]
        chunk_embeddings = None
        if other_chunks: 
            print(f"  Criando embeddings para {len(other_chunks)} chunks semânticos (só serão usados se necessário)...")
            chunk_embeddings = EMBEDDING_MODEL.encode(other_chunks, convert_to_tensor=True, show_progress_bar=False)
        
        for key in keys_for_llm:
            description = extraction_schema[key]
            
            # --- LÓGICA DE RECUPERAÇÃO HÍBRIDA DINÂMICA ---
            
            
            # 1. Sempre tente a Recuperação por Keyword primeiro
            print(f"  Buscando por keyword: '{key}'")
            keyword_snippets, max_score = get_keyword_snippets_and_score(pdf_text, key, k=3, snippet_size=3)

            # 2. "FORK" DA LÓGICA
            if max_score >= KEYWORD_CONFIDENCE_THRESHOLD:
                # PATH A: ALTA CONFIANÇA (LABEL ENCONTRADO)
                print(f"  -> Keyword HIT (Score: {max_score}). Usando snippets de keyword. PULANDO busca semântica.")
                context_snippets = keyword_snippets
            
            else:
                # PATH B: BAIXA CONFIANÇA (LABEL NÃO ENCONTRADO)
                # contexto posicional
                context_snippets = {first_chunk}
                print(f"  -> Keyword MISS (Score: {max_score}). Ignorando snippets de keyword. USANDO busca semântica.")
                if chunk_embeddings is not None and other_chunks:
                    query = f"{key}: {description}"
                    semantic_snippets = find_top_k_semantic_chunks(query, other_chunks, chunk_embeddings, k=3)
                    context_snippets.update(semantic_snippets)
                else:
                    print("  -> AVISO: Busca semântica pulada (sem chunks/embeddings).")

            # 4. Combine Contexts
            context = "\n---\n".join(context_snippets)
            
            # 5. Extract (API Call)
            print(f"  Chamando LLM para '{key}' (Contexto: {len(context)} caracteres)...")
            llm_response = call_llm_extractor(context, key, description)
            
            value = llm_response.get("value")
            regex = llm_response.get("regex")
            
            final_results[key] = value 
            
            if regex:
                print(f"  [APRENDIZADO] LLM gerou novo RegEx para '{key}': {regex}")
                new_heuristics_learned[key] = {"type": "regex", "pattern": regex}

    # --- PASSO 3: Salvar Conhecimento (AGORA EM MEMÓRIA) ---
    if new_heuristics_learned:
        print("\nSalvando novas heurísticas aprendidas no cache da sessão...")
        
        # Garante que o label exista no cache
        if label not in KNOWLEDGE_BASE_CACHE:
            KNOWLEDGE_BASE_CACHE[label] = {}
        
        # Atualiza o cache global
        KNOWLEDGE_BASE_CACHE[label].update(new_heuristics_learned)
        
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
    
    for item in json_data[:1]:
        pdf_file = "./data/files/" + item['pdf_path']
        reader = pypdf.PdfReader(pdf_file)
        # assume pdf only has one page
        pdf_text = reader.pages[0].extract_text()
        
        result = extract_data(
            item["label"],
            item["extraction_schema"],
            pdf_text
        )





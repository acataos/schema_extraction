import torch

# --- Model Settings ---
# O nome do seu modelo OpenAI
LLM_MODEL_NAME = "gpt-5-mini" 
# Modelo de embedding multilíngue
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- Retrieval Settings ---
# Limiar de confiança para a busca por keyword (0-100)
KEYWORD_CONFIDENCE_THRESHOLD = 80
# Quantos snippets de keyword/semântico recuperar
SNIPPET_K_KEYWORDS = 2
SNIPPET_K_SEMANTIC = 3
# Tamanho da "janela" de texto ao redor de uma keyword encontrada
SNIPPET_SIZE_KEYWORDS = 3

# --- Chunking Settings ---
# Otimizado para manter tabelas/blocos lógicos
CHUNK_SIZE = 80
CHUNK_OVERLAP = 20

# --- Global Device Setting ---
# Usa GPU se disponível, para acelerar os embeddings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

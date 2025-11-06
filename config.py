import torch

# --- Model Settings ---
LLM_MODEL_NAME = "gpt-5-mini"
# O Embedding Model só será usado para regras semânticas (fallback)
EMBEDDING_MODEL_PATH = 'sentence_transformer'

# --- Geometric/Rule Settings ---
# Quão perto (em pixels) duas caixas de texto precisam estar para
# serem consideradas "próximas"
PROXIMITY_THRESHOLD = 10
SEMANTIC_MATCH_THRESHOLD = 0.6
# Quantos pixels duas linhas podem variar em 'y' e ainda serem
# consideradas parte da mesma linha
ROW_TOLERANCE = 5

 # --- Snippet Retrieval Settings ---
SNIPPET_K_KEYWORDS = 2
SNIPPET_K_SEMANTIC = 0

# --- Global Device Setting ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

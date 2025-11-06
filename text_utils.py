import unicodedata
from collections import Counter
from thefuzz import fuzz

noise_chars = [",",".",":","'",'"']

CHAR_MAP = {
    "á": "a",
    "à": "a",
    "â": "a",
    "ã": "a",
    "ä": "a",
    "é": "e",
    "è": "e",
    "ê": "e",
    "ë": "e",
    "í": "i",
    "ì": "i",
    "î": "i",
    "ï": "i",
    "ó": "o",
    "ò": "o",
    "ô": "o",
    "õ": "o",
    "ö": "o",
    "ú": "u",
    "ù": "u",
    "û": "u",
    "ü": "u",
    "ç": "c",
    "Á": "A",
    "À": "A",
    "Â": "A",
    "Ã": "A",
    "Ä": "A",
    "É": "E",
    "È": "E",
    "Ê": "E",
    "Ë": "E",
    "Í": "I",
    "Ì": "I",
    "Î": "I",
    "Ï": "I",
    "Ó": "O",
    "Ò": "O",
    "Ô": "O",
    "Õ": "O",
    "Ö": "O",
    "Ú": "U",
    "Ù": "U",
    "Û": "U",
    "Ü": "U",
    "Ç": "C"
}

def normalize_text(s: str, noise_chars = None) -> str:
    """
    Removes diacritics using Unicode normalization and converts to lowercase.
    Ex: "Canção" -> "cancao"
    """
    if not s:
        return ""
    

    s = ''.join(CHAR_MAP.get(c, c) for c in s)
    if noise_chars is not None and isinstance(noise_chars, list):
        s = "".join(c for c in s if c not in noise_chars)
    
    # 3. Return lowercase
    return s.lower()


def fuzzy_match(query: str, text: str) -> int:
    """Retorna um score de similaridade (0-100) para texto normalizado."""
    return fuzz.ratio(normalize_text(query), normalize_text(text))

def calculate_stopwords(texts: list[str], threshold: float = 0.5) -> set[str]:
        """
        Analisa todas as descrições para encontrar palavras comuns
        específicas deste schema (ex: "profissional").
        """
        print("  Calculando stopwords específicas do schema...")
        if not texts:
            return set()

        word_counts = Counter()
        num_descriptions = len(texts)
        
        
        # Conta a frequência de palavras em *documentos* (descrições)
        for desc in texts:
            new_desc = normalize_text(desc, noise_chars)
            words_in_desc = set(new_desc.split())
            word_counts.update(words_in_desc)
            
        # Define o limiar (ex: 50% das descrições)
        min_doc_frequency = int(num_descriptions * threshold)
        if min_doc_frequency == 0:
            min_doc_frequency = 1
            
        # Palavras de parada são aquelas que aparecem em muitas descrições
        stopwords = {word for word, count in word_counts.items() if count >= min_doc_frequency and len(word) > 2}
        
        return stopwords

def stopword_filter(text: str, stopwords: set[str]) -> str:
    """Remove as stopwords específicas do schema de um texto."""
    if not text or not stopwords:
        return text
    
    words = text.split()
    filtered_words = [word for word in words if normalize_text(word, noise_chars) not in stopwords]
    
    return ' '.join(filtered_words)

def is_isolated_word_in(text: str, word: str) -> bool:
    """Verifica se 'word' aparece como palavra isolada em 'text'."""
    if not text or not word:
        return False
    
    normalized_text = normalize_text(text)
    normalized_word = normalize_text(word)
    normalized_text = normalized_text.replace("_"," ")
    normalized_word = normalized_word.replace("_"," ")
    
    words = normalized_text.split()
    return normalized_word in words

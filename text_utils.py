import unicodedata
from collections import Counter
from thefuzz import fuzz

noise_chars = [",",".",":","'",'"']

def normalize_text(s: str, noise_chars = []) -> str:
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
    s_no_diacritics = "".join(c for c in s_no_diacritics if c not in noise_chars)
    
    # 3. Return lowercase
    return s_no_diacritics.lower()


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

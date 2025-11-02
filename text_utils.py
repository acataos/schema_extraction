import unicodedata

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


from validate_docbr import CPF, CNH, CNPJ, CNS, PIS, TituloEleitoral, RENAVAM
import re

def clean_doc_id(doc_id: str) -> str:
    """Remove toda a pontuação de uma string de documento."""
    return re.sub(r'[.\-/]', '', doc_id)

def validate_cpf(cpf: str) -> bool:
    """Valida um CPF (limpo ou formatado) pelo dígito verificador."""
    if "*" in cpf:
        return True
    return CPF().validate(cpf)

def validate_cnpj(cnpj: str) -> bool:
    """Valida um CNPJ (limpo ou formatado) pelo dígito verificador."""
    if "*" in cnpj:
        return True
    return CNPJ().validate(cnpj)

def validate_cnh(cnh: str) -> bool:
    """Valida um CNH (limpo ou formatado) pelo dígito verificador."""
    if "*" in cnh:
        return True
    return CNH().validate(cnh)

def validate_cns(cns: str) -> bool:
    """Valida um CNS (limpo ou formatado) pelo dígito verificador."""
    if "*" in cns:
        return True
    return CNS().validate(cns)

def validate_renavam(renavam: str) -> bool:
    """Valida um RENAVAM (limpo ou formatado) pelo dígito verificador."""
    if "*" in renavam:
        return True
    return RENAVAM().validate(renavam)

def validate_pis(pis: str) -> bool:
    """Valida um RENAVAM (limpo ou formatado) pelo dígito verificador."""
    if "*" in pis:
        return True
    return PIS().validate(pis)

def validate_titulo_eleitoral(titulo_eleitoral: str) -> bool:
    """Valida um Titulo Eleitoral (limpo ou formatado) pelo dígito verificador."""
    if "*" in titulo_eleitoral:
        return True
    return TituloEleitoral().validate(titulo_eleitoral)

def validate_other_id(id: str):
    # first assert that its not a match for any other pattern
    for (pattern_type,pattern) in PATTERNS.items():
        if pattern_type == "other_id":
            continue
        is_veritable = pattern_type in VALIDATORS
        if re.fullmatch(pattern, id) and not is_veritable:
            return False
        elif re.fullmatch(pattern, id) and VALIDATORS[pattern_type](_clean_doc_id(id)):
            return False
    return True

PATTERNS = {
    "cpf": r"\b([\d*]{11}|[\d*]{3}\.[\d*]{3}\.[\d*]{3}-[\d*]{2})\b",
    "cnpj": r"\b([\d*]{14}|[\d*]{2}\.[\d*]{3}\.[\d*]{3}/[\d*]{4}-[\d*]{2})\b",
    "money": r"(?:R\$\s*)?\b[\d*]{1,3}(?:\.[\d*]{3})*,[\d*]{2}\b",
    "phone": r"\b(\([\d*]{2}\)\s*[\d*]{4,5}-[\d*]{4}|\([\d*]{2}\)\s*[\d*]{8,9}|([\d*]{2}\s*)?[\d*]{4,5}-[\d*]{4}|[\d*]{8,11})\b",
    "date": r"\b[\d*]{1,2}[/\-.][\d*]{1,2}[/\-.][\d*]{2,4}\b",
    "cep": r"\b([\d*]{8}|[\d*]{5}-[\d*]{3})\b",
    "cnh": r"\b[\d*]{11}\b",
    "pis": r"\b([\d*]{11}|[\d*]\.[\d*]{3}\.[\d*]{3}\.[\d*]{2}-[\d*])\b",
    "titulo_eleitoral": r"\b[\d*]{12}\b",
    "cns": r"\b[\d*]{15}\b",
    "rg": r"\b([\d*]{1,2}\.[\d*]{3}\.[\d*]{3}-[[\d*]X]|[\d*]{7,9})\b",
    "renavam": r"\b([\d*]{9}|[\d*]{11})\b",
    "other_id": r"\b([\d*]{5,}|[\d*]+[./-][\d*]+(?:[./-][\d*]+)*)\b",
}

VALIDATORS = {
    "cpf": validate_cpf,
    "cnpj": validate_cnpj,
    "cnh": validate_cnh,
    "cns": validate_cns,
    "renavam": validate_renavam,
    "titulo_eleitoral": validate_titulo_eleitoral,
    "pis": validate_pis,
    # "cep": validate_cep,
    "other_id": validate_other_id,
}

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
    "cep": "cep",
    "endereco": "cep",
    "logradouro": "cep",
    "contrato": "id",
    "numero": "id",
    "codigo": "id",
}
STRONG_PATTERNS = {
    "cpf",
    "cnpj",
    "cnh",
    "data",
    "cns",
    "cep"
    "pis",
    "rg",
    "renavam",
    "titulo_eleitoral",
}

import json
from openai import OpenAI
import config
from typing import Dict

def find_values_from_layout(client: OpenAI, context: str, schema_group: Dict[str, str]) -> Dict[str, any]:
    """
    Chama o LLM para extrair um GRUPO de valores, dado um contexto de layout.
    
    Retorna um dicionário com os valores encontrados.
    """
    
    # 1. Constrói a parte do prompt que lista as chaves
    schema_prompt = ""
    for key, description in schema_group.items():
        schema_prompt += f"  - Chave: '{key}', Descrição: '{description}'\n"

    # 2. Constrói os prompts do sistema e do usuário
    system_prompt = """
    Você é um assistente de extração de dados (em português).
    Sua tarefa é analisar um conjunto de trechos de texto de um documento
    e extrair os valores para um grupo de chaves.

    INSTRUÇÕES:
    -   Analise os "TRECHOS DE CONTEXTO" fornecidos.
    -   Para cada chave no "GRUPO DE CAMPOS PARA EXTRAIR", encontre o valor
        correspondente no contexto.
    -   Se um valor não for encontrado, retorne null para essa chave.
    -   Responda APENAS com um objeto JSON contendo os valores extraídos.

    EXEMPLO DE RESPOSTA:
    {
      "chave_1": "valor_extraido_1",
      "chave_2": 123.45,
      "chave_3": null
    }
    """
    
    human_prompt = f"""
    ---
    TRECHOS DE CONTEXTO (linhas de um documento):
    {context}
    ---
    GRUPO DE CAMPOS PARA EXTRAIR:
    {schema_prompt}
    ---
    OBJETO JSON DE RESPOSTA (apenas valores):
    """

    # 3. Executa a chamada à API
    if not client:
        print("ERRO: Cliente OpenAI não inicializado.")
        return {key: None for key in schema_group}

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ],
            response_format={"type": "json_object"}
        )
        response_text = response.choices[0].message.content
        return json.loads(response_text)
        
    except Exception as e:
        print(f"ERRO: Chamada ou parsing do LLM falhou: {e}")
        # Retorna null para tudo no grupo em caso de falha
        return {key: None for key in schema_group}

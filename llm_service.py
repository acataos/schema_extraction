import re
import json
from openai import OpenAI
import config

def _build_prompts(context: str, schema_group: dict) -> (str, str):
    """Helper "privado" para construir os prompts corretos."""
    is_single_key = len(schema_group) == 1
    
    if is_single_key:
        key, description = next(iter(schema_group.items()))
        
        system_prompt = """
        Você é um assistente de extração de dados (em português) e um especialista em RegEx.
        Sua tarefa é extrair um valor de um trecho de texto E gerar um padrão RegEx em Python adaptativo para automatizar essa extração no futuro.
        INSTRUÇÕES:
        1.  **Encontre o Valor:** Localize o valor exato para a chave solicitada.
        2.  **Identifique Âncoras:** Encontre os **rótulos ou cabeçalhos (headers) estáticos** que aparecem imediatamente *antes* e *depois* do valor no texto.
        3.  **Crie o RegEx:**
            * Use esses **rótulos estáticos** como âncoras.
            * **NÃO** use outros *valores* (como "JOANA D'ARC") dentro do padrão RegEx.
            * Use `(.*?)` (não-guloso) para o grupo de captura.
            * O RegEx deve ter **apenas um grupo de captura** `(...)`.
        4.  **Valor Não Encontrado:** Se o valor não for encontrado, retorne `null` para ambos os campos.
        5.  **Formato:** Responda APENAS com um objeto JSON.
        EXEMPLO DE RESPOSTA (para Chave Única):
        {
          "value": "Valor Extraído",
          "heuristic": {
            "type": "regex",
            "pattern": "Header Antes\\n(.*?)\\nHeader Depois"
          }
        }
        """
        
        human_prompt = f"""
        ---
        CONTEXTO:
        {context}
        ---
        CAMPO PARA EXTRAIR:
        Chave: "{key}"
        Descrição: "{description}"
        ---
        OBJETO JSON DE RESPOSTA:
        """
    
    else: # CENÁRIO DE GRUPO (Múltiplas Chaves)
        schema_prompt = ""
        for key, description in schema_group.items():
            schema_prompt += f"  - Chave: '{key}', Descrição: '{description}'\n"

        system_prompt = """
        Você é um assistente de extração de dados e um especialista em RegEx.
        Sua tarefa é extrair um GRUPO de valores de um texto E gerar um ÚNICO padrão RegEx em Python com MÚLTIPLOS grupos de captura para automatizar isso.
        INSTRUÇÕES:
        1.  **Encontre os Valores:** Localize todos os valores para o grupo de chaves solicitado.
        2.  **Crie o RegEx:**
            * Crie UM RegEx que encontre a linha ou bloco que contém TODOS os valores.
            * Use âncoras de header (ex: "Inscrição Seccional Subseção").
            * O RegEx deve ter **exatamente um grupo de captura (...) para cada chave**, na ordem em que aparecem.
        3.  **Formato:** Responda APENAS com um objeto JSON.
        EXEMPLO DE RESPOSTA (para um grupo de 3 chaves):
        {
          "values": {
            "chave_1": "valor1",
            "chave_2": "valor2",
            "chave_3": "valor3"
          },
          "heuristic": {
            "type": "regex_group",
            "pattern": "Header Único:\\s*(\\w+)\\s*(\\w+)\\s*(\\w+)",
            "group_mapping": ["chave_1", "chave_2", "chave_3"]
          }
        }
        """
        
        human_prompt = f"""
        ---
        CONTEXTO:
        {context}
        ---
        GRUPO DE CAMPOS PARA EXTRAIR:
        {schema_prompt}
        ---
        OBJETO JSON DE RESPOSTA:
        """

    return system_prompt, human_prompt

def _validate_and_format_response(response_text: str, schema_group: dict) -> dict:
    """Helper "privado" para validar a heurística e formatar a resposta."""
    is_single_key = len(schema_group) == 1
    data = json.loads(response_text)
    heuristic = data.get("heuristic")

    if is_single_key:
        key = next(iter(schema_group.keys()))
        
        if heuristic and heuristic.get("type") == "regex":
            try:
                re.compile(heuristic.get("pattern"), re.DOTALL | re.IGNORECASE)
            except re.error as e:
                print(f"AVISO: LLM (chave única) gerou RegEx inválido '{heuristic.get('pattern')}'. Descartando. Erro: {e}")
                heuristic = None
        
        return {
            "values": {key: data.get("value")},
            "heuristic": heuristic
        }
        
    else: # CENÁRIO DE GRUPO
        if heuristic and heuristic.get("type") == "regex_group":
            pattern = heuristic.get("pattern")
            mapping = heuristic.get("group_mapping")
            
            try:
                num_groups = re.compile(pattern).groups
                if not pattern or not mapping or (len(mapping) != num_groups):
                    print(f"AVISO: LLM gerou RegEx de grupo inválido. Mapeamento ({len(mapping)}) e grupos ({num_groups}) não batem. Descartando.")
                    heuristic = None
                else:
                    re.compile(pattern, re.DOTALL | re.IGNORECASE)
            except re.error as e:
                print(f"AVISO: LLM (grupo) gerou RegEx inválido '{pattern}'. Descartando. Erro: {e}")
                heuristic = None
            
            data["heuristic"] = heuristic
            
        return data

def call_llm_extractor(client: OpenAI, context: str, schema_group: dict) -> dict:
    """
    Chama o LLM para extrair um grupo de valores e aprender uma heurística.
    Função unificada e refatorada.
    """
    
    # 1. Construir os prompts corretos
    system_prompt, human_prompt = _build_prompts(context, schema_group)
    
    # 2. Executar a chamada
    if not client:
        print("ERRO: Cliente OpenAI não inicializado.")
        return {"values": None, "heuristic": None}
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ],
            response_format={"type": "json_object"}
        )
        response_text_to_parse = response.choices[0].message.content
    except Exception as e:
        print(f"ERRO: Chamada à API falhou para {list(schema_group.keys())}. Erro: {e}")
        return {"values": None, "heuristic": None}
    
    # 3. Validar e Formatar a Resposta
    try:
        return _validate_and_format_response(response_text_to_parse, schema_group)
    except Exception as e:
        print(f"ERRO: LLM retornou JSON inválido ou falhou na validação: {e} | JSON: {response_text_to_parse}")
        return {"values": None, "heuristic": None}

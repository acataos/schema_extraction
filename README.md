# Extrator Híbrido de Documentos por IA
Este projeto é uma aplicação de pipeline de extração de dados, projetada para analisar documentos PDF (como faturas, contratos ou formulários, com OCR já realizado) e extrair dados estruturados em formato JSON.

O projeto é um pipeline híbrido que combina a velocidade e a precisão de regras de processamento locais com a capacidade de interpretacão de texto de uma LLM. A interface do projeto se dá por meio de uma GUI (``app.py``) ou, alternativamente, por uma CLI (``main.py``).

## Desafios Centrais e Soluções
Extrair campos específicos de documentos variados é uma tarefa relativamente direta quando se pode usar uma LLM. No entanto, se tentamos uma aplicação ingênua, fica imediatamente claro que não só a LLM pode ser lenta para realizar a extração simultânea de uma variedade de campos, como também pode cometer erros se ela apenas tiver acesso ao texto bruto do documento, visto que sua formatação e layout podem ter sido comprometidos pelo OCR. Assim, no desenvolvimento desse projeto, foi fundamental entender como podemos fazer o melhor trabalho possível para fornecer dados limpos e ricos em contexto para a LLM. Enumero abaixo os principais desafios enfrentados e as soluções implementadas, em ordem crescente de complexidade:

1. Desafio: Recuperação de Layout

Como eu já mencionei, o layout do documento é crucial para estabelecer a relação entre chaves e valores, ajudando a identificar "tabelas" (chamo de tabela qualquer grupo de linhas alinhadas que mantém relação semântica) e "headers", por exemplo. O texto bruto não preserva essa informação. Felizmente, conseguimos recuperar também caixas de texto e suas coordenadas, possibilitanto a reconstrução do layout. Isso foi implementado por meio de uma classe Document que interpreta o PDF e organiza o texto de forma hierárquica, com spans (caixas de texto), linhas (grupos de spans na mesma linha) e tabelas (grupos de linhas alinhadas verticalmente). Esse trabalho permite reformatar o texto de forma a otimizar a compreensão da LLM. Além disso, também contribui para a resolução de outros desafios, como a geração de contexto relevante para cada chave e a recuperação imediata de valores.


2. Desafio: Geração de Contexto Relevante

Nessa etapa, o objetivo é repensar o que queremos que a LLM veja, na tentativa de quebrar a chamada em alguns poucos prompts curtos e ricos (reduzindo custos e aumentando a velocidade e a precisão). Aqui utilizamos algumas estratégias diferentes:
- Seguir instruções posicionais nas descrições das chaves (ex: "no canto superior esquerdo"). Como temos as coordenadas dos spans, conseguimos encontrar essas instruções nas descrições e decidir contexto relevante com base nelas.
- Similaridade gramatical das chaves: Frequentemente, as próprias chaves apresentam dicas valiosas de onde encontrar seus valores. Por exemplo, podemos inferir que o valor da chave "cidade" provavelmente está próximo trecho "Cidade", se esse trecho existir no documento. Podemo utilizar fuzzy matching (thefuzz) para encontrar essas palavras-chave no documento e coletar spans próximos como contexto.
- Similaridade semântica do conjunto chave-descrição: Utilizamos embeddings (sentence-transformers) para encontrar spans semanticamente semelhantes às descrições das chaves. Isso é especialmente útil quando as descrições contêm sinônimos ou termos relacionados que não aparecem literalmente no documento. Empiricamente, esse tipo de similaridade parece ser muito sujeito a ruído, então damos menos prioridade a esse tipo de busca de contexto, embora ainda o utilizemos.
- Busca de padrões: Algumas chaves têm padrões bem definidos (ex: CPF, CNPJ, CEP, datas, valores monetários). Podemos usar RegEx para encontrar todos os valores que correspondem a esses padrões no documento e usá-los como contexto para essas chaves.
- Busca categórica: Os valores de algumas chaves pertencem a um conjunto limitado de categorias (ex: a categoria presente numa carteira da OAB pode ser advogado, advogada, suplementar, estagiário ou estagiária). Podemos procurar por esses valores no documento e usá-los como contexto para essas chaves.

Uma vez que coletamos alguns spans ou linhas relevantes, é importante capturar também o contexto em torno deles. Para isso, usamos fortemente a organização hierárquica do Document, entendendo que spans na mesma linha possivelmente possuem relação semântica, assim como linhas na mesma tabela (em particular se dois spans estiverem alinhados verticalmente). Além disso, podemos usar também as distâncias entre linhas e spans para identificar blocos de texto e agregar contexto adicional.

3. Desafio: Resolução de Ambiguidade

Uma vez que estabelecemos o contexto relevante para cada chave, poderíamos simplesmente enviar tudo para a LLM e pedir que ela extraísse os valores de forma independente. No entanto, isso impossibilita a LLM de resolver ambiguidades entre chaves. Por exemplo, se temos duas chaves referentes a datas, pode ser não-trivial para a LLM entender qual data pertence a qual chave. Em um caso ainda mais crítico, pode ser só exista uma data viável para duas chaves diferentes. Se enviássemos as chaves independentemente, a LLM provavelmente atribuiria o mesmo valor para ambas, incorrendo em um erro. Assim, uma solução mais robusta é agrupar chaves "similares" em lotes, agregando seu contexto e enviando-as juntas para a LLM. Dessa forma, a LLM pode comparar as chaves e seus contextos, resolvendo ambiguidades de forma mais eficaz. Podemos decidir a similaridade entre chaves computando a similaridade semântica entre suas descrições, verificando a similaridade entre seus contextos (similaridade de Jaccard), entre outros méotodos. Além disso, podemos também aplicar nosso conhecimento acumulado sobre o label relevante para mostrar à LLM também chaves ausentes do schema, a fim de ajudar na desambiguação.
Uma abordagem de "Fast Path" (caminho rápido) que apenas usa RegEx simples é rápida, mas falha miseravelmente. Ela não consegue lidar com ambiguidades (ex: dois CPFs em um documento) e envia trabalho desnecessário para o LLM.

4. Desafio: Recuperação Imediata de Valores

As estratégias enumeradas até aqui já reduzem significativamente o custo em termos de tokens, além de aumentar a precisão. No entanto, a velocidade ainda pode ser um problema. Em particular, se acabamos por agregar uma grande quantidade de chaves em poucos lotes, o tempo de resposta da LLM pode ser alto, por conta do aumento da complexidade da tarefa em relação a procurar apenas uma chave. Assim, um próximo passo natural é tentar resolver o máximo de chaves possível localmente, sem a necessidade de enviar para a LLM. Para isso, conseguimos empregar algumas estratégias:

- Recuperação categórica: Para chaves que possuem um conjunto limitado de valores possíveis, podemos procurar esses valores no documento e, se encontrarmos apenas um deles, sabemos que ele provavelmente será o valor correto para essa chave (na prática, checamos para garantir que nenhuma outra chave pode ter esse valor como categoria. Ainda assim, podemos incorrer em falsos positivos, mas assumimos que eles serão raros e que o saldo será positivo). Assim, imediatamente retornamos esse valor e removemos essa chave do conjunto de chaves que precisam ser enviadas para a LLM.
- Recuperação de padrões: Similarmente, para chaves que possuem padrões bem definidos, podemos procurar todos os valores que correspondem a esse padrão no documento. A identificação de padrões pode ser delicada, então usamos o layout para procurar a chave correspondente a spans próximos ao valor encontrado, retornando o valor a encontrarmos (fuzzy matching). Isso reduz drasticamente o número de chaves que precisam ser enviadas para a LLM.

Além de reduzir o número de chaves enviadas para a LLM, essas estratégias também ajudam a "mascarar" valores já encontrados, simplificando o contexto passado para outras chaves e reduzindo o ruído no prompt enviado para a LLM.

5. Desafio: Aprendizado contínuo (não resolvido)

Para obter o melhor resultado possível, seria ideal ter um sistema que aprende mais a cada etapa. Até agora, o sistema implementado utiliza o conhecimento acumulado de labels para ajudar na tomada de decisões por meio da LLM. Entretanto, esse aprendizado poderia ser expandido significativamente se pedíssemos para a LLM inferir regras ou RegEx que tornariam desnecssárias possíveis chamadas fururas para a LLM. Os experimentos tentados nessa direção não foram bem-sucedidos, ou violando restrições de tempo (chamadas com pedidos de inferência de RegEx são substancialmente mais lentas, violando a restrição de 10s) ou sendo pouco robustas e, consequentemente, inúteis. Entretanto, creio que o potencial dessa abordagem é grande e resultaria em um ganho significativo de desempenho ao longo do tempo.

## Instalação e Uso
Requisitos

O projeto requer Python 3.13 e as seguintes bibliotecas principais:

```
```
Bash
pip install -r requirements.txt
```
(Conteúdo de requirements.txt)
```

```
openai
sentence-transformers
torch
thefuzz
PyMuPDF
PyQt6
validate-docbr
...
```

Configuração

Defina a variável de ambiente OPENAI_API_KEY com sua chave de API da OpenAI.

```
export OPENAI_API_KEY="sua_chave_aqui"  # Linux/Mac
```

### Executando a Aplicação

#### CLI

Prepare sua Fila de Jobs: Crie um arquivo ``fila.json`` (ou qualquer nome) que contenha uma lista de jobs. O pdf_path deve ser relativo ao diretório onde o ``fila.json`` está.

JSON
```
[
    {
      "label": "carteira_oab",
      "extraction_schema": {
        "nome": "Nome do profissional, normalmente no canto superior esquerdo da imagem",
        "inscricao": "Número de inscrição do profissional",
        "seccional": "Seccional do profissional",
        "subsecao": "Subseção à qual o profissional faz parte",
        "categoria": "Categoria, pode ser ADVOGADO, ADVOGADA, SUPLEMENTAR, ESTAGIARIO, ESTAGIARIA",
        "endereco_profissional": "Endereço do profissional",
        "telefone_profissional": "Telefone do profissional",
        "situacao": "Situação do profissional, normalmente no canto inferior direito."
      },
      "pdf_path": "oab_1.pdf"
    },
    {
      "label": "carteira_oab",
      "extraction_schema": {
        "nome": "Nome do profissional, normalmente no canto superior esquerdo da imagem",
        "inscricao": "Número de inscrição do profissional",
        "seccional": "Seccional do profissional",
        "subsecao": "Subseção à qual o profissional faz parte",
        "categoria": "Categoria, pode ser ADVOGADO, ADVOGADA, SUPLEMENTAR, ESTAGIARIO, ESTAGIARIA",
        "endereco_profissional": "Endereço do profissional",
        "situacao": "Situação do profissional, normalmente no canto inferior direito."
      },
      "pdf_path": "oab_2.pdf"
    },
]
```

Chame na sua linha de comando (no ambiente onde as dependências foram instaladas):
```
python main.py /path/to/fila.json
```

Os resultados serão impressos no console e um arquivo ``output.json`` será gerado com os resultados estruturados na pasta ``data/output/``. Cuidado para não sobrescrever arquivos existentes.

#### GUI:

Basta chamar

```
python app.py
```

Use a Aplicação:

Clique em "Carregar Fila (.json)..." e selecione seu fila.json.

O "Diretório Base" será preenchido automaticamente. Você pode alterá-lo se os PDFs estiverem em outro lugar. Assumimos que todos os PDFs estão no mesmo diretório.

Clique em "Extrair Próximo" para processar um de cada vez, ou "Extrair Tudo" para processar em lote.

Clique em "Salvar Resultados..." para exportar o JSON final para o arquivo indicado.

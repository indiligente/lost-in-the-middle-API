# Protocolo experimental de *Lost in the Middle*

Este documento registra o desenho experimental descrito em *Lost in the
Middle: How Language Models Use Long Contexts* (Liu et al., TACL 2024,
DOI `10.1162/tacl_a_00638`) e o relaciona com este projeto. Ele deve ser a
fonte de verdade para a automacao dos experimentos.

Mapa de verificacao no PDF:

- QA controlado: paginas PDF 3-5, paginas impressas 159-161, Figuras 2-5;
- chave-valor: paginas PDF 6-7, paginas impressas 162-163, Figuras 6-7;
- arquitetura, consulta e instruction tuning: paginas PDF 7-9, Figuras 8-10;
- estudo retriever-reader: pagina PDF 10, pagina impressa 166, Figura 11;
- controles, GPT-4 e Llama-2: paginas PDF 15-17, Figuras 12-16.

## 1. Pergunta de pesquisa

O trabalho mede se modelos de linguagem usam de forma robusta informacao em
contextos longos. Os autores variam de forma controlada:

1. o tamanho do contexto;
2. a posicao da informacao relevante;
3. o modelo avaliado.

O resultado central e uma curva frequentemente em formato de U: o desempenho
tende a ser maior quando a informacao relevante esta no inicio ou no fim e
menor quando ela esta no meio.

O artigo usa duas tarefas controladas principais e um estudo complementar:

- QA com multiplos documentos;
- recuperacao sintetica de chave-valor;
- estudo retriever-reader de QA aberto.

## 2. Modelos avaliados

### 2.1. Experimentos principais

Os experimentos principais de QA e chave-valor usam **6 configuracoes de
modelo**, pertencentes a quatro familias:

| Configuracao | Tipo | Janela declarada |
| --- | --- | ---: |
| MPT-30B-Instruct | aberto, decoder-only | 8.192 tokens |
| LongChat-13B (16K) | aberto, decoder-only | 16.384 tokens |
| GPT-3.5-Turbo-0613 | fechado, OpenAI API | 4K tokens |
| GPT-3.5-Turbo-16K-0613 | fechado, OpenAI API | 16K tokens |
| Claude-1.3 | fechado, Anthropic API | 8K tokens |
| Claude-1.3 (100K) | fechado, Anthropic API | 100K tokens |

Os autores usam decodificacao gulosa e um conjunto padrao de prompts. Para as
versoes OpenAI, o artigo especifica a versao `0613`.

Nos scripts publicados para os modelos locais, isso corresponde a temperatura
`0`, `top_p = 1` e limite de `100` novos tokens. Tamanho de lote e numero de
GPUs sao detalhes operacionais e variam conforme o modelo e o hardware.

### 2.2. Modelos das analises adicionais

Outras secoes acrescentam:

- Flan-T5-XXL e Flan-UL2, para comparar encoder-decoder com decoder-only;
- MPT-30B base, para comparar com MPT-30B-Instruct;
- GPT-4-0613 (8K), em uma amostra de 500 casos QA;
- Llama-2 7B, 13B e 70B, cada um nas versoes base e chat.

Contando cada variante como uma configuracao separada, o artigo apresenta
**16 configuracoes unicas**: 6 principais, 2 Flan, 1 MPT base, 1 GPT-4 e 6
Llama-2. O numero 6 e o correto quando nos referimos somente ao conjunto
principal compartilhado entre QA e chave-valor.

## 3. QA com multiplos documentos

### 3.1. Dados

- Fonte: NaturalQuestions-Open.
- Amostra: **2.655 perguntas** cuja resposta longa anotada e um paragrafo, e
  nao uma lista ou tabela.
- Corpus de recuperacao: dump da Wikipedia do fim de 2018.
- Documento: passagem de Wikipedia com no maximo 100 tokens.
- Recuperador: Contriever ajustado em MS-MARCO.
- Cada contexto controlado possui exatamente um documento gold e `k - 1`
  distratores recuperados que nao contem respostas anotadas do NQ.
- Os distratores sao apresentados por relevancia decrescente no experimento
  principal.

O documento gold e o paragrafo da Wikipedia anotado pelo NaturalQuestions.
Mover esse documento altera somente sua posicao; a pergunta, o gold e os
distratores devem permanecer os mesmos entre posicoes.

### 3.2. Grade experimental exata

| Total de documentos | Aproximacao de contexto | Posicoes no artigo (1-based) | Indices nos arquivos (0-based) | Casos por ponto |
| ---: | ---: | --- | --- | ---: |
| 10 | ~2K tokens | 1, 5, 10 | `0, 4, 9` | 2.655 |
| 20 | ~4K tokens | 1, 5, 10, 15, 20 | `0, 4, 9, 14, 19` | 2.655 |
| 30 | ~6K tokens | 1, 5, 10, 15, 20, 25, 30 | `0, 4, 9, 14, 19, 24, 29` | 2.655 |

Sao **15 condicoes de tamanho-posicao**. Para um modelo que suporte todas as
condicoes, isso corresponde a 39.825 respostas de QA, antes dos baselines.

Os arquivos locais em `qa_data/` confirmam os 2.655 registros para cada uma
das 15 condicoes.

### 3.3. Prompt principal

O prompt pede uma resposta de alta qualidade usando somente os resultados de
busca fornecidos, avisa que alguns podem ser irrelevantes, lista os documentos
numerados com titulo e texto, e coloca a pergunta depois dos documentos.

O template preservado neste repositorio e
`src/lost_in_the_middle/prompts/qa.prompt`.

```text
Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).

{search_results}

Question: {question}
Answer:
```

Cada documento e serializado como
`Document [N](Title: <titulo>) <texto>`, com numeracao iniciada em 1.

### 3.4. Baselines

Os autores contextualizam o resultado com:

- **closed-book**: somente a pergunta, sem documentos;
- **oracle**: somente o documento que contem a resposta.

Resultados da Tabela 1:

| Modelo | Closed-book | Oracle |
| --- | ---: | ---: |
| LongChat-13B (16K) | 35,0% | 83,4% |
| MPT-30B-Instruct | 31,5% | 81,9% |
| GPT-3.5-Turbo | 56,1% | 88,3% |
| GPT-3.5-Turbo (16K) | 56,0% | 88,6% |
| Claude-1.3 | 48,3% | 76,1% |
| Claude-1.3 (100K) | 48,2% | 76,4% |

### 3.5. Metrica

O artigo chama a metrica de acuracia: um caso esta correto quando alguma
resposta anotada pelo NaturalQuestions aparece na saida do modelo.

Na implementacao publicada, `best_subspan_em`:

1. considera somente a primeira linha da resposta;
2. converte para minusculas;
3. remove pontuacao ASCII e os artigos `a`, `an`, `the`;
4. normaliza espacos;
5. retorna 1 se alguma resposta gold normalizada for substring da predicao.

A media dos valores 0/1 e a acuracia reportada.

## 4. Recuperacao chave-valor

### 4.1. Dados

- Entrada: objeto JSON serializado com pares chave-valor.
- Chaves e valores: UUIDs aleatorios unicos de 128 bits.
- Consulta: uma chave presente no objeto.
- Gold: o valor associado a essa chave.
- Distratores: todos os demais pares.
- Amostra: **500 objetos por tamanho de contexto**.

O mesmo objeto deve ser reutilizado em todas as posicoes de um tamanho; move-se
o par gold e preserva-se a ordem relativa dos demais pares.

### 4.2. Grade experimental exata

| Pares no JSON | Aproximacao de contexto | Posicoes no artigo (1-based) | Indices usados pelo codigo (0-based) | Casos por ponto |
| ---: | ---: | --- | --- | ---: |
| 75 | ~4K tokens | 1, 25, 50, 75 | `0, 24, 49, 74` | 500 |
| 140 | ~8K tokens | 1, 35, 70, 105, 140 | `0, 34, 69, 104, 139` | 500 |
| 300 | ~16K tokens | 1, 50, 100, 150, 200, 250, 300 | `0, 49, 99, 149, 199, 249, 299` | 500 |

Sao **16 condicoes de tamanho-posicao**. Para um modelo que suporte todas as
condicoes, isso corresponde a 8.000 respostas.

Os arquivos locais em `kv_retrieval_data/` confirmam 500 registros para cada
um dos tres tamanhos.

### 4.3. Prompt e metrica

O prompt pede para extrair o valor correspondente a uma chave, apresenta o
JSON, mostra a chave depois do objeto e termina com `Corresponding value:`. O
template esta em `src/lost_in_the_middle/prompts/kv_retrieval.prompt`.

```text
Extract the value corresponding to the specified key in the JSON object below.

JSON data:
{formatted_kv_records}

Key: "{key}"
Corresponding value:
```

Um caso recebe acuracia 1 quando o valor correto aparece na resposta do modelo,
ignorando diferencas entre maiusculas e minusculas. A metrica agregada e a
media dos 500 casos.

## 5. Estudo retriever-reader de QA aberto

O artigo tambem avalia o efeito de recuperar mais documentos sem garantir que
exista exatamente um gold:

- consultas do mesmo subconjunto de NaturalQuestions-Open;
- Contriever ajustado em MS-MARCO;
- documentos da Wikipedia ordenados por relevancia;
- `k = 5, 10, 20, 30, 40, 50` documentos;
- comparacao entre recall do recuperador e acuracia do leitor;
- uso das 6 configuracoes principais.

O recall continua crescendo, mas a acuracia do leitor satura antes. Passar de
20 para 50 documentos melhora apenas cerca de 1,5 ponto no GPT-3.5-Turbo e 1
ponto no Claude-1.3, com aumento de contexto, latencia e custo.

## 6. Analises e controles adicionais

### 6.1. Arquitetura

Flan-T5-XXL e Flan-UL2 sao comparados a MPT-30B-Instruct e LongChat. Os modelos
encoder-decoder sao mais robustos dentro dos comprimentos vistos no treino,
mas tambem desenvolvem a curva em U quando avaliados em sequencias maiores.

### 6.2. Contextualizacao consciente da consulta

Os autores repetem a consulta antes e depois dos documentos ou pares. Isso
leva o KV a desempenho quase perfeito nos tres tamanhos, mas nao remove a
sensibilidade posicional no QA.

### 6.3. Instruction tuning

MPT-30B base e comparado a MPT-30B-Instruct no QA de 20 documentos. Ambos
mantem a curva em U, embora o modelo instruido tenha desempenho absoluto maior.

### 6.4. Controles de QA no apendice

- subconjunto de perguntas sem ambiguidade;
- distratores aleatorios da Wikipedia em vez de hard negatives recuperados;
- embaralhamento dos distratores e instrucao explicita de ordem aleatoria;
- GPT-4-0613 (8K) em 500 perguntas aleatorias, com 20 documentos;
- Llama-2 7B, 13B e 70B, base e chat, com 20 documentos.

Para Llama-2, 20 dos 2.655 casos sao descartados porque excedem 4.096 tokens,
restando 2.635 casos. GPT-4 usa as cinco posicoes do contexto de 20 documentos.

## 7. Relacao com o estado atual deste projeto

### 7.1. O que ja foi validado

- Um caso QA com 20 documentos e gold no indice 9 passou pelo OpenRouter e
  recebeu `best_subspan_em = 1.0`. O indice 9 pertence a grade original.
- Um caso KV com 75 pares e gold no indice 9 passou e recebeu acuracia 1.0.
  Esse indice foi adequado para validar o pipeline, mas **nao pertence a grade
  principal do artigo**, que usa `0, 24, 49, 74` para 75 pares.
- Prompts, formato JSONL e avaliadores originais estao conectados ao cliente de
  API.

### 7.2. Diferencas do piloto atual

| Item | Artigo | Piloto atual |
| --- | --- | --- |
| Casos QA por ponto | 2.655 | 1 executado de um piloto com 10 |
| Casos KV por ponto | 500 | 1 executado de um piloto com 10 |
| QA | 10, 20 e 30 documentos | 20 documentos, indice 9 |
| KV | 75, 140 e 300 pares | 75 pares, indice 9 |
| Modelo | configuracao fixa | `openrouter/free`, rota dinamica |
| Repetibilidade | modelo identificado | modelo retornado deve ser registrado por chamada |

O roteador gratuito e suficiente para validar a engenharia, mas nao deve ser
tratado como um unico modelo em uma analise cientifica: ele pode encaminhar
chamadas diferentes para modelos diferentes.

## 8. Requisitos para a automacao

A automacao deve ser orientada por uma matriz explicita, e nao por nomes de
arquivos inferidos. Para cada execucao, deve registrar:

- tarefa, tamanho e indice gold;
- dataset e hash ou versao do caso;
- provedor, modelo solicitado e modelo retornado;
- temperatura, `top_p`, limite de saida e controles de raciocinio;
- hash do prompt e resposta bruta;
- tokens de entrada, saida e total;
- latencia, status, erro e motivo de termino;
- metrica por caso e media agregada.

Hierarquia recomendada:

```text
results/experiments/
  <provider>/
    <model>/
      qa/<document_count>/gold_at_<index>/
        predictions.jsonl.gz
        scored.jsonl.gz
        summary.json
      kv/<pair_count>/gold_at_<index>/
        predictions.jsonl.gz
        scored.jsonl.gz
        summary.json
```

Comportamentos obrigatorios:

1. retomar sem repetir casos concluidos;
2. separar erro de API de resposta incorreta;
3. nao truncar prompts silenciosamente;
4. limitar concorrencia e respeitar `429`/`Retry-After`;
5. salvar incrementalmente depois de cada caso;
6. avaliar somente respostas concluidas;
7. nunca misturar modelos retornados diferentes em uma mesma curva sem
   identificacao explicita;
8. gerar um manifesto da rodada antes de iniciar chamadas pagas.

## 9. Escopo recomendado para a proxima etapa

Antes da reproducao completa, automatizar em tres niveis:

1. **smoke test:** 1 caso por condicao, usando a grade original;
2. **piloto:** 10 casos por condicao, para validar retomada, limites e custos;
3. **reproducao:** 2.655 casos por ponto no QA e 500 por ponto no KV, apenas
   com modelo fixo e orcamento aprovado.

Essa progressao valida a automacao sem confundir um teste tecnico com os
resultados cientificos do artigo.

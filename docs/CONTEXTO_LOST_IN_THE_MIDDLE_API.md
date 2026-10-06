# Continuidade do projeto: Lost in the Middle via API

> Nota de atualizacao: a descricao exata do desenho experimental do artigo e a
> matriz que deve orientar a automacao estao em `docs/PROTOCOLO_ARTIGO.md`.
> Este arquivo preserva tambem o historico das decisoes do projeto.

Atualizado em 1 de outubro de 2026. Este documento registra as decisões e o estado confirmado da conversa até esta data. Não pressupõe que os arquivos entregues pelo assistente já tenham sido instalados no computador do usuário.

## 1. Objetivo e decisão central

Reproduzir os protocolos experimentais de *Lost in the Middle: How Language Models Use Long Contexts*, utilizando modelos hospedados e acessados por API. Investigar como a posição da informação relevante e o tamanho do contexto afetam a acurácia.

Repositório original: https://github.com/nelson-liu/lost-in-the-middle

O usuário quer exatamente esta divisão:

| Etapa | Implementação desejada |
| --- | --- |
| Geração dos dados | Scripts originais. |
| Montagem dos prompts | `prompting.py` e templates originais. |
| Obtenção das respostas | Script adaptado da estrutura original, chamando um cliente de API. |
| Avaliação | Avaliadores originais, recebendo os campos que já esperam. |
| Gráficos | Código próprio; não encontramos scripts de gráficos ou notebooks na árvore `main` consultada. |

**Prioridade: reutilizar o código original sempre que possível.** Não reinventar geração, prompting ou métricas. Preservar os scripts originais e adicionar versões de obtenção de respostas para API.

Os serviços desejados são **Groq e OpenRouter**. Groq não é Grok/xAI; houve uma correção explícita do usuário. Nenhum modelo específico foi escolhido nesta conversa.

Usar outros modelos por API reproduz o protocolo, mas não implica reproduzir os resultados numéricos dos modelos do artigo.

## 2. Preferências de trabalho

- Trabalhar em etapas pequenas, explicar os comandos e validar antes de ampliar.
- O usuário prefere criar arquivos `.py` e executá-los, em vez de colar blocos `python - <<'PY'` no terminal.
- Não rodar modelos localmente nesta versão; não instalar o conjunto completo de dependências de inferência local.
- A automação em lote foi discutida, mas o usuário decidiu explicitamente adiá-la: primeiro conectar a API, testar os casos já preparados e só depois automatizar.
- Reutilizar seu projeto existente, em vez de começar um sistema independente que descarte o trabalho atual.
- Não tratar sugestões anteriores de arquitetura como implementação concluída.

## 3. Ambiente local conhecido

Projeto observado no terminal do usuário:

```text
/home/indiligente/Desktop/projects/lost-in-the-middle-API
```

- Git: branch `main`; o prompt indica arquivos não rastreados, mas não foi inspecionado o status completo.
- Conda: ambiente `lost-in-the-middle`.
- Python: `3.9.25`.
- Ambiente Conda em `/home/indiligente/miniconda3/envs/lost-in-the-middle`.
- Não remover dependências básicas do Conda, como OpenSSL, certificados, bibliotecas do sistema e o próprio Python.
- Houve timeout na instalação via PyPI; o usuário confirmou que era instabilidade de rede e que conseguiu instalar as dependências.
- Não temos acesso direto confirmado a esse computador ou a esse checkout. Os arquivos anexados são cópias de um projeto inicial, não uma sincronização do checkout atual.

Os comandos abaixo pressupõem execução na raiz do projeto. Existem scripts originais em `scripts/`, como confirmado pela execução local. A presença e integridade de `src/` e dos templates devem ser conferidas antes do teste de API.

## 4. As duas tarefas

### 4.1. Recuperação de chave–valor (UUIDs)

Referência no PDF: seção 3.1, página impressa 162, página 6 do PDF; continuação e tamanhos experimentais na página 163. Figura 6 mostra o prompt. O artigo utiliza 75, 140 e 300 pares, com 500 exemplos para cada tamanho.

Script original:

```text
scripts/data/make_kv_retrieval_data.py
```

Funcionamento conferido no código:

1. Gera chaves e valores com `uuid.uuid4()`.
2. Mantém as chaves em um dicionário até obter a quantidade solicitada.
3. Converte os pares para uma lista ordenada.
4. Escolhe aleatoriamente um par como alvo.
5. Salva um caso por linha, com esta estrutura:

```json
{
  "ordered_kv_records": [["UUID_CHAVE", "UUID_VALOR"]],
  "key": "UUID_CHAVE_ALVO",
  "value": "UUID_VALOR_ESPERADO"
}
```

O exemplo acima é apenas estrutural: o piloto real possui 75 pares por caso. O output esperado vem do próprio par gerado; não é produzido por outra IA.

O gerador escolhe o alvo aleatoriamente, mas não prepara sozinho toda a comparação de posições. Para comparar posições, reutilizar o mesmo caso e mover o mesmo par alvo, preservando a ordem relativa dos demais pares. Não gerar outro dicionário para cada posição.

### 4.2. Perguntas e documentos (QA)

Referência no PDF: seção 2.1, página impressa 159, página 3 do PDF; continuação na página 160.

Não estamos gerando frases/perguntas novas com IA. As perguntas e respostas anotadas vêm do NaturalQuestions; os documentos são passagens da Wikipédia. O artigo seleciona 2.655 perguntas cuja resposta longa é um parágrafo, excluindo listas e tabelas.

Script original:

```text
scripts/data/make_qa_data_from_retrieval_results.py
```

O script usa resultados de recuperação previamente disponibilizados pelos autores. Seleciona os primeiros `k - 1` documentos recuperados com `hasanswer == False`, marca-os como distratores e insere o trecho gold (`nq_annotated_gold.chunked_long_answer`) no índice solicitado. As respostas aceitas permanecem nas anotações do caso.

O artigo descreve passagens de até 100 tokens; o README descreve `chunked_long_answer` como aproximadamente 100 palavras. Não assumir que esses dois comprimentos são idênticos. Para reproduzir, preservar os dados disponibilizados.

O README, em “Generating new multi-document QA data”, documenta o download:

```text
https://nlp.stanford.edu/data/nfliu/lost-in-the-middle/nq-open-contriever-msmarco-retrieved-documents.jsonl.gz
```

No projeto do usuário, o caminho adotado é `data/nq-retrieval.jsonl.gz`. `wget` é somente a ferramenta de download; os casos prontos em `qa_data/` também poderiam ser usados sem esse download. Escolhemos baixá-lo para testar o script original de preparação.

Cada caso preparado contém `question`, `answers`, `ctxs` e outros metadados. Não enviar o registro inteiro à API: respostas esperadas, `isgold`, `hasanswer` e anotações são controles de avaliação, não conteúdo do prompt.

## 5. O que já foi concluído e confirmado

### UUIDs

O usuário executou:

```bash
python scripts/data/make_kv_retrieval_data.py --num-keys 75 --num-examples 10 --output-path data/piloto/kv-75.jsonl.gz
python validar_uuid.py
```

Saída confirmada:

```text
OK: 10 casos, 75 pares por caso e respostas esperadas consistentes.
```

A validação conferiu quantidade de casos, quantidade de pares, chaves distintas e consistência entre a chave consultada e o valor esperado. Não foi uma validação de respostas de modelo.

### QA

O usuário confirmou o download dos dados. Foi orientado a criar `scripts/data/preparar_qa_piloto.py`, que lê os primeiros 10 registros não vazios e salva:

```text
data/piloto/nq-retrieval-10.jsonl.gz
```

Houve um `FileNotFoundError` porque esse piloto ainda não estava disponível no caminho esperado; o bloqueio foi resolvido, como comprovado pela validação posterior.

Montagem executada:

```bash
python scripts/data/make_qa_data_from_retrieval_results.py --input-path data/piloto/nq-retrieval-10.jsonl.gz --num-total-documents 20 --gold-index 9 --output-path data/piloto/qa-20-pos9.jsonl.gz
python scripts/evaluation/validar_qa.py
```

Saída confirmada:

```text
OK: 10 casos, 20 documentos por caso e documento correto no índice 9.
```

Exemplo mostrado pelo usuário:

- Pergunta: `who got the first nobel prize in physics`
- Respostas aceitas: `['Wilhelm Conrad Röntgen']`
- Título gold: `List of Nobel laureates in Physics`
- O trecho gold informa explicitamente que Röntgen recebeu o primeiro prêmio em 1901.

O índice 9 é a décima posição. A validação conferiu um único gold nessa posição e as marcações dos distratores. A conferência das marcações não prova que todo distrator seja semanticamente incapaz de responder; o artigo discute ambiguidades no Apêndice A.

## 6. Código inicial enviado pelo usuário

Foram anexados `gerar_dicionario.py`, `dicionario.json`, `test.py` e `README(1).md`.

- `gerar_dicionario.py`: cria um único dicionário com 75 pares usando bibliotecas nativas.
- `dicionario.json`: contém 75 pares.
- `test.py`: consulta uma chave definida manualmente; usa Ollama em `localhost:11434`, modelo `qwen3:8b`, temperatura zero, configuração de contexto e saída; mede tempo e verifica a presença do valor esperado.
- O prompt inicial está em português e acrescenta uma instrução para responder apenas com o valor. É uma variante do experimento, não o template original.
- A estimativa `caracteres // 4` não garante que UUIDs caibam na janela de contexto.

Esse código demonstra um teste individual, mas a direção atual é usar os geradores e prompts originais e obter as respostas via API.

## 7. Dependências e chaves

Requirements mínimo inicial sugerido e instalado pelo usuário:

```text
tqdm>=4.65,<5
xopen>=1.7,<3
pydantic>=1.10,<2
groq
```

`tqdm` e `xopen` são usados pelos dois geradores; o prompting original usa dataclasses do Pydantic. A restrição a Pydantic 1 foi escolhida para conservar compatibilidade com esse código.

O pacote de conexão posteriormente entregue acrescenta:

```text
openai>=1.30,<2
python-dotenv>=1,<2
```

**Pendente:** acrescentar `regex` para usar `src/lost_in_the_middle/metrics.py`. Não assumir que já esteja instalado explicitamente ou que o requirements completo para a integração final esteja validado.

O `setup.py` original lê o requirements completo. `pip install -e .` sem cuidados pode instalar dependências de inferência local. Quando necessário, instalar o pacote com `--no-deps`, depois das dependências selecionadas, ou incluir `src/` corretamente no caminho de imports.

Chaves propostas no `.env`:

```dotenv
GROQ_API_KEY=
OPENROUTER_API_KEY=
```

Orientações dadas: `chmod 600 .env`; ignorar `.env` e `.env.*` no Git, permitindo `.env.example`; nunca salvar chaves em logs/resultados. O `.env` continua sendo texto puro. Não foi confirmada a configuração das chaves pelo usuário.

## 8. Pacote de conexão entregue: estado e limitações

Foi entregue `conexao_api_groq_openrouter.zip`, com:

| Arquivo | Responsabilidade |
| --- | --- |
| `api_client.py` | Comunicação com Groq e OpenRouter. |
| `scripts/api/test_api.py` | Teste de um caso KV ou QA, construção do prompt original, envio e registro. |
| `config.json` | Serviço, modelo, tarefa, dataset, índice e parâmetros. |
| `.env.example` | Nomes das variáveis, sem chaves. |
| `requirementsAPI.txt` | Dependências de preparação e API. |
| `docs/README_API.md` | Instalação e execução. |

`api_client.py` utiliza o SDK nativo `groq` e o cliente `OpenAI` apontado para `https://openrouter.ai/api/v1`. Recebe um prompt pronto e retorna texto, modelo solicitado/retornado, ID, uso de tokens, duração, motivo de término e fornecedor de backend quando presente.

Não gera dados, não monta prompts e não calcula métricas. Envia apenas uma mensagem de usuário por chamada, sem histórico. Retries automáticos foram desativados no piloto.

`scripts/api/test_api.py` busca o prompting original em `src/` ou `vendor/lost-in-the-middle/src/` e usa os templates originais. Inclui `--dry-run`, que salva o prompt sem credenciais nem chamada de API. Salva resultados individuais em JSON.

Configuração inicial: primeiro caso de KV, temperatura 0, limite de 200 tokens de saída e timeout de 120 segundos. Modelo está vazio para o usuário preencher. Esses parâmetros devem ser conferidos para o modelo escolhido; raciocínio/thinking não foi configurado.

**Verificações realizadas pelo assistente:** sintaxe compatível com Python 3.9; dry-run com prompting/templates originais para KV e um exemplo sintético de QA; clientes testados com SDKs simulados. Isso NÃO confirma autenticação, chamada real ou compatibilidade dos parâmetros com um modelo específico.

**Não confirmado pelo usuário:** extração/instalação do pacote, configuração das chaves, seleção do modelo, dry-run local ou qualquer chamada real a Groq/OpenRouter.

Após a entrega, o usuário explicitou a preferência por maior reutilização da estrutura original. Assim, o pacote é uma base de conexão, não a arquitetura definitiva da avaliação.

## 9. Avaliação original: decisão e detalhes conferidos

Arquivos originais:

```text
scripts/evaluation/evaluate_kv_responses.py
scripts/evaluation/evaluate_qa_responses.py
src/lost_in_the_middle/metrics.py
```

### KV

Lê `value` e `model_answer`. Acerto = valor esperado presente na resposta, ignorando maiúsculas/minúsculas. Calcula a média e opcionalmente salva `metric_accuracy` por caso.

O `scripts/api/test_api.py` entregue possui uma verificação inicial sensível a maiúsculas/minúsculas e igualdade estrita adicional. Não tratá-la como reprodução exata da avaliação. A preferência agora é executar o avaliador original.

### QA

Lê `answers` e `model_answer`. O script considera apenas a primeira linha da resposta e aplica `best_subspan_em`.

Normalização em `metrics.py`: minúsculas, remoção da pontuação ASCII, remoção dos artigos ingleses `a`, `an`, `the` e normalização de espaços. Verifica se alguma resposta aceita normalizada aparece na resposta normalizada. Calcula a média e opcionalmente salva `metric_best_subspan_em`.

Não substituir pela igualdade de toda a resposta, outra normalização ou uma avaliação por LLM. Preservar a resposta bruta no arquivo de predictions para deixar o processamento ao avaliador original.

O pacote inicial marca QA como avaliação pendente; a métrica oficial ainda não foi integrada.

### Formato de saída desejado para a adaptação

Preservar o caso original e acrescentar `model_answer`, além de metadados de execução. Salvar um registro por linha em `.jsonl` ou `.jsonl.gz` para compatibilidade com os avaliadores.

O teste inicial entregue usa `text` e `expected_answers` em JSON individual. Precisará ser adaptado ou convertido: `text` → `model_answer`; respostas de QA → `answers`; valor KV → `value`.

Exemplos futuros, dependentes de termos criado esses arquivos de predictions:

```bash
python scripts/evaluation/evaluate_kv_responses.py --input-path results/kv-predictions.jsonl.gz --output-path results/kv-scored.jsonl.gz
python scripts/evaluation/evaluate_qa_responses.py --input-path results/qa-predictions.jsonl.gz --output-path results/qa-scored.jsonl.gz
```

## 10. Próximos passos acordados

1. Inspecionar os scripts originais `get_kv_responses_from_mpt.py`, `get_kv_responses_from_longchat.py` e os equivalentes de QA para adaptar sua estrutura de leitura/prompt/saída. Os nomes foram identificados, mas seus fluxos completos não foram analisados nesta conversa.
2. Conservar o cliente de API como camada de transporte e criar um script de obtenção de respostas para API, no estilo original. Evitar importação de módulos de inferência local nesse caminho.
3. Conferir um prompt de UUID com os casos já validados.
4. Configurar um modelo e chave e realizar uma chamada real. Registrar a resposta no formato esperado pelo avaliador original e executar a avaliação KV.
5. Repetir com um caso de QA e executar seu avaliador original.
6. Somente depois preparar variantes de posição e automatizar os lotes, persistência e retomada.
7. Criar gráficos próprios sobre resultados já avaliados. Eixo de posição versus acurácia, separando modelos e tamanhos de contexto.

Primeiro marco: **um caso original de KV → prompt original → API → predictions compatíveis → avaliador original**. Segundo marco: o mesmo fluxo para QA.

## 11. Cuidados experimentais e pendências

- Fixar os mesmos casos entre posições; manter a ordem relativa dos distratores.
- Registrar posição com índice iniciado em zero e tamanho em documentos ou pares; não confundir essas unidades com tokens.
- Fixar commit original e conservar corpus/configuração para reprodução. O commit ainda não foi registrado.
- Não tratar erros de autenticação, limite de API, contexto excedido ou transporte como respostas incorretas do modelo; reportá-los separadamente.
- Não truncar silenciosamente o input; verificar limites do modelo escolhido e registrar o motivo de término da saída.
- Registrar provedor, modelo, parâmetros, prompt, resposta, tokens e latência. Temperatura zero não garante determinismo absoluto.
- OpenRouter pode escolher diferentes fornecedores de inferência para o mesmo modelo. O cliente inicial não fixa uma rota; definir essa política antes do experimento formal e registrar a rota quando disponível.
- Controle de raciocínio depende do modelo/serviço. O limite inicial de 200 tokens pode precisar de ajuste; registrar mudanças em vez de alterar condições sem documentação.
- Ainda não há escolha definitiva de modelos, contagem formal de casos, orçamento ou execução em lote.
- Os 10 casos iniciais são um teste técnico; não representam uma amostra suficiente para conclusões fortes sobre o fenômeno.
- Nenhuma chamada real às APIs nem gráfico foi confirmado nesta conversa.

## 12. Referências práticas

- README original: https://github.com/nelson-liu/lost-in-the-middle/blob/main/README.md
- Experimentos e avaliação: https://github.com/nelson-liu/lost-in-the-middle/blob/main/EXPERIMENTS.md
- Prompting: https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/prompting.py
- Métricas: https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/metrics.py
- Groq: https://console.groq.com/docs/quickstart
- OpenRouter: https://openrouter.ai/docs/quickstart

**Orientação para o próximo chat:** continuar da preferência final por scripts originais. Não refazer a geração dos pilotos já validados. Não começar pela automação em lote. Conferir o código atual do checkout antes de editar e não assumir que o pacote de conexão entregue já esteja integrado.

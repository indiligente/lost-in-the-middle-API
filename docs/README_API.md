# Teste inicial: Groq e OpenRouter

Copie os arquivos deste pacote para a raiz do seu projeto, onde estao scripts/,
src/ e data/. Tambem funciona com o original em vendor/lost-in-the-middle/.
Mantenha a pasta prompts/ original junto de prompting.py. Este pacote nao
inclui dados, codigo dos autores, modelos ou chaves.

## Instalacao

No ambiente Conda ativo:

```bash
python -m pip install -r requirementsAPI.txt
python -m pip check
```

O requirementsAPI.txt fornecido substitui o anterior para acrescentar os SDKs.
Nao execute pip install -e . com as dependencias completas do repositorio
original. O teste inclui src/ na busca de imports automaticamente.

## Chaves

Se ja tem .env, acrescente apenas a chave que faltar. Caso contrario:

```bash
cp .env.example .env
chmod 600 .env
nano .env
```

Acrescente ao seu .gitignore (preserve o conteudo existente):

```gitignore
.env
.env.*
!.env.example
results/
```

Nao compartilhe .env. Uma variavel ja exportada no terminal prevalece sobre .env.

## Conferir o prompt, sem API

```bash
python scripts/api/test_api.py --dry-run
```

Le o primeiro caso de UUID e salva o prompt em results/. Nao exige chave nem
modelo. Usa get_kv_retrieval_prompt original, sem traducao ou instrucoes novas.

## Chamar a Groq

Escolha um ID de modelo disponivel na sua conta. Nao use literalmente ID_DO_MODELO.

```bash
python scripts/api/test_api.py --provider groq --model ID_DO_MODELO
```

## Chamar o OpenRouter

```bash
python scripts/api/test_api.py --provider openrouter --model ID_DO_MODELO_OPENROUTER
```

Voce tambem pode salvar provider e model em config.json e executar:

```bash
python scripts/api/test_api.py
```

O OpenRouter pode rotear um modelo por diferentes fornecedores de inferencia.
Este primeiro cliente registra o fornecedor retornado quando disponivel; ainda
nao fixa uma rota. Fixar a rota sera uma configuracao necessaria se essa
variacao precisar ser controlada no experimento.

## Selecao inicial de modelos

O `config.json` fica apontado para `openrouter/free` no OpenRouter. Ele usa
somente modelos gratuitos, mas pode selecionar um modelo diferente a cada
execucao. O primeiro piloto bem-sucedido foi roteado para
`nvidia/nemotron-3-super-120b-a12b:free`. Para uma rota fixa, uma alternativa
gratuita catalogada e `google/gemma-4-31b-it:free`, embora ela estivesse
temporariamente limitada durante esta verificacao. Para a Groq, o candidato geral mais barato
com preco publicado entre os modelos ativos e `openai/gpt-oss-20b`: US$ 0,075
por milhao de tokens de entrada e US$ 0,30 por milhao de tokens de saida.
O cliente desativa o raciocinio exposto nesse modelo para preservar a resposta
final esperada pelos avaliadores originais.

O OpenRouter informou limite de chave de 100, com 100 restantes, e creditos
pagos iguais a zero no momento da verificacao. A API da Groq informa modelos e
precos, mas nao expoe por essa chave uma cota total de billing; o saldo e os
limites da conta devem ser confirmados no painel Groq antes de uma rodada grande.

## Conferir / executar um caso de QA

```bash
python scripts/api/test_api.py --task qa --dataset data/piloto/qa-20-pos9.jsonl.gz --dry-run
python scripts/api/test_api.py --task qa --dataset data/piloto/qa-20-pos9.jsonl.gz --provider groq --model ID_DO_MODELO
```

Cada execucao envia somente um caso em uma nova requisicao, sem historico.
Respostas esperadas e marcacoes gold nao sao enviadas ao modelo. Os resultados
ficam em JSON, com prompt, modelo, tokens, duracao e motivo de termino.
UUID e avaliado por presenca do valor na resposta, mais igualdade estrita como
medida adicional. A metrica oficial de QA ainda precisa ser integrada; o teste
de QA por enquanto salva a resposta para inspecao.

Temperatura e limite de saida precisam ser aceitos pelo modelo escolhido. Para
modelos que nao estejam na selecao inicial, reasoning/thinking pode consumir o
limite de saida antes de retornar texto. finish_reason=length indica que a
resposta atingiu o limite. Nao ha truncamento local do prompt nem estimativa de
tokens por caracteres. Erros HTTP sao registrados separadamente de acertos.

## Falhas comuns

- FileNotFoundError: confira o caminho do dataset.
- ImportError: instale requirementsAPI.txt e mantenha o src/ original.
- Chave ausente: configure a chave do provedor selecionado.
- HTTP 401: chave invalida ou revogada.
- HTTP 402 no OpenRouter: confira creditos da conta.
- HTTP 400/404: confira modelo, parametros e limite de contexto.
- HTTP 429: confira limites de requisicoes e tokens do servico.

Nao ha retries automaticos nem execucao em lote nesta etapa.

## Gerar predictions compatíveis com os avaliadores originais

O script abaixo usa os prompts originais, chama Groq ou OpenRouter via
`api_client.py` e salva um `.jsonl.gz` com `model_answer`, no formato lido pelos
avaliadores originais.

Primeiro, confira sem chamar API:

```bash
python scripts/api/get_responses_from_api.py \
  --task kv \
  --input-path data/piloto/kv-75.jsonl.gz \
  --output-path results/kv-api-dry-run.jsonl.gz \
  --provider groq \
  --model ID_DO_MODELO \
  --gold-index 9 \
  --limit 1 \
  --dry-run
```

Depois, com chave e modelo configurados:

```bash
python scripts/api/get_responses_from_api.py \
  --task kv \
  --input-path data/piloto/kv-75.jsonl.gz \
  --output-path results/kv-api-predictions.jsonl.gz \
  --provider groq \
  --model ID_DO_MODELO \
  --gold-index 9 \
  --limit 1
```

Avalie com o avaliador original:

```bash
python scripts/evaluation/evaluate_kv_responses.py \
  --input-path results/kv-api-predictions.jsonl.gz \
  --output-path results/kv-api-scored.jsonl.gz
```

Para QA:

```bash
python scripts/api/get_responses_from_api.py \
  --task qa \
  --input-path data/piloto/qa-20-pos9.jsonl.gz \
  --output-path results/qa-api-predictions.jsonl.gz \
  --provider groq \
  --model ID_DO_MODELO \
  --limit 1

PYTHONPATH=src python scripts/evaluation/evaluate_qa_responses.py \
  --input-path results/qa-api-predictions.jsonl.gz \
  --output-path results/qa-api-scored.jsonl.gz
```

## Coletar uma matriz de arquivos e posicoes

O manifesto `experiments/smoke-openrouter.json` descreve as 31 condicoes da
grade principal do artigo: 15 de QA e 16 de KV. O smoke test coleta um caso
por condicao. Ele nao calcula metricas.

Esse manifesto usa `openrouter/free` somente para validar transporte,
persistencia e retomada com baixo custo. O roteador pode escolher modelos
diferentes e alguns podem consumir o limite de saida com raciocinio. Antes de
uma coleta destinada a analise, copie o manifesto, use um `experiment_id`
novo, selecione um modelo fixo e confira `max_output_tokens`.

Valide todos os arquivos, posicoes gold e prompts sem chamar a API:

```bash
python scripts/api/run_experiment.py \
  --manifest experiments/smoke-openrouter.json \
  --dry-run
```

Para validar apenas duas condicoes:

```bash
python scripts/api/run_experiment.py \
  --manifest experiments/smoke-openrouter.json \
  --dry-run \
  --only qa-10-gold-0 \
  --only kv-75-gold-0
```

Depois de conferir o modelo, a cota e o dry-run, inicie a coleta real:

```bash
python scripts/api/run_experiment.py \
  --manifest experiments/smoke-openrouter.json
```

Cada condicao recebe seu proprio diretorio:

```text
results/experiments/smoke-openrouter-official-grid/
  manifest.json
  collection_summary.json
  qa/10_documents/gold_at_0/
    predictions.jsonl.gz
    errors.jsonl.gz
  kv/75_pairs/gold_at_0/
    predictions.jsonl.gz
    errors.jsonl.gz
```

`predictions.jsonl.gz` contem somente respostas concluidas e preserva o
exemplo original, o prompt, a resposta e os metadados da API. Erros de
validacao ou transporte ficam em `errors.jsonl.gz` e nao devem ser enviados
ao avaliador como respostas do modelo.

### Retomada e limites

Cada caso possui um `record_id` deterministico. Ao executar novamente o mesmo
manifesto, respostas ja presentes em `predictions.jsonl.gz` sao ignoradas;
casos que terminaram em erro sao tentados novamente. A escrita ocorre depois
de cada resposta, sem esperar a condicao inteira terminar.

O bloco `collection` do manifesto controla casos por condicao, retries,
intervalo entre requisicoes e espera exponencial. Respostas `429`, timeouts e
erros temporarios `5xx` sao tentados novamente. A coleta continua nas demais
condicoes quando um caso falha.

Use um `experiment_id` novo ao alterar modelo, parametros ou grade. O
`manifest.json` copiado para o diretorio de resultados e imutavel: a automacao
recusa misturar configuracoes diferentes no mesmo experimento.

### Avaliacao permanece separada

O coletor nao cria `scored.jsonl.gz` nem calcula acuracia. Depois de confirmar
que a coleta esta completa, os arquivos `predictions.jsonl.gz` podem ser
passados aos avaliadores em `scripts/evaluation/`, seguindo o mesmo fluxo do
repositorio original.

## Testes automatizados

Ative o ambiente do projeto e execute:

```bash
conda activate lost-in-the-middle
python -m pytest -q
```

Os testes de API usam clientes simulados. Eles validam manifestos, construcao
de prompts, persistencia, retomada e retries sem ler chaves, acessar a rede ou
consumir cota dos provedores.

## Baselines QA

Os baselines Oracle e closed-book usam manifestos separados para impedir que
prompts com e sem documentos sejam misturados na mesma rodada:

```text
experiments/qa-oracle-openrouter.json
experiments/qa-closedbook-openrouter.json
```

Os equivalentes para Groq usam `openai/gpt-oss-20b`:

```text
experiments/smoke-groq.json
experiments/qa-oracle-groq.json
experiments/qa-closedbook-groq.json
```

O cliente envia `include_reasoning: false` para esse modelo, para priorizar a
resposta final esperada pelos avaliadores. A disponibilidade do modelo foi
confirmada pela API da Groq antes da criacao dos manifestos; a cota total da
conta ainda deve ser conferida no painel antes de executar a grade completa.

Ambos usam `qa_data/nq-open-oracle.jsonl.gz`, com 2.655 perguntas. No Oracle,
o unico documento gold e enviado ao modelo. No closed-book, o arquivo fornece
a pergunta e as respostas esperadas, mas `closedbook: true` faz o coletor
enviar somente a pergunta.

Valide os dois sem chamadas de API:

```bash
python scripts/api/run_experiment.py \
  --manifest experiments/qa-oracle-openrouter.json \
  --dry-run

python scripts/api/run_experiment.py \
  --manifest experiments/qa-closedbook-openrouter.json \
  --dry-run
```

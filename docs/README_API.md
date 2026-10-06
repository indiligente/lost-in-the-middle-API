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

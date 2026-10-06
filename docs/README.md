# Documentacao do projeto

Este diretorio concentra a documentacao especifica da reproducao via API.

- [README da API](README_API.md): instalacao, configuracao, modelos e comandos.
- [Contexto e estado](CONTEXTO_LOST_IN_THE_MIDDLE_API.md): decisoes, pilotos e proximos passos.
- [Protocolo do artigo](PROTOCOLO_ARTIGO.md): matriz experimental, modelos, amostras, metricas e requisitos de automacao.
- [Experimentos originais](EXPERIMENTS.md): protocolos originais de QA e KV.

Os scripts seguem a mesma separacao:

- `scripts/api/`: chamadas a provedores, teste individual e coleta por manifesto.
- `scripts/data/`: geracao e preparacao dos dados.
- `scripts/evaluation/`: metricas e validacoes.
- `scripts/original/`: inferencia local original dos autores.

Os planos de coleta ficam em `experiments/`. O manifesto inicial
`experiments/smoke-openrouter.json` percorre um caso de cada condicao oficial
sem misturar a etapa posterior de avaliacao.

O `README.md` na raiz continua sendo o ponto de entrada padrao do repositorio
original. Os resultados locais ficam em `results/` e os dados de piloto em
`data/piloto/`.

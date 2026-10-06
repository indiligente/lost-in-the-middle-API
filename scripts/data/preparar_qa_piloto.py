import gzip
import json
from pathlib import Path

entrada = Path("data/nq-retrieval.jsonl.gz")
saida = Path("data/piloto/nq-retrieval-10.jsonl.gz")

saida.parent.mkdir(parents=True, exist_ok=True)

casos = []

with gzip.open(entrada, "rt", encoding="utf-8") as arquivo:
    for linha in arquivo:
        if not linha.strip():
            continue

        casos.append(json.loads(linha))

        if len(casos) == 10:
            break

if len(casos) != 10:
    raise ValueError(f"Esperávamos 10 casos, mas encontramos {len(casos)}.")

with gzip.open(saida, "wt", encoding="utf-8") as arquivo:
    for caso in casos:
        arquivo.write(json.dumps(caso, ensure_ascii=False) + "\n")

print(f"OK: {len(casos)} casos salvos em {saida}")
print("Primeira pergunta:", casos[0]["question"])
print("Respostas esperadas:", casos[0]["answers"])

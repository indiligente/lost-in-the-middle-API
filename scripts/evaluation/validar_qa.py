import gzip
import json

caminho = "data/piloto/qa-20-pos9.jsonl.gz"

with gzip.open(caminho, "rt", encoding="utf-8") as arquivo:
    casos = [json.loads(linha) for linha in arquivo if linha.strip()]

assert len(casos) == 10, "Quantidade de casos diferente de 10."

for indice, caso in enumerate(casos):
    documentos = caso["ctxs"]

    assert caso["question"], f"Caso {indice}: pergunta ausente."
    assert caso["answers"], f"Caso {indice}: respostas ausentes."
    assert len(documentos) == 20, f"Caso {indice}: quantidade de documentos incorreta."

    assert sum(doc["isgold"] for doc in documentos) == 1, (
        f"Caso {indice}: deve existir exatamente um documento correto."
    )
    assert documentos[9]["isgold"] is True, (
        f"Caso {indice}: documento correto fora do índice 9."
    )
    assert documentos[9]["hasanswer"] is True

    assert all(
        doc["hasanswer"] is False
        for posicao, doc in enumerate(documentos)
        if posicao != 9
    ), f"Caso {indice}: distrator marcado como contendo resposta."

print("OK: 10 casos, 20 documentos por caso e documento correto no índice 9.")

primeiro = casos[0]
print("\nPrimeira pergunta:", primeiro["question"])
print("Respostas esperadas:", primeiro["answers"])
print("Título do documento correto:", primeiro["ctxs"][9]["title"])
print("Texto do documento correto:", primeiro["ctxs"][9]["text"])

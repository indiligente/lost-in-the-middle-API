"""Executa um caso de UUID ou QA usando os templates originais."""
import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_client import ClienteAPI


def local_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_case(path, index):
    if index < 0:
        raise ValueError("case-index deve ser zero ou maior.")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        current = 0
        for line in stream:
            if not line.strip():
                continue
            if current == index:
                return json.loads(line)
            current += 1
    raise ValueError("Caso {} nao encontrado em {}.".format(index, path))


def original_prompt(task, case):
    # Aceita repositorio original na raiz ou clonado em vendor/.
    candidates = [ROOT / "src", ROOT / "vendor/lost-in-the-middle/src"]
    for source in reversed(candidates):
        if (source / "lost_in_the_middle/prompting.py").is_file():
            sys.path.insert(0, str(source))
    try:
        from lost_in_the_middle.prompting import (
            Document, get_kv_retrieval_prompt, get_qa_prompt,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Nao foi possivel importar o prompting original. Mantenha src/ e "
            "src/lost_in_the_middle/prompts/ do repositorio dos autores, "
            "na raiz ou em vendor/lost-in-the-middle/."
        ) from exc

    if task == "kv":
        pairs = case["ordered_kv_records"]
        keys = [pair[0] for pair in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("Caso contem chaves duplicadas.")
        if dict(pairs)[case["key"]] != case["value"]:
            raise ValueError("Resposta esperada inconsistente com os pares.")
        return (
            get_kv_retrieval_prompt(pairs, case["key"],
                                    query_aware_contextualization=False),
            [case["value"]], len(pairs), keys.index(case["key"]),
        )

    docs = case["ctxs"]
    positions = [i for i, doc in enumerate(docs) if doc.get("isgold")]
    if len(positions) != 1 or not case["answers"]:
        raise ValueError("QA deve ter um documento gold e respostas esperadas.")
    prompt = get_qa_prompt(
        case["question"], [Document.from_dict(doc) for doc in docs],
        mention_random_ordering=False, query_aware_contextualization=False,
    )
    return prompt, case["answers"], len(docs), positions[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--provider", choices=["groq", "openrouter"])
    parser.add_argument("--model")
    parser.add_argument("--task", choices=["kv", "qa"])
    parser.add_argument("--dataset")
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--dry-run", action="store_true",
                        help="Salva o prompt sem carregar chaves nem chamar a API.")
    args = parser.parse_args()

    with local_path(args.config).open(encoding="utf-8") as stream:
        config = json.load(stream)
    for name in ("provider", "model", "task", "dataset", "case_index"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value

    task = config.get("task", "kv")
    if task not in ("kv", "qa"):
        raise ValueError("task deve ser kv ou qa.")
    index = config.get("case_index", 0)
    dataset = local_path(config["dataset"])
    case = read_case(dataset, index)
    prompt, expected, size, gold_index = original_prompt(task, case)
    output_dir = local_path(config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    prompt_path = output_dir / (run_id + "-prompt.txt")
    prompt_path.write_text(prompt, encoding="utf-8")
    print("Tarefa: {} | Caso: {} | Tamanho: {} | Indice correto: {}".format(
        task, index, size, gold_index))
    print("Prompt salvo em:", prompt_path)
    if args.dry_run:
        print("Dry-run concluido. Nenhuma chamada a API realizada.")
        return

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    client = ClienteAPI(
        provider=config["provider"], model=config.get("model", ""),
        temperature=config.get("temperature", 0),
        max_output_tokens=config.get("max_output_tokens", 200),
        timeout_seconds=config.get("timeout_seconds", 120),
    )
    record = {
        "run_id": run_id, "task": task, "dataset": str(dataset),
        "case_index": index, "context_size": size, "gold_index": gold_index,
        "prompt": prompt, "expected_answers": expected,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "parameters": {"temperature": client.temperature,
                       "max_output_tokens": client.max_output_tokens,
                       "timeout_seconds": config.get("timeout_seconds", 120),
                       "max_retries": 0},
        "provider": client.provider, "requested_model": client.model,
    }
    failed = False
    try:
        result = client.gerar(prompt)
        record.update(result.to_dict())
        record["status"] = "success"
        if task == "kv":
            record["value_in_response"] = expected[0] in result.text
            record["strict_exact_match"] = result.text.strip() == expected[0]
        else:
            # Avaliacao oficial de QA sera integrada posteriormente.
            record["evaluation"] = "pending_official_qa_metric"
        print("Resposta:", repr(result.text))
        print("Esperado:", expected)
        if task == "kv":
            print("Valor presente na resposta:", record["value_in_response"])
        print("Tempo: {:.2f}s | Tokens entrada: {} | Tokens saida: {}".format(
            result.duration_seconds, result.prompt_tokens, result.completion_tokens))
        print("Motivo de termino:", result.finish_reason)
    except Exception as exc:
        failed = True
        record.update(status="api_error", error_type=type(exc).__name__,
                      http_status=getattr(exc, "status_code", None))
        # Nao salvar corpo de erro, headers ou credenciais.
        print("Falha na API: {} | HTTP: {}".format(
            record["error_type"], record["http_status"]), file=sys.stderr)
    finally:
        result_path = output_dir / (run_id + "-result.json")
        result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        client.close()
        print("Resultado salvo em:", result_path)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

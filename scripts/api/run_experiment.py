#!/usr/bin/env python3
"""Collect API predictions for every condition declared in a manifest."""

import argparse
import copy
import gzip
import hashlib
import json
import logging
import pathlib
import re
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from tqdm import tqdm
from xopen import xopen

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
load_dotenv(ROOT / ".env", override=False)
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from api_client import ClienteAPI  # noqa: E402
from scripts.api.get_responses_from_api import (  # noqa: E402
    build_kv_example,
    build_qa_example,
)

logger = logging.getLogger(__name__)

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def resolve_path(path_value):
    path = pathlib.Path(path_value)
    return path if path.is_absolute() else ROOT / path


def project_relative(path):
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def load_manifest(path):
    with pathlib.Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    required = {"schema_version", "experiment_id", "provider", "model", "conditions"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError("Manifest is missing fields: {}".format(", ".join(missing)))
    if manifest["schema_version"] != 1:
        raise ValueError("Only manifest schema_version=1 is supported.")
    if not SAFE_ID.match(manifest["experiment_id"]):
        raise ValueError("experiment_id may contain only letters, numbers, '.', '_' and '-'.")
    if manifest["provider"] not in ("groq", "openrouter"):
        raise ValueError("provider must be groq or openrouter.")
    if not isinstance(manifest["model"], str) or not manifest["model"].strip():
        raise ValueError("model must be a non-empty string.")
    if not isinstance(manifest["conditions"], list) or not manifest["conditions"]:
        raise ValueError("conditions must be a non-empty list.")

    seen_ids = set()
    seen_coordinates = set()
    for condition in manifest["conditions"]:
        for field in ("id", "task", "input_path", "context_size", "gold_index"):
            if field not in condition:
                raise ValueError("Every condition must define {}.".format(field))
        condition_id = condition["id"]
        if not SAFE_ID.match(condition_id):
            raise ValueError("Invalid condition id: {}".format(condition_id))
        if condition_id in seen_ids:
            raise ValueError("Duplicate condition id: {}".format(condition_id))
        seen_ids.add(condition_id)
        if condition["task"] not in ("qa", "kv"):
            raise ValueError("Invalid task in {}: {}".format(condition_id, condition["task"]))
        if not isinstance(condition["context_size"], int) or condition["context_size"] < 1:
            raise ValueError("context_size must be positive in {}.".format(condition_id))
        if not isinstance(condition["gold_index"], int) or condition["gold_index"] < 0:
            raise ValueError("gold_index must be non-negative in {}.".format(condition_id))
        coordinates = (
            condition["task"],
            condition["context_size"],
            condition["gold_index"],
        )
        if coordinates in seen_coordinates:
            raise ValueError(
                "Conditions would share an output directory: {}.".format(condition_id)
            )
        seen_coordinates.add(coordinates)
        input_path = resolve_path(condition["input_path"])
        if not input_path.is_file():
            raise FileNotFoundError("Input for {} not found: {}".format(condition_id, input_path))

    generation = manifest.get("generation", {})
    collection = manifest.get("collection", {})
    if generation.get("max_output_tokens", 100) < 1:
        raise ValueError("generation.max_output_tokens must be positive.")
    if generation.get("timeout_seconds", 120) <= 0:
        raise ValueError("generation.timeout_seconds must be positive.")
    if collection.get("cases_per_condition", 1) < 1:
        raise ValueError("collection.cases_per_condition must be positive.")
    if collection.get("max_retries", 3) < 0:
        raise ValueError("collection.max_retries cannot be negative.")


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_examples(path, limit):
    with xopen(path) as stream:
        seen = 0
        for source_index, line in enumerate(stream):
            if not line.strip():
                continue
            yield source_index, json.loads(line)
            seen += 1
            if seen >= limit:
                return


def validate_and_build(condition, example, generation):
    task = condition["task"]
    context_size = condition["context_size"]
    gold_index = condition["gold_index"]
    if task == "kv":
        records = example.get("ordered_kv_records", [])
        if len(records) != context_size:
            raise ValueError(
                "Expected {} KV pairs, found {}.".format(context_size, len(records))
            )
        if gold_index >= len(records):
            raise ValueError("gold_index {} is outside the KV context.".format(gold_index))
        return build_kv_example(
            input_example=example,
            gold_index=gold_index,
            query_aware_contextualization=generation.get(
                "query_aware_contextualization", False
            ),
        )

    documents = example.get("ctxs", [])
    if len(documents) != context_size:
        raise ValueError(
            "Expected {} QA documents, found {}.".format(context_size, len(documents))
        )
    gold_positions = [index for index, document in enumerate(documents) if document.get("isgold") is True]
    if gold_positions != [gold_index]:
        raise ValueError(
            "Manifest gold_index={} but the QA example has gold at {}.".format(
                gold_index, gold_positions
            )
        )
    return build_qa_example(
        input_example=example,
        closedbook=generation.get("closedbook", False),
        prompt_mention_random_ordering=generation.get(
            "prompt_mention_random_ordering", False
        ),
        use_random_ordering=generation.get("use_random_ordering", False),
        query_aware_contextualization=generation.get(
            "query_aware_contextualization", False
        ),
    )


def condition_directory(root, condition):
    unit = "documents" if condition["task"] == "qa" else "pairs"
    return (
        root
        / condition["task"]
        / "{}_{}".format(condition["context_size"], unit)
        / "gold_at_{}".format(condition["gold_index"])
    )


def read_record_ids(path):
    record_ids = set()
    if not path.exists():
        return record_ids
    with gzip.open(str(path), "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_id"):
                record_ids.add(record["record_id"])
    return record_ids


def append_jsonl_gz(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(path), "at", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def response_status_code(error):
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        return status_code
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None)


def retry_after_seconds(error):
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def call_with_retries(client, prompt, max_retries, retry_base_seconds):
    attempt = 0
    while True:
        attempt += 1
        try:
            return client.gerar(prompt), attempt
        except Exception as error:
            status_code = response_status_code(error)
            error_name = error.__class__.__name__.lower()
            transient_transport_error = status_code is None and any(
                marker in error_name for marker in ("connection", "timeout")
            )
            retryable = status_code in RETRYABLE_STATUS_CODES or transient_transport_error
            if not retryable or attempt > max_retries:
                try:
                    error.api_attempts = attempt
                except Exception:
                    pass
                raise
            delay = retry_after_seconds(error)
            if delay is None:
                delay = retry_base_seconds * (2 ** (attempt - 1))
            logger.warning(
                "API attempt %d failed (%s); retrying in %.1fs",
                attempt,
                status_code or error.__class__.__name__,
                delay,
            )
            time.sleep(delay)


def prediction_record(
    output_example,
    prompt,
    result,
    attempts,
    manifest,
    condition,
    input_path,
    input_sha256,
    source_index,
    source_example_sha256,
    record_id,
):
    generation = manifest.get("generation", {})
    record = copy.deepcopy(output_example)
    record.update(
        {
            "record_id": record_id,
            "experiment_id": manifest["experiment_id"],
            "condition_id": condition["id"],
            "task": condition["task"],
            "context_size": condition["context_size"],
            "model_gold_index": condition["gold_index"],
            "source_path": project_relative(input_path),
            "source_file_sha256": input_sha256,
            "source_index": source_index,
            "source_example_sha256": source_example_sha256,
            "collected_at": utc_now(),
            "model_prompt": prompt,
            "model_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "model_answer": result.text,
            "model": manifest["model"],
            "model_provider": manifest["provider"],
            "model_temperature": generation.get("temperature", 0.0),
            "model_top_p": generation.get("top_p", 1.0),
            "model_max_new_tokens": generation.get("max_output_tokens", 100),
            "api_timeout_seconds": generation.get("timeout_seconds", 120),
            "api_attempts": attempts,
            "api_status": "success",
            "api_response_id": result.response_id,
            "api_returned_model": result.returned_model,
            "api_finish_reason": result.finish_reason,
            "api_prompt_tokens": result.prompt_tokens,
            "api_completion_tokens": result.completion_tokens,
            "api_total_tokens": result.total_tokens,
            "api_duration_seconds": result.duration_seconds,
            "api_backend_provider": result.backend_provider,
        }
    )
    return record


def error_record(
    error,
    manifest,
    condition,
    input_path,
    input_sha256,
    source_index,
    source_example_sha256,
    prompt,
    record_id,
):
    return {
        "record_id": record_id,
        "experiment_id": manifest["experiment_id"],
        "condition_id": condition["id"],
        "task": condition["task"],
        "context_size": condition["context_size"],
        "gold_index": condition["gold_index"],
        "source_path": project_relative(input_path),
        "source_file_sha256": input_sha256,
        "source_index": source_index,
        "source_example_sha256": source_example_sha256,
        "model_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "provider": manifest["provider"],
        "model": manifest["model"],
        "api_status": "error",
        "api_status_code": response_status_code(error),
        "api_attempts": getattr(error, "api_attempts", None),
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "failed_at": utc_now(),
    }


def write_manifest_snapshot(output_root, manifest, source_path):
    snapshot = copy.deepcopy(manifest)
    snapshot["manifest_sha256"] = sha256_json(manifest)
    snapshot["manifest_source"] = project_relative(pathlib.Path(source_path))
    snapshot_path = output_root / "manifest.json"
    if snapshot_path.exists():
        with snapshot_path.open(encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous.get("manifest_sha256") != snapshot["manifest_sha256"]:
            raise ValueError(
                "The output directory already contains a different manifest. "
                "Use a new experiment_id."
            )
        return
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def collect_condition(
    client,
    manifest,
    condition,
    output_root,
    input_hashes,
    limit_override,
    dry_run,
):
    generation = manifest.get("generation", {})
    collection = manifest.get("collection", {})
    limit = limit_override or condition.get(
        "cases", collection.get("cases_per_condition", 1)
    )
    input_path = resolve_path(condition["input_path"])
    input_key = str(input_path.resolve())
    if input_key not in input_hashes:
        logger.info("Hashing %s", project_relative(input_path))
        input_hashes[input_key] = file_sha256(input_path)
    input_sha256 = input_hashes[input_key]

    output_dir = condition_directory(output_root, condition)
    predictions_path = output_dir / "predictions.jsonl.gz"
    errors_path = output_dir / "errors.jsonl.gz"
    dry_run_path = output_dir / "dry-run.jsonl.gz"
    completed = read_record_ids(predictions_path)
    stats = {
        "planned": limit,
        "success": 0,
        "error": 0,
        "skipped": 0,
        "dry_run": 0,
        "stored_predictions": 0,
        "unresolved_errors": 0,
        "validated_records": 0,
    }

    examples = iter_examples(input_path, limit)
    for source_index, example in tqdm(examples, total=limit, desc=condition["id"]):
        example_sha256 = sha256_json(example)
        record_id = hashlib.sha256(
            "{}:{}:{}:{}".format(
                manifest["experiment_id"], condition["id"], source_index, example_sha256
            ).encode("utf-8")
        ).hexdigest()
        if not dry_run and record_id in completed:
            stats["skipped"] += 1
            continue

        try:
            output_example, prompt = validate_and_build(condition, example, generation)
        except Exception as error:
            record = error_record(
                error,
                manifest,
                condition,
                input_path,
                input_sha256,
                source_index,
                example_sha256,
                "",
                record_id,
            )
            record["error_stage"] = "validation"
            append_jsonl_gz(errors_path, record)
            stats["error"] += 1
            continue

        if dry_run:
            append_jsonl_gz(
                dry_run_path,
                {
                    "record_id": record_id,
                    "experiment_id": manifest["experiment_id"],
                    "condition_id": condition["id"],
                    "task": condition["task"],
                    "context_size": condition["context_size"],
                    "gold_index": condition["gold_index"],
                    "source_path": project_relative(input_path),
                    "source_index": source_index,
                    "source_example_sha256": example_sha256,
                    "model_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "api_status": "dry_run",
                    "validated_at": utc_now(),
                },
            )
            stats["dry_run"] += 1
            continue

        try:
            result, attempts = call_with_retries(
                client=client,
                prompt=prompt,
                max_retries=collection.get("max_retries", 3),
                retry_base_seconds=collection.get("retry_base_seconds", 2.0),
            )
            record = prediction_record(
                output_example,
                prompt,
                result,
                attempts,
                manifest,
                condition,
                input_path,
                input_sha256,
                source_index,
                example_sha256,
                record_id,
            )
            append_jsonl_gz(predictions_path, record)
            completed.add(record_id)
            stats["success"] += 1
        except Exception as error:
            append_jsonl_gz(
                errors_path,
                error_record(
                    error,
                    manifest,
                    condition,
                    input_path,
                    input_sha256,
                    source_index,
                    example_sha256,
                    prompt,
                    record_id,
                ),
            )
            stats["error"] += 1

        interval = collection.get("request_interval_seconds", 0.0)
        if interval > 0:
            time.sleep(interval)

    if dry_run:
        stats["validated_records"] = len(read_record_ids(dry_run_path))
    else:
        successful_ids = read_record_ids(predictions_path)
        failed_ids = read_record_ids(errors_path)
        stats["stored_predictions"] = len(successful_ids)
        stats["unresolved_errors"] = len(failed_ids - successful_ids)
    return stats


def run(manifest_path, dry_run=False, only=None, limit_override=None):
    manifest = load_manifest(manifest_path)
    selected = [
        condition
        for condition in manifest["conditions"]
        if not only or condition["id"] in only
    ]
    if only:
        unknown = sorted(set(only) - {condition["id"] for condition in selected})
        if unknown:
            raise ValueError("Unknown condition ids: {}".format(", ".join(unknown)))
    if not selected:
        raise ValueError("No conditions selected.")

    output_base = resolve_path(manifest.get("output_dir", "results/experiments"))
    output_root = output_base / manifest["experiment_id"]
    write_manifest_snapshot(output_root, manifest, manifest_path)

    generation = manifest.get("generation", {})
    client = None
    if not dry_run:
        client = ClienteAPI(
            provider=manifest["provider"],
            model=manifest["model"],
            temperature=generation.get("temperature", 0.0),
            top_p=generation.get("top_p", 1.0),
            max_output_tokens=generation.get("max_output_tokens", 100),
            timeout_seconds=generation.get("timeout_seconds", 120),
        )

    summary = {
        "experiment_id": manifest["experiment_id"],
        "mode": "dry_run" if dry_run else "collection",
        "started_at": utc_now(),
        "provider": manifest["provider"],
        "model": manifest["model"],
        "conditions": {},
    }
    input_hashes = {}
    try:
        for condition in selected:
            logger.info("Collecting condition %s", condition["id"])
            summary["conditions"][condition["id"]] = collect_condition(
                client=client,
                manifest=manifest,
                condition=condition,
                output_root=output_root,
                input_hashes=input_hashes,
                limit_override=limit_override,
                dry_run=dry_run,
            )
    finally:
        if client is not None:
            client.close()

    summary["finished_at"] = utc_now()
    summary["totals"] = {
        key: sum(condition_stats[key] for condition_stats in summary["conditions"].values())
        for key in (
            "planned",
            "success",
            "error",
            "skipped",
            "dry_run",
            "stored_predictions",
            "unresolved_errors",
            "validated_records",
        )
    }
    summary_name = "dry-run-summary.json" if dry_run else "collection_summary.json"
    (output_root / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Results written to %s", output_root)
    incomplete = summary["totals"]["error"] or summary["totals"]["unresolved_errors"]
    return 1 if incomplete else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to the experiment manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and prompts without API calls.")
    parser.add_argument(
        "--only",
        action="append",
        help="Run only this condition id. May be provided more than once.",
    )
    parser.add_argument(
        "--limit-per-condition",
        type=int,
        help="Temporary case limit for validation; the manifest remains unchanged.",
    )
    args = parser.parse_args()
    if args.limit_per_condition is not None and args.limit_per_condition < 1:
        parser.error("--limit-per-condition must be positive.")
    return run(
        manifest_path=args.manifest,
        dry_run=args.dry_run,
        only=args.only,
        limit_override=args.limit_per_condition,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    sys.exit(main())

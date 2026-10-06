import gzip
import json

import pytest

from scripts.api.run_experiment import collect_condition, write_manifest_snapshot


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def gerar(self, prompt):
        self.calls += 1
        return self.result


class UnauthorizedError(Exception):
    status_code = 401


class FailingClient:
    def __init__(self):
        self.calls = 0

    def gerar(self, prompt):
        self.calls += 1
        raise UnauthorizedError("invalid key")


def make_manifest():
    return {
        "experiment_id": "collection-test",
        "provider": "openrouter",
        "model": "test-model",
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 100,
            "timeout_seconds": 120,
        },
        "collection": {"cases_per_condition": 1, "max_retries": 0},
    }


def make_condition(input_path):
    return {
        "id": "kv-3-gold-0",
        "task": "kv",
        "input_path": str(input_path),
        "context_size": 3,
        "gold_index": 0,
    }


def read_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_success_is_saved_as_prediction(
    tmp_path, write_jsonl_gz, kv_example, fake_result
):
    input_path = write_jsonl_gz(tmp_path / "input.jsonl.gz", [kv_example])
    output_root = tmp_path / "results"
    client = FakeClient(fake_result)

    stats = collect_condition(
        client,
        make_manifest(),
        make_condition(input_path),
        output_root,
        {},
        None,
        False,
    )

    predictions = output_root / "kv/3_pairs/gold_at_0/predictions.jsonl.gz"
    records = read_records(predictions)
    assert client.calls == 1
    assert stats["success"] == 1
    assert stats["stored_predictions"] == 1
    assert len(records) == 1
    assert records[0]["model_answer"] == "fake answer"
    assert records[0]["api_status"] == "success"
    assert not any(key.startswith("metric_") for key in records[0])


def test_completed_prediction_is_not_requested_again(
    tmp_path, write_jsonl_gz, kv_example, fake_result
):
    input_path = write_jsonl_gz(tmp_path / "input.jsonl.gz", [kv_example])
    output_root = tmp_path / "results"
    client = FakeClient(fake_result)
    arguments = (
        client,
        make_manifest(),
        make_condition(input_path),
        output_root,
        {},
        None,
        False,
    )

    collect_condition(*arguments)
    second_stats = collect_condition(*arguments)

    predictions = output_root / "kv/3_pairs/gold_at_0/predictions.jsonl.gz"
    assert client.calls == 1
    assert second_stats["skipped"] == 1
    assert len(read_records(predictions)) == 1


def test_api_error_is_saved_separately(tmp_path, write_jsonl_gz, kv_example):
    input_path = write_jsonl_gz(tmp_path / "input.jsonl.gz", [kv_example])
    output_root = tmp_path / "results"
    client = FailingClient()

    stats = collect_condition(
        client,
        make_manifest(),
        make_condition(input_path),
        output_root,
        {},
        None,
        False,
    )

    condition_root = output_root / "kv/3_pairs/gold_at_0"
    assert not (condition_root / "predictions.jsonl.gz").exists()
    errors = read_records(condition_root / "errors.jsonl.gz")
    assert client.calls == 1
    assert stats["error"] == 1
    assert stats["unresolved_errors"] == 1
    assert errors[0]["api_status"] == "error"
    assert errors[0]["api_status_code"] == 401


def test_manifest_snapshot_rejects_changed_configuration(tmp_path):
    output_root = tmp_path / "experiment"
    source = tmp_path / "manifest.json"
    source.write_text("{}", encoding="utf-8")
    manifest = {"experiment_id": "snapshot-test", "model": "model-a"}
    write_manifest_snapshot(output_root, manifest, source)

    changed = {"experiment_id": "snapshot-test", "model": "model-b"}
    with pytest.raises(ValueError, match="different manifest"):
        write_manifest_snapshot(output_root, changed, source)

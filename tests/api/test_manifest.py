import copy

import pytest

from scripts.api.run_experiment import validate_manifest


def make_manifest(input_path):
    return {
        "schema_version": 1,
        "experiment_id": "test-experiment",
        "provider": "openrouter",
        "model": "test-model",
        "generation": {"max_output_tokens": 100, "timeout_seconds": 120},
        "collection": {"cases_per_condition": 1, "max_retries": 0},
        "conditions": [
            {
                "id": "qa-2-gold-1",
                "task": "qa",
                "input_path": str(input_path),
                "context_size": 2,
                "gold_index": 1,
            }
        ],
    }


def test_valid_manifest_is_accepted(tmp_path):
    input_path = tmp_path / "input.jsonl.gz"
    input_path.touch()

    validate_manifest(make_manifest(input_path))


@pytest.mark.parametrize("field", ["schema_version", "experiment_id", "provider", "model", "conditions"])
def test_required_manifest_fields(field, tmp_path):
    input_path = tmp_path / "input.jsonl.gz"
    input_path.touch()
    manifest = make_manifest(input_path)
    del manifest[field]

    with pytest.raises(ValueError, match="missing fields"):
        validate_manifest(manifest)


def test_missing_input_is_rejected(tmp_path):
    manifest = make_manifest(tmp_path / "missing.jsonl.gz")

    with pytest.raises(FileNotFoundError):
        validate_manifest(manifest)


def test_duplicate_condition_id_is_rejected(tmp_path):
    input_path = tmp_path / "input.jsonl.gz"
    input_path.touch()
    manifest = make_manifest(input_path)
    manifest["conditions"].append(copy.deepcopy(manifest["conditions"][0]))

    with pytest.raises(ValueError, match="Duplicate condition id"):
        validate_manifest(manifest)


def test_conditions_cannot_share_output_directory(tmp_path):
    input_path = tmp_path / "input.jsonl.gz"
    input_path.touch()
    manifest = make_manifest(input_path)
    duplicate = copy.deepcopy(manifest["conditions"][0])
    duplicate["id"] = "another-condition"
    manifest["conditions"].append(duplicate)

    with pytest.raises(ValueError, match="share an output directory"):
        validate_manifest(manifest)


@pytest.mark.parametrize("provider", ["invalid", "", None])
def test_invalid_provider_is_rejected(provider, tmp_path):
    input_path = tmp_path / "input.jsonl.gz"
    input_path.touch()
    manifest = make_manifest(input_path)
    manifest["provider"] = provider

    with pytest.raises(ValueError, match="provider"):
        validate_manifest(manifest)

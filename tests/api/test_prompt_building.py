import pytest

from scripts.api.run_experiment import validate_and_build


def test_builds_standard_qa_prompt(qa_example):
    condition = {"task": "qa", "context_size": 2, "gold_index": 1}

    output, prompt = validate_and_build(condition, qa_example, {})

    assert "Document [1](Title: Distractor)" in prompt
    assert "Document [2](Title: Gold)" in prompt
    assert "Question: Who discovered the test element?" in prompt
    assert len(output["model_documents"]) == 2


def test_builds_oracle_prompt(oracle_example):
    condition = {"task": "qa", "context_size": 1, "gold_index": 0}

    output, prompt = validate_and_build(condition, oracle_example, {"closedbook": False})

    assert "Document [1](Title: Gold)" in prompt
    assert len(output["model_documents"]) == 1


def test_builds_closedbook_prompt_without_documents(oracle_example):
    condition = {"task": "qa", "context_size": 1, "gold_index": 0}

    output, prompt = validate_and_build(condition, oracle_example, {"closedbook": True})

    assert prompt == "Question: Who discovered the test element?\nAnswer:"
    assert output["model_documents"] == []
    assert "Ada Example" not in prompt


def test_repositions_kv_gold_pair(kv_example):
    condition = {"task": "kv", "context_size": 3, "gold_index": 0}

    output, prompt = validate_and_build(condition, kv_example, {})

    assert output["model_ordered_kv_records"][0] == ["key-b", "value-b"]
    assert output["model_gold_index"] == 0
    assert 'Key: "key-b"' in prompt


def test_qa_gold_position_must_match_manifest(qa_example):
    condition = {"task": "qa", "context_size": 2, "gold_index": 0}

    with pytest.raises(ValueError, match="gold_index"):
        validate_and_build(condition, qa_example, {})


def test_context_size_must_match_input(kv_example):
    condition = {"task": "kv", "context_size": 75, "gold_index": 0}

    with pytest.raises(ValueError, match="Expected 75 KV pairs"):
        validate_and_build(condition, kv_example, {})

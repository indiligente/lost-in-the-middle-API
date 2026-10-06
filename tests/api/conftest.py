import gzip
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def qa_example():
    return {
        "question": "Who discovered the test element?",
        "answers": ["Ada Example"],
        "ctxs": [
            {
                "title": "Distractor",
                "text": "This passage does not contain the answer.",
                "isgold": False,
            },
            {
                "title": "Gold",
                "text": "Ada Example discovered the test element.",
                "isgold": True,
            },
        ],
    }


@pytest.fixture
def oracle_example():
    return {
        "question": "Who discovered the test element?",
        "answers": ["Ada Example"],
        "ctxs": [
            {
                "title": "Gold",
                "text": "Ada Example discovered the test element.",
                "isgold": True,
            }
        ],
    }


@pytest.fixture
def kv_example():
    return {
        "key": "key-b",
        "value": "value-b",
        "ordered_kv_records": [
            ["key-a", "value-a"],
            ["key-b", "value-b"],
            ["key-c", "value-c"],
        ],
    }


@pytest.fixture
def write_jsonl_gz():
    def write(path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        return path

    return write


@pytest.fixture
def fake_result():
    return SimpleNamespace(
        text="fake answer",
        response_id="response-id",
        returned_model="returned-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
        duration_seconds=0.01,
        backend_provider="fake-backend",
    )

#!/usr/bin/env python3
"""Get KV or QA responses from an API provider.

This mirrors the original response scripts, but replaces local model inference
with the shared Groq/OpenRouter client in api_client.py.
"""
import argparse
import dataclasses
import hashlib
import json
import logging
import pathlib
import random
import sys
from copy import deepcopy

from tqdm import tqdm
from xopen import xopen
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
load_dotenv(ROOT / ".env", override=False)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from api_client import ClienteAPI  # noqa: E402
from lost_in_the_middle.prompting import (  # noqa: E402
    Document,
    get_closedbook_qa_prompt,
    get_kv_retrieval_prompt,
    get_qa_prompt,
)

logger = logging.getLogger(__name__)
random.seed(0)


def build_kv_example(input_example, gold_index, query_aware_contextualization):
    output_example = deepcopy(input_example)
    ordered_kv_records = deepcopy(input_example["ordered_kv_records"])
    key = input_example["key"]
    value = input_example["value"]

    original_kv_index = ordered_kv_records.index([key, value])
    original_kv = ordered_kv_records.pop(original_kv_index)
    ordered_kv_records.insert(gold_index, original_kv)

    prompt = get_kv_retrieval_prompt(
        data=ordered_kv_records,
        key=key,
        query_aware_contextualization=query_aware_contextualization,
    )
    output_example["model_ordered_kv_records"] = ordered_kv_records
    output_example["model_gold_index"] = gold_index
    return output_example, prompt


def build_qa_example(
    input_example,
    closedbook,
    prompt_mention_random_ordering,
    use_random_ordering,
    query_aware_contextualization,
):
    output_example = deepcopy(input_example)
    question = input_example["question"]
    if closedbook:
        documents = []
        prompt = get_closedbook_qa_prompt(question)
    else:
        documents = [Document.from_dict(ctx) for ctx in deepcopy(input_example["ctxs"])]
        if not documents:
            raise ValueError("QA example does not contain documents.")

        if use_random_ordering:
            (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
            original_gold_document = documents[original_gold_index]
            distractors = [doc for doc in documents if doc.isgold is False]
            random.shuffle(distractors)
            distractors.insert(original_gold_index, original_gold_document)
            documents = distractors

        prompt = get_qa_prompt(
            question,
            documents,
            mention_random_ordering=prompt_mention_random_ordering,
            query_aware_contextualization=query_aware_contextualization,
        )

    output_example["model_documents"] = [dataclasses.asdict(document) for document in documents]
    output_example["model_prompt_mention_random_ordering"] = prompt_mention_random_ordering
    output_example["model_use_random_ordering"] = use_random_ordering
    return output_example, prompt


def iter_examples(input_path, limit):
    seen = 0
    with xopen(input_path) as fin:
        for line in fin:
            if not line.strip():
                continue
            yield json.loads(line)
            seen += 1
            if limit is not None and seen >= limit:
                return


def main(
    task,
    input_path,
    provider,
    model,
    temperature,
    top_p,
    max_new_tokens,
    timeout_seconds,
    output_path,
    limit,
    gold_index,
    closedbook,
    prompt_mention_random_ordering,
    use_random_ordering,
    query_aware_contextualization,
    dry_run,
):
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    client = None
    if not dry_run:
        client = ClienteAPI(
            provider=provider,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_new_tokens,
            timeout_seconds=timeout_seconds,
        )

    try:
        with xopen(output_path, "w") as fout:
            for input_example in tqdm(iter_examples(input_path, limit)):
                if task == "kv":
                    if gold_index is None:
                        raise ValueError("--gold-index is required for task=kv.")
                    output_example, prompt = build_kv_example(
                        input_example=input_example,
                        gold_index=gold_index,
                        query_aware_contextualization=query_aware_contextualization,
                    )
                else:
                    output_example, prompt = build_qa_example(
                        input_example=input_example,
                        closedbook=closedbook,
                        prompt_mention_random_ordering=prompt_mention_random_ordering,
                        use_random_ordering=use_random_ordering,
                        query_aware_contextualization=query_aware_contextualization,
                    )

                output_example["model_prompt"] = prompt
                output_example["model_prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                output_example["model"] = model
                output_example["model_provider"] = provider
                output_example["model_temperature"] = temperature
                output_example["model_top_p"] = top_p
                output_example["model_max_new_tokens"] = max_new_tokens
                output_example["api_timeout_seconds"] = timeout_seconds

                if dry_run:
                    output_example["model_answer"] = ""
                    output_example["api_status"] = "dry_run"
                else:
                    result = client.gerar(prompt)
                    output_example["model_answer"] = result.text
                    output_example["api_status"] = "success"
                    output_example["api_response_id"] = result.response_id
                    output_example["api_returned_model"] = result.returned_model
                    output_example["api_finish_reason"] = result.finish_reason
                    output_example["api_prompt_tokens"] = result.prompt_tokens
                    output_example["api_completion_tokens"] = result.completion_tokens
                    output_example["api_total_tokens"] = result.total_tokens
                    output_example["api_duration_seconds"] = result.duration_seconds
                    output_example["api_backend_provider"] = result.backend_provider

                fout.write(json.dumps(output_example, ensure_ascii=False) + "\n")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["kv", "qa"], required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--provider", choices=["groq", "openrouter"], default="groq")
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gold-index", type=int)
    parser.add_argument("--closedbook", action="store_true")
    parser.add_argument("--prompt-mention-random-ordering", action="store_true")
    parser.add_argument("--use-random-ordering", action="store_true")
    parser.add_argument("--query-aware-contextualization", action="store_true")
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        task=args.task,
        input_path=args.input_path,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=args.timeout_seconds,
        output_path=args.output_path,
        limit=args.limit,
        gold_index=args.gold_index,
        closedbook=args.closedbook,
        prompt_mention_random_ordering=args.prompt_mention_random_ordering,
        use_random_ordering=args.use_random_ordering,
        query_aware_contextualization=args.query_aware_contextualization,
        dry_run=args.dry_run,
    )
    logger.info("finished running %s", sys.argv[0])

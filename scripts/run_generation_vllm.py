#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
import sys
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.code_utils import assemble_candidate
from tddexp.data import load_tasks, make_irrelevant_tests
from tddexp.io import append_jsonl, ensure_dir, read_jsonl
from tddexp.model_inputs import (
    extract_prompt_token_ids,
    prepare_prompt_inputs,
    prompt_token_hash,
)
from tddexp.prompts import build_prompt, select_condition_tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate code in offline batches with vLLM.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["nl_only", "nl_tests", "shuffled_tests", "irrelevant_tests"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--visible-tests", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument(
        "--thinking",
        choices=("auto", "enabled", "disabled"),
        default="enabled",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--order-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--job-offset", type=int, default=0)
    parser.add_argument("--job-count", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="outputs/debug_vllm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_runtime_environment()
    output_dir = ensure_dir(args.output_dir)
    generations_path = output_dir / "generations.jsonl"
    if generations_path.exists() and not args.resume:
        generations_path.unlink()
    generations_path.touch()

    tasks = load_tasks(args.dataset, dataset_name=args.dataset_name)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    irrelevant = make_irrelevant_tests(tasks, args.visible_tests)
    tasks = [
        task
        for idx, task in enumerate(tasks)
        if idx % args.num_shards == args.shard_index
    ]
    jobs = build_jobs(tasks, irrelevant, args)
    expanded = []
    for repeat in range(args.repeats):
        current = [{**j, 'repeat': repeat} for j in jobs]
        if args.order_seed is not None:
            random.Random(args.order_seed + repeat).shuffle(current)
        expanded.extend(current)
    jobs = expanded[args.job_offset:None if args.job_count is None else args.job_offset + args.job_count]
    if args.resume:
        completed = {(r["task_id"], r["condition"], r.get('repeat', 0)) for r in read_jsonl(generations_path)}
        jobs = [j for j in jobs if (j["task_id"], j["condition"], j['repeat']) not in completed]
    if not jobs:
        print("All requested generations are already present.")
        return
    print(
        f"Running vLLM shard {args.shard_index}/{args.num_shards} with "
        f"{len(tasks)} tasks and {len(jobs)} task-condition prompts."
    )

    if args.dry_run:
        for job in jobs:
            row = make_output_row(
                job,
                completion=job["canonical_code"],
                candidate=job["canonical_code"],
                args=args,
                backend_version="dry-run",
            )
            append_jsonl(generations_path, row)
        print(f"Wrote {len(jobs)} dry-run generations to {generations_path}")
        return

    tokenizer, renderer = load_prompt_components(args.model_path, args.tokenizer_path)
    for job in jobs:
        prepared = prepare_prompt_inputs(
            tokenizer,
            renderer,
            job["prompt"],
            args.prompt_format,
            args.thinking,
            return_tensors=None,
        )
        token_ids = extract_prompt_token_ids(prepared)
        job["prompt_token_ids"] = token_ids
        job["prompt_token_hash"] = prompt_token_hash(token_ids)
        job["prompt_token_count"] = len(token_ids)

    longest_prompt = max((job["prompt_token_count"] for job in jobs), default=0)
    required_length = longest_prompt + args.max_new_tokens
    if required_length > args.max_model_len:
        raise SystemExit(
            f"--max-model-len={args.max_model_len} is smaller than the longest "
            f"prompt plus generation budget ({longest_prompt}+{args.max_new_tokens}="
            f"{required_length})."
        )

    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=args.model_path,
        runner="generate",
        tokenizer=args.tokenizer_path or args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        dtype=args.dtype,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        skip_special_tokens=True,
    )

    progress = tqdm(total=len(jobs), desc="vllm prompts")
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        prompts = [
            TokensPrompt(prompt_token_ids=job["prompt_token_ids"])
            for job in batch
        ]
        # Use the same enqueue/step schedule as LLM.generate, persisting each
        # finished request instead of waiting for the longest item in a batch.
        request_ids = llm.enqueue(prompts, sampling, use_tqdm=False)
        states = llm.llm_engine.output_processor.request_states
        pending = {states[request_id].external_req_id: job for request_id, job in zip(request_ids, batch)}
        for request_output in finished_requests(llm):
            job = pending.pop(request_output.request_id)
            if len(request_output.outputs) != 1:
                raise RuntimeError(
                    f"Expected one completion, received {len(request_output.outputs)}."
                )
            generated = request_output.outputs[0]
            completion = generated.text
            candidate = assemble_candidate(
                job["prompt"],
                completion,
                entry_point=job["entry_point"],
                prompt_is_code_prefix=is_code_prefix_dataset(job["dataset_name"]),
            )
            row = make_output_row(
                job,
                completion=completion,
                candidate=candidate,
                args=args,
                backend_version=vllm_version,
            )
            row["finish_reason"] = generated.finish_reason
            row["stop_reason"] = generated.stop_reason
            row["generated_token_count"] = len(generated.token_ids)
            append_jsonl(generations_path, row)
            progress.update(1)
        if pending: raise RuntimeError('Engine finished with incomplete requests.')
    progress.close()
    print(f"Wrote {len(jobs)} vLLM generations to {generations_path}")


def finished_requests(llm):
    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            if output.finished:
                yield output


def build_jobs(tasks: list[Any], irrelevant: dict[str, list[str]], args: argparse.Namespace) -> list[dict]:
    jobs: list[dict] = []
    for task in tasks:
        visible, hidden = task.split_tests(args.visible_tests)
        conditions = list((task.metadata or {}).get('condition_prompts', {})) if args.conditions == ['auto'] else args.conditions
        if not conditions:
            raise ValueError('Automatic conditions require a prepared protocol dataset.')
        for condition in conditions:
            prompt_tests = select_condition_tests(
                task,
                condition,
                visible_tests=visible,
                irrelevant_tests=irrelevant.get(task.task_id),
            )
            prompt = build_prompt(
                task,
                condition,
                visible_tests=visible,
                irrelevant_tests=irrelevant.get(task.task_id),
            )
            jobs.append(
                {
                    "task_id": task.task_id,
                    "dataset_name": task.dataset_name,
                    "condition": condition,
                    "prompt": prompt,
                    "prompt_tests": prompt_tests,
                    "visible_tests": visible,
                    "hidden_tests": hidden,
                    "all_tests": task.tests,
                    "entry_point": task.entry_point,
                    "canonical_code": task.canonical_code,
                }
            )
    return jobs


def make_output_row(
    job: dict,
    *,
    completion: str,
    candidate: str,
    args: argparse.Namespace,
    backend_version: str,
) -> dict:
    return {
        "task_id": job["task_id"],
        "dataset_name": job["dataset_name"],
        "condition": job["condition"],
        "repeat": job.get('repeat', 0),
        "prompt": job["prompt"],
        "completion": completion,
        "candidate_code": candidate,
        "prompt_tests": job["prompt_tests"],
        "visible_tests": job["visible_tests"],
        "hidden_tests": job["hidden_tests"],
        "all_tests": job["all_tests"],
        "entry_point": job["entry_point"],
        "state_path": None,
        "model_path": args.model_path,
        "prompt_format": args.prompt_format,
        "thinking": args.thinking,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "generation_backend": "vllm",
        "generation_backend_version": backend_version,
        "generation_seed": args.seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "order_seed": args.order_seed + job.get('repeat', 0) if args.order_seed is not None else None,
        "prompt_token_hash": job.get("prompt_token_hash"),
        "prompt_token_count": job.get("prompt_token_count"),
    }


def load_prompt_components(model_path: str, tokenizer_path: str | None = None) -> tuple[Any, Any]:
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if config.model_type in {"qwen3_5", "qwen3_5_moe"}:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        return processor.tokenizer, processor
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    return tokenizer, tokenizer


def is_code_prefix_dataset(dataset_name: str) -> bool:
    return not dataset_name.lower().startswith(
        ("livecodebench", "mbppplus", "humanevalplus", "controlled_")
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise SystemExit('--repeats must be positive')
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.max_num_seqs < 1:
        raise SystemExit("--max-num-seqs must be >= 1")
    if args.max_num_batched_tokens < args.max_num_seqs:
        raise SystemExit("--max-num-batched-tokens must be >= --max-num-seqs")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")


def configure_runtime_environment() -> None:
    local_tmp = (ROOT / ".tmp").resolve()
    local_tmp.mkdir(exist_ok=True)
    for variable in ("TMPDIR", "TMP", "TEMP"):
        value = os.environ.get(variable)
        if not value or not Path(value).is_absolute():
            os.environ[variable] = str(local_tmp)

    # One outer process already owns each GPU. Avoid a redundant EngineCore
    # child and use the native sampler for deterministic greedy decoding so
    # FlashInfer does not JIT-compile sampling kernels that are never needed.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


if __name__ == "__main__":
    main()

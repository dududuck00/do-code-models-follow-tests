#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.code_utils import assemble_candidate
from tddexp.data import load_tasks, make_irrelevant_tests
from tddexp.io import append_jsonl, ensure_dir
from tddexp.modeling import LocalCausalLM
from tddexp.prompts import build_prompt, select_condition_tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--conditions", nargs="+", default=["nl_only", "nl_tests", "shuffled_tests", "irrelevant_tests"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1, help="Split tasks into this many shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Run only this zero-based shard index.")
    parser.add_argument(
        "--visible-tests",
        type=int,
        default=None,
        help=(
            "Number of public tests to include. Defaults to all public tests for "
            "LiveCodeBench and 3 tests for assertion-style datasets."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    parser.add_argument(
        "--thinking",
        choices=("auto", "enabled", "disabled"),
        default="auto",
        help="Thinking mode passed to chat templates. Only applies with --prompt-format chat.",
    )
    parser.add_argument("--collect-states", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow real model generation without CUDA.")
    parser.add_argument("--dry-run", action="store_true", help="Use canonical code instead of loading a model.")
    parser.add_argument("--output-dir", default="outputs/debug")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    states_dir = ensure_dir(output_dir / "states")
    generations_path = output_dir / "generations.jsonl"
    if generations_path.exists():
        generations_path.unlink()

    tasks = load_tasks(args.dataset, dataset_name=args.dataset_name)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    validate_shard_args(args.num_shards, args.shard_index)
    irrelevant = make_irrelevant_tests(tasks, args.visible_tests)
    tasks = [
        task
        for idx, task in enumerate(tasks)
        if idx % args.num_shards == args.shard_index
    ]

    model = None
    if not args.dry_run:
        preflight_device(args.device, args.allow_cpu)
        model = LocalCausalLM(
            args.model_path,
            device=args.device,
            dtype=args.dtype,
            prompt_format=args.prompt_format,
            thinking=args.thinking,
        )

    total = len(tasks) * len(args.conditions)
    print(
        f"Running shard {args.shard_index}/{args.num_shards} with "
        f"{len(tasks)} tasks and {len(args.conditions)} conditions."
    )
    for task in tqdm(tasks, total=len(tasks), desc="tasks"):
        visible, hidden = task.split_tests(args.visible_tests)
        for condition in args.conditions:
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
            state_path = states_dir / f"{safe_id(task.task_id)}__{condition}.npz"
            if args.dry_run:
                completion = task.canonical_code
                saved_state_path = None
                candidate = task.canonical_code
            else:
                assert model is not None
                output = model.generate(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    collect_states=args.collect_states,
                    state_path=state_path,
                )
                completion = output.completion
                saved_state_path = output.state_path
                candidate = assemble_candidate(
                    prompt,
                    completion,
                    entry_point=task.entry_point,
                    prompt_is_code_prefix=is_code_prefix_dataset(task.dataset_name),
                )

            append_jsonl(
                generations_path,
                {
                    "task_id": task.task_id,
                    "dataset_name": task.dataset_name,
                    "condition": condition,
                    "prompt": prompt,
                    "completion": completion,
                    "candidate_code": candidate,
                    "prompt_tests": prompt_tests,
                    "visible_tests": visible,
                    "hidden_tests": hidden,
                    "all_tests": task.tests,
                    "entry_point": task.entry_point,
                    "state_path": saved_state_path,
                    "model_path": args.model_path,
                    "prompt_format": args.prompt_format,
                    "thinking": args.thinking,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
            )

    print(f"Wrote {total} generations to {generations_path}")


def safe_id(task_id: str) -> str:
    return task_id.replace("/", "_").replace(":", "_")


def is_code_prefix_dataset(dataset_name: str) -> bool:
    return not dataset_name.lower().startswith(("livecodebench", "mbppplus", "humanevalplus", "controlled_"))


def validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")


def preflight_device(device: str, allow_cpu: bool) -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for real model generation. Install it with "
            "`pip install -r requirements-model.txt`."
        ) from exc

    wants_cuda = device == "auto" or device.startswith("cuda")
    if wants_cuda and not torch.cuda.is_available() and not allow_cpu:
        raise SystemExit(
            "CUDA is not available, so Qwen2.5-Coder-7B would run on CPU and be extremely slow. "
            "Run this target on a GPU node, or pass `--allow-cpu` for tiny debugging only."
        )


if __name__ == "__main__":
    main()

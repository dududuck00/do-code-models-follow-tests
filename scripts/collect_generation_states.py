#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.io import append_jsonl, ensure_dir, read_jsonl
from tddexp.model_inputs import prompt_token_hash
from tddexp.modeling import LocalCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay generated prompts with Transformers and save prompt-end states."
    )
    parser.add_argument("--generations", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--states-dir", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument(
        "--thinking",
        choices=("auto", "enabled", "disabled"),
        default="enabled",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_shard_args(args.num_shards, args.shard_index)
    rows = read_jsonl(args.generations)
    validate_rows(rows, args)
    selected = [
        row
        for idx, row in enumerate(rows)
        if idx % args.num_shards == args.shard_index
    ]

    output_dir = ensure_dir(args.output_dir)
    states_dir = ensure_dir(args.states_dir or (output_dir / "states"))
    output_path = output_dir / "generations.jsonl"
    if output_path.exists():
        output_path.unlink()
    output_path.touch()

    print(
        f"Collecting Transformers states for shard {args.shard_index}/{args.num_shards}: "
        f"{len(selected)} prompts."
    )
    model = None
    transformers_version = "dry-run"
    if not args.dry_run:
        import transformers

        transformers_version = transformers.__version__
        model = LocalCausalLM(
            args.model_path,
            device=args.device,
            dtype=args.dtype,
            prompt_format=args.prompt_format,
            thinking=args.thinking,
        )

    for row in tqdm(selected, desc="state prompts"):
        updated = dict(row)
        if args.dry_run:
            updated["state_path"] = None
            updated["state_backend"] = "dry-run"
            updated["state_backend_version"] = transformers_version
            append_jsonl(output_path, updated)
            continue

        assert model is not None
        prompt = str(row["prompt"])
        token_ids = model.prompt_token_ids(prompt)
        actual_hash = prompt_token_hash(token_ids)
        expected_hash = row.get("prompt_token_hash")
        if expected_hash is not None and actual_hash != expected_hash:
            raise RuntimeError(
                "Prompt token mismatch between vLLM generation and Transformers "
                f"state replay for {row['task_id']} / {row['condition']}: "
                f"vLLM={expected_hash}, Transformers={actual_hash}."
            )

        state_path = states_dir / f"{safe_id(str(row['task_id']))}__{row['condition']}.npz"
        if not (args.resume and state_path.exists()):
            model.collect_prompt_state(prompt, state_path)

        updated["state_path"] = str(state_path)
        updated["state_backend"] = "transformers"
        updated["state_backend_version"] = transformers_version
        updated["state_prompt_token_hash"] = actual_hash
        updated["state_prompt_token_count"] = len(token_ids)
        updated["prompt_tokens_verified"] = expected_hash is not None
        append_jsonl(output_path, updated)

    print(f"Wrote {len(selected)} rows with state metadata to {output_path}")


def validate_rows(rows: list[dict], args: argparse.Namespace) -> None:
    if not rows:
        raise SystemExit("No generation rows found.")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("task_id")), str(row.get("condition")))
        if key in seen:
            raise SystemExit(f"Duplicate generation row: {key}")
        seen.add(key)
        if row.get("prompt_format") != args.prompt_format:
            raise SystemExit(
                f"Prompt format mismatch for {key}: row={row.get('prompt_format')}, "
                f"collector={args.prompt_format}"
            )
        if row.get("thinking") != args.thinking:
            raise SystemExit(
                f"Thinking mode mismatch for {key}: row={row.get('thinking')}, "
                f"collector={args.thinking}"
            )
        if str(row.get("model_path")) != args.model_path:
            raise SystemExit(
                f"Model path mismatch for {key}: row={row.get('model_path')}, "
                f"collector={args.model_path}"
            )


def validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")


def safe_id(task_id: str) -> str:
    return task_id.replace("/", "_").replace(":", "_")


if __name__ == "__main__":
    main()

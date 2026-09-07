#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one batched vLLM generation shard per GPU and merge the rows."
    )
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["nl_only", "nl_tests", "shuffled_tests", "irrelevant_tests"],
    )
    parser.add_argument("--limit", type=int, default=None)
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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("No GPU ids were provided.")

    output_dir = Path(args.output_dir)
    shards_dir = output_dir / "vllm_shards"
    logs_dir = output_dir / "logs" / "vllm"
    shards_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    local_tmp = (ROOT / ".tmp").resolve()
    local_tmp.mkdir(exist_ok=True)
    for variable in ("TMPDIR", "TMP", "TEMP"):
        value = env_base.get(variable)
        if not value or not Path(value).is_absolute():
            env_base[variable] = str(local_tmp)

    processes: list[tuple[int, str, subprocess.Popen, object]] = []
    for shard_index, gpu in enumerate(gpus):
        shard_dir = shards_dir / f"shard_{shard_index}"
        log_path = logs_dir / f"shard_{shard_index}.log"
        cmd = build_shard_command(args, shard_index, len(gpus), shard_dir)
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard_index, gpu, process, log_handle))
        print(f"Launched vLLM shard {shard_index}/{len(gpus)} on GPU {gpu}; log={log_path}")

    failures: list[tuple[int, str, int]] = []
    for shard_index, gpu, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append((shard_index, gpu, return_code))

    if failures:
        for shard_index, gpu, return_code in failures:
            print(
                f"vLLM shard {shard_index} on GPU {gpu} failed with exit code "
                f"{return_code}; see {logs_dir / f'shard_{shard_index}.log'}",
                file=sys.stderr,
            )
        raise SystemExit(1)

    shard_dirs = [str(shards_dir / f"shard_{idx}") for idx in range(len(gpus))]
    merged_path = output_dir / "generations_vllm.jsonl"
    merge_cmd = [
        sys.executable,
        "scripts/merge_generation_shards.py",
        "--shard-dirs",
        *shard_dirs,
        "--output",
        str(merged_path),
        "--expect-conditions",
        *args.conditions,
    ]
    subprocess.run(merge_cmd, cwd=ROOT, check=True)
    print(f"All vLLM shards finished. Merged generations: {merged_path}")


def build_shard_command(
    args: argparse.Namespace,
    shard_index: int,
    num_shards: int,
    shard_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_generation_vllm.py",
        "--dataset",
        args.dataset,
        "--dataset-name",
        args.dataset_name,
        "--model-path",
        args.model_path,
        "--conditions",
        *args.conditions,
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--dtype",
        args.dtype,
        "--prompt-format",
        args.prompt_format,
        "--thinking",
        args.thinking,
        "--batch-size",
        str(args.batch_size),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(shard_dir),
    ]
    if not args.enable_prefix_caching:
        cmd.append("--no-enable-prefix-caching")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.visible_tests is not None:
        cmd.extend(["--visible-tests", str(args.visible_tests)])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def parse_gpus(values: list[str]) -> list[str]:
    gpus: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                gpus.append(item)
    return gpus


if __name__ == "__main__":
    main()

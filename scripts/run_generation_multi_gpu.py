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
        description="Launch sharded generation across multiple GPUs and merge the results."
    )
    parser.add_argument("--gpus", nargs="+", required=True, help="GPU ids, e.g. 0,1,2,3 or 0 1 2 3.")
    parser.add_argument("--dataset", default="data/livecodebench_release_v6_minus_v5.jsonl")
    parser.add_argument("--dataset-name", default="livecodebench_v6_minus_v5")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--conditions", nargs="+", default=["nl_only", "nl_tests", "shuffled_tests", "irrelevant_tests"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--visible-tests", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    parser.add_argument(
        "--thinking",
        choices=("auto", "enabled", "disabled"),
        default="auto",
    )
    parser.add_argument("--collect-states", dest="collect_states", action="store_true", default=True)
    parser.add_argument("--no-collect-states", dest="collect_states", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="outputs/livecodebench_v6_minus_v5_qwen_full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("No GPU ids were provided.")

    output_dir = Path(args.output_dir)
    shards_dir = output_dir / "shards"
    logs_dir = output_dir / "logs"
    shards_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    local_tmp = ROOT / ".tmp"
    local_tmp.mkdir(exist_ok=True)
    env_base.setdefault("TMPDIR", str(local_tmp))
    env_base.setdefault("TMP", str(local_tmp))
    env_base.setdefault("TEMP", str(local_tmp))

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
        print(f"Launched shard {shard_index}/{len(gpus)} on GPU {gpu}; log={log_path}")

    failures: list[tuple[int, str, int]] = []
    for shard_index, gpu, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append((shard_index, gpu, return_code))

    if failures:
        for shard_index, gpu, return_code in failures:
            print(
                f"Shard {shard_index} on GPU {gpu} failed with exit code {return_code}; "
                f"see {logs_dir / f'shard_{shard_index}.log'}",
                file=sys.stderr,
            )
        raise SystemExit(1)

    shard_dirs = [str(shards_dir / f"shard_{idx}") for idx in range(len(gpus))]
    merged_path = output_dir / "generations.jsonl"
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
    print(f"All shards finished. Merged generations: {merged_path}")


def parse_gpus(values: list[str]) -> list[str]:
    gpus: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                gpus.append(item)
    return gpus


def build_shard_command(args: argparse.Namespace, shard_index: int, num_shards: int, shard_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_generation.py",
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
        "--device",
        "cuda:0",
        "--dtype",
        args.dtype,
        "--prompt-format",
        args.prompt_format,
        "--thinking",
        args.thinking,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--output-dir",
        str(shard_dir),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.visible_tests is not None:
        cmd.extend(["--visible-tests", str(args.visible_tests)])
    if args.collect_states:
        cmd.append("--collect-states")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


if __name__ == "__main__":
    main()

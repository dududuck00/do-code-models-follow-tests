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
        description="Collect prompt-end states with one Transformers replica per GPU."
    )
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conditions", nargs="*", default=None)
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
    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("No GPU ids were provided.")

    output_dir = Path(args.output_dir)
    shards_dir = output_dir / "state_shards"
    states_dir = output_dir / "states"
    logs_dir = output_dir / "logs" / "states"
    shards_dir.mkdir(parents=True, exist_ok=True)
    states_dir.mkdir(parents=True, exist_ok=True)
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
        cmd = build_shard_command(
            args,
            shard_index,
            len(gpus),
            shard_dir,
            states_dir,
        )
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
        print(
            f"Launched state shard {shard_index}/{len(gpus)} on GPU {gpu}; "
            f"log={log_path}"
        )

    failures: list[tuple[int, str, int]] = []
    for shard_index, gpu, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append((shard_index, gpu, return_code))

    if failures:
        for shard_index, gpu, return_code in failures:
            print(
                f"State shard {shard_index} on GPU {gpu} failed with exit code "
                f"{return_code}; see {logs_dir / f'shard_{shard_index}.log'}",
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
    ]
    if args.conditions:
        merge_cmd.extend(["--expect-conditions", *args.conditions])
    subprocess.run(merge_cmd, cwd=ROOT, check=True)
    print(f"All state shards finished. Final generations: {merged_path}")


def build_shard_command(
    args: argparse.Namespace,
    shard_index: int,
    num_shards: int,
    shard_dir: Path,
    states_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/collect_generation_states.py",
        "--generations",
        args.generations,
        "--model-path",
        args.model_path,
        "--output-dir",
        str(shard_dir),
        "--states-dir",
        str(states_dir),
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
    ]
    if args.resume:
        cmd.append("--resume")
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

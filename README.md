# TDD Interpretability Experiments

This repository studies how visible tests constrain code generation and change internal representations.

## Repository contents

Git tracks source code, scripts, tests, configuration, dependency files, frozen experiment datasets, source benchmarks, tokenizer assets, and synthetic-test candidate pools. Runtime outputs, model weights, virtual environments, temporary files, manuscript files, release archives, and duplicated synthetic benchmark variants stay local. The complete results archive contains the experiment outputs and manuscript.

The original LiveCodeBench source dataset is stored as `data/livecodebench_release_v6_minus_v5.jsonl.gz`. To restore the JSONL path used by data-preparation and historical-experiment scripts, run:

```bash
gzip -dk data/livecodebench_release_v6_minus_v5.jsonl.gz
```

The frozen three-experiment datasets are available directly under `data/controlled/`.

## Controlled experiments

The active protocol is in [experiment_design.md](experiment_design.md), with executable settings in `configs/controlled.yaml`. It includes matched I/O controls, paired semantic rules, and fixed-size suites selected by independently validated fault detection.

```bash
mkdir -p .tmp
export TMPDIR="$PWD/.tmp" TMP="$PWD/.tmp" TEMP="$PWD/.tmp"
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/audit_controls.py
.venv/bin/python scripts/run_controlled_experiments.py --model qwen25 --gpu 0 --experiments semantic --generate-only
# Run evaluation inside the code-execution sandbox.
.venv/bin/python scripts/run_controlled_experiments.py --model qwen25 --gpu 0 --experiments semantic --evaluate-only
```

Frozen tasks are in `data/controlled/`; generated programs, task-level grades, and paired summaries are in `outputs/controlled/`. Existing tasks remain paired across three main and semantic runs. Semantic outcomes are paired within each run, averaged within instances, and bootstrapped by specification family. Quality evaluation excludes every candidate-pool input, including candidates that were not selected for display.

All three experiments cover Qwen2.5-Coder-7B, DeepSeek-Coder-6.7B, Qwen3.6-27B, Qwen3.5-9B, and Qwen3.8-27B. Model keys and local paths are listed in `configs/controlled.yaml`. Qwen3.8 main generation uses two replicas with two GPUs each. Its `protocol.json` records all generation commands, fixed request ranges, GPU IDs, and the merge command; set `CUDA_VISIBLE_DEVICES` to the IDs for each replica. `execution_overrides` supplies its tensor-parallel configuration. Other model/experiment combinations use one GPU.

The frozen task files contain the official and held-out evaluators; `evaluate_controlled.py` reads those evaluators by task ID.

The official LiveCodeBench checkout is configured by `evaluation.lcb_root`. The checker runs in timed child processes using pipes. DeepSeek uses the original ByteLevel tokenizer assets with a generic fast-tokenizer loader under `data/model_tokenizers/` to preserve whitespace on Transformers 5.

## Results and manuscript

After all evaluations finish, regenerate the result tables and report, then compile with the AAAI template's required pdfLaTeX engine:

```bash
.venv/bin/python -B scripts/export_controlled_paper.py
.venv/bin/python -B scripts/analyze_semantic_families.py
.venv/bin/python -B scripts/write_controlled_report.py
bash scripts/build_paper.sh
.venv/bin/python -B scripts/package_results.py
```

The export requires all fifteen experiment/model result files. The report is `outputs/analysis/controlled_report.md`; the PDF is `aaai/AuthorKit27/AuthorKit27/paper.pdf`. The build script uses the project-local TinyTeX installation when available, or the system pdfLaTeX and BibTeX tools.

## Environment

The project-local virtual environment is `.venv`.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

For real local-model generation, install the model dependencies on a machine with the right PyTorch build:

```bash
pip install -r requirements-model.txt
```

On GPU nodes, prefer the PyTorch command matching the node's CUDA version from the official PyTorch selector, then install `accelerate`.

The default model identifier is:

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

## Quick Smoke Test

This does not load the 7B model. It uses reference code to verify dataset loading, prompt construction, and test execution.

```bash
source .venv/bin/activate
python scripts/run_smoke.py --limit 3
```

## First Real Generation Run

```bash
source .venv/bin/activate
python scripts/run_generation.py \
  --dataset data/humaneval.jsonl \
  --dataset-name humaneval \
  --model-path Qwen/Qwen2.5-Coder-7B-Instruct \
  --conditions nl_only nl_tests shuffled_tests irrelevant_tests \
  --limit 20 \
  --visible-tests 3 \
  --max-new-tokens 256 \
  --collect-states \
  --output-dir outputs/humaneval_qwen_mvp
```

Then evaluate and train probes:

```bash
python scripts/evaluate_generations.py \
  --generations outputs/humaneval_qwen_mvp/generations.jsonl \
  --output outputs/humaneval_qwen_mvp/eval_results.jsonl

python scripts/train_probe.py \
  --eval-results outputs/humaneval_qwen_mvp/eval_results.jsonl \
  --states-dir outputs/humaneval_qwen_mvp/states \
  --output outputs/humaneval_qwen_mvp/probe_results.json
```

## LiveCodeBench Run

Install data-loading dependencies:

```bash
source .venv/bin/activate
pip install -r requirements-data.txt
```

Download/convert LiveCodeBench code generation lite. For the main experiment, prefer the newest tasks added in `release_v6` relative to `release_v5`:

```bash
make prepare-lcb-v6-new
```

By default, this also strips sample/example tests from `question_content` and stores the result in `question_content_no_public_tests`. The raw `public_test_cases` field is kept separately and is only injected in test-conditioned prompts such as `nl_tests`.

Run the first LiveCodeBench generation experiment:

```bash
make generate-lcb-v6-new
```

For LiveCodeBench, omitting `--visible-tests` uses all `public_test_cases` attached to
each problem. Pass `--visible-tests N` only when you intentionally want to truncate the
visible tests for an ablation.

Export generations in the format expected by the official LiveCodeBench custom evaluator:

```bash
make export-lcb-nl-tests
make export-lcb-nl-only
make export-lcb-shuffled
make export-lcb-irrelevant
```

Then evaluate these JSON files with the official LiveCodeBench runner, using its custom evaluator. The exported files have this shape:

```json
[
  {"question_id": "question-id", "code_list": ["...python code..."]}
]
```

We intentionally do not use `scripts/evaluate_generations.py` for LiveCodeBench because LiveCodeBench mixes stdin-style and platform-style code tasks. The official runner should provide the pass/fail labels; after that, those labels can be joined back with `outputs/livecodebench_qwen_mvp/states/*.npz` for probing.

## Qwen3.6-27B LiveCodeBench Robustness Run

Qwen3.6 uses a multimodal conditional-generation architecture and requires a
recent Transformers release. Create the environment with Python 3.10 or 3.11:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-qwen36.txt
```

The experiment uses the model chat template and thinking mode, with greedy
decoding so paired prompt-condition comparisons remain deterministic. First
verify that one A800 can load a BF16 model replica:

```bash
make smoke-lcb-qwen36
```

For faster generation, keep vLLM in a separate environment so its compiled
Torch dependencies do not modify the Transformers environment:

```bash
uv venv .venv-vllm --python 3.12 --seed
uv pip install --python .venv-vllm/bin/python vllm --torch-backend=auto
```

The accelerated pipeline uses vLLM only for batched autoregressive generation,
then replays the exact prompt with Transformers to collect the final prompt
token's hidden state from every layer. A SHA-256 hash of the prompt token IDs is
checked across both stages. The finalized Qwen3.6 robustness protocol uses
greedy decoding with thinking disabled and at most 8192 new tokens. Thinking
enabled was rejected for the main run because most generations exhausted the
output budget before closing the reasoning segment. Run the two-stage smoke
test:

```bash
make smoke-lcb-qwen36-vllm
```

Then run both stages of the original-public-test experiment on eight GPUs:

```bash
make run-lcb-qwen36-vllm-mgpu GPUS=0,1,2,3,4,5,6,7
```

The first stage writes `generations_vllm.jsonl`; the second writes the final
`generations.jsonl` plus `states/*.npz`. The original Transformers-only target
`generate-lcb-qwen36-mgpu` remains available as a reference implementation.

After restoring the official LiveCodeBench repository, evaluate and analyze:

```bash
make evaluate-lcb-qwen36 LCB_ROOT=/path/to/LiveCodeBench
make analyze-lcb-qwen36-behavior
make analyze-lcb-qwen36-hidden-shift
```

The synthetic high5 robustness condition is:

```bash
make run-lcb-qwen36-synth-high5-vllm-mgpu GPUS=0,1,2,3,4,5,6,7
make evaluate-lcb-qwen36-synth-high5 LCB_ROOT=/path/to/LiveCodeBench
make analyze-lcb-qwen36-synth-high5-behavior
make analyze-lcb-qwen36-synth-high5-hidden-shift
```

## Repair Datasets

For QuixBugs, clone or download the benchmark into `data/QuixBugs`, then run:

```bash
make prepare-quixbugs
```

This writes:

```text
data/quixbugs_repair.jsonl
```

For BugsInPy, place the checkout at `data/BugsInPy`, then run:

```bash
make prepare-bugsinpy
```

This creates a manifest for later curation:

```text
data/bugsinpy_manifest.jsonl
```

## Notes

- `visible_tests` are put into the prompt. For LiveCodeBench, the default is all public tests per task.
- `hidden_tests` are used as the correctness label.
- If the dataset has only a few tests, hidden tests fall back to all tests, which is acceptable for a smoke test but not for final reporting.

# Do Code Language Models Follow Tests? Paired Interventions on Program Behavior

Replication artifact for the FSE 2027 submission. The current manuscript is [paper/paper_fse2027.pdf](paper/paper_fse2027.pdf); editable sources, twelve tables, and three figures are in [paper/](paper/).

## What is included

- Five models: Qwen2.5-Coder-7B-Instruct, Qwen3.5-9B, Qwen3.6-27B, Qwen3.8-27B, and DeepSeek-Coder-6.7B-Instruct.
- Frozen tasks, condition prompts, evaluators, donor mappings, and fixed test-selection pools under `data/controlled/`.
- All **67,650 generated-program records**, compressed individually as `outputs/controlled/{experiment}/{model}/generations.jsonl.gz`, and their task-level `eval_results.jsonl` files.
- Representation and equivalent-wording analyses: prompt inventories, per-instance/per-layer distances, behavioral labels, grouped folds, held-out predictions, and summaries. The actual activation tensors and model weights are not included.
- Visible-assertion replay records and failure examples underlying the manuscript's failure analysis.

| Experiment | Tasks or instances | Conditions | Runs | Models | Generations |
|---|---:|---:|---:|---:|---:|
| Matched benchmark controls | 710 | 4 | 3 | 5 | 42,600 |
| Other-task test extension | 710 | 1 | 3 | 5 | 10,650 |
| Paired semantics and capability controls | 120 | 6 | 3 | 5 | 10,800 |
| Fixed three-test selection | 180 | 4 | 1 | 5 | 3,600 |
| **Total** | | | | | **67,650** |

The benchmark experiment including the extension has 53,250 generations. Representation analysis reuses existing generated programs: 1,200 original prompt states and 2,400 equivalent-wording states were collected, with no additional code generations. Repeated generations are grouped within tasks; the twenty semantic families remain the family-level analysis units.

## Recheck results without a GPU

Python 3.10+ is sufficient for the release verifier; it uses the standard library and does not execute generated code.

```bash
mkdir -p .tmp
export TMPDIR="$PWD/.tmp" TMP="$PWD/.tmp" TEMP="$PWD/.tmp"
python scripts/verify_release.py
```

This checks unique task/condition/run keys, matching generation/evaluation coverage, all 67,650 records, and the frozen manuscript's behavioral point estimates. It recomputes semantic switching, other-task contrasts' underlying rates, held-out probe MAE/MSE and constant baselines, and the visible-versus-hidden failure count. Results are written to `.tmp/release_verification.json`.

Figure rebuilding uses only the included table values and plot inputs:

```bash
pip install -r paper/Figures/requirements.txt
python paper/Figures/build_figures.py
```

Open `paper/paper.tex` in Overleaf with XeLaTeX, or compile it locally with XeLaTeX/BibTeX or Tectonic. The included PDF is the checked submission build. Current table sources are maintained in `paper/Tables/`; older `export_*paper.py` scripts preserve the historical export format and are not the current FSE typesetting pipeline.

## Evidence map

| Paper material | Saved evidence and entry points |
|---|---|
| Figure 1 and Table 1: intervention and rule families | `data/controlled/semantic/tasks.jsonl`, `src/tddexp/semantic_tasks.py`, `paper/Figures/build_figures.py` |
| Tables 2–3 and Figure 2: rule adoption, repeats, family profiles | `outputs/controlled/semantic/`, `outputs/analysis/semantic_families.json`, `scripts/analyze_controlled.py`, `scripts/analyze_semantic_families.py` |
| Visible adherence, default-rule transitions and program examples | `outputs/semantic_visible/`, `outputs/semantic_cases/`, `scripts/semantic_visible/` |
| Table 4: failed-program diagnoses | `outputs/semantic_representation/behavior_labels.jsonl`, `label_summary.json`, `scripts/label_semantic_representation.py` |
| Tables 5–8 and Figure 3: matched benchmark conditions, repeats, other-task tests | `outputs/controlled/main/`, `outputs/controlled/unrelated/`, `outputs/analysis/unrelated_results.json`, `data/controlled/unrelated/donor_mapping.jsonl` |
| Table 9: fixed-size test selection | `data/controlled/quality/`, `outputs/controlled/quality/`, `outputs/analysis/control_audit/quality_detection.json` |
| Table 10: equivalent wording | `outputs/semantic_surface/analysis.json`, `pair_layer_metrics.csv`, `scripts/analyze_semantic_surface.py` |
| Table 11: representation–behavior association | `outputs/semantic_representation/representation_analysis.json`, `pair_layer_metrics.csv`, `scripts/analyze_semantic_representation.py` |
| Table 12: held-out-family prediction | `outputs/semantic_representation/family_probe.json`, `family_probe_predictions.jsonl`, `family_folds.json`, `scripts/probe_semantic_representation.py` |

The manuscript's frozen, compact evidence snapshots are also under `paper/Results/`. Original statistical analysis outputs are retained there; the current paper displays point estimates and repeated-run results.

## Rerun generation, evaluation, or representation extraction

Install dependencies in a new virtual environment; choose a PyTorch build suitable for the machine before installing model dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-model.txt
```

`configs/controlled.yaml` contains public model identifiers accepted by the model loaders. Replace these values with local checkpoint directories if desired. The original DeepSeek tokenizer is included under `data/model_tokenizers/`. `configs/unrelated.yaml`, `configs/semantic_representation.yaml`, and `configs/semantic_surface.yaml` define the extensions. Model-specific optional requirements are retained in the other requirements files.

```bash
python scripts/run_controlled_experiments.py --model qwen25 --gpu 0 --experiments semantic --generate-only
# Evaluate generated programs inside an isolated code-execution environment.
python scripts/run_controlled_experiments.py --model qwen25 --gpu 0 --experiments semantic --evaluate-only
```

To use the included generations with tools expecting uncompressed JSONL:

```bash
find outputs/controlled -name 'generations.jsonl.gz' -exec gzip -dk {} \;
```

Do this before running the visible-replay scripts or the original representation preparation/collection scripts. `scripts/semantic_visible/summarize_visible_hidden.py` recomputes visible/hidden summaries from saved evaluations without re-executing model code. `analyze_visible.py` executes the generated programs and should be run in an isolated environment. `reevaluate_timeouts.py` is the retained historical repair of eleven provisional timeout records, not a step required for the final saved records.

Representation/probe refitting requires regenerating the omitted activation tensors using the provided `prepare_*`, `run_*`, and `collect_generation_states.py` scripts; the CPU verifier checks saved predictions and metrics without those tensors. The probe is linear ridge regression in kernel form, selected by inner family-held-out MAE and evaluated on outer held-out families.

LiveCodeBench execution uses the official `run_test` implementation. Configure its checkout with `evaluation.lcb_root`; the default is `.tmp/LiveCodeBench`. The frozen benchmark tasks and official/held-out test inputs are included. Restore the original source dataset for historical preparation scripts with `gzip -dk data/livecodebench_release_v6_minus_v5.jsonl.gz`.

## Provenance and exclusions

This release contains copies of saved experiment artifacts. Machine-specific paths in configurations and provenance strings were replaced by public model identifiers or relative/redacted local paths. The experiment outcomes and numerical values were preserved. `artifact_manifest.json` records coverage; `checksums.sha256` records the released file hashes.

Model weights, activation tensor caches, runtime environments, machine logs, and exploratory runs that do not enter the current paper are omitted. The generation records, prompt token hashes, prompts, candidate programs, task-level grades, diagnostic outcomes, and derived numerical records needed to inspect the reported behavior are included. Rerunning model inference or tensor-level probes therefore requires downloading the models and recomputing the omitted states.

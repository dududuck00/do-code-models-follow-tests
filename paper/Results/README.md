# Experiment summaries

The original controlled_results.json and semantic_families.json are copied from the author-supplied tdd_results_20260907 archive.
They retain the unrounded results, paired intervals, per-run gains/losses,
and comparisons between models. No new model completions were generated. Later targeted execution of saved case programs is described below.
The original four-condition study contains 57,000 generations and 57,000 unique
evaluation records: main 42,600, semantic 10,800, quality 3600.

Raw evaluation records were independently checked against the supplied means,
per-run gains/losses, and all 400 family metric cells (1360 consistency checks).
The archive's confidence-interval endpoints are retained, not newly recomputed.
Main and semantic experiments have three runs; quality has one.
Tables and figures display rounded values; consult these JSON files for precision.

Model order in the manuscript: Qwen2.5-7B, Qwen3.5-9B, Qwen3.6-27B,
Qwen3.8-27B, DeepSeek-6.7B. The first and last are the Instruct checkpoints
identified in Section 3.1. Historical experiments retain their original coverage.
The full raw archive and execution protocols should be placed in the anonymous
repository when its URL is ready; they are not required to compile this paper.

## Other-task control

`unrelated_report.json` and `unrelated_runs.csv` transcribe the author-provided rounded report supplied on 2026-09-08. They add 10,650 generations to the original 57,000, for a reported total of 67,650. The original four-condition records are reused. Per-task donor mappings, generation records, and unrelated_control_audit.json have not yet been inspected locally for this addition. The supplied paired differences and intervals are preserved rather than recomputed from rounded pass rates. The original JSON files remain unchanged and do not include this fifth condition.

## Historical Synthetic high3

`historical_qwen36_high3.json` records the author-supplied Qwen3.6 high3 summary added on 2026-09-08. It supplies the historical high3 row now retained outside the main text in historical_synthetic_and_controls.tex. The paired interval and p-value are transcribed from the report; no task-level recomputation was performed. The 67,650 generation count refers to the controlled experiments and excludes these historical explorations.


## Paired representation analysis

semantic_representation_report.json aggregates the five-model author reports: 1200 unique model-prompt forwards, 3600 existing behavior records, 120 instances per model, and 20 families. DeepSeek uses the same fixed folds, seed 0, and unchanged probe protocol. deepseek_representation_report.json preserves the additional precision and diagnostic detail. DeepSeek peaks at layer 17/32, while Qwen peaks are late. All five adjusted correlation intervals cross zero. All MAE/MSE point estimates exceed their constant baselines; the MAE-excess interval excludes zero for Qwen2.5, Qwen3.5, and DeepSeek. The 52 unique diagnostic programs refer only to Qwen, not the five-model total. The original association and probe interval endpoints remain transcribed, not recomputed. Raw states were subsequently inspected for the wording-control distances described below; folds and probe predictions were not re-audited.

## Same-rule wording controls

wording_control_report.json contains full-precision final-layer means and contrasts, obtained from the server analysis. semantic_surface_analysis.json and semantic_surface_pair_layer_metrics.csv retain all layers, both metrics, and both variants. semantic_surface_prompt_validation.json records AST and reference-verdict equivalence. semantic_surface_local_audit.json records independent recomputation of all layer means and family-bootstrap intervals from 108,000 metric rows. semantic_surface_state_audit.json records read-only remote checking of all 3600 saved states against the final-layer metric means. Each model contributes 120 instances in 20 equally sized families; aggregates are arithmetic means. The two added variants require 2400 forwards and reuse 1200 original states. No new completions were generated. Table 10 scales every displayed distance and interval by 1000; unscaled full precision is retained here.

## Existing-program failure cases

semantic_failure_cases.json contains selected existing Qwen3.8 prompts, programs and evaluation labels for interval_boundary, merge_touching and substring_case, together with derived counts. These are the three families with zero paired switching: 18 instances, three runs each. NL-only follows A in all 54 records; B-test completions follow A in 51 and neither rule in three; explicit B passes in all 54. The examples use instance 0 and the first run. Interval and substring examples were replayed locally to verify the outputs described in Section 4.2; the merge comparisons were inspected in the saved code. No model generation was rerun. Aggregated proportions for the two 27B models use all A/B-test behavior counts: 139/146 and 119/123 target-rule failures pass the other rule. These are descriptive completion-record proportions, not paired switching rates or independent sample counts.

historical_synthetic_and_controls.tex preserves the earlier exploratory table and transformation-audit narrative. It is not included by paper.tex and is not an appendix. Original numbers and historical scope remain available for replication.

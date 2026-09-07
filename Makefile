PYTHON := .venv/bin/python
VLLM_PYTHON ?= .venv-vllm/bin/python
QWEN25_MODEL ?= Qwen/Qwen2.5-Coder-7B-Instruct
HUMANEVALPLUS_OUTPUT ?= outputs/humanevalplus_qwen_structured
MBPPPLUS_ABLATION_OUTPUT ?= outputs/mbppplus_qwen_ablation
HUMANEVALPLUS_ABLATION_OUTPUT ?= outputs/humanevalplus_qwen_ablation
PROBE_BACKEND ?= torch
PROBE_DEVICE ?= cuda
PROBE_EPOCHS ?= 200
PROBE_LR ?= 0.01
PROBE_WEIGHT_DECAY ?= 0.0001
TEST_MECHANISM_CONDITIONS ?= nl_tests_k1 nl_tests_k2 nl_tests_k3 asserts_only_k1 asserts_only_k2 asserts_only_k3 nl_tests_low1 nl_tests_high1 nl_tests_low2 nl_tests_high2 nl_tests_diverse3 asserts_only_low1 asserts_only_high1
MBPPPLUS_TEST_MECHANISM_OUTPUT ?= outputs/mbppplus_qwen_test_mechanism
HUMANEVALPLUS_TEST_MECHANISM_OUTPUT ?= outputs/humanevalplus_qwen_test_mechanism
SYNTH_MODEL ?= deepseek-v4-flash
SYNTH_NUM_CANDIDATES ?= 12
SYNTH_MAX_TOKENS ?= 4096
SYNTH_TIMEOUT ?= 180
SYNTH_RETRIES ?= 5
SYNTH_SLEEP ?= 0.5
SYNTH_WORKERS ?= 4
SYNTH_MAX_WEAK_PER_TASK ?= 4
SYNTH_KILL_RATE_WEIGHT ?= 10.0
SYNTH_SELECTOR ?= high
SYNTH_COUNT ?= 3
SYNTH_SETTINGS ?= high1 high2 high3 low3 random3 diverse3
SYNTH_ANALYSIS_DIR ?= outputs/analysis/synthetic_tests
LCB_SYNTH_TEST_DIR ?= data/synthetic_public_tests/livecodebench_v6_new
LCB_SYNTH_SELECTOR ?= high
LCB_SYNTH_COUNT ?= 5
LCB_SYNTH_DATASET ?= $(LCB_SYNTH_TEST_DIR)/livecodebench_v6_new_synth_$(LCB_SYNTH_SELECTOR)$(LCB_SYNTH_COUNT).jsonl
LCB_SYNTH_OUTPUT ?= outputs/livecodebench_v6_new_qwen_synthetic_$(LCB_SYNTH_SELECTOR)$(LCB_SYNTH_COUNT)
LCB_SYNTH_CONDITIONS ?= nl_only nl_tests shuffled_tests irrelevant_tests
MBPPPLUS_SYNTH_TEST_DIR ?= data/synthetic_public_tests/mbppplus
HUMANEVALPLUS_SYNTH_TEST_DIR ?= data/synthetic_public_tests/humanevalplus
MBPPPLUS_SYNTH_DATASET ?= $(MBPPPLUS_SYNTH_TEST_DIR)/mbppplus_synth_$(SYNTH_SELECTOR)$(SYNTH_COUNT).jsonl
HUMANEVALPLUS_SYNTH_DATASET ?= $(HUMANEVALPLUS_SYNTH_TEST_DIR)/humanevalplus_synth_$(SYNTH_SELECTOR)$(SYNTH_COUNT).jsonl
MBPPPLUS_SYNTH_OUTPUT ?= outputs/mbppplus_qwen_synthetic_$(SYNTH_SELECTOR)$(SYNTH_COUNT)
HUMANEVALPLUS_SYNTH_OUTPUT ?= outputs/humanevalplus_qwen_synthetic_$(SYNTH_SELECTOR)$(SYNTH_COUNT)
QWEN36_MODEL ?= Qwen/Qwen3.6-27B
QWEN36_DTYPE ?= bfloat16
QWEN36_MAX_NEW_TOKENS ?= 8192
QWEN36_THINKING ?= disabled
QWEN36_TEMPERATURE ?= 0.0
QWEN36_TOP_P ?= 1.0
QWEN36_CONDITIONS ?= nl_only nl_tests shuffled_tests irrelevant_tests
QWEN36_LCB_OUTPUT ?= outputs/livecodebench_v6_new_qwen36_27b_nothink
QWEN36_LCB_SYNTH_OUTPUT ?= outputs/livecodebench_v6_new_qwen36_27b_synth_high5_nothink
QWEN36_VLLM_BATCH_SIZE ?= 32
QWEN36_VLLM_MAX_NUM_SEQS ?= 8
QWEN36_VLLM_MAX_BATCHED_TOKENS ?= 2048
QWEN36_VLLM_MAX_MODEL_LEN ?= 16384
QWEN36_VLLM_GPU_MEMORY_UTILIZATION ?= 0.90
QWEN36_VLLM_SMOKE_OUTPUT ?= outputs/debug_lcb_qwen36_vllm_smoke
QWEN36_SMOKE_GPU ?= 0
LCB_ROOT ?= third_party/LiveCodeBench
LCB_DATA_LOCAL ?= third_party/livecodebench/code_generation_lite

.PHONY: test-controlled audit-controlled prepare-controlled
test-controlled:
	mkdir -p .tmp
	TMPDIR=$(CURDIR)/.tmp TMP=$(CURDIR)/.tmp TEMP=$(CURDIR)/.tmp $(PYTHON) -B -m unittest discover -s tests -v

audit-controlled:
	TMPDIR=$(CURDIR)/.tmp TMP=$(CURDIR)/.tmp TEMP=$(CURDIR)/.tmp $(PYTHON) -B scripts/audit_controls.py

prepare-controlled:
	TMPDIR=$(CURDIR)/.tmp TMP=$(CURDIR)/.tmp TEMP=$(CURDIR)/.tmp $(PYTHON) -B scripts/prepare_main_protocol.py
	TMPDIR=$(CURDIR)/.tmp TMP=$(CURDIR)/.tmp TEMP=$(CURDIR)/.tmp $(PYTHON) -B scripts/prepare_semantic_protocol.py
	TMPDIR=$(CURDIR)/.tmp TMP=$(CURDIR)/.tmp TEMP=$(CURDIR)/.tmp $(PYTHON) -B scripts/prepare_quality_protocol.py --target-tasks 180

.PHONY: smoke generate-mvp evaluate-mvp summarize-mvp probe-mvp
.PHONY: generate-mbpp-mgpu evaluate-mbpp summarize-mbpp generate-humaneval-mgpu evaluate-humaneval-full summarize-humaneval-full
.PHONY: generate-humanevalplus-mgpu evaluate-humanevalplus summarize-humanevalplus generate-mbppplus-mgpu evaluate-mbppplus summarize-mbppplus
.PHONY: prepare-lcb prepare-lcb-v6-new generate-lcb generate-lcb-v6-new generate-lcb-v6-new-full-mgpu export-lcb-nl-tests export-lcb-nl-only export-lcb-shuffled export-lcb-irrelevant
.PHONY: export-lcb-v6-new-nl-tests export-lcb-v6-new-nl-only export-lcb-v6-new-shuffled export-lcb-v6-new-irrelevant prepare-quixbugs prepare-bugsinpy
.PHONY: eval-lcb-v6-new-nl-only eval-lcb-v6-new-nl-tests eval-lcb-v6-new-shuffled eval-lcb-v6-new-irrelevant analyze-behavior
.PHONY: probe-mbppplus-success probe-mbppplus-condition probe-mbppplus-rescue
.PHONY: probe-humanevalplus-success probe-humanevalplus-condition probe-humanevalplus-rescue probe-interpretability
.PHONY: generate-mbppplus-ablation-mgpu evaluate-mbppplus-ablation summarize-mbppplus-ablation
.PHONY: generate-humanevalplus-ablation-mgpu evaluate-humanevalplus-ablation summarize-humanevalplus-ablation
.PHONY: merge-mbppplus-ablation merge-humanevalplus-ablation summarize-mbppplus-combined summarize-humanevalplus-combined analyze-ablation-behavior ablation-evalplus
.PHONY: generate-mbppplus-test-mechanism-mgpu evaluate-mbppplus-test-mechanism summarize-mbppplus-test-mechanism merge-mbppplus-test-mechanism summarize-mbppplus-test-mechanism-combined analyze-mbppplus-hidden-shift
.PHONY: generate-humanevalplus-test-mechanism-mgpu evaluate-humanevalplus-test-mechanism summarize-humanevalplus-test-mechanism merge-humanevalplus-test-mechanism summarize-humanevalplus-test-mechanism-combined analyze-humanevalplus-hidden-shift analyze-test-mechanism-behavior test-mechanism-evalplus
.PHONY: prepare-mbppplus-synthetic-tests generate-mbppplus-synthetic-mgpu evaluate-mbppplus-synthetic summarize-mbppplus-synthetic
.PHONY: prepare-humanevalplus-synthetic-tests generate-humanevalplus-synthetic-mgpu evaluate-humanevalplus-synthetic summarize-humanevalplus-synthetic
.PHONY: merge-synthetic-analysis analyze-synthetic-behavior analyze-mbppplus-synthetic-hidden-shift analyze-humanevalplus-synthetic-hidden-shift analyze-synthetic-hidden-shift
.PHONY: prepare-lcb-synthetic-tests generate-lcb-synthetic-mgpu evaluate-lcb-synthetic analyze-lcb-synthetic-behavior analyze-lcb-synthetic-hidden-shift
.PHONY: generate-lcb-qwen36-mgpu evaluate-lcb-qwen36 analyze-lcb-qwen36-behavior analyze-lcb-qwen36-hidden-shift
.PHONY: generate-lcb-qwen36-synth-high5-mgpu evaluate-lcb-qwen36-synth-high5 analyze-lcb-qwen36-synth-high5-behavior analyze-lcb-qwen36-synth-high5-hidden-shift
.PHONY: smoke-lcb-qwen36
.PHONY: smoke-lcb-qwen36-vllm generate-lcb-qwen36-vllm-mgpu collect-lcb-qwen36-states-mgpu run-lcb-qwen36-vllm-mgpu
.PHONY: generate-lcb-qwen36-synth-high5-vllm-mgpu collect-lcb-qwen36-synth-high5-states-mgpu run-lcb-qwen36-synth-high5-vllm-mgpu

smoke:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_smoke.py --limit 3

generate-mvp:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation.py \
		--dataset data/humaneval.jsonl \
		--dataset-name humaneval \
		--model-path $(QWEN25_MODEL) \
		--conditions nl_only nl_tests shuffled_tests irrelevant_tests \
		--limit 20 \
		--visible-tests 3 \
		--device cuda \
		--max-new-tokens 256 \
		--collect-states \
		--output-dir outputs/humaneval_qwen_mvp

evaluate-mvp:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_generations.py \
		--generations outputs/humaneval_qwen_mvp/generations.jsonl \
		--output outputs/humaneval_qwen_mvp/eval_results.jsonl \
		--tmp-root .tmp

summarize-mvp:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results outputs/humaneval_qwen_mvp/eval_results.jsonl \
		--output outputs/humaneval_qwen_mvp/summary.csv

probe-mvp:
	$(PYTHON) scripts/train_probe.py \
		--eval-results outputs/humaneval_qwen_mvp/eval_results.jsonl \
		--states-dir outputs/humaneval_qwen_mvp/states \
		--output outputs/humaneval_qwen_mvp/probe_results.json

generate-mbpp-mgpu:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation_multi_gpu.py \
		--gpus $(GPUS) \
		--model-path $(QWEN25_MODEL) \
		--dataset data/mbpp.jsonl \
		--dataset-name mbpp \
		--visible-tests 1 \
		--max-new-tokens 512 \
		--output-dir outputs/mbpp_qwen_full

evaluate-mbpp:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_generations.py \
		--generations outputs/mbpp_qwen_full/generations.jsonl \
		--output outputs/mbpp_qwen_full/eval_results.jsonl \
		--tmp-root .tmp

summarize-mbpp:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results outputs/mbpp_qwen_full/eval_results.jsonl \
		--output outputs/mbpp_qwen_full/summary.csv

generate-humaneval-mgpu:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation_multi_gpu.py \
		--gpus $(GPUS) \
		--model-path $(QWEN25_MODEL) \
		--dataset data/humaneval.jsonl \
		--dataset-name humaneval \
		--visible-tests 1 \
		--max-new-tokens 512 \
		--output-dir outputs/humaneval_qwen_full

evaluate-humaneval-full:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_generations.py \
		--generations outputs/humaneval_qwen_full/generations.jsonl \
		--output outputs/humaneval_qwen_full/eval_results.jsonl \
		--tmp-root .tmp

summarize-humaneval-full:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results outputs/humaneval_qwen_full/eval_results.jsonl \
		--output outputs/humaneval_qwen_full/summary.csv

generate-humanevalplus-mgpu:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation_multi_gpu.py \
		--gpus $(GPUS) \
		--model-path $(QWEN25_MODEL) \
		--dataset data/humanevalplus.jsonl \
		--dataset-name humanevalplus \
		--visible-tests 3 \
		--max-new-tokens 512 \
		--output-dir $(HUMANEVALPLUS_OUTPUT)

evaluate-humanevalplus:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_evalplus_generations.py \
		--generations $(HUMANEVALPLUS_OUTPUT)/generations.jsonl \
		--dataset data/humanevalplus.jsonl \
		--output $(HUMANEVALPLUS_OUTPUT)/eval_results.jsonl \
		--tmp-root .tmp

summarize-humanevalplus:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results $(HUMANEVALPLUS_OUTPUT)/eval_results.jsonl \
		--output $(HUMANEVALPLUS_OUTPUT)/summary.csv

generate-mbppplus-mgpu:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation_multi_gpu.py \
		--gpus $(GPUS) \
		--model-path $(QWEN25_MODEL) \
		--dataset data/mbppplus.jsonl \
		--dataset-name mbppplus \
		--visible-tests 3 \
		--max-new-tokens 512 \
		--output-dir outputs/mbppplus_qwen_full

evaluate-mbppplus:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_evalplus_generations.py \
		--generations outputs/mbppplus_qwen_full/generations.jsonl \
		--dataset data/mbppplus.jsonl \
		--output outputs/mbppplus_qwen_full/eval_results.jsonl \
		--tmp-root .tmp

summarize-mbppplus:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results outputs/mbppplus_qwen_full/eval_results.jsonl \
		--output outputs/mbppplus_qwen_full/summary.csv

generate-mbppplus-ablation-mgpu:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation_multi_gpu.py \
		--gpus $(GPUS) \
		--model-path $(QWEN25_MODEL) \
		--dataset data/mbppplus.jsonl \
		--dataset-name mbppplus \
		--conditions asserts_only test_names_only \
		--visible-tests 3 \
		--max-new-tokens 512 \
		--output-dir $(MBPPPLUS_ABLATION_OUTPUT)

evaluate-mbppplus-ablation:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_evalplus_generations.py \
		--generations $(MBPPPLUS_ABLATION_OUTPUT)/generations.jsonl \
		--dataset data/mbppplus.jsonl \
		--output $(MBPPPLUS_ABLATION_OUTPUT)/eval_results.jsonl \
		--tmp-root .tmp

summarize-mbppplus-ablation:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results $(MBPPPLUS_ABLATION_OUTPUT)/eval_results.jsonl \
		--output $(MBPPPLUS_ABLATION_OUTPUT)/summary.csv

merge-mbppplus-ablation:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/merge_eval_results.py \
		--inputs outputs/mbppplus_qwen_full/eval_results.jsonl $(MBPPPLUS_ABLATION_OUTPUT)/eval_results.jsonl \
		--output outputs/analysis/ablation/mbppplus_eval_results_combined.jsonl

summarize-mbppplus-combined:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results outputs/analysis/ablation/mbppplus_eval_results_combined.jsonl \
		--output outputs/analysis/ablation/mbppplus_summary_combined.csv

generate-humanevalplus-ablation-mgpu:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation_multi_gpu.py \
		--gpus $(GPUS) \
		--model-path $(QWEN25_MODEL) \
		--dataset data/humanevalplus.jsonl \
		--dataset-name humanevalplus \
		--conditions asserts_only test_names_only \
		--visible-tests 3 \
		--max-new-tokens 512 \
		--output-dir $(HUMANEVALPLUS_ABLATION_OUTPUT)

evaluate-humanevalplus-ablation:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_evalplus_generations.py \
		--generations $(HUMANEVALPLUS_ABLATION_OUTPUT)/generations.jsonl \
		--dataset data/humanevalplus.jsonl \
		--output $(HUMANEVALPLUS_ABLATION_OUTPUT)/eval_results.jsonl \
		--tmp-root .tmp

summarize-humanevalplus-ablation:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results $(HUMANEVALPLUS_ABLATION_OUTPUT)/eval_results.jsonl \
		--output $(HUMANEVALPLUS_ABLATION_OUTPUT)/summary.csv

merge-humanevalplus-ablation:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/merge_eval_results.py \
		--inputs $(HUMANEVALPLUS_OUTPUT)/eval_results.jsonl $(HUMANEVALPLUS_ABLATION_OUTPUT)/eval_results.jsonl \
		--output outputs/analysis/ablation/humanevalplus_eval_results_combined.jsonl

summarize-humanevalplus-combined:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results outputs/analysis/ablation/humanevalplus_eval_results_combined.jsonl \
		--output outputs/analysis/ablation/humanevalplus_summary_combined.csv

analyze-ablation-behavior:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/analyze_behavior_flips.py \
		--run mbppplus=outputs/analysis/ablation/mbppplus_eval_results_combined.jsonl \
		--run humanevalplus=outputs/analysis/ablation/humanevalplus_eval_results_combined.jsonl \
		--pairs nl_only:nl_tests nl_only:asserts_only nl_only:test_names_only asserts_only:nl_tests test_names_only:nl_tests shuffled_tests:nl_tests \
		--output-dir outputs/analysis/ablation/behavior

ablation-evalplus: generate-mbppplus-ablation-mgpu evaluate-mbppplus-ablation summarize-mbppplus-ablation merge-mbppplus-ablation summarize-mbppplus-combined generate-humanevalplus-ablation-mgpu evaluate-humanevalplus-ablation summarize-humanevalplus-ablation merge-humanevalplus-ablation summarize-humanevalplus-combined analyze-ablation-behavior

generate-mbppplus-test-mechanism-mgpu:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation_multi_gpu.py \
		--gpus $(GPUS) \
		--model-path $(QWEN25_MODEL) \
		--dataset data/mbppplus.jsonl \
		--dataset-name mbppplus \
		--conditions $(TEST_MECHANISM_CONDITIONS) \
		--visible-tests 3 \
		--max-new-tokens 512 \
		--output-dir $(MBPPPLUS_TEST_MECHANISM_OUTPUT)

evaluate-mbppplus-test-mechanism:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_evalplus_generations.py \
		--generations $(MBPPPLUS_TEST_MECHANISM_OUTPUT)/generations.jsonl \
		--dataset data/mbppplus.jsonl \
		--output $(MBPPPLUS_TEST_MECHANISM_OUTPUT)/eval_results.jsonl \
		--tmp-root .tmp

summarize-mbppplus-test-mechanism:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results $(MBPPPLUS_TEST_MECHANISM_OUTPUT)/eval_results.jsonl \
		--output $(MBPPPLUS_TEST_MECHANISM_OUTPUT)/summary.csv

merge-mbppplus-test-mechanism:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/merge_eval_results.py \
		--inputs outputs/mbppplus_qwen_full/eval_results.jsonl $(MBPPPLUS_TEST_MECHANISM_OUTPUT)/eval_results.jsonl \
		--output outputs/analysis/test_mechanism/mbppplus_eval_results_combined.jsonl

summarize-mbppplus-test-mechanism-combined:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results outputs/analysis/test_mechanism/mbppplus_eval_results_combined.jsonl \
		--output outputs/analysis/test_mechanism/mbppplus_summary_combined.csv

analyze-mbppplus-hidden-shift:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/analyze_hidden_state_shift.py \
		--eval-results outputs/analysis/test_mechanism/mbppplus_eval_results_combined.jsonl \
		--baseline-condition nl_only \
		--output-dir outputs/analysis/test_mechanism/mbppplus_hidden_shift

generate-humanevalplus-test-mechanism-mgpu:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation_multi_gpu.py \
		--gpus $(GPUS) \
		--model-path $(QWEN25_MODEL) \
		--dataset data/humanevalplus.jsonl \
		--dataset-name humanevalplus \
		--conditions $(TEST_MECHANISM_CONDITIONS) \
		--visible-tests 3 \
		--max-new-tokens 512 \
		--output-dir $(HUMANEVALPLUS_TEST_MECHANISM_OUTPUT)

evaluate-humanevalplus-test-mechanism:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_evalplus_generations.py \
		--generations $(HUMANEVALPLUS_TEST_MECHANISM_OUTPUT)/generations.jsonl \
		--dataset data/humanevalplus.jsonl \
		--output $(HUMANEVALPLUS_TEST_MECHANISM_OUTPUT)/eval_results.jsonl \
		--tmp-root .tmp

summarize-humanevalplus-test-mechanism:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results $(HUMANEVALPLUS_TEST_MECHANISM_OUTPUT)/eval_results.jsonl \
		--output $(HUMANEVALPLUS_TEST_MECHANISM_OUTPUT)/summary.csv

merge-humanevalplus-test-mechanism:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/merge_eval_results.py \
		--inputs $(HUMANEVALPLUS_OUTPUT)/eval_results.jsonl $(HUMANEVALPLUS_TEST_MECHANISM_OUTPUT)/eval_results.jsonl \
		--output outputs/analysis/test_mechanism/humanevalplus_eval_results_combined.jsonl

summarize-humanevalplus-test-mechanism-combined:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results outputs/analysis/test_mechanism/humanevalplus_eval_results_combined.jsonl \
		--output outputs/analysis/test_mechanism/humanevalplus_summary_combined.csv

analyze-humanevalplus-hidden-shift:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/analyze_hidden_state_shift.py \
		--eval-results outputs/analysis/test_mechanism/humanevalplus_eval_results_combined.jsonl \
		--baseline-condition nl_only \
		--output-dir outputs/analysis/test_mechanism/humanevalplus_hidden_shift

analyze-test-mechanism-behavior:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/analyze_behavior_flips.py \
		--run mbppplus=outputs/analysis/test_mechanism/mbppplus_eval_results_combined.jsonl \
		--run humanevalplus=outputs/analysis/test_mechanism/humanevalplus_eval_results_combined.jsonl \
		--pairs nl_only:nl_tests_k1 nl_only:nl_tests_k2 nl_only:nl_tests_k3 nl_only:nl_tests_high1 nl_only:nl_tests_low1 nl_tests_low1:nl_tests_high1 nl_tests_low2:nl_tests_high2 nl_tests_k1:nl_tests_k3 asserts_only_k1:asserts_only_k3 \
		--output-dir outputs/analysis/test_mechanism/behavior

test-mechanism-evalplus: generate-mbppplus-test-mechanism-mgpu evaluate-mbppplus-test-mechanism summarize-mbppplus-test-mechanism merge-mbppplus-test-mechanism summarize-mbppplus-test-mechanism-combined analyze-mbppplus-hidden-shift generate-humanevalplus-test-mechanism-mgpu evaluate-humanevalplus-test-mechanism summarize-humanevalplus-test-mechanism merge-humanevalplus-test-mechanism summarize-humanevalplus-test-mechanism-combined analyze-humanevalplus-hidden-shift analyze-test-mechanism-behavior

prepare-mbppplus-synthetic-tests:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/generate_synthetic_public_tests.py \
		--dataset data/mbppplus.jsonl \
		--dataset-name mbppplus \
		--output-dir $(MBPPPLUS_SYNTH_TEST_DIR) \
		--model $(SYNTH_MODEL) \
		--num-candidates $(SYNTH_NUM_CANDIDATES) \
		--max-tokens $(SYNTH_MAX_TOKENS) \
		--timeout $(SYNTH_TIMEOUT) \
		--retries $(SYNTH_RETRIES) \
		--sleep $(SYNTH_SLEEP) \
		--workers $(SYNTH_WORKERS) \
		--weak-generations outputs/mbppplus_qwen_full/eval_results.jsonl \
		--max-weak-per-task $(SYNTH_MAX_WEAK_PER_TASK) \
		--kill-rate-weight $(SYNTH_KILL_RATE_WEIGHT) \
		--resume

generate-mbppplus-synthetic-mgpu:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation_multi_gpu.py \
		--gpus $(GPUS) \
		--model-path $(QWEN25_MODEL) \
		--dataset $(MBPPPLUS_SYNTH_DATASET) \
		--dataset-name mbppplus \
		--conditions nl_tests asserts_only \
		--visible-tests $(SYNTH_COUNT) \
		--max-new-tokens 512 \
		--output-dir $(MBPPPLUS_SYNTH_OUTPUT)

evaluate-mbppplus-synthetic:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_evalplus_generations.py \
		--generations $(MBPPPLUS_SYNTH_OUTPUT)/generations.jsonl \
		--dataset $(MBPPPLUS_SYNTH_DATASET) \
		--output $(MBPPPLUS_SYNTH_OUTPUT)/eval_results.jsonl \
		--tmp-root .tmp

summarize-mbppplus-synthetic:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results $(MBPPPLUS_SYNTH_OUTPUT)/eval_results.jsonl \
		--output $(MBPPPLUS_SYNTH_OUTPUT)/summary.csv

prepare-humanevalplus-synthetic-tests:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/generate_synthetic_public_tests.py \
		--dataset data/humanevalplus.jsonl \
		--dataset-name humanevalplus \
		--output-dir $(HUMANEVALPLUS_SYNTH_TEST_DIR) \
		--model $(SYNTH_MODEL) \
		--num-candidates $(SYNTH_NUM_CANDIDATES) \
		--max-tokens $(SYNTH_MAX_TOKENS) \
		--timeout $(SYNTH_TIMEOUT) \
		--retries $(SYNTH_RETRIES) \
		--sleep $(SYNTH_SLEEP) \
		--workers $(SYNTH_WORKERS) \
		--weak-generations outputs/humanevalplus_qwen_structured/eval_results.jsonl \
		--max-weak-per-task $(SYNTH_MAX_WEAK_PER_TASK) \
		--kill-rate-weight $(SYNTH_KILL_RATE_WEIGHT) \
		--resume

generate-humanevalplus-synthetic-mgpu:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation_multi_gpu.py \
		--gpus $(GPUS) \
		--model-path $(QWEN25_MODEL) \
		--dataset $(HUMANEVALPLUS_SYNTH_DATASET) \
		--dataset-name humanevalplus \
		--conditions nl_tests asserts_only \
		--visible-tests $(SYNTH_COUNT) \
		--max-new-tokens 512 \
		--output-dir $(HUMANEVALPLUS_SYNTH_OUTPUT)

evaluate-humanevalplus-synthetic:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_evalplus_generations.py \
		--generations $(HUMANEVALPLUS_SYNTH_OUTPUT)/generations.jsonl \
		--dataset $(HUMANEVALPLUS_SYNTH_DATASET) \
		--output $(HUMANEVALPLUS_SYNTH_OUTPUT)/eval_results.jsonl \
		--tmp-root .tmp

summarize-humanevalplus-synthetic:
	$(PYTHON) scripts/summarize_results.py \
		--eval-results $(HUMANEVALPLUS_SYNTH_OUTPUT)/eval_results.jsonl \
		--output $(HUMANEVALPLUS_SYNTH_OUTPUT)/summary.csv

merge-synthetic-analysis:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/merge_synthetic_analysis.py \
		--settings $(SYNTH_SETTINGS) \
		--output-dir $(SYNTH_ANALYSIS_DIR)

analyze-synthetic-behavior: merge-synthetic-analysis
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/analyze_behavior_flips.py \
		--run mbppplus=$(SYNTH_ANALYSIS_DIR)/mbppplus_synthetic_eval_results_tagged.jsonl \
		--run humanevalplus=$(SYNTH_ANALYSIS_DIR)/humanevalplus_synthetic_eval_results_tagged.jsonl \
		--pairs nl_only:orig_nl_tests nl_only:synth_high1_nl_tests nl_only:synth_high2_nl_tests nl_only:synth_high3_nl_tests synth_high1_nl_tests:synth_high2_nl_tests synth_high2_nl_tests:synth_high3_nl_tests synth_high1_nl_tests:synth_high3_nl_tests nl_only:synth_low3_nl_tests nl_only:synth_random3_nl_tests nl_only:synth_diverse3_nl_tests synth_low3_nl_tests:synth_high3_nl_tests synth_random3_nl_tests:synth_high3_nl_tests synth_diverse3_nl_tests:synth_high3_nl_tests synth_high3_asserts_only:synth_high3_nl_tests \
		--output-dir $(SYNTH_ANALYSIS_DIR)/behavior

analyze-mbppplus-synthetic-hidden-shift: merge-synthetic-analysis
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/analyze_hidden_state_shift.py \
		--eval-results $(SYNTH_ANALYSIS_DIR)/mbppplus_synthetic_eval_results_tagged.jsonl \
		--baseline-condition nl_only \
		--conditions orig_nl_tests synth_high1_nl_tests synth_high2_nl_tests synth_high3_nl_tests synth_low3_nl_tests synth_random3_nl_tests synth_diverse3_nl_tests synth_high1_asserts_only synth_high2_asserts_only synth_high3_asserts_only synth_low3_asserts_only synth_random3_asserts_only synth_diverse3_asserts_only \
		--output-dir $(SYNTH_ANALYSIS_DIR)/mbppplus_hidden_shift

analyze-humanevalplus-synthetic-hidden-shift: merge-synthetic-analysis
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/analyze_hidden_state_shift.py \
		--eval-results $(SYNTH_ANALYSIS_DIR)/humanevalplus_synthetic_eval_results_tagged.jsonl \
		--baseline-condition nl_only \
		--conditions orig_nl_tests synth_high1_nl_tests synth_high2_nl_tests synth_high3_nl_tests synth_low3_nl_tests synth_random3_nl_tests synth_diverse3_nl_tests synth_high1_asserts_only synth_high2_asserts_only synth_high3_asserts_only synth_low3_asserts_only synth_random3_asserts_only synth_diverse3_asserts_only \
		--output-dir $(SYNTH_ANALYSIS_DIR)/humanevalplus_hidden_shift

analyze-synthetic-hidden-shift: analyze-mbppplus-synthetic-hidden-shift analyze-humanevalplus-synthetic-hidden-shift

prepare-lcb-synthetic-tests:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/generate_lcb_synthetic_public_tests.py \
		--dataset data/livecodebench_release_v6_minus_v5.jsonl \
		--output-dir $(LCB_SYNTH_TEST_DIR) \
		--model $(SYNTH_MODEL) \
		--num-candidates $(SYNTH_NUM_CANDIDATES) \
		--max-tokens $(SYNTH_MAX_TOKENS) \
		--timeout $(SYNTH_TIMEOUT) \
		--retries $(SYNTH_RETRIES) \
		--sleep $(SYNTH_SLEEP) \
		--workers $(SYNTH_WORKERS) \
		--counts 3 5 \
		--selectors high random diverse \
		--reprocess-existing \
		--resume

generate-lcb-synthetic-mgpu:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation_multi_gpu.py \
		--gpus $(GPUS) \
		--model-path $(QWEN25_MODEL) \
		--dataset $(LCB_SYNTH_DATASET) \
		--dataset-name livecodebench_v6_minus_v5 \
		--conditions $(LCB_SYNTH_CONDITIONS) \
		--max-new-tokens 1024 \
		--output-dir $(LCB_SYNTH_OUTPUT)

evaluate-lcb-synthetic:
	@mkdir -p $(LCB_SYNTH_OUTPUT)/eval
	@for condition in $(LCB_SYNTH_CONDITIONS); do \
		echo "Evaluating $$condition"; \
		TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_lcb_subset.py \
			--dataset data/livecodebench_release_v6_minus_v5.jsonl \
			--generations $(LCB_SYNTH_OUTPUT)/generations.jsonl \
			--condition $$condition \
			--output-dir $(LCB_SYNTH_OUTPUT)/eval; \
	done

analyze-lcb-synthetic-behavior:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/analyze_behavior_flips.py \
		--run livecodebench_v6_new_synth_$(LCB_SYNTH_SELECTOR)$(LCB_SYNTH_COUNT)=$(LCB_SYNTH_OUTPUT)/eval \
		--pairs nl_only:nl_tests nl_only:shuffled_tests nl_only:irrelevant_tests shuffled_tests:nl_tests irrelevant_tests:nl_tests \
		--output-dir $(LCB_SYNTH_OUTPUT)/behavior

analyze-lcb-synthetic-hidden-shift:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/merge_eval_results.py \
		--inputs $(foreach c,$(LCB_SYNTH_CONDITIONS),$(LCB_SYNTH_OUTPUT)/eval/$(c)_eval_all.jsonl) \
		--output $(LCB_SYNTH_OUTPUT)/eval_results_combined.jsonl
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/analyze_hidden_state_shift.py \
		--eval-results $(LCB_SYNTH_OUTPUT)/eval_results_combined.jsonl \
		--baseline-condition nl_only \
		--conditions nl_tests shuffled_tests irrelevant_tests \
		--output-dir $(LCB_SYNTH_OUTPUT)/hidden_shift

generate-lcb-qwen36-mgpu:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation_multi_gpu.py \
		--gpus $(GPUS) \
		--dataset data/livecodebench_release_v6_minus_v5.jsonl \
		--dataset-name livecodebench_v6_minus_v5 \
		--model-path $(QWEN36_MODEL) \
		--conditions $(QWEN36_CONDITIONS) \
		--prompt-format chat \
		--thinking $(QWEN36_THINKING) \
		--dtype $(QWEN36_DTYPE) \
		--max-new-tokens $(QWEN36_MAX_NEW_TOKENS) \
		--temperature $(QWEN36_TEMPERATURE) \
		--top-p $(QWEN36_TOP_P) \
		--output-dir $(QWEN36_LCB_OUTPUT)

generate-lcb-qwen36-vllm-mgpu:
	TMPDIR=$(CURDIR)/.tmp TMP=$(CURDIR)/.tmp TEMP=$(CURDIR)/.tmp $(VLLM_PYTHON) scripts/run_generation_vllm_multi_gpu.py \
		--gpus $(GPUS) \
		--dataset data/livecodebench_release_v6_minus_v5.jsonl \
		--dataset-name livecodebench_v6_minus_v5 \
		--model-path $(QWEN36_MODEL) \
		--conditions $(QWEN36_CONDITIONS) \
		--prompt-format chat \
		--thinking $(QWEN36_THINKING) \
		--dtype $(QWEN36_DTYPE) \
		--max-new-tokens $(QWEN36_MAX_NEW_TOKENS) \
		--temperature $(QWEN36_TEMPERATURE) \
		--top-p $(QWEN36_TOP_P) \
		--batch-size $(QWEN36_VLLM_BATCH_SIZE) \
		--max-num-seqs $(QWEN36_VLLM_MAX_NUM_SEQS) \
		--max-num-batched-tokens $(QWEN36_VLLM_MAX_BATCHED_TOKENS) \
		--max-model-len $(QWEN36_VLLM_MAX_MODEL_LEN) \
		--gpu-memory-utilization $(QWEN36_VLLM_GPU_MEMORY_UTILIZATION) \
		--output-dir $(QWEN36_LCB_OUTPUT)

collect-lcb-qwen36-states-mgpu:
	TMPDIR=$(CURDIR)/.tmp TMP=$(CURDIR)/.tmp TEMP=$(CURDIR)/.tmp $(PYTHON) scripts/collect_generation_states_multi_gpu.py \
		--gpus $(GPUS) \
		--generations $(QWEN36_LCB_OUTPUT)/generations_vllm.jsonl \
		--model-path $(QWEN36_MODEL) \
		--output-dir $(QWEN36_LCB_OUTPUT) \
		--conditions $(QWEN36_CONDITIONS) \
		--prompt-format chat \
		--thinking $(QWEN36_THINKING) \
		--dtype $(QWEN36_DTYPE)

run-lcb-qwen36-vllm-mgpu:
	$(MAKE) generate-lcb-qwen36-vllm-mgpu GPUS="$(GPUS)"
	$(MAKE) collect-lcb-qwen36-states-mgpu GPUS="$(GPUS)"

smoke-lcb-qwen36:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation.py \
		--dataset data/livecodebench_release_v6_minus_v5.jsonl \
		--dataset-name livecodebench_v6_minus_v5 \
		--model-path $(QWEN36_MODEL) \
		--conditions nl_only nl_tests \
		--limit 2 \
		--device cuda:0 \
		--prompt-format chat \
		--thinking $(QWEN36_THINKING) \
		--dtype $(QWEN36_DTYPE) \
		--max-new-tokens $(QWEN36_MAX_NEW_TOKENS) \
		--temperature $(QWEN36_TEMPERATURE) \
		--top-p $(QWEN36_TOP_P) \
		--collect-states \
		--output-dir outputs/debug_lcb_qwen36_smoke

smoke-lcb-qwen36-vllm:
	CUDA_VISIBLE_DEVICES=$(QWEN36_SMOKE_GPU) TMPDIR=$(CURDIR)/.tmp TMP=$(CURDIR)/.tmp TEMP=$(CURDIR)/.tmp \
		$(VLLM_PYTHON) scripts/run_generation_vllm.py \
		--dataset data/livecodebench_release_v6_minus_v5.jsonl \
		--dataset-name livecodebench_v6_minus_v5 \
		--model-path $(QWEN36_MODEL) \
		--conditions nl_only nl_tests \
		--limit 2 \
		--prompt-format chat \
		--thinking $(QWEN36_THINKING) \
		--dtype $(QWEN36_DTYPE) \
		--max-new-tokens $(QWEN36_MAX_NEW_TOKENS) \
		--temperature $(QWEN36_TEMPERATURE) \
		--top-p $(QWEN36_TOP_P) \
		--batch-size 4 \
		--max-num-seqs 2 \
		--max-num-batched-tokens $(QWEN36_VLLM_MAX_BATCHED_TOKENS) \
		--max-model-len $(QWEN36_VLLM_MAX_MODEL_LEN) \
		--gpu-memory-utilization $(QWEN36_VLLM_GPU_MEMORY_UTILIZATION) \
		--output-dir $(QWEN36_VLLM_SMOKE_OUTPUT)/vllm_raw
	CUDA_VISIBLE_DEVICES=$(QWEN36_SMOKE_GPU) TMPDIR=$(CURDIR)/.tmp TMP=$(CURDIR)/.tmp TEMP=$(CURDIR)/.tmp \
		$(PYTHON) scripts/collect_generation_states.py \
		--generations $(QWEN36_VLLM_SMOKE_OUTPUT)/vllm_raw/generations.jsonl \
		--model-path $(QWEN36_MODEL) \
		--output-dir $(QWEN36_VLLM_SMOKE_OUTPUT) \
		--states-dir $(QWEN36_VLLM_SMOKE_OUTPUT)/states \
		--device cuda:0 \
		--prompt-format chat \
		--thinking $(QWEN36_THINKING) \
		--dtype $(QWEN36_DTYPE)

evaluate-lcb-qwen36:
	@mkdir -p $(QWEN36_LCB_OUTPUT)/eval
	@for condition in $(QWEN36_CONDITIONS); do \
		echo "Evaluating $$condition"; \
		TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_lcb_subset.py \
			--livecodebench-root $(LCB_ROOT) \
			--dataset data/livecodebench_release_v6_minus_v5.jsonl \
			--generations $(QWEN36_LCB_OUTPUT)/generations.jsonl \
			--condition $$condition \
			--output-dir $(QWEN36_LCB_OUTPUT)/eval; \
	done

analyze-lcb-qwen36-behavior:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/analyze_behavior_flips.py \
		--run qwen36_27b_original=$(QWEN36_LCB_OUTPUT)/eval \
		--pairs nl_only:nl_tests nl_only:shuffled_tests nl_only:irrelevant_tests shuffled_tests:nl_tests irrelevant_tests:nl_tests \
		--output-dir $(QWEN36_LCB_OUTPUT)/behavior

analyze-lcb-qwen36-hidden-shift:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/merge_eval_results.py \
		--inputs $(foreach c,$(QWEN36_CONDITIONS),$(QWEN36_LCB_OUTPUT)/eval/$(c)_eval_all.jsonl) \
		--output $(QWEN36_LCB_OUTPUT)/eval_results_combined.jsonl
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/analyze_hidden_state_shift.py \
		--eval-results $(QWEN36_LCB_OUTPUT)/eval_results_combined.jsonl \
		--baseline-condition nl_only \
		--conditions nl_tests shuffled_tests irrelevant_tests \
		--output-dir $(QWEN36_LCB_OUTPUT)/hidden_shift

generate-lcb-qwen36-synth-high5-mgpu:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation_multi_gpu.py \
		--gpus $(GPUS) \
		--dataset data/synthetic_public_tests/livecodebench_v6_new/livecodebench_v6_new_synth_high5.jsonl \
		--dataset-name livecodebench_v6_minus_v5 \
		--model-path $(QWEN36_MODEL) \
		--conditions $(QWEN36_CONDITIONS) \
		--prompt-format chat \
		--thinking $(QWEN36_THINKING) \
		--dtype $(QWEN36_DTYPE) \
		--max-new-tokens $(QWEN36_MAX_NEW_TOKENS) \
		--temperature $(QWEN36_TEMPERATURE) \
		--top-p $(QWEN36_TOP_P) \
		--output-dir $(QWEN36_LCB_SYNTH_OUTPUT)

generate-lcb-qwen36-synth-high5-vllm-mgpu:
	TMPDIR=$(CURDIR)/.tmp TMP=$(CURDIR)/.tmp TEMP=$(CURDIR)/.tmp $(VLLM_PYTHON) scripts/run_generation_vllm_multi_gpu.py \
		--gpus $(GPUS) \
		--dataset data/synthetic_public_tests/livecodebench_v6_new/livecodebench_v6_new_synth_high5.jsonl \
		--dataset-name livecodebench_v6_minus_v5 \
		--model-path $(QWEN36_MODEL) \
		--conditions $(QWEN36_CONDITIONS) \
		--prompt-format chat \
		--thinking $(QWEN36_THINKING) \
		--dtype $(QWEN36_DTYPE) \
		--max-new-tokens $(QWEN36_MAX_NEW_TOKENS) \
		--temperature $(QWEN36_TEMPERATURE) \
		--top-p $(QWEN36_TOP_P) \
		--batch-size $(QWEN36_VLLM_BATCH_SIZE) \
		--max-num-seqs $(QWEN36_VLLM_MAX_NUM_SEQS) \
		--max-num-batched-tokens $(QWEN36_VLLM_MAX_BATCHED_TOKENS) \
		--max-model-len $(QWEN36_VLLM_MAX_MODEL_LEN) \
		--gpu-memory-utilization $(QWEN36_VLLM_GPU_MEMORY_UTILIZATION) \
		--output-dir $(QWEN36_LCB_SYNTH_OUTPUT)

collect-lcb-qwen36-synth-high5-states-mgpu:
	TMPDIR=$(CURDIR)/.tmp TMP=$(CURDIR)/.tmp TEMP=$(CURDIR)/.tmp $(PYTHON) scripts/collect_generation_states_multi_gpu.py \
		--gpus $(GPUS) \
		--generations $(QWEN36_LCB_SYNTH_OUTPUT)/generations_vllm.jsonl \
		--model-path $(QWEN36_MODEL) \
		--output-dir $(QWEN36_LCB_SYNTH_OUTPUT) \
		--conditions $(QWEN36_CONDITIONS) \
		--prompt-format chat \
		--thinking $(QWEN36_THINKING) \
		--dtype $(QWEN36_DTYPE)

run-lcb-qwen36-synth-high5-vllm-mgpu:
	$(MAKE) generate-lcb-qwen36-synth-high5-vllm-mgpu GPUS="$(GPUS)"
	$(MAKE) collect-lcb-qwen36-synth-high5-states-mgpu GPUS="$(GPUS)"

evaluate-lcb-qwen36-synth-high5:
	@mkdir -p $(QWEN36_LCB_SYNTH_OUTPUT)/eval
	@for condition in $(QWEN36_CONDITIONS); do \
		echo "Evaluating $$condition"; \
		TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_lcb_subset.py \
			--livecodebench-root $(LCB_ROOT) \
			--dataset data/livecodebench_release_v6_minus_v5.jsonl \
			--generations $(QWEN36_LCB_SYNTH_OUTPUT)/generations.jsonl \
			--condition $$condition \
			--output-dir $(QWEN36_LCB_SYNTH_OUTPUT)/eval; \
	done

analyze-lcb-qwen36-synth-high5-behavior:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/analyze_behavior_flips.py \
		--run qwen36_27b_synth_high5=$(QWEN36_LCB_SYNTH_OUTPUT)/eval \
		--pairs nl_only:nl_tests nl_only:shuffled_tests nl_only:irrelevant_tests shuffled_tests:nl_tests irrelevant_tests:nl_tests \
		--output-dir $(QWEN36_LCB_SYNTH_OUTPUT)/behavior

analyze-lcb-qwen36-synth-high5-hidden-shift:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/merge_eval_results.py \
		--inputs $(foreach c,$(QWEN36_CONDITIONS),$(QWEN36_LCB_SYNTH_OUTPUT)/eval/$(c)_eval_all.jsonl) \
		--output $(QWEN36_LCB_SYNTH_OUTPUT)/eval_results_combined.jsonl
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/analyze_hidden_state_shift.py \
		--eval-results $(QWEN36_LCB_SYNTH_OUTPUT)/eval_results_combined.jsonl \
		--baseline-condition nl_only \
		--conditions nl_tests shuffled_tests irrelevant_tests \
		--output-dir $(QWEN36_LCB_SYNTH_OUTPUT)/hidden_shift

analyze-behavior:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/analyze_behavior_flips.py \
		--run mbppplus=outputs/mbppplus_qwen_full/eval_results.jsonl \
		--run humanevalplus=outputs/humanevalplus_qwen_structured/eval_results.jsonl \
		--run livecodebench_v6_new=outputs/livecodebench_v6_minus_v5_qwen_full_fixed/eval \
		--output-dir outputs/analysis/behavior

probe-mbppplus-success:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/train_probe.py \
		--eval-results outputs/mbppplus_qwen_full/eval_results.jsonl \
		--probe success \
		--backend $(PROBE_BACKEND) \
		--device $(PROBE_DEVICE) \
		--torch-epochs $(PROBE_EPOCHS) \
		--torch-lr $(PROBE_LR) \
		--torch-weight-decay $(PROBE_WEIGHT_DECAY) \
		--output outputs/analysis/probes/mbppplus_success.json

probe-mbppplus-condition:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/train_probe.py \
		--eval-results outputs/mbppplus_qwen_full/eval_results.jsonl \
		--probe condition \
		--backend $(PROBE_BACKEND) \
		--device $(PROBE_DEVICE) \
		--torch-epochs $(PROBE_EPOCHS) \
		--torch-lr $(PROBE_LR) \
		--torch-weight-decay $(PROBE_WEIGHT_DECAY) \
		--output outputs/analysis/probes/mbppplus_condition.json

probe-mbppplus-rescue:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/train_probe.py \
		--eval-results outputs/mbppplus_qwen_full/eval_results.jsonl \
		--probe rescue \
		--base-condition nl_only \
		--target-condition nl_tests \
		--backend $(PROBE_BACKEND) \
		--device $(PROBE_DEVICE) \
		--torch-epochs $(PROBE_EPOCHS) \
		--torch-lr $(PROBE_LR) \
		--torch-weight-decay $(PROBE_WEIGHT_DECAY) \
		--output outputs/analysis/probes/mbppplus_rescue.json

probe-humanevalplus-success:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/train_probe.py \
		--eval-results outputs/humanevalplus_qwen_structured/eval_results.jsonl \
		--probe success \
		--backend $(PROBE_BACKEND) \
		--device $(PROBE_DEVICE) \
		--torch-epochs $(PROBE_EPOCHS) \
		--torch-lr $(PROBE_LR) \
		--torch-weight-decay $(PROBE_WEIGHT_DECAY) \
		--output outputs/analysis/probes/humanevalplus_success.json

probe-humanevalplus-condition:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/train_probe.py \
		--eval-results outputs/humanevalplus_qwen_structured/eval_results.jsonl \
		--probe condition \
		--backend $(PROBE_BACKEND) \
		--device $(PROBE_DEVICE) \
		--torch-epochs $(PROBE_EPOCHS) \
		--torch-lr $(PROBE_LR) \
		--torch-weight-decay $(PROBE_WEIGHT_DECAY) \
		--output outputs/analysis/probes/humanevalplus_condition.json

probe-humanevalplus-rescue:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/train_probe.py \
		--eval-results outputs/humanevalplus_qwen_structured/eval_results.jsonl \
		--probe rescue \
		--base-condition nl_only \
		--target-condition nl_tests \
		--backend $(PROBE_BACKEND) \
		--device $(PROBE_DEVICE) \
		--torch-epochs $(PROBE_EPOCHS) \
		--torch-lr $(PROBE_LR) \
		--torch-weight-decay $(PROBE_WEIGHT_DECAY) \
		--output outputs/analysis/probes/humanevalplus_rescue.json

probe-interpretability: probe-mbppplus-success probe-mbppplus-condition probe-mbppplus-rescue probe-humanevalplus-success probe-humanevalplus-condition probe-humanevalplus-rescue

prepare-lcb:
	$(PYTHON) scripts/prepare_livecodebench.py \
		--version-tag release_v6 \
		--output data/livecodebench_release_v6.jsonl

prepare-lcb-v6-new:
	$(PYTHON) scripts/prepare_livecodebench.py \
		--local-path $(LCB_DATA_LOCAL) \
		--version-tag release_v6 \
		--exclude-version-tag release_v5 \
		--output data/livecodebench_release_v6_minus_v5.jsonl

generate-lcb:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation.py \
		--dataset data/livecodebench_release_v6.jsonl \
		--dataset-name livecodebench \
		--model-path $(QWEN25_MODEL) \
		--conditions nl_only nl_tests shuffled_tests irrelevant_tests \
		--limit 50 \
		--device cuda \
		--max-new-tokens 1024 \
		--collect-states \
		--output-dir outputs/livecodebench_qwen_mvp

generate-lcb-v6-new:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation.py \
		--dataset data/livecodebench_release_v6_minus_v5.jsonl \
		--dataset-name livecodebench_v6_minus_v5 \
		--model-path $(QWEN25_MODEL) \
		--conditions nl_only nl_tests shuffled_tests irrelevant_tests \
		--limit 50 \
		--device cuda \
		--max-new-tokens 1024 \
		--collect-states \
		--output-dir outputs/livecodebench_v6_minus_v5_qwen_mvp

generate-lcb-v6-new-full-mgpu:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/run_generation_multi_gpu.py \
		--gpus $(GPUS) \
		--model-path $(QWEN25_MODEL)

export-lcb-nl-tests:
	$(PYTHON) scripts/export_lcb_custom_outputs.py \
		--generations outputs/livecodebench_qwen_mvp/generations.jsonl \
		--condition nl_tests \
		--output outputs/livecodebench_qwen_mvp/lcb_custom_outputs_nl_tests.json

export-lcb-nl-only:
	$(PYTHON) scripts/export_lcb_custom_outputs.py \
		--generations outputs/livecodebench_qwen_mvp/generations.jsonl \
		--condition nl_only \
		--output outputs/livecodebench_qwen_mvp/lcb_custom_outputs_nl_only.json

export-lcb-shuffled:
	$(PYTHON) scripts/export_lcb_custom_outputs.py \
		--generations outputs/livecodebench_qwen_mvp/generations.jsonl \
		--condition shuffled_tests \
		--output outputs/livecodebench_qwen_mvp/lcb_custom_outputs_shuffled_tests.json

export-lcb-irrelevant:
	$(PYTHON) scripts/export_lcb_custom_outputs.py \
		--generations outputs/livecodebench_qwen_mvp/generations.jsonl \
		--condition irrelevant_tests \
		--output outputs/livecodebench_qwen_mvp/lcb_custom_outputs_irrelevant_tests.json

export-lcb-v6-new-nl-tests:
	$(PYTHON) scripts/export_lcb_custom_outputs.py \
		--generations outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/generations.jsonl \
		--condition nl_tests \
		--output outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/lcb_custom_outputs_nl_tests.json

export-lcb-v6-new-nl-only:
	$(PYTHON) scripts/export_lcb_custom_outputs.py \
		--generations outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/generations.jsonl \
		--condition nl_only \
		--output outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/lcb_custom_outputs_nl_only.json

export-lcb-v6-new-shuffled:
	$(PYTHON) scripts/export_lcb_custom_outputs.py \
		--generations outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/generations.jsonl \
		--condition shuffled_tests \
		--output outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/lcb_custom_outputs_shuffled_tests.json

export-lcb-v6-new-irrelevant:
	$(PYTHON) scripts/export_lcb_custom_outputs.py \
		--generations outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/generations.jsonl \
		--condition irrelevant_tests \
		--output outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/lcb_custom_outputs_irrelevant_tests.json

eval-lcb-v6-new-nl-only:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_lcb_subset.py \
		--generations outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/generations.jsonl \
		--condition nl_only \
		--output-dir outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/eval

eval-lcb-v6-new-nl-tests:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_lcb_subset.py \
		--generations outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/generations.jsonl \
		--condition nl_tests \
		--output-dir outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/eval

eval-lcb-v6-new-shuffled:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_lcb_subset.py \
		--generations outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/generations.jsonl \
		--condition shuffled_tests \
		--output-dir outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/eval

eval-lcb-v6-new-irrelevant:
	TMPDIR=.tmp TMP=.tmp TEMP=.tmp $(PYTHON) scripts/evaluate_lcb_subset.py \
		--generations outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/generations.jsonl \
		--condition irrelevant_tests \
		--output-dir outputs/livecodebench_v6_minus_v5_qwen_mvp_fixed/eval

prepare-quixbugs:
	$(PYTHON) scripts/prepare_quixbugs.py \
		--quixbugs-root data/QuixBugs \
		--output data/quixbugs_repair.jsonl

prepare-bugsinpy:
	$(PYTHON) scripts/prepare_bugsinpy_manifest.py \
		--bugsinpy-root data/BugsInPy \
		--output data/bugsinpy_manifest.jsonl

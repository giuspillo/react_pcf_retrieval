SHELL := /bin/bash

conda_env = pcf-scout
# Python interpreter (override e.g. with: make evaluate-ground-truth PYTHON=python)
PYTHON ?= conda run --no-capture-output -n $(conda_env) python

GIT_COMMIT = $(shell git rev-parse --short HEAD 2>/dev/null || echo "No commit / Not a git repo")
GIT_BRANCH = $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "Unknown branch")
RUNNING_USER = $(shell whoami 2>/dev/null || echo "Unknown user")
OS_INFO = $(shell uname -srm 2>/dev/null || echo "Unknown OS")

define run_pipeline_step
    @mkdir -p outputs/pipeline_logs
    @TIMESTAMP=$$(date '+%Y%m%d_%H%M%S'); \
    LOG_FILE="outputs/pipeline_logs/$(1)_$${TIMESTAMP}.log"; \
    echo "====================================================================================================" | tee -a $$LOG_FILE; \
    echo "Starting Step | $(1)" | tee -a $$LOG_FILE; \
    echo "Git Branch    | $(GIT_BRANCH)" | tee -a $$LOG_FILE; \
    echo "Git Commit    | $(GIT_COMMIT)" | tee -a $$LOG_FILE; \
    echo "Running User  | $(RUNNING_USER)" | tee -a $$LOG_FILE; \
    echo "Environment   | $(OS_INFO)" | tee -a $$LOG_FILE; \
    echo "Start Time    | $$(date '+%Y-%m-%d %H:%M:%S')" | tee -a $$LOG_FILE; \
    echo "====================================================================================================" | tee -a $$LOG_FILE; \
    echo "" | tee -a $$LOG_FILE; \
    echo "====================================================================================================" | tee -a $$LOG_FILE; \
    echo "Configurations" | tee -a $$LOG_FILE; \
    echo "====================================================================================================" | tee -a $$LOG_FILE; \
    for f in config/*.yaml; do \
        if [ -f "$$f" ]; then \
            echo "" | tee -a $$LOG_FILE; \
            echo "  ==> $$f <==" | tee -a $$LOG_FILE; \
            awk '/^[[:space:]]*#/ {next} /^[[:space:]]*$$/ {next} {print "    " $$0}' $$f | tee -a $$LOG_FILE; \
        fi \
    done; \
	echo "" | tee -a $$LOG_FILE; \
    echo "====================================================================================================" | tee -a $$LOG_FILE; \
    echo "" | tee -a $$LOG_FILE; \
    echo "====================================================================================================" | tee -a $$LOG_FILE; \
    echo "Process Execution" | tee -a $$LOG_FILE; \
    echo "====================================================================================================" | tee -a $$LOG_FILE; \
    START_TIME=$$(date +%s); \
    $(PYTHON) -u $(2) 2>&1 | tee -a $$LOG_FILE; \
    EXIT_CODE=$${PIPESTATUS[0]}; \
    END_TIME=$$(date +%s); \
    ELAPSED=$$(($$END_TIME - $$START_TIME)); \
    MINUTES=$$(($$ELAPSED / 60)); \
    SECONDS=$$(($$ELAPSED % 60)); \
    if [ $$EXIT_CODE -eq 0 ]; then STATUS="SUCCESS"; else STATUS="FAILED"; fi; \
	echo "" | tee -a $$LOG_FILE; \
    echo "====================================================================================================" | tee -a $$LOG_FILE; \
    echo "" | tee -a $$LOG_FILE; \
    echo "====================================================================================================" | tee -a $$LOG_FILE; \
    echo "Finished Step | $(1)" | tee -a $$LOG_FILE; \
    echo "Status        | $$STATUS" | tee -a $$LOG_FILE; \
    echo "Duration      | $$ELAPSED seconds ($${MINUTES}m $${SECONDS}s)" | tee -a $$LOG_FILE; \
    echo "Exit Code     | $$EXIT_CODE" | tee -a $$LOG_FILE; \
    echo "====================================================================================================" | tee -a $$LOG_FILE
endef

# ===================================
# PCF-SCOUT: PCF ESTIMATION
# (ground truth or use-case catalog, see config/emissions_enrichment.yaml)
# ===================================
run-pcf-scout:
	$(call run_pipeline_step,run_pcf_scout,-m pcf_scout.run)

# ===================================
# ZERO-SHOT BASELINE
# (settings in config/zero_shot.yaml)
# ===================================
run-zero-shot:
	$(call run_pipeline_step,run_zero_shot,-m pcf_scout.zero_shot)

# ===================================
# GROUND TRUTH: PAPER ANALYSES
# ===================================
evaluate-ground-truth:
	$(call run_pipeline_step,evaluate_ground_truth,ground_truth/evaluate.py)

# ===================================
# USE CASE: DATA PREPROCESSING
# ===================================
preprocess-data:
	$(call run_pipeline_step,preprocess_data,-m use_case.src.preprocess)

# ===================================
# USE CASE: MODEL TRAINING
# ===================================
train-recsys:
	$(call run_pipeline_step,train_recsys,-m use_case.src.train)

# ===================================
# USE CASE: MODEL INFERENCE
# +
# SUSTAINABILITY-AWARE RECOMMENDATION
# RE-RANKING
# ===================================
predict-recommendations:
	$(call run_pipeline_step,predict_recommendations,-m use_case.src.predict)

# ===================================
# USE CASE: MODEL EVALUATION
# ===================================
evaluate-recsys:
	$(call run_pipeline_step,evaluate_recsys,-m use_case.src.evaluate)

.PHONY: run-pcf-scout run-zero-shot evaluate-ground-truth preprocess-data train-recsys predict-recommendations evaluate-recsys

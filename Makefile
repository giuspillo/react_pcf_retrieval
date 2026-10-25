conda_env = eco-amazon-electronics

GIT_COMMIT = $(shell git rev-parse --short HEAD 2>/dev/null || echo "No commit / Not a git repo")
GIT_BRANCH = $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "Unknown branch")
RUNNING_USER = $(shell whoami 2>/dev/null || echo "Unknown user")
OS_INFO = $(shell uname -srm 2>/dev/null || echo "Unknown OS")

define run_pipeline_step
    @mkdir -p logs/pipeline/electronics
    @TIMESTAMP=$$(date '+%Y%m%d_%H%M%S'); \
    LOG_FILE="logs/pipeline/electronics/$(1)_$${TIMESTAMP}.log"; \
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
    for f in src/config/*.yaml; do \
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
    conda run --no-capture-output -n $(conda_env) python -u -m $(2) 2>&1 | tee -a $$LOG_FILE; \
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
# DATA PREPROCESSING
# ===================================
preprocess-data:
	$(call run_pipeline_step,preprocess_data,src.data.preprocess)

# ===================================
# EMISSION DATA ENRICHMENT
# ===================================
enrich-data-with-emissions:
	$(call run_pipeline_step,enrich_data_with_emissions,src.data.enrich_with_emissions)

# ===================================
# MODEL TRAINING
# ===================================
train-recsys:
	$(call run_pipeline_step,train_recsys,src.modeling.train)

# ===================================
# MODEL INFERENCE
# +
# SUSTAINABILITY-AWARE RECOMMENDATION
# RE-RANKING
# ===================================
predict-recommendations:
	$(call run_pipeline_step,predict_recommendations,src.modeling.predict)

# ===================================
# MODEL EVALUATION
# ===================================
evaluate-recsys:
	$(call run_pipeline_step,evaluate_recsys,src.modeling.evaluate)

"""pcf_scout/const.py

Constants for running PCF-Scout on the ground truth and on the use-case catalog.
"""

# ============================================================================
# DATASETS
# ============================================================================

# Dataset (domain) names
DATASET_NAME_ELEC = "electronics"
DATASET_NAME_CLOTH = "clothing"
DATASET_NAME_HK = "home_and_kitchen"

# Supported datasets
SUPPORTED_DATASETS = [DATASET_NAME_ELEC, DATASET_NAME_CLOTH, DATASET_NAME_HK]

# ============================================================================
# LLMS
# ============================================================================

# LLM names (<provider>:<model>, as expected by LangChain)
LLM_NAME_G25F = "google_genai:gemini-2.5-flash"
LLM_NAME_O3M = "openai:o3-mini"

# Supported LLMs
SUPPORTED_LLMS = [LLM_NAME_G25F, LLM_NAME_O3M]

# LLM keys used in data files and directory names
LLM_KEYS = {
    LLM_NAME_G25F: "gemini_2_5_flash",
    LLM_NAME_O3M: "openai_o3_mini",
}

# ============================================================================
# DIRECTORIES AND PATHS
# ============================================================================

# Configuration
CONFIG_DIR = "config"
CONFIG_FILE_FORMAT = "*.yaml"

# Ground truth (input: released data, one file per domain)
GROUND_TRUTH_DATA_DIR = "ground_truth/data"

# Use-case catalog (input: Amazon Reviews'23 item metadata of the catalog)
USE_CASE_RAW_ITEM_METADATA_PATH = (
    f"use_case/data/{DATASET_NAME_ELEC}/raw/item_metadata.jsonl"
)

# Outputs of new runs (kept separate from the released data)
OUTPUTS_DIR = "outputs"

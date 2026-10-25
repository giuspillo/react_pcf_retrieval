"""use_case/src/const.py

Project constants.
"""

# ============================================================================
# DATASETS
# ============================================================================

# Dataset names
DATASET_NAME_ELEC = "electronics"
DATASET_NAME_CLOTH = "clothing"
DATASET_NAME_HK = "home_and_kitchen"

# Supported datasets
SUPPORTED_DATASETS = [DATASET_NAME_ELEC, DATASET_NAME_CLOTH, DATASET_NAME_HK]

# Dataset raw user reviews columns
DATASET_RAW_UR_USER_ID_COL = "user_id"
DATASET_RAW_UR_ASIN_COL = "asin"
DATASET_RAW_UR_PARENT_ASIN_COL = "parent_asin"
DATASET_RAW_UR_TIMESTAMP_COL = "timestamp"
DATASET_RAW_UR_RATING_COL = "rating"

# Dataset processed user reviews columns
DATASET_PROCESSED_UR_USER_ID_COL = "user_id:token"
DATASET_PROCESSED_UR_ITEM_ID_COL = "item_id:token"
DATASET_PROCESSED_UR_RATING_COL = "rating:float"
DATASET_PROCESSED_UR_TIMESTAMP_COL = "timestamp:float"

# ============================================================================
# MAPS
# ============================================================================

# User map columns
MAP_UMAP_USER_ID_COL = "user_id"
MAP_UMAP_USER_INDEX_COL = "user_index"

# Item map columns
MAP_IMAP_ITEM_ID_COL = "item_id"
MAP_IMAP_ITEM_INDEX_COL = "item_index"
MAP_IMAP_PARENT_ASIN_COL = "parent_asin"

# ============================================================================
# MODELS AND LLMS
# ============================================================================

# Model names
MODEL_NAME_BPR = "BPR"
MODEL_NAME_LIGHTGCN = "LightGCN"

# Supported models
SUPPORTED_MODELS = [MODEL_NAME_BPR, MODEL_NAME_LIGHTGCN]

# Supported model hyperparameters
MODEL_SUPPORTED_PARAMS = {
    "BPR": {
        "embedding_size",
        "learning_rate",
        "weight_decay",
        "train_batch_size",
    },
    "LightGCN": {
        "embedding_size",
        "n_layers",
        "reg_weight",
        "learning_rate",
        "weight_decay",
        "train_batch_size",
    },
}

# LLM names
LLM_NAME_G25F = "google_genai:gemini-2.5-flash"
LLM_NAME_O3M = "openai:o3-mini"

# Supported LLMs
SUPPORTED_LLMS = [LLM_NAME_G25F, LLM_NAME_O3M]

# ============================================================================
# RE-RANKING ALPHAS
# ============================================================================

# Re-ranking alpha values
RERANKING_ALPHA_SUS = 0.25
RERANKING_ALPHA_BAL = 0.5
RERANKING_ALPHA_REL = 0.75
RERANKING_ALPHA_PURE = 1.0

# Supported re-ranking alpha values
SUPPORTED_RERANKING_ALPHAS = [
    RERANKING_ALPHA_SUS,
    RERANKING_ALPHA_BAL,
    RERANKING_ALPHA_REL,
    RERANKING_ALPHA_PURE,
]

# ============================================================================
# TUNING
# ============================================================================

# Tuning validation metric
TUNING_VAL_METRIC = {
    "name": "best_valid_score",
    "mode": "max",
}

# ============================================================================
# DIRECTORIES AND PATHS
# ============================================================================

# Source directory
SRC_DIR = "src"

# Data directory
DATA_DIR = "data"

# Data raw directories
DATA_RAW_DIR = f"{DATA_DIR}/raw"
DATA_RAW_AR23_DIR = f"{DATA_RAW_DIR}/amazon_reviews_23"
DATA_RAW_AR23_IM_DIR = f"{DATA_RAW_AR23_DIR}/item_metadata"
DATA_RAW_AR23_UR_DIR = f"{DATA_RAW_AR23_DIR}/user_reviews"
DATA_RAW_GT_DIR = f"{DATA_RAW_DIR}/ground_truths"

# Data interim directories
DATA_INTERIM_DIR = f"{DATA_DIR}/interim"
DATA_INTERIM_MAPS_DIR = f"{DATA_INTERIM_DIR}/maps"

# Data processed directories
DATA_PROCESSED_DIR = f"{DATA_DIR}/processed"
DATA_PROCESSED_AR23_DIR = f"{DATA_PROCESSED_DIR}/amazon_reviews_23"
DATA_PROCESSED_AR23_IM_DIR = f"{DATA_PROCESSED_AR23_DIR}/item_metadata"
DATA_PROCESSED_AR23_IM_G25F_DIR = (
    f"{DATA_PROCESSED_AR23_IM_DIR}/{LLM_NAME_G25F}"
)
DATA_PROCESSED_AR23_UR_DIR = f"{DATA_PROCESSED_AR23_DIR}/user_reviews"
DATA_PROCESSED_AR23_UR_ELEC_DIR = (
    f"{DATA_PROCESSED_AR23_UR_DIR}/{DATASET_NAME_ELEC}"
)
DATA_PROCESSED_GT_DIR = f"{DATA_PROCESSED_DIR}/ground_truths"
DATA_PROCESSED_GT_G25F_DIR = f"{DATA_PROCESSED_GT_DIR}/{LLM_NAME_G25F}"
DATA_PROCESSED_GT_O3M_DIR = f"{DATA_PROCESSED_GT_DIR}/{LLM_NAME_O3M}"

# Data raw paths
DATA_RAW_AR23_IM_ELEC_PATH = (
    f"{DATA_RAW_AR23_IM_DIR}/{DATASET_NAME_ELEC}.jsonl"
)
DATA_RAW_AR23_UR_ELEC_PATH = (
    f"{DATA_RAW_AR23_UR_DIR}/{DATASET_NAME_ELEC}.jsonl"
)
DATA_RAW_GT_ELEC_PATH = f"{DATA_RAW_GT_DIR}/{DATASET_NAME_ELEC}.jsonl"
DATA_RAW_GT_CLOTH_PATH = f"{DATA_RAW_GT_DIR}/{DATASET_NAME_CLOTH}.jsonl"
DATA_RAW_GT_HK_PATH = f"{DATA_RAW_GT_DIR}/{DATASET_NAME_HK}.jsonl"

# Data interim paths
DATA_INTERIM_MAPS_UMAP_PATH = f"{DATA_INTERIM_MAPS_DIR}/user_map.tsv"
DATA_INTERIM_MAPS_IMAP_PATH = f"{DATA_INTERIM_MAPS_DIR}/item_map.tsv"

# Data processed paths
DATA_PROCESSED_AR23_IM_G25F_ELEC_PATH = (
    f"{DATA_PROCESSED_AR23_IM_G25F_DIR}/{DATASET_NAME_ELEC}.jsonl"
)
DATA_PROCESSED_AR23_UR_ELEC_FULL_PATH = (
    f"{DATA_PROCESSED_AR23_UR_ELEC_DIR}/{DATASET_NAME_ELEC}.inter"
)
DATA_PROCESSED_AR23_UR_ELEC_TRAIN_PATH = (
    f"{DATA_PROCESSED_AR23_UR_ELEC_DIR}/{DATASET_NAME_ELEC}.train.inter"
)
DATA_PROCESSED_AR23_UR_ELEC_TEST_PATH = (
    f"{DATA_PROCESSED_AR23_UR_ELEC_DIR}/{DATASET_NAME_ELEC}.test.inter"
)
DATA_PROCESSED_AR23_UR_ELEC_VALID_PATH = (
    f"{DATA_PROCESSED_AR23_UR_ELEC_DIR}/{DATASET_NAME_ELEC}.valid.inter"
)
DATA_PROCESSED_GT_G25F_ELEC_PATH = (
    f"{DATA_PROCESSED_GT_G25F_DIR}/{DATASET_NAME_ELEC}.jsonl"
)
DATA_PROCESSED_GT_O3M_ELEC_PATH = (
    f"{DATA_PROCESSED_GT_O3M_DIR}/{DATASET_NAME_ELEC}.jsonl"
)
DATA_PROCESSED_GT_G25F_CLOTH_PATH = (
    f"{DATA_PROCESSED_GT_G25F_DIR}/{DATASET_NAME_CLOTH}.jsonl"
)
DATA_PROCESSED_GT_O3M_CLOTH_PATH = (
    f"{DATA_PROCESSED_GT_O3M_DIR}/{DATASET_NAME_CLOTH}.jsonl"
)
DATA_PROCESSED_GT_G25F_HK_PATH = (
    f"{DATA_PROCESSED_GT_G25F_DIR}/{DATASET_NAME_HK}.jsonl"
)
DATA_PROCESSED_GT_O3M_HK_PATH = (
    f"{DATA_PROCESSED_GT_O3M_DIR}/{DATASET_NAME_HK}.jsonl"
)

# Model directories
MODELS_DIR = "models"
MODELS_ELEC_DIR = f"{MODELS_DIR}/{DATASET_NAME_ELEC}"

# Models best parameters
MODELS_BEST_PARAMS_FILE_NAME = "best_params.json"

# Models BPR directories
MODELS_ELEC_BPR_DIR = f"{MODELS_ELEC_DIR}/{MODEL_NAME_BPR}"
MODELS_ELEC_BPR_PREDS_DIR = f"{MODELS_ELEC_BPR_DIR}/predictions"
MODELS_ELEC_BPR_PREDS_G25F_DIR = f"{MODELS_ELEC_BPR_PREDS_DIR}/{LLM_NAME_G25F}"

# Models LightGCN directories
MODELS_ELEC_LIGHTGCN_DIR = f"{MODELS_ELEC_DIR}/{MODEL_NAME_LIGHTGCN}"
MODELS_ELEC_LIGHTGCN_PREDS_DIR = f"{MODELS_ELEC_LIGHTGCN_DIR}/predictions"
MODELS_ELEC_LIGHTGCN_PREDS_G25F_DIR = (
    f"{MODELS_ELEC_LIGHTGCN_PREDS_DIR}/{LLM_NAME_G25F}"
)

# Models BPR paths
MODELS_ELEC_BPR_PATH = f"{MODELS_ELEC_BPR_DIR}/{MODEL_NAME_BPR}.pth"
MODELS_ELEC_BPR_PREDS_G25F_SUS_PATH = f"{MODELS_ELEC_BPR_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_SUS).replace('.', '_')}.pth"
MODELS_ELEC_BPR_PREDS_G25F_BAL_PATH = f"{MODELS_ELEC_BPR_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_BAL).replace('.', '_')}.pth"
MODELS_ELEC_BPR_PREDS_G25F_REL_PATH = f"{MODELS_ELEC_BPR_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_REL).replace('.', '_')}.pth"
MODELS_ELEC_BPR_PREDS_G25F_PURE_PATH = f"{MODELS_ELEC_BPR_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_PURE).replace('.', '_')}.pth"

# Models LightGCN paths
MODELS_ELEC_LIGHTGCN_PATH = (
    f"{MODELS_ELEC_LIGHTGCN_DIR}/{MODEL_NAME_LIGHTGCN}.pth"
)
MODELS_ELEC_LIGHTGCN_PREDS_G25F_SUS_PATH = f"{MODELS_ELEC_LIGHTGCN_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_SUS).replace('.', '_')}.pth"
MODELS_ELEC_LIGHTGCN_PREDS_G25F_BAL_PATH = f"{MODELS_ELEC_LIGHTGCN_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_BAL).replace('.', '_')}.pth"
MODELS_ELEC_LIGHTGCN_PREDS_G25F_REL_PATH = f"{MODELS_ELEC_LIGHTGCN_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_REL).replace('.', '_')}.pth"
MODELS_ELEC_LIGHTGCN_PREDS_G25F_PURE_PATH = f"{MODELS_ELEC_LIGHTGCN_PREDS_G25F_DIR}/alpha_{str(RERANKING_ALPHA_PURE).replace('.', '_')}.pth"

# Config directory
CONFIG_DIR = f"{SRC_DIR}/config"

# ============================================================================
# FILE FORMATS
# ============================================================================

# Configuration files
CONFIG_FILE_FORMAT = "*.yaml"

# Model checkpoint files
MODEL_CHECKPOINT_FILE_FORMAT = "*.pth"

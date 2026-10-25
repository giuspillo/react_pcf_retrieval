"""use_case/src/train.py

Train RecBole models with hyperparameter tuning.
"""

import glob
import hashlib
import json
import os
import shutil

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator
from recbole.quick_start import run_recbole

from src.const import (
    DATA_PROCESSED_AR23_UR_DIR,
    DATASET_NAME_ELEC,
    MODEL_CHECKPOINT_FILE_FORMAT,
    MODEL_NAME_BPR,
    MODEL_NAME_LIGHTGCN,
    MODEL_SUPPORTED_PARAMS,
    MODELS_BEST_PARAMS_FILE_NAME,
    MODELS_ELEC_BPR_DIR,
    MODELS_ELEC_BPR_PATH,
    MODELS_ELEC_BPR_PREDS_DIR,
    MODELS_ELEC_LIGHTGCN_DIR,
    MODELS_ELEC_LIGHTGCN_PATH,
    MODELS_ELEC_LIGHTGCN_PREDS_DIR,
    SUPPORTED_DATASETS,
    SUPPORTED_MODELS,
    TUNING_VAL_METRIC,
)
from src.utils import load_config

# Load configuration
config = load_config()

# Determine paths based on dataset name
assert config["dataset"]["name"] in SUPPORTED_DATASETS, (
    f"Supported dataset: {SUPPORTED_DATASETS}, got: {config['dataset']['name']}"
)
if config["dataset"]["name"] == DATASET_NAME_ELEC:
    # Map models to their registry information
    model_registry = {
        MODEL_NAME_BPR: {
            "checkpoint_dir": MODELS_ELEC_BPR_DIR,
            "model_path": MODELS_ELEC_BPR_PATH,
            "preds_dir": MODELS_ELEC_BPR_PREDS_DIR,
        },
        MODEL_NAME_LIGHTGCN: {
            "checkpoint_dir": MODELS_ELEC_LIGHTGCN_DIR,
            "model_path": MODELS_ELEC_LIGHTGCN_PATH,
            "preds_dir": MODELS_ELEC_LIGHTGCN_PREDS_DIR,
        },
    }

    # Build checkpoint directories and directories to keep
    checkpoint_dirs = {
        model: os.path.abspath(model_registry[model]["checkpoint_dir"])
        for model in SUPPORTED_MODELS
    }
    keep_dirs = {
        os.path.abspath(model_registry[model]["preds_dir"])
        for model in SUPPORTED_MODELS
    }


def _trainable(
    config: dict,
    base_config: dict,
    dataset: str,
    model: str,
    data_path: str,
    checkpoint_dir: str,
    enable_tune: bool,
) -> None:
    """Train a RecBole model with optional hyperparameter tuning via Ray Tune.

    This function serves as the training entry point. It:
    1. Filters the sampled hyperparameters to retain only those supported by
       the target model
    2. Merges the filtered hyperparameters with the base RecBole configuration
    3. Creates a unique checkpoint directory per trial to avoid conflicts
    4. Trains the model using RecBole
    5. Reports the validation metric back to Ray Tune (if enabled)

    Args:
        config (dict):
            Hyperparameters sampled by Ray Tune for the current trial.

        base_config (dict):
            Base RecBole configuration shared across all trials.

        dataset (str):
            Name of the dataset to train on.

        model (str):
            Name of the RecBole model to train.

        data_path (str):
            Absolute path to the processed dataset directory.

        checkpoint_dir (str):
            Base directory where model checkpoints will be stored.

        enable_tune (bool):
            Whether the function is executed within a Ray Tune context.
            If True, reports metrics to Ray Tune and creates isolated
            checkpoint directories per trial.

    Returns:
        None
    """
    # Filter hyperparameters based on model's supported parameters
    supported_params = MODEL_SUPPORTED_PARAMS.get(model, set())
    filtered_config = {
        k: v for k, v in config.items() if k in supported_params
    }

    # Merge base RecBole config with trial-specific hyperparameters from Ray Tune
    final_config = {**base_config, **filtered_config}

    # Create unique checkpoint directory for this trial to avoid conflicts
    # when different trials have different hyperparameters
    if enable_tune:
        # Create a hash from the trial's hyperparameters to ensure uniqueness
        config_str = str(sorted(filtered_config.items()))
        trial_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        unique_checkpoint_dir = os.path.join(checkpoint_dir, trial_hash)
        os.makedirs(unique_checkpoint_dir, exist_ok=True)
    else:
        unique_checkpoint_dir = checkpoint_dir

    # Add data path and checkpoint directory to the config
    final_config["data_path"] = data_path
    final_config["checkpoint_dir"] = unique_checkpoint_dir

    # Train the model
    result = run_recbole(
        dataset=dataset,
        model=model,
        config_dict=final_config,
    )

    # Keep track of the best validation score
    if enable_tune:
        tune.report({TUNING_VAL_METRIC["name"]: result["best_valid_score"]})


def train_recsys() -> None:
    """Run hyperparameter tuning and final training for supported RecBole models.

    This function orchestrates the full training pipeline using Ray Tune. It:
    1. Initializes a Ray runtime environment
    2. Defines the hyperparameter search space
    3. Runs hyperparameter optimization independently for each supported model
    4. Collects the best configuration per model
    5. Shuts down Ray after tuning is complete
    6. Retrains each model using its best hyperparameters on the full setup
    For each model:
    - A dedicated checkpoint directory is used
    - Hyperparameter search is performed using a scheduler (ASHA) and a
      basic variant generator
    - The best trial is selected based on the configured validation metric

    Returns:
        None
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            address=config["ray_tune"]["address"],
            num_cpus=config["ray_tune"]["num_cpus"],
            num_gpus=config["ray_tune"]["num_gpus"],
            ignore_reinit_error=config["ray_tune"]["ignore_reinit_error"],
            include_dashboard=config["ray_tune"]["include_dashboard"],
            dashboard_host=config["ray_tune"]["dashboard_host"],
            dashboard_port=config["ray_tune"]["dashboard_port"],
            configure_logging=config["ray_tune"]["configure_logging"],
            logging_level=config["ray_tune"]["logging_level"],
            logging_format=config["ray_tune"]["logging_format"],
            log_to_driver=config["ray_tune"]["log_to_driver"],
        )

    # Define parameter space for Ray Tune
    param_space = {k: tune.choice(v) for k, v in config["param_space"].items()}

    # Find best hyperparameters for each model via Ray Tune (or load from existing JSON)
    best_hyperparams = {}
    for model in SUPPORTED_MODELS:
        # Get checkpoint directory
        checkpoint_dir = checkpoint_dirs[model]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Path to JSON with best params stored in the model root
        best_params_path = os.path.join(
            checkpoint_dir,
            MODELS_BEST_PARAMS_FILE_NAME,
        )

        # If a JSON with best params already exists, load and skip HPO for this model
        if os.path.exists(best_params_path):
            with open(best_params_path) as fh:
                best_hyperparams[model] = json.load(fh)
            continue

        # Define a scheduler
        scheduler = ASHAScheduler(
            max_t=config["ray_tune"]["max_t"],
            grace_period=config["ray_tune"]["grace_period"],
        )

        # Define a search algorithm
        search_alg = BasicVariantGenerator(
            random_state=config["ray_tune"]["random_state"],
        )

        # Run Ray Tune to find the best hyperparameters
        # and keep track of the results for each model
        tuner = tune.Tuner(
            tune.with_parameters(
                _trainable,
                base_config=config["recbole"],
                dataset=config["dataset"]["name"],
                model=model,
                data_path=os.path.abspath(DATA_PROCESSED_AR23_UR_DIR),
                checkpoint_dir=checkpoint_dir,
                enable_tune=True,
            ),
            param_space=param_space,
            tune_config=tune.TuneConfig(
                mode=TUNING_VAL_METRIC["mode"],
                metric=TUNING_VAL_METRIC["name"],
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=config["ray_tune"]["num_samples"],
            ),
        )
        res = tuner.fit()

        # Extract best config and filter to supported params
        best_config = res.get_best_result().config
        supported_params = MODEL_SUPPORTED_PARAMS.get(model, set())
        best_hyperparams[model] = {
            k: v for k, v in best_config.items() if k in supported_params
        }

    # Shutdown Ray after tuning is complete
    ray.shutdown()

    # Clean up any temporary checkpoint directories created during tuning
    for model in SUPPORTED_MODELS:
        checkpoint_root = checkpoint_dirs[model]
        for name in os.listdir(checkpoint_root):
            path = os.path.join(checkpoint_root, name)
            if path in keep_dirs:
                continue
            if os.path.isdir(path):
                shutil.rmtree(path)

    # Final training for each model with the best hyperparameters (from JSON or HPO)
    for model in SUPPORTED_MODELS:
        # Get the best hyperparameters for the current model
        best_config = best_hyperparams.get(model, {})

        # Get checkpoint directory
        checkpoint_dir = checkpoint_dirs[model]

        # Retrain the model with its best hyperparameters found or loaded
        _trainable(
            config=best_config,
            base_config=config["recbole"],
            dataset=config["dataset"]["name"],
            model=model,
            data_path=os.path.abspath(DATA_PROCESSED_AR23_UR_DIR),
            checkpoint_dir=checkpoint_dir,
            enable_tune=False,
        )

        # Rename the checkpoint file to the final model path
        ckpt = max(
            glob.glob(
                os.path.join(checkpoint_dir, MODEL_CHECKPOINT_FILE_FORMAT),
            ),
            key=os.path.getctime,
        )
        target_path = os.path.abspath(model_registry[model]["model_path"])
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        os.replace(ckpt, target_path)


if __name__ == "__main__":
    train_recsys()

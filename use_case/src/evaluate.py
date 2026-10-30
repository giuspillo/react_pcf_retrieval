"""use_case/src/evaluate.py

Evaluate trained RecBole models.
"""

import os

import numpy as np
import pandas as pd
import torch
from recbole.data.dataset import Dataset
from recbole.evaluator import Evaluator
from recbole.quick_start import load_data_and_model
from tqdm import tqdm

from use_case.src.const import (
    DATA_INTERIM_MAPS_IMAP_PATH,
    DATA_PROCESSED_AR23_IM_G25F_ELEC_PATH,
    DATASET_NAME_ELEC,
    LLM_NAME_G25F,
    MODEL_NAME_BPR,
    MODEL_NAME_LIGHTGCN,
    MODELS_ELEC_BPR_PATH,
    MODELS_ELEC_BPR_PREDS_G25F_BAL_PATH,
    MODELS_ELEC_BPR_PREDS_G25F_PURE_PATH,
    MODELS_ELEC_BPR_PREDS_G25F_REL_PATH,
    MODELS_ELEC_BPR_PREDS_G25F_SUS_PATH,
    MODELS_ELEC_LIGHTGCN_PATH,
    MODELS_ELEC_LIGHTGCN_PREDS_G25F_BAL_PATH,
    MODELS_ELEC_LIGHTGCN_PREDS_G25F_PURE_PATH,
    MODELS_ELEC_LIGHTGCN_PREDS_G25F_REL_PATH,
    MODELS_ELEC_LIGHTGCN_PREDS_G25F_SUS_PATH,
    RERANKING_ALPHA_BAL,
    RERANKING_ALPHA_PURE,
    RERANKING_ALPHA_REL,
    RERANKING_ALPHA_SUS,
    SUPPORTED_DATASETS,
    SUPPORTED_MODELS,
    RESULTS_ELEC_PATH,
    SUPPORTED_RERANKING_ALPHAS,
)
from use_case.src.utils import load_config, load_jsonl

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
            "model_path": MODELS_ELEC_BPR_PATH,
            "preds_paths": {
                LLM_NAME_G25F: {
                    RERANKING_ALPHA_SUS: MODELS_ELEC_BPR_PREDS_G25F_SUS_PATH,
                    RERANKING_ALPHA_BAL: MODELS_ELEC_BPR_PREDS_G25F_BAL_PATH,
                    RERANKING_ALPHA_REL: MODELS_ELEC_BPR_PREDS_G25F_REL_PATH,
                    RERANKING_ALPHA_PURE: MODELS_ELEC_BPR_PREDS_G25F_PURE_PATH,
                },
            },
        },
        MODEL_NAME_LIGHTGCN: {
            "model_path": MODELS_ELEC_LIGHTGCN_PATH,
            "preds_paths": {
                LLM_NAME_G25F: {
                    RERANKING_ALPHA_SUS: MODELS_ELEC_LIGHTGCN_PREDS_G25F_SUS_PATH,
                    RERANKING_ALPHA_BAL: MODELS_ELEC_LIGHTGCN_PREDS_G25F_BAL_PATH,
                    RERANKING_ALPHA_REL: MODELS_ELEC_LIGHTGCN_PREDS_G25F_REL_PATH,
                    RERANKING_ALPHA_PURE: MODELS_ELEC_LIGHTGCN_PREDS_G25F_PURE_PATH,
                },
            },
        },
    }

    # Map models to their enriched item metadata paths
    emission_data_paths = {
        LLM_NAME_G25F: DATA_PROCESSED_AR23_IM_G25F_ELEC_PATH,
    }

# Ensure reproducibility by setting random seeds
np.random.seed(config["evaluation"]["seed"])
torch.manual_seed(config["evaluation"]["seed"])
torch.cuda.manual_seed_all(config["evaluation"]["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _compute_avg_emissions(
    reranked_items: list,
    item_map: dict,
    emission_data: dict,
    top_k: int,
) -> float:
    """Compute system-level average emissions of recommended items.

    This function calculates the average emission footprint of the recommender
    system by aggregating emissions across all recommended items (top-k per
    user) and then averaging over the total number of valid items.

    Args:
        reranked_items (list):
            List of reranked recommendation lists, one per user.

        item_map (dict):
            Mapping from internal item IDs to external identifiers.

        emission_data (dict):
            Mapping from item identifiers to emission values.

        top_k (int):
            Number of top-ranked items considered per user.

    Returns:
        float:
            System-level average emission per recommended item.
    """
    # Compute the average emissions for the top-k recommended
    # items across all users
    tot_emissions = 0.0
    count = 0
    for user_items in reranked_items:
        for item_id in user_items[:top_k]:
            asin = item_map.get(int(item_id))
            emission_value = emission_data.get(asin)
            if emission_value is not None:
                tot_emissions += emission_value
                count += 1
    return tot_emissions / count if count > 0 else 0.0


def _compute_evaluation_metrics(
    preds_data: dict,
    top_k: int,
    dataset: Dataset,
    item_map: dict,
    emission_data: dict,
    item_popularity: list[tuple[int, int]],
    config: dict,
) -> dict:
    """Compute evaluation metrics for a single recommendation run.

    This function evaluates a reranked recommendation output using both
    standard RecBole metrics and sustainability-aware metrics. It performs:
    1. Truncation of predicted item lists to the specified top-k
    2. Construction of the RecBole evaluation input structure
    3. Execution of RecBole's Evaluator for ranking metrics
    4. Computation of average emissions over recommended items
    5. Aggregation of all metrics into a unified result dictionary

    Args:
        preds_data (dict):
            Dictionary containing model predictions and evaluation artifacts:
            - reranked_item_ids: list of ranked item IDs per user
            - reranked_items: full reranked lists per user
            - pos_matrix: binary relevance matrix
            - pos_len: number of relevant items per user
            - model: model name identifier
            - alpha: reranking parameter value

        top_k (int):
            Number of top-ranked items considered for evaluation.

        dataset (Dataset):
            RecBole dataset object.

        item_map (dict):
            Mapping from internal item indices to external item identifiers.

        emission_data (dict):
            Mapping from item identifiers to emission values.

        item_popularity (list[tuple[int, int]]):
            List of (item_id, count) pairs representing item popularity
            in the training set.

        config (dict):
            RecBole evaluation configuration dictionary, extended locally
            with the current top-K setting.

    Returns:
        dict:
            Dictionary containing:
            - MODEL: model identifier
            - ALPHA: reranking weight parameter
            - TOP-K: evaluated cutoff
            - EMISSIONS: average emission score
            - Standard RecBole metrics specified in the configuration
            - F1: harmonic mean of precision and recall (if precision and recall
              are available)
    """
    # Get prediction data
    reranked_item_ids = np.array(preds_data["reranked_item_ids"])[:, :top_k]
    reranked_item_ids = torch.tensor(reranked_item_ids)
    pos_matrix = torch.tensor(preds_data["pos_matrix"])[:, :top_k]
    pos_len = torch.tensor(preds_data["pos_len"]).view(-1, 1)

    # Overwrite RecBole configuration to set the current top-k value
    config = {
        **config,
        "topk": [top_k],
    }

    # Input format for RecBole Evaluator
    struct = {
        "rec.topk": torch.cat((pos_matrix, pos_len), dim=1).cpu(),
        "rec.items": reranked_item_ids.cpu(),
        "data.num_items": dataset.item_num,
        "data.count_items": item_popularity,
    }

    # Compute evaluation metrics using RecBole Evaluator
    results = Evaluator(config).evaluate(struct)

    # Compute average emissions for the top-k recommended items
    # across all users
    avg_emissions = _compute_avg_emissions(
        reranked_items=preds_data["reranked_item_ids"],
        item_map=item_map,
        emission_data=emission_data,
        top_k=top_k,
    )

    # Aggregate evaluation results
    base_results = results
    results = {
        "MODEL": preds_data["model"],
        "ALPHA": preds_data["alpha"],
        "TOP-K": top_k,
        "EMISSIONS": avg_emissions,
        **{
            k.split("@")[0].upper(): v
            for k, v in base_results.items()
            if isinstance(k, str) and "@" in k
        },
        "F1": (lambda p, r: (2 * p * r / (p + r)) if (p + r) > 0 else 0.0)(
            {
                k.split("@")[0].upper(): v
                for k, v in base_results.items()
                if isinstance(k, str) and "@" in k
            }.get("PRECISION"),
            {
                k.split("@")[0].upper(): v
                for k, v in base_results.items()
                if isinstance(k, str) and "@" in k
            }.get("RECALL"),
        ),
    }

    return results


def evaluate_recsys() -> None:
    """Run evaluation for all recommendation models and configurations.

    This function performs a complete evaluation of the recommender system across
    all the configurations. The evaluation includes both standard RecBole
    and sustainability-aware metrics based on emissions. The pipeline steps are:
    1. Load trained recommendation models and associated datasets
    2. Load item-level emission data for sustainability evaluation
    3. Load (Item Index, Parent ASIN) mapping
    4. Compute item popularity statistics from the training set
    5. Iterate over all evaluation configurations
    6. Load precomputed reranked predictions
    7. Compute evaluation metrics using RecBole and sustainability scoring
    8. Aggregate all results

    Returns:
        None
    """
    # Load models and dataset
    models_bundle = {}
    for model in SUPPORTED_MODELS:
        models_bundle[model] = load_data_and_model(
            model_file=model_registry[model]["model_path"],
        )

    # Load emission data
    emission_data = {}
    for llm in emission_data_paths:
        emission_data[llm] = {
            d["parent_asin"]: d["co2e_kg"]["value"]
            for d in load_jsonl(emission_data_paths[llm])
            if d["co2e_kg"] and d["co2e_kg"]["value"] is not None
        }

    # Load (Item Index, Parent ASIN) mapping
    item_map_df = pd.read_csv(DATA_INTERIM_MAPS_IMAP_PATH, sep="\t")
    item_map = dict(
        zip(
            item_map_df["item_index"],
            item_map_df["parent_asin"],
        ),
    )

    # Compute item popularity in the training set as raw interaction
    # counts per item ID
    dataset = models_bundle[next(iter(SUPPORTED_MODELS))][2]
    train_item_ids = dataset.inter_feat[dataset.iid_field].numpy()
    item_popularity = np.bincount(
        train_item_ids,
        minlength=dataset.item_num,
    )
    item_popularity = list(enumerate(item_popularity))

    results = []
    # Calculate total count based on available LLMs per model
    total_count = 0
    for model in SUPPORTED_MODELS:
        total_count += (
            len(model_registry[model]["preds_paths"].keys())
            * len(SUPPORTED_RERANKING_ALPHAS)
            * len(config["recbole"]["topk"])
        )

    with tqdm(
        total=total_count,
        desc="Evaluating recommendation models",
    ) as pbar:
        for model in SUPPORTED_MODELS:
            for llm in model_registry[model]["preds_paths"].keys():
                for alpha in SUPPORTED_RERANKING_ALPHAS:
                    # Load prediction data
                    preds_path = model_registry[model]["preds_paths"][llm][
                        alpha
                    ]
                    preds_data = torch.load(preds_path, weights_only=False)

                    # Evaluate results for each top-k value
                    for top_k in config["recbole"]["topk"]:
                        res = _compute_evaluation_metrics(
                            preds_data=preds_data,
                            top_k=top_k,
                            dataset=dataset,
                            item_map=item_map,
                            emission_data=emission_data.get(llm, {}),
                            item_popularity=item_popularity,
                            config=config["recbole"],
                        )
                        results.append(res)
                        pbar.update(1)

    # Display results
    print("Evaluation Results")

    results_df = pd.DataFrame(results)
    for col in results_df.select_dtypes(include=["float64"]).columns:
        results_df[col] = results_df[col].round(6)

    # Get all numeric columns excluding MODEL, ALPHA, TOP-K
    all_cols = set(results_df.columns)
    exclude_cols = {"MODEL", "ALPHA", "TOP-K"}
    metrics = sorted([col for col in all_cols if col not in exclude_cols])
    top_k_values = sorted(results_df["TOP-K"].unique())
    alphas = sorted(results_df["ALPHA"].unique())

    # Display one table per model
    for model in SUPPORTED_MODELS:
        model_df = results_df[results_df["MODEL"] == model].copy()
        print(f"\n{'=' * 100}")
        print(f"Model: {model}")
        print(f"{'=' * 100}")

        for top_k in top_k_values:
            top_k_df = model_df[model_df["TOP-K"] == top_k].copy()
            print(f"\nTop-K: {top_k}")
            print("-" * 100)

            # Create pivot table: alpha on rows, metrics on columns
            data_for_table = []
            for alpha in alphas:
                row = {"Alpha": alpha}
                alpha_df = top_k_df[top_k_df["ALPHA"] == alpha]
                for metric in metrics:
                    if not alpha_df.empty:
                        value = alpha_df[metric].values[0]
                        row[metric] = f"{value:.4f}"
                    else:
                        row[metric] = "N/A"
                data_for_table.append(row)

            table_df = pd.DataFrame(data_for_table)
            print(table_df.to_string(index=False))

    print("\n" + "=" * 100 + "\n")

    # Save all results (one row per model, top-k and alpha), using the
    # RecBole metric names as column headers
    column_names = {
        "MODEL": "Model",
        "TOP-K": "Top-K",
        "ALPHA": "Alpha",
        "EMISSIONS": "Emissions",
        "AVERAGEPOPULARITY": "AveragePopularity",
        "GINIINDEX": "GiniIndex",
        "ITEMCOVERAGE": "ItemCoverage",
        "TAILPERCENTAGE": "TailPercentage",
        "RECALL": "Recall",
        "PRECISION": "Precision",
        "HIT": "Hit",
    }
    out_df = results_df.rename(
        columns=lambda c: column_names.get(c, c if c in {"F1", "MRR", "NDCG"} else c.title()),
    )
    first = ["Model", "Top-K", "Alpha"]
    out_df = out_df[first + sorted(c for c in out_df.columns if c not in first)]
    out_df = out_df.sort_values(first).reset_index(drop=True)
    os.makedirs(os.path.dirname(RESULTS_ELEC_PATH), exist_ok=True)
    out_df.to_csv(RESULTS_ELEC_PATH, sep="\t", index=False, float_format="%.4f")
    print(f"Results saved to {RESULTS_ELEC_PATH}")


if __name__ == "__main__":
    evaluate_recsys()

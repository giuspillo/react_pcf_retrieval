"""use_case/src/predict.py

Predict with trained RecBole models and apply sustainability-aware re-ranking.
"""

import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from recbole.data.dataloader import AbstractDataLoader
from recbole.data.dataset import Dataset
from recbole.model.abstract_recommender import AbstractRecommender
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
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
    SUPPORTED_LLMS,
    SUPPORTED_MODELS,
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
np.random.seed(config["inference"]["seed"])
torch.manual_seed(config["inference"]["seed"])
torch.cuda.manual_seed_all(config["inference"]["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _compute_sas_scores(
    emission_data: dict,
    item_ids: torch.Tensor,
    scores: list[float],
    dataset: Dataset,
    item_map: dict,
    alpha: float,
    global_min_emission: float,
    global_max_emission: float,
) -> np.ndarray:
    """Re-ranks a list of recommended items using a sustainability-aware scoring
    (SaS) function.

    This function combines recommendation relevance scores with emission data
    to produce a sustainability-aware ranking. Specifically, for each recommended item:
    1. Converts internal RecBole item IDs to external ASIN identifiers
    2. Retrieves the corresponding emission values
    3. Normalizes both recommendation scores and emission values
    4. Combines them using a weighted sustainability-aware score (SaS)
    5. Sorts items according to the resulting scores
    The final ranking balances recommendation relevance and sustainability
    according to the provided alpha parameter.

    Args:
        emission_data (dict):
            Mapping from external item identifiers (ASIN) to emission values.

        item_ids (torch.Tensor):
            Sequence of internal RecBole item IDs representing the
            recommended items for a user.

        scores (list[float]):
            Recommendation relevance scores associated with the input items.

        dataset (Dataset):
            RecBole dataset object.

        item_map (dict):
            Mapping from internal item indices to external ASIN identifiers.

        alpha (float):
            Weighting factor controlling the trade-off between recommendation
            relevance and sustainability.

        global_min_emission (float):
            Minimum emission value across all items, used for normalization.

        global_max_emission (float):
            Maximum emission value across all items, used for normalization.

    Returns:
        np.ndarray:
            Array containing the re-ranked internal item IDs ordered
            according to the sustainability-aware scores.
    """
    # Convert internal RecBole item IDs to external ASIN identifiers
    external_item_ids = []
    for item_id in item_ids:
        raw_token = dataset.id2token(dataset.iid_field, int(item_id))
        item_idx = int(raw_token)
        asin = item_map.get(item_idx)
        external_item_ids.append(asin)

    # Retrieve emission values for the recommended items
    emission_values = np.array(
        [emission_data.get(asin) for asin in external_item_ids],
    )

    # Apply logarithmic transformation to emphasize differences in lower emission ranges
    # (add 1 to avoid log(0) if global_min_emission is 0)
    log_emissions = np.log1p(emission_values)
    log_global_max = np.log1p(global_max_emission)
    log_global_min = np.log1p(global_min_emission)

    # Normalize emission values (min-max normalization based on log scale)
    if log_global_max != log_global_min:
        emission_values_norm = (log_global_max - log_emissions) / (
            log_global_max - log_global_min
        )
    else:
        emission_values_norm = np.zeros_like(emission_values)

    # Normalize recommendation scores (min-max normalization)
    scores = np.array(scores)
    if scores.max() != scores.min():
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores_norm = np.zeros_like(scores)

    # Compute sustainability-aware scores
    if alpha == RERANKING_ALPHA_PURE:
        sas_scores = scores_norm
    else:
        emission_values_norm = np.power(emission_values_norm, 3)
        sas_scores = alpha * scores_norm + (1 - alpha) * emission_values_norm

    # Sort items by sustainability-aware score in descending order
    sas_scores_sorted = np.argsort(sas_scores)[::-1]

    # Convert item IDs to NumPy and reorder them by sustainability-aware scores
    return item_ids.cpu().numpy()[sas_scores_sorted].copy()


def _compute_top_k_recommendations(
    model: AbstractRecommender,
    dataset: Dataset,
    test_data: AbstractDataLoader,
    top_k: int,
    chunk_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Computes top-k item recommendations for all users in the test set
    using a trained RecBole model.

    This function performs full-sort inference over all users contained in
    the test set. To ensure scalability and memory efficiency, users are
    processed in fixed-size chunks. For each chunk, the function computes the
    top-k item scores and corresponding item indices, and aggregates the results
    across all chunks.

    Args:
        model (AbstractRecommender):
            Trained RecBole recommendation model used for inference.

        dataset (Dataset):
            RecBole dataset object used to access metadata.

        test_data (AbstractDataLoader):
            RecBole dataloader containing the test interaction data.

        top_k (int):
            Number of top-ranked items to retrieve per user.

        chunk_size (int):
            Number of users processed per iteration to control memory usage.

        device (str):
            Computation device used for inference.

    Returns:
        tuple[torch.Tensor, torch.Tensor, np.ndarray]:
            - scores: Tensor of shape [num_users, top_k] containing predicted relevance scores
            - item_ids: Tensor of shape [num_users, top_k] containing recommended item indices
            - user_ids: Numpy array containing the internal user IDs
    """
    # Get all the internal RecBole user IDs from the test set
    user_ids = np.unique(
        test_data.dataset.inter_feat[dataset.uid_field].numpy(),
    )

    # Process users in chunks to manage memory usage
    tot_users = len(user_ids)
    scores = []
    item_ids = []
    for i in tqdm(
        range(math.ceil(tot_users / chunk_size)),
        desc=f"Computing top-{top_k} recommendations for {tot_users} users",
    ):
        # Calculate start and end indices for the current chunk
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, tot_users)

        # Get the internal user IDs for the current chunk
        chunk_user_ids = torch.tensor(user_ids[start_idx:end_idx])

        # Compute top-k scores and item IDs for the current chunk of users
        chunk_scores, chunk_item_ids = full_sort_topk(
            chunk_user_ids,
            model,
            test_data,
            k=top_k,
            device=device,
        )
        scores.append(chunk_scores.cpu())
        item_ids.append(chunk_item_ids.cpu())

    # Concatenate results from all chunks
    scores = torch.cat(scores, dim=0)
    item_ids = torch.cat(item_ids, dim=0)

    return scores, item_ids, user_ids


def _rerank_top_k_recommendations(
    scores: torch.Tensor,
    item_ids: torch.Tensor,
    user_ids: np.ndarray,
    dataset: Dataset,
    test_data: AbstractDataLoader,
    item_map: dict,
    emission_data: dict,
    alpha: float,
    top_k: int,
    global_min_emission: float,
    global_max_emission: float,
):
    """Re-ranks top-K item recommendations using a sustainability-aware scoring function
    and computes evaluation signals against ground truth interactions.

    This function takes pre-computed model scores and item rankings, applies a
    secondary ranking step based on a sustainability-aware objective, and
    evaluates the resulting ranked lists against the ground truth interactions
    in the test set. Specifically, for each user:
    1. Retrieves the top-K recommended items from the base recommender
    2. Applies sustainability-aware reranking using emission-aware item scores
    3. Truncates the reranked list to the final top-K items
    4. Computes binary relevance indicators (hit vector) against ground truth
    5. Stores the total number of relevant items per user

    Args:
        scores (torch.Tensor):
            Tensor of shape [num_users, top_k] containing predicted relevance
            scores from the base recommender.

        item_ids (torch.Tensor):
            Tensor of shape [num_users, top_k] containing item indices
            predicted by the base recommender.

        user_ids (np.ndarray):
            Numpy array containing the internal user IDs.

        dataset (Dataset):
            RecBole dataset object.

        test_data (AbstractDataLoader):
            Test dataloader containing user-item interaction data.

        item_map (dict):
            Mapping from internal item indices to external identifiers (ASIN).

        emission_data (dict):
            Mapping from item identifiers to emission values.

        alpha (float):
            Weighting factor controlling the trade-off between recommendation
            relevance and emission-aware scoring.

        top_k (int):
            Number of items to keep in the final reranked recommendation list.

        global_min_emission (float):
            Minimum emission value across all items, used for normalization.

        global_max_emission (float):
            Maximum emission value across all items, used for normalization.

    Returns:
        tuple:
            - pos_matrix (list[list[int]]):
                Binary relevance matrix indicating whether each reranked item
                is present in the ground truth for each user.

            - pos_len (list[int]):
                Number of relevant (ground-truth) items per user.

            - reranked_item_ids (list[list[int]]):
                Final reranked top-K item lists per user.
    """
    # Get all the internal RecBole user and item IDs from the test set
    gt_user_ids = test_data.dataset.inter_feat[dataset.uid_field].numpy()
    gt_item_ids = test_data.dataset.inter_feat[dataset.iid_field].numpy()

    # Build ground truth map (internal user ID -> set of interacted items)
    gt_map = defaultdict(set)
    for u, i in zip(gt_user_ids, gt_item_ids):
        gt_map[u].add(i)

    # Re-rank recommendations for each user
    tot_users = len(user_ids)
    pos_matrix = []
    pos_len = []
    reranked_item_ids = []
    for idx, user_id in enumerate(
        tqdm(
            user_ids,
            desc=f"Re-ranking top-{top_k} recommendations for {tot_users} users with alpha={alpha}",
        ),
    ):
        # Re-rank the top-k recommendations for the current user
        user_reranked_item_ids = _compute_sas_scores(
            emission_data=emission_data,
            item_ids=item_ids[idx],
            scores=scores[idx].tolist(),
            dataset=dataset,
            item_map=item_map,
            alpha=alpha,
            global_min_emission=global_min_emission,
            global_max_emission=global_max_emission,
        )

        # Keep track of the top-k re-ranked items for the current user
        reranked_item_ids.append(user_reranked_item_ids[:top_k])

        # Get the ground truth interacted items for the current user
        # to compute the position of relevant items in the re-ranked list
        user_gt = gt_map[user_id]
        hits = [
            1 if int(item) in user_gt else 0
            for item in user_reranked_item_ids[:top_k]
        ]

        # Keep track of the binary relevance indicators for items in the reranked
        # list and total number of relevant items per user
        pos_matrix.append(hits)
        pos_len.append(len(user_gt))

    return pos_matrix, pos_len, reranked_item_ids


def predict_recommendations() -> None:
    """Runs the full recommendation inference pipeline including:
    1. Loading trained recommendation models
    2. Loading emission data
    3. Loading (Item Index, Parent ASIN) mapping
    4. Computing top-K recommendations for all users
    5. Re-ranking recommendations using sustainability-aware scoring (SaS)
    6. Evaluating reranked lists against ground truth
    7. Saving results

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
    for llm in SUPPORTED_LLMS:
        try:
            emission_data[llm] = {
                d["parent_asin"]: d["co2e_kg"]["value"]
                for d in load_jsonl(emission_data_paths[llm])
                if d["co2e_kg"] and d["co2e_kg"]["value"] is not None
            }
        except Exception:
            emission_data[llm] = {}

    # Load (Item Index, Parent ASIN) mapping
    item_map_df = pd.read_csv(DATA_INTERIM_MAPS_IMAP_PATH, sep="\t")
    item_map = dict(
        zip(
            item_map_df["item_index"],
            item_map_df["parent_asin"],
        ),
    )

    # Compute top-k recommendations
    recommendations = {}
    for model in SUPPORTED_MODELS:
        scores, item_ids, user_ids = _compute_top_k_recommendations(
            model=models_bundle[model][1],
            dataset=models_bundle[model][2],
            test_data=models_bundle[model][5],
            top_k=config["inference"]["top_k"],
            chunk_size=config["inference"]["chunk_size"],
            device=models_bundle[model][0]["device"],
        )
        recommendations[model] = {
            "scores": scores,
            "item_ids": item_ids,
            "user_ids": user_ids,
        }

    # Re-rank recommendations varying the alpha parameter
    for llm in SUPPORTED_LLMS:
        # Skip LLM if emission data is not available
        if not emission_data.get(llm):
            continue

        # Compute global min and max emission values for normalization
        # across all items
        all_emissions = list(emission_data[llm].values())
        global_min_emission = min(all_emissions) if all_emissions else 0
        global_max_emission = max(all_emissions) if all_emissions else 1

        for model in SUPPORTED_MODELS:
            # Get the pre-computed scores and item IDs for the current model
            scores = recommendations[model]["scores"]
            item_ids = recommendations[model]["item_ids"]
            user_ids = recommendations[model]["user_ids"]
            for alpha in SUPPORTED_RERANKING_ALPHAS:
                # Re-rank the top-k recommendations and compute evaluation signals
                pos_matrix, pos_len, reranked_item_ids = (
                    _rerank_top_k_recommendations(
                        scores=scores,
                        item_ids=item_ids,
                        user_ids=user_ids,
                        dataset=models_bundle[model][2],
                        test_data=models_bundle[model][5],
                        item_map=item_map,
                        emission_data=emission_data[llm],
                        alpha=alpha,
                        top_k=config["inference"]["top_k"],
                        global_min_emission=global_min_emission,
                        global_max_emission=global_max_emission,
                    )
                )

                # Save the results for the current model and alpha
                results = {
                    "pos_matrix": pos_matrix,
                    "pos_len": pos_len,
                    "reranked_item_ids": reranked_item_ids,
                    "model": model,
                    "alpha": alpha,
                }
                preds_path = model_registry[model]["preds_paths"][llm][alpha]
                os.makedirs(os.path.dirname(preds_path), exist_ok=True)
                torch.save(results, preds_path)

    # Print summary of saved predictions
    print("Predict Recommendations Summary")

    summary_data = []
    total_predictions = 0

    for model in SUPPORTED_MODELS:
        for llm in model_registry[model]["preds_paths"].keys():
            for alpha in SUPPORTED_RERANKING_ALPHAS:
                pred_path = model_registry[model]["preds_paths"][llm][alpha]

                # Check if file exists and get size
                if os.path.exists(pred_path):
                    file_size_mb = os.path.getsize(pred_path) / (1024 * 1024)
                    # Load to get statistics
                    try:
                        pred_data = torch.load(pred_path, weights_only=False)
                        n_users = len(pred_data.get("reranked_item_ids", []))
                        n_items_per_user = (
                            len(pred_data.get("reranked_item_ids", [[]])[0])
                            if n_users > 0
                            else 0
                        )
                        total_predictions += n_users * n_items_per_user
                    except Exception:
                        n_users = "Error"
                        n_items_per_user = "Error"
                else:
                    file_size_mb = 0.0
                    n_users = "Missing"
                    n_items_per_user = "Missing"

                summary_data.append(
                    {
                        "Model": model,
                        "LLM": llm,
                        "Alpha": alpha,
                        "Num Users": n_users,
                        "Items Per User": n_items_per_user,
                        "File Size (MB)": f"{file_size_mb:.2f}",
                        "Path": pred_path,
                    },
                )

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    predict_recommendations()

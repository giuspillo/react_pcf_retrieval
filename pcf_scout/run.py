"""pcf_scout/run.py

Run PCF-Scout to estimate the PCF of products, either on the ground truth
(several calls per item, compared with the zero-shot baseline) or on the
use-case catalog (one call per item).

Usage (from the repository root):
    python -m pcf_scout.run

Settings are read from config/dataset.yaml and config/emissions_enrichment.yaml.
New estimates are written to outputs/, never over the released data.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm.asyncio import tqdm

from pcf_scout.agent.context import Context
from pcf_scout.agent.graph import graph
from pcf_scout.agent.state import InputState
from pcf_scout.const import (
    DATASET_NAME_ELEC,
    GROUND_TRUTH_DATA_DIR,
    LLM_KEYS,
    OUTPUTS_DIR,
    SUPPORTED_DATASETS,
    SUPPORTED_LLMS,
    USE_CASE_RAW_ITEM_METADATA_PATH,
)
from pcf_scout.utils import load_config, load_jsonl

# Load environment variables
load_dotenv()

# Load configuration
config = load_config()

# Determine dataset, data type and model
dataset_name = config["dataset"]["name"]
assert dataset_name in SUPPORTED_DATASETS, (
    f"Supported dataset: {SUPPORTED_DATASETS}, got: {dataset_name}"
)
is_ground_truth = config["emissions_enrichment"]["is_ground_truth"]
model = config["emissions_enrichment"]["model"]
assert model in SUPPORTED_LLMS, (
    f"Supported LLMs: {SUPPORTED_LLMS}, got: {model}"
)
llm_key = LLM_KEYS[model]

if not is_ground_truth and dataset_name != DATASET_NAME_ELEC:
    raise ValueError(
        f"The use-case catalog is only available for {DATASET_NAME_ELEC}. "
        "Set is_ground_truth: true for the other domains.",
    )

# Input and output paths
if is_ground_truth:
    raw_jsonl_file = f"{GROUND_TRUTH_DATA_DIR}/{dataset_name}.jsonl"
    processed_jsonl_file = (
        f"{OUTPUTS_DIR}/ground_truth/{llm_key}/{dataset_name}.jsonl"
    )
else:
    raw_jsonl_file = USE_CASE_RAW_ITEM_METADATA_PATH
    processed_jsonl_file = (
        f"{OUTPUTS_DIR}/use_case/{llm_key}/{dataset_name}.jsonl"
    )
Path(processed_jsonl_file).parent.mkdir(parents=True, exist_ok=True)


def _gt_calls(product_data: dict[str, Any]) -> list:
    """Return (creating it if needed) the list of PCF-Scout calls of a
    ground-truth item for the current LLM."""
    return product_data.setdefault("pcf_scout", {}).setdefault(llm_key, [])


def _valid_values(calls: list) -> list[float]:
    """Return the PCF values of the successful calls."""
    return [
        c["value"]
        for c in calls
        if isinstance(c, dict) and c.get("value") is not None
    ]


def _calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    This function computes RMSE, representing the square root of the average
    squared differences between true and predicted values. It is sensitive to
    large errors and provides a measure of the magnitude of prediction errors.

    Args:
        y_true (np.ndarray):
            True values.

        y_pred (np.ndarray):
            Predicted values.

    Returns:
        float:
            RMSE value in the same unit as the input data.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _calculate_me(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Error (ME).

    This function computes ME, representing the average difference between
    true and predicted values. It preserves the sign of the error to
    indicate systematic overestimation or underestimation.

    Args:
        y_true (np.ndarray):
            True values.

        y_pred (np.ndarray):
            Predicted values.

    Returns:
        float:
            ME value in the same unit as the input data.
    """
    return float(np.mean(y_true - y_pred))


def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE).

    This function computes MAPE, representing the average absolute percentage
    difference between true and predicted values.

    Args:
        y_true (np.ndarray):
            True values.

        y_pred (np.ndarray):
            Predicted values.

    Returns:
        float:
            MAPE value (0-100).
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def _calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Weighted Absolute Percentage Error (WAPE).

    This function computes WAPE, representing the total absolute error
    weighted by the sum of the true values. It is more robust than MAPE
    when dealing with true values close to or equal to zero.

    Args:
        y_true (np.ndarray):
            True values.

        y_pred (np.ndarray):
            Predicted values.

    Returns:
        float:
            WAPE value (0-100 under normal conditions, can exceed 100).
    """
    sum_absolute_error = np.sum(np.abs(y_true - y_pred))
    sum_true_values = np.sum(np.abs(y_true))

    if sum_true_values == 0:
        return 0.0 if sum_absolute_error == 0 else float("inf")

    return (sum_absolute_error / sum_true_values) * 100


def _calculate_ndcg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG).

    This function computes NDCG, a ranking metric that evaluates the quality of
    predicted rankings based on the true relevance of items.

    Args:
        y_true (np.ndarray):
            True values (relevance scores).

        y_pred (np.ndarray):
            Predicted values (ranking scores).

    Returns:
        float:
            NDCG value (0-1).
    """
    # Get indices that would sort predictions in descending order
    pred_indices = np.argsort(-y_pred)

    # Get the true values (relevance) in the order of predictions
    relevance = y_true[pred_indices]

    # Calculate DCG: sum of (relevance / log2(position + 1))
    positions = np.arange(1, len(relevance) + 1)
    dcg = np.sum(relevance / np.log2(positions + 1))

    # Calculate IDCG: DCG of ideal ranking (when items are sorted
    # by true values in descending order)
    ideal_relevance = np.sort(y_true)[::-1]
    ideal_positions = np.arange(1, len(ideal_relevance) + 1)
    idcg = np.sum(ideal_relevance / np.log2(ideal_positions + 1))

    # NDCG = DCG / IDCG
    return dcg / idcg if idcg > 0 else 0.0


def _calculate_rmse_to_mae_ratio(rmse: float, mae: float) -> float:
    """Calculate the ratio of RMSE to MAE.

    This function computes the ratio of RMSE to MAE, providing insight into
    the distribution of errors. A ratio close to 1 suggests that errors are
    uniformly distributed, while a higher ratio indicates the presence of
    larger outliers.

    Args:
        rmse (float):
            Root Mean Squared Error value.

        mae (float):
            Mean Absolute Error value.

    Returns:
        float:
            Ratio of RMSE to MAE (unitless).
    """
    if mae == 0.0:
        return float("inf") if rmse > 0 else 1.0

    return rmse / mae


def _calculate_metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute a comprehensive set of metrics.

    This function computes the following metrics:
    - Mean Absolute Error (MAE)
    - Mean Error (ME)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Percentage Error (MAPE)
    - Weighted Absolute Percentage Error (WAPE)
    - Normalized Discounted Cumulative Gain (NDCG)
    - Spearman's Rank Correlation Coefficient
    - RMSE to MAE Ratio

    Args:
        y_true (np.ndarray):
            Array of true values.

        y_pred (np.ndarray):
            Array of predicted values.

    Returns:
        dict[str, float]:
            Dictionary containing all computed metric values.
    """
    # Mask to filter out pairs where either true or predicted value is NaN
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    if len(y_true_filtered) == 0:
        return dict.fromkeys(
            [
                "MAE",
                "ME",
                "RMSE",
                "MAPE",
                "WAPE",
                "NDCG",
                "Spearman",
                "RMSE/MAE",
            ],
            0.0,
        )

    mae = mean_absolute_error(y_true_filtered, y_pred_filtered)
    me = _calculate_me(y_true_filtered, y_pred_filtered)
    rmse = _calculate_rmse(y_true_filtered, y_pred_filtered)
    mape = _calculate_mape(y_true_filtered, y_pred_filtered)
    wape = _calculate_wape(y_true_filtered, y_pred_filtered)
    ndcg = _calculate_ndcg(y_true_filtered, y_pred_filtered)
    spearman_corr, _ = spearmanr(y_true_filtered, y_pred_filtered)
    rmse_to_mae_ratio = _calculate_rmse_to_mae_ratio(rmse, mae)

    return {
        "MAE": mae,
        "ME": me,
        "RMSE": rmse,
        "MAPE": mape,
        "WAPE": wape,
        "NDCG": ndcg,
        "Spearman": spearman_corr,
        "RMSE/MAE": rmse_to_mae_ratio,
    }


async def _process_single_product(
    product_data: dict[str, Any],
    context: Context,
    semaphore: asyncio.Semaphore,
    is_ground_truth: bool,
) -> dict[str, Any]:
    """Process a single product asynchronously with concurrency limit.

    This function processes a single product by invoking the agent to compute the
    carbon footprint estimate. It uses an asyncio semaphore to limit the number
    of concurrent invocations. The function extracts the carbon footprint value
    from the agent's result and updates the product data.

    Args:
        product_data (dict[str, Any]):
            Product data dictionary.

        context (Context):
            Context configuration for the agent.

        semaphore (asyncio.Semaphore):
            Asyncio semaphore to limit concurrency.

        is_ground_truth (bool):
            Flag indicating whether the product data comes from the ground
            truth or not.

    Returns:
        dict[str, Any]:
            Updated product data.
    """
    async with semaphore:
        # Create input state for the agent with product data
        title = product_data["title"]
        input_state = InputState(product_data={"title": title})

        # Initialize the appropriate field in product data to store
        # the carbon footprint
        if is_ground_truth:
            _gt_calls(product_data)
        elif "co2e_kg" not in product_data:
            product_data["co2e_kg"] = None

        try:
            # Invoke the agent
            result = await graph.ainvoke(input=input_state, context=context)

            # Extract carbon footprint value from result
            # and save it in the product data
            co2e_kg = result["co2e_kg"]
            if is_ground_truth:
                _gt_calls(product_data).append(co2e_kg)
            else:
                product_data["co2e_kg"] = co2e_kg
        except Exception as e:
            print(f"Error processing '{title}': {e!s}")

            # If there's an error, set the carbon footprint value to
            # None to indicate failure
            if is_ground_truth:
                _gt_calls(product_data).append(None)
            else:
                product_data["co2e_kg"] = None

        return product_data


async def _calculate_metrics(
    products: list,
    compare_with_baseline: bool,
) -> None:
    """Calculate metrics from products list.

    This function computes metrics from a matrix where rows represent products
    and columns represent independent estimate calls. Metrics are calculated
    column-wise (across products for each call) while statistics are calculated
    row-wise (consistency within each product across calls). Optionally compares
    current metrics with baseline.

    Args:
        products (list):
            List of product data dictionaries.

        compare_with_baseline (bool):
            If True, compares current metrics with baseline metrics.

    Returns:
        None
    """
    # Extract true values
    true_values = np.array(
        [float(p["pcf_kg"]) for p in products],
    )
    num_products = len(products)
    num_calls = config["emissions_enrichment"]["num_estimates_per_product"]

    # Populate current prediction matrix (rows=products, columns=calls) with
    # estimates from products data
    current_preds_matrix = np.full((num_products, num_calls), np.nan)
    for row_idx, product_data in enumerate(products):
        current_estimates = _gt_calls(product_data)
        for col_idx in range(min(len(current_estimates), num_calls)):
            est = current_estimates[col_idx]
            if est is not None:
                val = est.get("value") if isinstance(est, dict) else est
                current_preds_matrix[row_idx, col_idx] = (
                    float(val) if val is not None else np.nan
                )

    # Calculate current column-wise metrics (across products for each call)
    current_call_metrics = [
        _calculate_metric_values(true_values, current_preds_matrix[:, i])
        for i in range(num_calls)
    ]
    current_metrics = (
        {
            k: np.mean([m[k] for m in current_call_metrics])
            for k in current_call_metrics[0]
        }
        if current_call_metrics
        else {}
    )

    # Calculate current row-wise statistics (within each product across calls)
    current_means = np.nanmean(current_preds_matrix, axis=1)
    current_stds = np.nanstd(current_preds_matrix, axis=1)
    current_metrics["StdDev"] = (
        float(np.nanmean(current_stds)) if len(current_stds) > 0 else 0.0
    )
    current_metrics["CV"] = (
        float(
            np.nanmean(
                np.where(
                    current_means != 0,
                    current_stds / current_means,
                    0.0,
                ),
            ),
        )
        if len(current_means) > 0
        else 0.0
    )

    baseline_metrics = {}
    if compare_with_baseline:
        # Populate baseline prediction matrix (rows=products, columns=calls) with
        # baseline estimates from products data
        baseline_preds_matrix = np.full((num_products, num_calls), np.nan)
        for row_idx, product_data in enumerate(products):
            baseline_estimates = product_data["zero_shot"][llm_key]
            baseline_preds_matrix[row_idx, : len(baseline_estimates)] = [
                float(x) if x is not None else np.nan
                for x in baseline_estimates[:num_calls]
            ]

        # Calculate baseline column-wise metrics (across products for each call)
        baseline_call_metrics = [
            _calculate_metric_values(true_values, baseline_preds_matrix[:, i])
            for i in range(num_calls)
        ]
        baseline_metrics = (
            {
                k: np.mean([m[k] for m in baseline_call_metrics])
                for k in baseline_call_metrics[0]
            }
            if baseline_call_metrics
            else {}
        )

        # Calculate baseline row-wise statistics (within each product across calls)
        baseline_means = np.nanmean(baseline_preds_matrix, axis=1)
        baseline_stds = np.nanstd(baseline_preds_matrix, axis=1)
        baseline_metrics["StdDev"] = (
            float(np.nanmean(baseline_stds)) if len(baseline_stds) > 0 else 0.0
        )
        baseline_metrics["CV"] = (
            float(
                np.nanmean(
                    np.where(
                        baseline_means != 0,
                        baseline_stds / baseline_means,
                        0.0,
                    ),
                ),
            )
            if len(baseline_means) > 0
            else 0.0
        )

    # Baseline metrics (if baseline comparison is enabled)
    if compare_with_baseline:
        print("\nBaseline System Results:")
        print("-" * 100)
        for metric_name, metric_value in baseline_metrics.items():
            if isinstance(metric_value, (float, np.floating)):
                print(f"  {metric_name:12s}: {metric_value:12.4f}")
            else:
                print(f"  {metric_name:12s}: {metric_value}")

    # Current system metrics
    print("\nCurrent System Results:")
    print("-" * 100)
    for metric_name, metric_value in current_metrics.items():
        if isinstance(metric_value, (float, np.floating)):
            print(f"  {metric_name:12s}: {metric_value:12.4f}")
        else:
            print(f"  {metric_name:12s}: {metric_value}")

    # Baseline-current system comparison (if baseline comparison is enabled)
    if compare_with_baseline:
        print("\nBaseline vs. Current System Results:")
        print("-" * 100)
        for metric_name in baseline_metrics:
            baseline_val = baseline_metrics[metric_name]
            current_val = current_metrics[metric_name]

            # For ranking metrics, higher is better
            if metric_name in ["NDCG", "Spearman"]:
                improvement = current_val - baseline_val
                improvement_pct = (
                    (improvement / abs(baseline_val) * 100)
                    if baseline_val != 0
                    else 0
                )
                direction = "↑" if improvement > 0 else "↓"

            # For error metrics and statistics, lower is better
            else:
                improvement = baseline_val - current_val
                improvement_pct = (
                    (improvement / abs(baseline_val) * 100)
                    if baseline_val != 0
                    else 0
                )
                direction = "↓" if improvement > 0 else "↑"

            print(
                f"  {metric_name:12s}: {improvement:+.4f} ({improvement_pct:+.2f}%) {direction}",
            )


async def enrich_data_with_emissions() -> None:
    """Enrich product with emissions data.

    This function enriches product data with emissions estimates by processing
    products from a raw JSONL file. It checks if emissions data is already
    computed for each product and only processes those that require it. The
    function uses asynchronous processing with concurrency limits to efficiently
    compute emissions estimates for products. After processing, it updates the
    product data and writes it back to a processed JSONL file. If the data being
    processed is ground truth data, it also calculates and displays metrics
    comparing the estimates with the ground truth values.

    Returns:
        None
    """
    # Load products from JSONL file and check if all the
    # emissions are already computed
    processed_lookup = {}
    if Path(processed_jsonl_file).exists():
        for p in load_jsonl(str(Path(processed_jsonl_file))):
            key = p.get("parent_asin") or p.get("id") or p.get("title")
            if key:
                processed_lookup[key] = p

    products = []
    products_to_process = []
    for idx, product_data in enumerate(load_jsonl(str(Path(raw_jsonl_file)))):
        lookup_key = (
            product_data.get("parent_asin")
            or product_data.get("id")
            or product_data.get("title")
        )
        if lookup_key in processed_lookup:
            existing_data = processed_lookup[lookup_key]
        else:
            existing_data = product_data
            # New ground-truth run: start from an empty list of calls for
            # the current LLM (the released calls are not reused)
            if is_ground_truth:
                existing_data.setdefault("pcf_scout", {})[llm_key] = []

        # Save product data
        products.append(existing_data)

        # Check if emissions are already computed for this product
        if is_ground_truth:
            if (
                len(_valid_values(_gt_calls(existing_data)))
                < config["emissions_enrichment"]["num_estimates_per_product"]
            ):
                # This product does not have the required number of
                # emission estimates
                products_to_process.append((idx, existing_data))
        elif (
            not isinstance(existing_data.get("co2e_kg"), dict)
            or existing_data["co2e_kg"].get("value") is None
        ):
            # This product does not have a valid emission value
            products_to_process.append((idx, existing_data))

    # If all products already have the required emission data and
    # we are processing ground truth data, skip processing and directly
    # calculate metrics comparing emissions estimates with the ground truth value
    if (
        len(products_to_process) == 0
        and is_ground_truth
    ):
        print(
            f"All estimates already computed in {processed_jsonl_file}, calculating metrics...",
        )
        await _calculate_metrics(
            products,
            compare_with_baseline=config["emissions_enrichment"][
                "compare_with_baseline"
            ],
        )
        return

    # If all products already have the required emission data and we are
    # processing non-ground truth data, skip processing
    if (
        len(products_to_process) == 0
        and not is_ground_truth
    ):
        print(f"All estimates already computed in {processed_jsonl_file}!")

    # Regardless of whether we are processing ground truth data or not,
    # if there are products to process, we need to compute the estimates
    # for those products
    else:
        print(
            f"Identified {len(products_to_process)} products to process from {raw_jsonl_file}.",
        )

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(
        config["emissions_enrichment"]["max_concurrent_requests"],
    )

    # One log directory per LLM and dataset
    logs_dir = (
        Path(config["emissions_enrichment"]["logs_dir"])
        / llm_key
        / dataset_name
    )
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Process products
    tasks = []
    task_to_product_idx = {}

    for idx, product_data in products_to_process:
        # Calculate how many runs are needed for this product
        # based on whether we are processing ground truth data
        # or not and how many valid estimates it already has
        # (in case of ground truth data)
        if is_ground_truth:
            calls_done = len(_gt_calls(product_data))
            runs_needed = config["emissions_enrichment"][
                "num_estimates_per_product"
            ] - len(_valid_values(_gt_calls(product_data)))
        else:
            calls_done = 0
            runs_needed = 1

        # Generate the required number of tasks for this product based on
        # the runs needed
        for run_idx in range(runs_needed):
            safe_title = "_".join(
                [
                    w
                    for w in product_data["title"]
                    .replace("/", " ")
                    .replace("\\", " ")
                    .replace(":", " ")
                    .split()
                    if w
                ],
            )
            # Truncate safe_title to avoid filesystem filename length limits
            safe_title = safe_title[:200]
            # Ground truth: number logs by call (failed calls included), so
            # that new calls never overwrite older logs
            log_suffix = (
                f"({calls_done + run_idx + 1})" if is_ground_truth else ""
            )

            task = asyncio.create_task(
                _process_single_product(
                    product_data=product_data,
                    context=Context(
                        model=config["emissions_enrichment"]["model"],
                        model_temperature=config["emissions_enrichment"][
                            "model_temperature"
                        ],
                        search_web_max_results=config["emissions_enrichment"][
                            "search_web_max_results"
                        ],
                        search_web_type=config["emissions_enrichment"][
                            "search_web_type"
                        ],
                        search_web_max_calls=config["emissions_enrichment"][
                            "search_web_max_calls"
                        ],
                        enable_logging=config["emissions_enrichment"][
                            "enable_logging"
                        ],
                        logs_dir=str(logs_dir),
                        logs_path=f"{safe_title}{log_suffix}.log",
                    ),
                    semaphore=semaphore,
                    is_ground_truth=is_ground_truth,
                ),
            )
            tasks.append(task)
            task_to_product_idx[task] = idx

    if not tasks:
        return

    # Use tqdm to display progress bar while processing products
    with tqdm(
        total=len(tasks),
        desc="Enriching products with emission data...",
    ) as pbar:
        # Iterate over tasks as they complete to monitor progress and save
        # intermediate results to disk so progress is not lost on failure
        for next_to_complete in asyncio.as_completed(tasks):
            try:
                # Await the task and get the updated product data
                res = await next_to_complete
                # Map task -> product index and update the products list
                idx = task_to_product_idx.get(next_to_complete)
                if idx is not None and res is not None:
                    products[idx] = res
            except Exception as e:
                print(f"\nError processing a single product run: {e}")
            finally:
                # Save the current state of all products after each completed run
                try:
                    with open(processed_jsonl_file, "w") as f:
                        f.writelines(
                            json.dumps(product) + "\n" for product in products
                        )
                except Exception as e:
                    print(f"Error writing interim results: {e}")

                pbar.update(1)

    # Sort ground-truth items by their true PCF in descending order before saving
    if is_ground_truth:
        products.sort(
            key=lambda x: (
                x["pcf_kg"] if x.get("pcf_kg") is not None else float("-inf")
            ),
            reverse=True,
        )

    # Write fully updated and sorted products data back to processed JSONL file
    with open(processed_jsonl_file, "w") as f:
        f.writelines(json.dumps(product) + "\n" for product in products)

    if is_ground_truth:
        # Calculate and display metrics
        print("Calculating metrics...")
        await _calculate_metrics(
            products,
            compare_with_baseline=config["emissions_enrichment"][
                "compare_with_baseline"
            ],
        )


if __name__ == "__main__":
    asyncio.run(enrich_data_with_emissions())

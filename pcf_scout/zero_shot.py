"""pcf_scout/zero_shot.py

Zero-shot LLM baseline: estimate the PCF of products with a single prompt,
without web search.

Usage (from the repository root):
    python -m pcf_scout.zero_shot

Settings are read from config/dataset.yaml and config/zero_shot.yaml.
New estimates are written to outputs/zero_shot/, never over the released data:
  - ground truth: item["zero_shot"][<llm>] is the list of estimated values (as in
    the released data) and item["zero_shot_answers"][<llm>] keeps the full answers;
  - use-case catalog: item["co2e_kg"] = {"value", "source", "explanation"}.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from tqdm.asyncio import tqdm

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

ZERO_SHOT_PROMPT = """You are an expert in life cycle analysis (LCA) and CO2e emission calculation for {product_category} products.
You must estimate the CO2e emissions, based on the entire life cycle (cradle to grave), for the following {product_category} product.

Product data: {product_data}

INSTRUCTIONS:
1. FIRST, check if there are any official carbon footprint reports or environmental product declarations (EPD) from the manufacturer for this specific product. If found, use these official values as your primary source.

2. If NO official manufacturer reports are available, then estimate emissions following these protocols:
- GHG Protocol Product Standard for system boundaries and calculation methodology
- ISO 14040/14044 for Life Cycle Assessment principles
- PAS 2050 and ISO/TS 14067 for carbon footprint calculation guidelines

3. For estimation, consider:
- Main materials composition
- Manufacturing processes
- Transportation
- Use phase energy consumption
- End-of-life disposal

4. Use the most recent emission factors and scientific data available
5. Document your sources and assumptions in the explanation
6. Clearly state if you’re using manufacturer data or estimation

Reply ONLY with a JSON object containing these exact fields:
{{
  "co2e_kg": <number>,
  "source": <if "manufacturer report" or "estimation">,
  "explanation": "<detailed explanation including data source>"
}}
Do not include any markdown formatting or additional JSON wrappers."""

# Product category used in the prompt for each dataset
PRODUCT_CATEGORIES = {
    "electronics": "Electronics",
    "clothing": "Clothing",
    "home_and_kitchen": "Home and Kitchen",
}

# Load environment variables
load_dotenv()

# Load configuration
config = load_config()
settings = config["zero_shot"]

# Determine dataset, data type and model
dataset_name = config["dataset"]["name"]
assert dataset_name in SUPPORTED_DATASETS, (
    f"Supported dataset: {SUPPORTED_DATASETS}, got: {dataset_name}"
)
is_ground_truth = settings["is_ground_truth"]
model_name = settings["model"]
assert model_name in SUPPORTED_LLMS, (
    f"Supported LLMs: {SUPPORTED_LLMS}, got: {model_name}"
)
llm_key = LLM_KEYS[model_name]

if not is_ground_truth and dataset_name != DATASET_NAME_ELEC:
    raise ValueError(
        f"The use-case catalog is only available for {DATASET_NAME_ELEC}. "
        "Set is_ground_truth: true for the other domains.",
    )

# Input and output paths
if is_ground_truth:
    input_jsonl_file = f"{GROUND_TRUTH_DATA_DIR}/{dataset_name}.jsonl"
    output_jsonl_file = (
        f"{OUTPUTS_DIR}/zero_shot/ground_truth/{llm_key}/{dataset_name}.jsonl"
    )
else:
    input_jsonl_file = USE_CASE_RAW_ITEM_METADATA_PATH
    output_jsonl_file = (
        f"{OUTPUTS_DIR}/zero_shot/use_case/{llm_key}/{dataset_name}.jsonl"
    )
Path(output_jsonl_file).parent.mkdir(parents=True, exist_ok=True)


def _parse_answer(content: str) -> dict[str, Any]:
    """Parse the JSON answer of the LLM.

    This function removes markdown code fences, parses the JSON object and
    casts the estimated value to float. If the answer cannot be parsed, the
    value is None and the raw answer is kept as explanation.

    Args:
        content (str):
            The raw answer of the LLM.

    Returns:
        dict[str, Any]:
            Dictionary with the keys "value", "source" and "explanation".
    """
    try:
        answer = json.loads(
            content.replace("```json", "").replace("```", "").strip(),
        )
        return {
            "value": float(answer["co2e_kg"]),
            "source": answer.get("source"),
            "explanation": answer.get("explanation"),
        }
    except Exception:
        return {"value": None, "source": None, "explanation": content}


def _answers(product_data: dict[str, Any]) -> list:
    """Return (creating it if needed) the list of zero-shot answers of a
    ground-truth item for the current LLM."""
    return product_data.setdefault("zero_shot_answers", {}).setdefault(
        llm_key,
        [],
    )


def _needed(product_data: dict[str, Any]) -> int:
    """Return how many zero-shot estimates are still needed for a product."""
    if is_ground_truth:
        valid = [a for a in _answers(product_data) if a["value"] is not None]
        return settings["num_estimates_per_product"] - len(valid)
    co2e_kg = product_data.get("co2e_kg")
    return int(not isinstance(co2e_kg, dict) or co2e_kg.get("value") is None)


async def _estimate(
    product_data: dict[str, Any],
    model: Any,
    semaphore: asyncio.Semaphore,
) -> None:
    """Ask the LLM for one zero-shot estimate and store it in the product data.

    Args:
        product_data (dict[str, Any]):
            Product data dictionary (updated in place).

        model (Any):
            The chat model.

        semaphore (asyncio.Semaphore):
            Asyncio semaphore to limit concurrency.

    Returns:
        None
    """
    async with semaphore:
        prompt = ZERO_SHOT_PROMPT.format(
            product_category=PRODUCT_CATEGORIES[dataset_name],
            product_data=json.dumps(
                {
                    k: product_data[k]
                    for k in settings["product_fields"]
                    if k in product_data
                },
                ensure_ascii=False,
            ),
        )
        try:
            response = await model.ainvoke([HumanMessage(content=prompt)])
            answer = _parse_answer(str(response.content))
        except Exception as e:
            print(f"Error processing '{product_data.get('title')}': {e!s}")
            answer = {"value": None, "source": None, "explanation": str(e)}

        if is_ground_truth:
            _answers(product_data).append(answer)
            product_data.setdefault("zero_shot", {})[llm_key] = [
                a["value"] for a in _answers(product_data) if a["value"] is not None
            ]
        else:
            product_data["co2e_kg"] = answer


def _key(product_data: dict[str, Any]) -> str:
    return (
        product_data.get("parent_asin")
        or product_data.get("id")
        or product_data.get("title")
    )


async def run_zero_shot() -> None:
    """Run the zero-shot baseline on the ground truth or on the catalog.

    Products already estimated in the output file are skipped, so that an
    interrupted run can be resumed.

    Returns:
        None
    """
    # Resume from previous outputs, if any
    done = {}
    if Path(output_jsonl_file).exists():
        done = {_key(p): p for p in load_jsonl(output_jsonl_file)}

    products = []
    for product_data in load_jsonl(input_jsonl_file):
        if _key(product_data) in done:
            product_data = done[_key(product_data)]
        elif is_ground_truth:
            # New run: do not reuse the released estimates for this LLM
            product_data.setdefault("zero_shot", {})[llm_key] = []
            product_data.setdefault("zero_shot_answers", {})[llm_key] = []
        products.append(product_data)

    model = init_chat_model(
        model=model_name,
        temperature=settings["model_temperature"],
    )
    semaphore = asyncio.Semaphore(settings["max_concurrent_requests"])
    tasks = [
        asyncio.create_task(_estimate(p, model, semaphore))
        for p in products
        for _ in range(max(_needed(p), 0))
    ]
    print(f"Running {len(tasks)} zero-shot calls on {input_jsonl_file}.")

    def save() -> None:
        with open(output_jsonl_file, "w", encoding="utf-8") as f:
            f.writelines(
                json.dumps(p, ensure_ascii=False) + "\n" for p in products
            )

    with tqdm(total=len(tasks), desc="Zero-shot estimation") as pbar:
        for next_to_complete in asyncio.as_completed(tasks):
            await next_to_complete
            save()
            pbar.update(1)
    save()

    missing = sum(_needed(p) > 0 for p in products)
    print(f"Saved to {output_jsonl_file} ({missing} products still missing estimates).")


if __name__ == "__main__":
    asyncio.run(run_zero_shot())

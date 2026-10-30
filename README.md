# PCF-Scout

Repository for the paper *"PCF-Scout: Agentic Approaches for Estimating E-commerce Product Carbon Footprint in Emissions-aware Applications"* (submitted to ECIR 2027, resource track).

The resource has two components and a use case:

1. **PCF-Scout**, a ReAct agent (LangGraph) that retrieves the official Product Carbon Footprint (PCF, in kg CO2e) of a product from the web, or estimates it when no official value is available. Each answer comes with its source (`official` / `estimated`) and an explanation.
2. **A ground truth of 582 products** (194 per domain: Electronics, Clothing, Home and Kitchen) with official PCF values from Environmental Product Declarations (EPDs). For every item it includes the PCF-Scout calls and the zero-shot baseline estimates with Gemini 2.5 Flash and OpenAI o3-mini, together with the full execution logs of the agent.
3. **A use case** of sustainability-aware recommendation on Amazon Reviews'23 *Electronics*: PCF-Scout annotates the catalog, and the recommendations of BPR and LightGCN are re-ranked with the Sustainability-aware Score (SaS).

## Repository structure

```txt
.
├── config/                              <- YAML configuration of all the pipelines
├── pcf_scout/                           <- (1) The agent
│   ├── agent/                           <- LangGraph ReAct agent
│   │   ├── const.py                     <- Agent constants (defaults, graph nodes, tools)
│   │   ├── context.py                   <- Runtime configuration (LLM, web search, logging)
│   │   ├── graph.py                     <- Agent graph and execution logic
│   │   ├── prompts.py                   <- Prompts
│   │   ├── state.py                     <- Execution state
│   │   └── tools.py                     <- search_web tool (Exa)
│   ├── const.py  utils.py               <- Paths and helpers
│   ├── run.py                           <- Run PCF-Scout on the ground truth or on the use-case catalog
│   └── zero_shot.py                     <- Zero-shot LLM baseline (same prompt as below)
├── ground_truth/                        <- (2) The ground truth
│   ├── data/                            <- One JSONL file per domain (GT value + all calls, see below)
│   ├── logs/                            <- (DVC) Execution logs of PCF-Scout (one file per call) + index.tsv
│   ├── evaluate.py                      <- Reproduces the ground-truth analyses of the paper
│   └── results/                         <- Output of evaluate.py (TSV)
├── use_case/                            <- (3) Sustainability-aware recommendation
│   ├── data/electronics/
│   │   ├── raw/                         <- (DVC) Amazon Reviews'23 user reviews and item metadata of the catalog
│   │   ├── maps/                        <- User and item maps
│   │   ├── electronics{,.train,.valid,.test}.inter   <- RecBole atomic files (temporal split per user)
│   │   └── pcf_annotations/             <- PCF of the 9,710 catalog products estimated by PCF-Scout
│   ├── logs/                            <- (DVC) Execution logs of PCF-Scout on the catalog
│   ├── src/                             <- preprocess, train, predict (SaS re-ranking), evaluate
│   └── results/electronics.tsv          <- All recommendation results (k = 5, 10, 20; alpha = 0.25, 0.5, 0.75, 1.0)
├── Makefile
└── pyproject.toml
```

Folders marked with (DVC) are not stored in Git: see [Large files](#large-files-dvc-and-zenodo) to download them.

## Ground truth format

`ground_truth/data/<domain>.jsonl` contains one item per line:

```json
{
  "id": "clothing_001",
  "domain": "clothing",
  "title": "Eddy Rider Leather Jacket Black",
  "pcf_kg": 43.6,
  "emission_class": "high",
  "zero_shot": {
    "gemini_2_5_flash": [78.1, 78.1, 41.0, 41.0],
    "openai_o3_mini": [120.0, 70.0, 70.0, 70.0]
  },
  "pcf_scout": {
    "gemini_2_5_flash": [
      {"call": 1, "value": 150.0, "source": "estimated", "explanation": "..."},
      ...
    ],
    "openai_o3_mini": [...]
  }
}
```

- `pcf_kg`: official PCF from the EPD (kg CO2e per product).
- `emission_class`: `low` / `medium` / `high`, from the 33rd and 67th percentiles of `pcf_kg` within the domain.
- `zero_shot`: 4 independent zero-shot estimates per LLM (see `pcf_scout/zero_shot.py`).
- `pcf_scout`: up to 4 independent calls of PCF-Scout per LLM, each with value, source (`official` or `estimated`) and explanation. Calls that were interrupted (e.g., o3-mini runs failing during web search) are missing; with Gemini 2.5 Flash on Electronics, most items have a single call. `ground_truth/evaluate.py` only uses the calls covering at least half of the items (see its docstring for the full protocol).

`ground_truth/logs/<llm>/<domain>/` contains the execution log of each agent call (input, web-search queries and results, answer). Logs of calls that were repeated (e.g., interrupted runs) are kept as well; `ground_truth/logs/index.tsv` lists every log with its item, whether it produced an answer, the number of web searches, the answer, and whether the answer is one of the released calls.

## Use-case data

`use_case/data/electronics/` contains the 15-core subset of the Amazon Reviews'23 *Electronics* reviews used in the paper (21,751 users, 11,495 items, 464,457 interactions) in RecBole atomic format:

- `rating`: binarized feedback, 1 for 5-star reviews and 0 for 1–4-star reviews (threshold in `config/preprocessing.yaml`);
- split: temporal per user, with the oldest 80% of each user's interactions (rounded down) for training, the next 10% for validation, and the remaining most recent ones for test (ratios in `config/preprocessing.yaml`); validation and test interactions with items never seen in training are removed. The resulting sets contain 363,022 / 35,117 / 65,862 interactions;
- `maps/`: mapping between RecBole indices and Amazon user IDs / item ASINs (with their parent ASIN, which links each item to its PCF annotation);
- `pcf_annotations/gemini_2_5_flash.jsonl`: PCF estimated by PCF-Scout (Gemini 2.5 Flash) for each of the 9,710 parent products of the catalog (`value`, `source`, `explanation`).

`make preprocess-data` regenerates the atomic files and the maps from `raw/user_reviews.jsonl`.

## Large files (DVC and Zenodo)

The following folders are versioned with [DVC](https://dvc.org) and stored on [DagsHub](https://dagshub.com/<user>/<repo>), not in Git:

| Folder | Content | Size |
|---|---|---|
| `ground_truth/logs/` | Execution logs of PCF-Scout on the ground truth (4,446 files) | 90 MB |
| `use_case/logs/` | Execution logs of PCF-Scout on the use-case catalog (9,672 files) | 289 MB |
| `use_case/data/electronics/raw/` | Amazon Reviews'23 user reviews and item metadata of the catalog | 296 MB |

Everything else (ground truth data, use-case splits and maps, PCF annotations of the catalog, results) is in Git. The large files are only needed to inspect the agent logs and to re-run the use-case preprocessing or PCF-Scout on the catalog.

**Option 1: DVC.** After cloning the repository and installing the project (see [Setup](#setup)):

```bash
dvc pull
```

**Option 2: Zenodo.** The same folders are archived on Zenodo ([DOI](https://doi.org/<zenodo-doi>)) as three zip files. Download them and unzip them in the repository root, which restores the folders above:

```bash
unzip ground_truth_logs.zip
unzip use_case_logs.zip
unzip use_case_raw_data.zip
```

## Setup

```bash
conda create -n pcf-scout python=3.14
conda activate pcf-scout
pip install -e .
pip install "dvc[s3]"   # only to download the large files with DVC
```

API keys are read from the environment (or from a `.env` file): `GOOGLE_API_KEY` (Gemini), `OPENAI_API_KEY` (o3-mini) and `EXA_API_KEY` (web search).

## Usage

All commands are run from the repository root.

| Step | Command |
|---|---|
| Run PCF-Scout (settings in `config/dataset.yaml` and `config/emissions_enrichment.yaml`) | `make run-pcf-scout` |
| Run the zero-shot baseline (settings in `config/zero_shot.yaml`) | `make run-zero-shot` |
| Reproduce the ground-truth analyses (standard library only) | `make evaluate-ground-truth` |
| Use case: preprocess the reviews | `make preprocess-data` |
| Use case: tune and train BPR and LightGCN | `make train-recsys` |
| Use case: recommend and re-rank with SaS | `make predict-recommendations` |
| Use case: evaluate (writes `use_case/results/electronics.tsv`) | `make evaluate-recsys` |

`make <target> PYTHON=python` runs a step with the current interpreter instead of the conda environment.

**Running PCF-Scout.** With `is_ground_truth: true`, the agent is run on `ground_truth/data/<dataset>.jsonl` for the selected LLM, `num_estimates_per_product` times per item (4 in the paper). With `is_ground_truth: false`, it annotates the use-case catalog (`use_case/data/electronics/raw/item_metadata.jsonl`, one call per product). New estimates and logs are written to `outputs/`, never over the released data; interrupted runs are resumed from there.

## Zero-shot baseline prompt

Used by `pcf_scout/zero_shot.py`, with `{product_category}` set to the domain name and the product title as product data.

```text
You are an expert in life cycle analysis (LCA) and CO2e emission calculation for {product_category} products.
You must estimate the CO2e emissions, based on the entire life cycle (cradle to grave), for the following {product_category} product.

Product data: {json.dumps(product_data, ensure_ascii=False)}

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
{
  "co2e_kg": <number>,
  "source": <if "manufacturer report" or "estimation">,
  "explanation": "<detailed explanation including data source>"
}
Do not include any markdown formatting or additional JSON wrappers.
```

## Compatibility Notes

This section tracks direct code modifications applied to external libraries to ensure compatibility between dependencies, frameworks, and runtime versions.

> **IMPORTANT**: These changes are not part of the original upstream projects and are applied locally.

### RecBole (v1.2.0)

#### 1. Compatibility with NumPy (v2.4.4)

Fix for NumPy deprecated alias removal.

##### 1.1 `recbole/config/configurator.py`

Replace:

```python
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.str = np.str_
np.long = np.int_
np.unicode = np.unicode_
```

With:

```python
np.bool = bool
np.int = int
np.float = float
np.complex = complex
np.object = object
np.str = str
np.long = int
np.unicode = str
```

##### 1.2 `recbole/evaluator/metrics.py`

At the beginning of the script (after imports), add:

```python
np.float = float
```

#### 2. Compatibility with PyTorch (v2.11.0)

Fix for PyTorch checkpoint loading behavior change.

##### 2.1 `recbole/trainer/trainer.py`

Replace:

```python
checkpoint = torch.load(checkpoint_file, map_location=self.device)
```

With:

```python
checkpoint = torch.load(
    checkpoint_file,
    map_location=self.device,
    weights_only=False
)
```

##### 2.2 `recbole/quick_start/quick_start.py`

Replace:

```python
checkpoint = torch.load(model_file)
```

With:

```python
checkpoint = torch.load(model_file, weights_only=False)
```

#### 3. Compatibility with SciPy (v1.17.1)

Fix for SciPy internal API removal.

##### 3.1 `recbole/model/general_recommender/lightgcn.py`

Replace:

```python
A._update(data_dict)
```

With:

```python
for (i, j), val in data_dict.items():
    A[i, j] = val
```

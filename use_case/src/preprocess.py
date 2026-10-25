"""use_case/data/preprocess.py

Preprocess data.
"""

import pandas as pd

from src.const import (
    DATA_INTERIM_MAPS_IMAP_PATH,
    DATA_INTERIM_MAPS_UMAP_PATH,
    DATA_PROCESSED_AR23_UR_ELEC_FULL_PATH,
    DATA_PROCESSED_AR23_UR_ELEC_TEST_PATH,
    DATA_PROCESSED_AR23_UR_ELEC_TRAIN_PATH,
    DATA_PROCESSED_AR23_UR_ELEC_VALID_PATH,
    DATA_RAW_AR23_UR_ELEC_PATH,
    DATASET_NAME_ELEC,
    DATASET_PROCESSED_UR_ITEM_ID_COL,
    DATASET_PROCESSED_UR_RATING_COL,
    DATASET_PROCESSED_UR_TIMESTAMP_COL,
    DATASET_PROCESSED_UR_USER_ID_COL,
    DATASET_RAW_UR_ASIN_COL,
    DATASET_RAW_UR_PARENT_ASIN_COL,
    DATASET_RAW_UR_RATING_COL,
    DATASET_RAW_UR_TIMESTAMP_COL,
    DATASET_RAW_UR_USER_ID_COL,
    MAP_IMAP_ITEM_ID_COL,
    MAP_IMAP_ITEM_INDEX_COL,
    MAP_IMAP_PARENT_ASIN_COL,
    MAP_UMAP_USER_ID_COL,
    MAP_UMAP_USER_INDEX_COL,
    SUPPORTED_DATASETS,
)
from src.utils import load_config

# Load configuration
config = load_config()

# Map datasets to their configurations
dataset_registry = {
    DATASET_NAME_ELEC: {
        "raw_path": DATA_RAW_AR23_UR_ELEC_PATH,
        "processed_full_path": DATA_PROCESSED_AR23_UR_ELEC_FULL_PATH,
        "processed_train_path": DATA_PROCESSED_AR23_UR_ELEC_TRAIN_PATH,
        "processed_valid_path": DATA_PROCESSED_AR23_UR_ELEC_VALID_PATH,
        "processed_test_path": DATA_PROCESSED_AR23_UR_ELEC_TEST_PATH,
    },
}

# Determine paths based on dataset name
assert config["dataset"]["name"] in SUPPORTED_DATASETS, (
    f"Supported dataset: {SUPPORTED_DATASETS}, got: {config['dataset']['name']}"
)
assert config["dataset"]["name"] in dataset_registry, (
    f"Dataset configuration missing for: {config['dataset']['name']}"
)

dataset_name = config["dataset"]["name"]
paths = dataset_registry[dataset_name]
raw_user_reviews_path = paths["raw_path"]
processed_user_reviews_path = paths["processed_full_path"]
processed_user_reviews_train_path = paths["processed_train_path"]
processed_user_reviews_valid_path = paths["processed_valid_path"]
processed_user_reviews_test_path = paths["processed_test_path"]


def _deduplicate_user_reviews(
    user_reviews: pd.DataFrame,
    keep: str,
) -> pd.DataFrame:
    """Remove duplicate user-item interactions, keeping the specified occurrence.

    For duplicate (user_id, item_id) pairs, this function removes all but one
    occurrence.

    Args:
        user_reviews (pd.DataFrame):
            User reviews dataset with columns:
            - "user_id": Unique identifier for users.
            - "asin": Item identifier.

        keep (str):
            Which duplicate to keep.

    Returns:
        pd.DataFrame:
            Deduplicated user reviews dataset.
    """
    # Remove duplicate user-item interactions
    user_reviews = user_reviews.drop_duplicates(
        subset=[DATASET_RAW_UR_USER_ID_COL, DATASET_RAW_UR_ASIN_COL],
        keep=keep,
    ).reset_index(drop=True)

    # Sanity check
    assert not user_reviews.duplicated(
        subset=[DATASET_RAW_UR_USER_ID_COL, DATASET_RAW_UR_ASIN_COL],
    ).any()

    return user_reviews


def _build_user_item_maps(
    user_reviews: pd.DataFrame,
) -> tuple[dict, list[tuple]]:
    """Build user and item maps from user reviews and save them.

    This function creates and saves two maps:
    - A map from user IDs to integer indices.
    - A map from item IDs (ASIN) to integer indices, enriched with parent
      ASIN information.
    Both maps are built from the provided user reviews dataset.

    Args:
        user_reviews (pd.DataFrame):
            User reviews dataset with at least the following columns:
            - "user_id": Unique identifier for users.
            - "asin": Item identifier.
            - "parent_asin": Parent item identifier.

    Returns:
        tuple[dict, list[tuple]]:
            A tuple containing:
            - map_users (dict): (User ID, User Index) map.
            - map_items (list[tuple]): List of (Item ID, Item Index, Parent ASIN) tuples.
    """
    # Create (User ID, User Index) map
    map_users = {
        user_id: i
        for i, user_id in enumerate(
            sorted(user_reviews[DATASET_RAW_UR_USER_ID_COL].unique()),
        )
    }

    # Create (Item ID, Item Index) map
    map_items = {
        item_id: i
        for i, item_id in enumerate(
            sorted(user_reviews[DATASET_RAW_UR_ASIN_COL].unique()),
        )
    }

    # Create (ASIN, Parent ASIN) map
    asin_to_parent = (
        user_reviews.drop_duplicates(DATASET_RAW_UR_ASIN_COL)
        .set_index(DATASET_RAW_UR_ASIN_COL)[DATASET_RAW_UR_PARENT_ASIN_COL]
        .to_dict()
    )

    # Create (Item ID, Item Index, Parent ASIN) map,
    # overriding (Item ID, Item Index) map
    map_items = [
        (item_id, item_index, asin_to_parent.get(item_id, item_id))
        for item_id, item_index in map_items.items()
    ]

    # Save maps
    pd.DataFrame(
        list(map_users.items()),
        columns=[MAP_UMAP_USER_ID_COL, MAP_UMAP_USER_INDEX_COL],
    ).to_csv(DATA_INTERIM_MAPS_UMAP_PATH, sep="\t", index=False)

    pd.DataFrame(
        map_items,
        columns=[
            MAP_IMAP_ITEM_ID_COL,
            MAP_IMAP_ITEM_INDEX_COL,
            MAP_IMAP_PARENT_ASIN_COL,
        ],
    ).to_csv(DATA_INTERIM_MAPS_IMAP_PATH, sep="\t", index=False)

    # Sanity checks
    assert len(map_users) == user_reviews[DATASET_RAW_UR_USER_ID_COL].nunique()
    assert len(map_items) == user_reviews[DATASET_RAW_UR_ASIN_COL].nunique()
    assert set(map_users.values()) == set(range(len(map_users)))
    assert set(idx for _, idx, _ in map_items) == set(range(len(map_items)))

    return map_users, map_items


def _binarize_user_reviews(
    user_reviews: pd.DataFrame,
    map_users: dict,
    map_items: list[tuple],
    threshold: int,
) -> pd.DataFrame:
    """Binarize ratings and save user-item interactions in RecBole format.

    This function transforms the raw user reviews into a binarized interaction
    matrix suitable for recommendation systems, applying three key transformations:
    1. Maps original user IDs and item ASINs to sequential indices.
    2. Binarizes ratings so that those greater than or equal to the positive threshold
       become 1 (positive) all others become 0 (negative), enabling implicit feedback
       scenarios.
    3. Saves in RecBole format.

    Args:
        user_reviews (pd.DataFrame):
            Raw user reviews dataset containing at least the following columns:
            - "user_id": Unique identifier for users.
            - "asin": Item identifier.
            - "rating": Numerical rating values to binarize.
            - "timestamp": Interaction timestamp.

        map_users (dict):
            (User ID, User Index) map.

        map_items (list[tuple]):
            List of (Item ID, Item Index, Parent ASIN) tuples.

        threshold (int):
            Rating threshold for binarization. Ratings >= threshold become 1.

    Returns:
        pd.DataFrame:
            The binarized user reviews dataset.
    """
    # Binarize ratings and save user-item interactions in RecBole format
    binarized_user_reviews = pd.DataFrame(
        {
            DATASET_PROCESSED_UR_USER_ID_COL: user_reviews[
                DATASET_RAW_UR_USER_ID_COL
            ].map(
                map_users,
            ),
            DATASET_PROCESSED_UR_ITEM_ID_COL: user_reviews[
                DATASET_RAW_UR_ASIN_COL
            ].map(
                {item_id: item_index for item_id, item_index, _ in map_items},
            ),
            DATASET_PROCESSED_UR_RATING_COL: (
                user_reviews[DATASET_RAW_UR_RATING_COL] >= threshold
            ).astype(int),
            DATASET_PROCESSED_UR_TIMESTAMP_COL: (
                pd.to_datetime(
                    user_reviews[DATASET_RAW_UR_TIMESTAMP_COL],
                ).astype("int64")
            ),
        },
    ).reset_index(drop=True)

    binarized_user_reviews.to_csv(
        processed_user_reviews_path,
        sep="\t",
        index=False,
    )

    # Sanity checks
    assert not binarized_user_reviews.isna().any().any()
    assert set(
        binarized_user_reviews[DATASET_PROCESSED_UR_RATING_COL].unique(),
    ).issubset({0, 1})
    assert binarized_user_reviews[DATASET_PROCESSED_UR_USER_ID_COL].min() >= 0
    assert binarized_user_reviews[DATASET_PROCESSED_UR_ITEM_ID_COL].min() >= 0

    return binarized_user_reviews


def _split_user_reviews(
    user_reviews: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
) -> None:
    """Split user reviews by temporal order.

    Performs a user-aware temporal split, ensuring that for each individual
    user, older interactions are allocated to train, more recent to validation,
    and the most recent to test. This approach prevents temporal leakage and
    creates a more realistic evaluation scenario. Additionally, the function
    ensures that validation and test sets contain only items present in the
    training set, preventing cold-start problems during model evaluation.

    For each user, interactions are ordered by timestamp and split as:
    - Train: oldest interaction
    - Validation: more recent interactions
    - Test: most recent interactions

    Args:
        user_reviews (pd.DataFrame):
            Binarized user-item interactions with columns:
            - "user_id:token": Mapped user index
            - "item_id:token": Mapped item index
            - "rating:float": Binary rating 0 or 1
            - "timestamp:float": Interaction timestamp

        train_ratio (float):
            Proportion of data for training set.

        valid_ratio (float):
            Proportion of data for validation set.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            A tuple containing train, validation, and test dataframes.
    """
    # Initialize lists to accumulate splits across all users
    train_splits = []
    valid_splits = []
    test_splits = []

    # Group interactions by user and process each user's history temporally
    # to maintain chronological order within user interactions
    for _, user_interactions in user_reviews.groupby(
        DATASET_PROCESSED_UR_USER_ID_COL,
        sort=True,
    ):
        # Calculate split boundaries based on number of user interactions
        # ensuring each split respects the configured ratios
        user_interactions = user_interactions.sort_values(
            by=DATASET_PROCESSED_UR_TIMESTAMP_COL,
        )
        num_interactions = len(user_interactions)
        train_size = int(
            train_ratio * num_interactions,
        )
        valid_size = int(
            valid_ratio * num_interactions,
        )

        # Split user's temporal sequence: old -> train, recent -> valid,
        # newest -> test maintaining strict temporal ordering within each
        # user's interaction history
        train_splits.append(user_interactions.iloc[:train_size])
        valid_splits.append(
            user_interactions.iloc[train_size : train_size + valid_size],
        )
        test_splits.append(user_interactions.iloc[train_size + valid_size :])

    # Concatenate all user splits into final split matrices
    train_df = pd.concat(train_splits, ignore_index=True)
    valid_df = pd.concat(valid_splits, ignore_index=True)
    test_df = pd.concat(test_splits, ignore_index=True)

    # Filter validation and test to contain only items present in training set,
    # preventing cold-start problems where model evaluates on unseen items
    train_items = set(train_df[DATASET_PROCESSED_UR_ITEM_ID_COL].unique())
    valid_df = valid_df[
        valid_df[DATASET_PROCESSED_UR_ITEM_ID_COL].isin(train_items)
    ]
    test_df = test_df[
        test_df[DATASET_PROCESSED_UR_ITEM_ID_COL].isin(train_items)
    ]

    # Save all splits in RecBole format
    train_df.to_csv(processed_user_reviews_train_path, sep="\t", index=False)
    valid_df.to_csv(processed_user_reviews_valid_path, sep="\t", index=False)
    test_df.to_csv(processed_user_reviews_test_path, sep="\t", index=False)

    # Sanity checks
    assert not train_df.isna().any().any()
    assert not valid_df.isna().any().any()
    assert not test_df.isna().any().any()
    train_items = set(train_df[DATASET_PROCESSED_UR_ITEM_ID_COL])
    assert set(valid_df[DATASET_PROCESSED_UR_ITEM_ID_COL]).issubset(
        train_items,
    )
    assert set(test_df[DATASET_PROCESSED_UR_ITEM_ID_COL]).issubset(train_items)
    assert len(train_df) > 0
    assert train_df[DATASET_PROCESSED_UR_USER_ID_COL].nunique() > 0

    return train_df, valid_df, test_df


def preprocess_data() -> None:
    """Execute the complete preprocessing pipeline for data.

    This function orchestrates the entire data preprocessing workflow, which
    consists of the following stages:
    1. Data Loading: Load raw user reviews sorted by user and time.
    2. Deduplication: Remove duplicate user-item interactions (keep most recent).
    3. Mapping: Build user and item ID to index mappings
    4. Binarization: Convert ratings to implicit feedback (0 or 1)
    5. Splitting: Partition interactions into train/validation/test sets
    All outputs are persisted to disk.

    Returns:
        None
    """
    # Load raw user reviews and sort chronologically by user and timestamp
    # to maintain temporal order of interactions for each user
    user_reviews = pd.read_json(raw_user_reviews_path, lines=True).sort_values(
        by=[DATASET_RAW_UR_USER_ID_COL, DATASET_RAW_UR_TIMESTAMP_COL],
    )
    n_raw = len(user_reviews)
    n_users_raw = user_reviews[DATASET_RAW_UR_USER_ID_COL].nunique()
    n_items_raw = user_reviews[DATASET_RAW_UR_ASIN_COL].nunique()

    # Remove duplicate user-item interactions, keeping the most recent rating
    n_before = len(user_reviews)
    user_reviews = _deduplicate_user_reviews(
        user_reviews,
        keep=config["preprocessing"]["deduplication_keep"],
    )
    n_after_dedup = len(user_reviews)
    n_users_after_dedup = user_reviews[DATASET_RAW_UR_USER_ID_COL].nunique()
    n_items_after_dedup = user_reviews[DATASET_RAW_UR_ASIN_COL].nunique()

    # Create and persist user/item ID to index mappings
    map_users, map_items = _build_user_item_maps(user_reviews)

    # Binarize ratings using positive threshold, creating implicit feedback
    # matrix suitable for recommendation models
    binarized_user_reviews = _binarize_user_reviews(
        user_reviews,
        map_users,
        map_items,
        threshold=config["preprocessing"]["bin_rating_threshold"],
    )
    n_total_interactions = len(binarized_user_reviews)
    n_positives = binarized_user_reviews[DATASET_PROCESSED_UR_RATING_COL].sum()
    n_negatives = n_total_interactions - n_positives
    n_users_binarized = binarized_user_reviews[
        DATASET_PROCESSED_UR_USER_ID_COL
    ].nunique()
    n_items_binarized = binarized_user_reviews[
        DATASET_PROCESSED_UR_ITEM_ID_COL
    ].nunique()

    # Split binarized interactions into train/valid/test sets and persist
    # each split separately for downstream model training and evaluation
    train_df, valid_df, test_df = _split_user_reviews(
        binarized_user_reviews,
        train_ratio=config["preprocessing"]["train_ratio"],
        valid_ratio=config["preprocessing"]["valid_ratio"],
    )

    # Print summary
    print("\nData Preprocessing Summary")

    # Stage-by-stage statistics
    summary_stats = pd.DataFrame(
        {
            "Stage": [
                "Raw",
                "After Deduplication",
                "After Binarization",
                "Train Split",
                "Validation Split",
                "Test Split",
            ],
            "Interactions": [
                n_raw,
                n_after_dedup,
                n_total_interactions,
                len(train_df),
                len(valid_df),
                len(test_df),
            ],
            "Unique Users": [
                n_users_raw,
                n_users_after_dedup,
                n_users_binarized,
                train_df[DATASET_PROCESSED_UR_USER_ID_COL].nunique(),
                valid_df[DATASET_PROCESSED_UR_USER_ID_COL].nunique(),
                test_df[DATASET_PROCESSED_UR_USER_ID_COL].nunique(),
            ],
            "Unique Items": [
                n_items_raw,
                n_items_after_dedup,
                n_items_binarized,
                train_df[DATASET_PROCESSED_UR_ITEM_ID_COL].nunique(),
                valid_df[DATASET_PROCESSED_UR_ITEM_ID_COL].nunique(),
                test_df[DATASET_PROCESSED_UR_ITEM_ID_COL].nunique(),
            ],
        },
    )

    for col in ["Interactions", "Unique Users", "Unique Items"]:
        summary_stats[col] = summary_stats[col].map(lambda x: f"{x:,}")

    print("\nProcessing Stages")
    print(summary_stats.to_markdown(index=False))

    # Interaction quality summary
    print("\nInteraction Quality")
    quality_stats = pd.DataFrame(
        {
            "Metric": [
                "Total Interactions",
                "Positive (rating >= threshold)",
                "Negative (rating < threshold)",
                "Positive Ratio",
            ],
            "Value": [
                f"{n_total_interactions:,}",
                f"{n_positives:,}",
                f"{n_negatives:,}",
                f"{(n_positives / n_total_interactions * 100):.2f}%",
            ],
        },
    )
    print(quality_stats.to_markdown(index=False))

    # Sparsity metrics
    print("\nSparsity")
    sparsity_stats = pd.DataFrame(
        {
            "Set": ["Train", "Validation", "Test"],
            "Interactions": [
                f"{len(train_df):,}",
                f"{len(valid_df):,}",
                f"{len(test_df):,}",
            ],
            "Density": [
                f"{len(train_df) / (train_df[DATASET_PROCESSED_UR_USER_ID_COL].nunique() * train_df[DATASET_PROCESSED_UR_ITEM_ID_COL].nunique()):.4f}",
                f"{len(valid_df) / (valid_df[DATASET_PROCESSED_UR_USER_ID_COL].nunique() * valid_df[DATASET_PROCESSED_UR_ITEM_ID_COL].nunique()):.4f}",
                f"{len(test_df) / (test_df[DATASET_PROCESSED_UR_USER_ID_COL].nunique() * test_df[DATASET_PROCESSED_UR_ITEM_ID_COL].nunique()):.4f}",
            ],
        },
    )
    print(sparsity_stats.to_markdown(index=False))


if __name__ == "__main__":
    preprocess_data()

"""use_case/src/utils.py

Utility functions.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import yaml

from use_case.src.const import CONFIG_DIR, CONFIG_FILE_FORMAT


def load_config() -> dict:
    """Load and merge all YAML configuration files.

    This function reads all YAML files and merges them into a single
    nested dictionary.

    Returns:
        dict:
            Merged configuration dictionary containing all YAML settings.
    """
    config = {}

    # Iterate over all YAML config files in deterministic order
    for f in sorted(Path(CONFIG_DIR).glob(CONFIG_FILE_FORMAT)):
        with open(f, encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

            # Stack-based recursive merge to avoid deep recursion
            stack = [(config, data)]
            while stack:
                base, update = stack.pop()
                for k, v in update.items():
                    # If both sides are dicts, merge recursively
                    if (
                        k in base
                        and isinstance(base[k], dict)
                        and isinstance(v, dict)
                    ):
                        stack.append((base[k], v))
                    else:
                        base[k] = v

    return config


def load_jsonl(path: str) -> Iterator[dict]:
    """Load a JSONL file.

    Each non-empty line in the file is parsed independently as a JSON object.
    Empty lines are ignored. This function yields items lazily to avoid loading
    the entire file into memory, making it suitable for large datasets.

    Args:
        path (str):
            Path to the JSONL file.

    Yields:
        Iterator[dict]:
            Parsed JSON object from each valid line in the file.
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines to avoid parsing errors
            if line:
                yield json.loads(line)

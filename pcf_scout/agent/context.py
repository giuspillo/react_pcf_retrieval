"""pcf_scout/agent/context.py

This module defines configurable parameters for the agent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from .const import (
    DEFAULT_ENABLE_LOGGING,
    DEFAULT_LOGS_DIR,
    DEFAULT_LOGS_PATH,
    DEFAULT_MODEL,
    DEFAULT_MODEL_TEMPERATURE,
    DEFAULT_SEARCH_WEB_MAX_CALLS,
    DEFAULT_SEARCH_WEB_MAX_RESULTS,
    DEFAULT_SEARCH_WEB_TYPE,
)


@dataclass(kw_only=True)
class Context:
    """The context for the agent.

    This class defines the configurable parameters for the agent.

    Attributes:
        model (Annotated[str, {"__template_metadata__": {"kind": "llm"}}]):
            The name of the language model to use for the agent's main interactions.
            ('provider-name/model-name').

        model_temperature (float):
            The temperature to use for the language model (0.0-1.0).

        search_web_max_results (int):
            The maximum number of web search results to return for each query (1-100).

        search_web_type (str):
            The type of web search to perform ("fast", "auto", "deep").

        search_web_max_calls (int):
            The maximum number of web search calls the agent can make per run (> 0).

        enable_logging (bool):
            Whether to enable logging of agent execution to files (True/False).

        logs_dir (str):
            The directory path for execution logs if logging is enabled.

        logs_path (str):
            The file path for the execution log if logging is enabled.
    """

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default=DEFAULT_MODEL,
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "('provider-name/model-name')",
        },
    )

    model_temperature: float = field(
        default=DEFAULT_MODEL_TEMPERATURE,
        metadata={
            "description": "The temperature to use for the language model (0.0-1.0).",
        },
    )

    search_web_max_results: int = field(
        default=DEFAULT_SEARCH_WEB_MAX_RESULTS,
        metadata={
            "description": "The maximum number of web search results to return for each query (1-100).",
        },
    )

    search_web_type: str = field(
        default=DEFAULT_SEARCH_WEB_TYPE,
        metadata={
            "description": 'The type of web search to perform ("fast", "auto", "deep").',
        },
    )

    search_web_max_calls: int = field(
        default=DEFAULT_SEARCH_WEB_MAX_CALLS,
        metadata={
            "description": "The maximum number of web search calls the agent can make per run (> 0).",
        },
    )

    enable_logging: bool = field(
        default=DEFAULT_ENABLE_LOGGING,
        metadata={
            "description": "Whether to enable logging of agent execution to files (True/False).",
        },
    )

    logs_dir: str = field(
        default=DEFAULT_LOGS_DIR,
        metadata={
            "description": "The directory path for execution logs if logging is enabled.",
        },
    )

    logs_path: str = field(
        default=DEFAULT_LOGS_PATH,
        metadata={
            "description": "The file path for the execution log if logging is enabled.",
        },
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args.

        This function iterates over the fields of the dataclass and checks if
        any of them were not initialized. If a field was not initialized, it
        attempts to fetch its value from an environment variable with the
        same name (converted to uppercase). If the environment variable is not
        set, it retains the default value defined in the dataclass.

        Returns:
            None
        """
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                env_val = os.environ.get(f.name.upper())
                if env_val is not None:
                    # Field annotations are strings (postponed evaluation),
                    # so cast using the type of the default value
                    if isinstance(f.default, bool):
                        value = env_val.strip().lower() in {"1", "true", "yes"}
                    elif f.default is None:
                        value = float(env_val)
                    else:
                        value = type(f.default)(env_val)
                    setattr(self, f.name, value)

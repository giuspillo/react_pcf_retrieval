"""ecoamazon/src/state.py

This module defines the state structures for the agent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from .const import DEFAULT_INITIAL_MESSAGE, DEFAULT_SEARCH_WEB_COUNT


@dataclass
class InputState:
    """Input state for the agent.

    This class is used to define the initial state and structure of incoming data,
    representing a narrower interface to the outside world.

    Attributes:
        product_data (dict[str, Any]):
            The data of the product for which the carbon footprint is being computed.
    """

    product_data: dict[str, Any] = field(
        default=dict,
        metadata={
            "description": "The data of the product for which the carbon footprint is being computed.",
        },
    )


@dataclass
class State(InputState):
    """State of the agent.

    This class represents the state of the agent, storing any information needed
    throughout the agent's lifecycle.

    Attributes:
        messages (Annotated[Sequence[AnyMessage], add_messages]):
            Messages tracking the primary execution state of the agent.

        initial_message (AnyMessage):
            The first user message to preserve context across iterations.

        search_web_count (int):
            Tracks the number of times search web has been called per agent run.

        co2e_kg (dict[str, Any]):
            The final carbon footprint result.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list,
        metadata={
            "description": "Messages tracking the primary execution state of the agent.",
        },
    )

    initial_message: AnyMessage = field(
        default=DEFAULT_INITIAL_MESSAGE,
        metadata={
            "description": "The first user message to preserve context across iterations.",
        },
    )

    search_web_count: int = field(
        default=DEFAULT_SEARCH_WEB_COUNT,
        metadata={
            "description": "Tracks the number of times search web has been called per agent run.",
        },
    )

    co2e_kg: dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "The final carbon footprint result."},
    )

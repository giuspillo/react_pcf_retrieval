"""pcf_scout/agent/tools.py

This module provides tools which can be used by the agent.
"""

from collections.abc import Callable
from typing import Any

from langchain_exa import ExaSearchRetriever
from langgraph.runtime import get_runtime

from .const import SEARCH_WEB_HIGHLIGHTS
from .context import Context


async def search_web(query: str) -> Any:
    """Search for web results.

    This function performs a web search using the Exa search engine.

    Args:
        query (str):
            The search query.

    Returns:
        Any:
            The search results.
    """
    runtime = get_runtime(Context)

    # Wrap the ExaSearchRetriever with the runtime context and invoke it
    wrapped = ExaSearchRetriever(
        k=runtime.context.search_web_max_results,
        type=runtime.context.search_web_type,
        highlights=SEARCH_WEB_HIGHLIGHTS,
    )
    response = await wrapped.ainvoke(query)

    # Compress the response to only include title, url, and
    # highlights for each result
    compressed_response = []
    for result in response:
        compressed_result = {
            "title": result.metadata["title"],
            "url": result.metadata["url"],
            "highlights": result.metadata.get("highlights") or [],
        }
        compressed_response.append(compressed_result)

    return compressed_response


TOOLS: list[Callable[..., Any]] = [search_web]

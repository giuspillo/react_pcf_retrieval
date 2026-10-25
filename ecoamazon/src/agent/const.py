"""ecoamazon/src/const.py

This module defines constants for the agent.
"""

# Model
DEFAULT_MODEL = "google_genai:gemini-2.5-flash"
DEFAULT_MODEL_TEMPERATURE = None

# Search web
DEFAULT_SEARCH_WEB_MAX_RESULTS = 10
DEFAULT_SEARCH_WEB_TYPE = "auto"
DEFAULT_SEARCH_WEB_MAX_CALLS = 3
DEFAULT_SEARCH_WEB_COUNT = 0
SEARCH_WEB_HIGHLIGHTS = True

# Messages
DEFAULT_INITIAL_MESSAGE = None

# Logging
DEFAULT_ENABLE_LOGGING = True
DEFAULT_LOGS_PATH = "agent.log"
DEFAULT_LOGS_DIR = "logs"
LOGS_WIDTH = 100
LOGS_CONTENT_WIDTH = LOGS_WIDTH - 4

# Graph
GRAPH_NAME = "agent"

# Graph nodes
CALL_MODEL_GRAPH_NODE = "call_model"
TOOLS_GRAPH_NODE = "tools"
ANSWER_GRAPH_NODE = "answer"
START_GRAPH_NODE = "__start__"
END_GRAPH_NODE = "__end__"

# Tools
SEARCH_WEB_TOOL = "search_web"

"""pcf_scout/agent/graph.py

This module defines a ReAct agent with LangGraph — LangChain.
"""

import json
import textwrap
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from .const import (
    ANSWER_GRAPH_NODE,
    CALL_MODEL_GRAPH_NODE,
    END_GRAPH_NODE,
    GRAPH_NAME,
    LOGS_CONTENT_WIDTH,
    LOGS_WIDTH,
    SEARCH_WEB_TOOL,
    START_GRAPH_NODE,
    TOOLS_GRAPH_NODE,
)
from .context import Context
from .prompts import (
    AFTER_TOOL_MESSAGE_PROMPT,
    OUTPUT_FORMAT_INSTRUCTIONS,
    SYSTEM_PROMPT,
)
from .state import InputState, State
from .tools import TOOLS


def _log_entry(runtime: Runtime[Context], text: str) -> None:
    """Write log entry to file if logging is enabled.

    This function handles the actual writing of log entries to the
    log file.

    Args:
        runtime (Runtime[Context]):
            The runtime environment.

        text (str):
            The text to write to the log file.

    Returns:
        None
    """
    if not runtime.context.enable_logging:
        return

    log_dir = Path.cwd() / runtime.context.logs_dir
    log_dir.mkdir(exist_ok=True)

    with open(log_dir / runtime.context.logs_path, "a") as f:
        f.write(text)


def _log_wrapped(
    runtime: Runtime[Context],
    prefix_first: str,
    prefix_next: str,
    content: str,
) -> None:
    """Write wrapped content preserving indentation across multiple lines.

    This function automatically wraps long content to fit within log width,
    maintaining proper indentation for subsequent lines.

    Args:
        runtime (Runtime[Context]):
            The runtime environment.

        prefix_first (str):
            Indentation prefix for the first line.

        prefix_next (str):
            Indentation prefix for continuation lines.

        content (str):
            The content to wrap and log.

    Returns:
        None
    """
    wrapped = textwrap.wrap(
        content,
        width=LOGS_CONTENT_WIDTH - len(prefix_first),
        replace_whitespace=False,
        drop_whitespace=False,
    )

    if not wrapped:
        return

    _log_entry(runtime, f"{prefix_first}{wrapped[0]}\n")

    for line in wrapped[1:]:
        _log_entry(runtime, f"{prefix_next}{line}\n")


def _log_initialization(state: InputState, runtime: Runtime[Context]) -> None:
    """Log agent initialization with input configuration.

    This function logs the initial input configuration and parameters of the agent
    at the start of execution, providing a clear record of the context for the run.

    Args:
        state (InputState):
            The initial input state.

        runtime (Runtime[Context]):
            The runtime environment.

    Returns:
        None
    """
    log_dir = Path.cwd() / runtime.context.logs_dir
    log_dir.mkdir(exist_ok=True)

    log_file_path = log_dir / runtime.context.logs_path

    # Overwrite the log file at the start of each run with the
    # initialization details
    with open(log_file_path, "w") as f:
        separator = "-" * LOGS_WIDTH

        f.write("Input\n")
        f.write(separator + "\n")

        f.write(f"Product data: {state.product_data}\n")
        f.write(f"Model: {runtime.context.model}\n")
        f.write(f"Model Temperature: {runtime.context.model_temperature}\n")
        f.write(
            f"Search Web Max Calls: {runtime.context.search_web_max_calls}\n",
        )
        f.write(
            f"Search Web Max Results: {runtime.context.search_web_max_results}\n",
        )
        f.write(f"Search Web Type: {runtime.context.search_web_type}\n")
        f.write("\n")
        f.write("Execution\n")
        f.write(separator + "\n")
        f.write("Start\n")


def _log_model_call(
    runtime: Runtime[Context],
    call_num: int,
    tool_name: str | None,
    query: str | None,
) -> None:
    """Log model call with tool invocation.

    This function logs each model call along with the tool it decided to call
    (if any) and the query. It formats the log entries to fit within the defined
    log width and maintains a clear structure for readability.

    Args:
        runtime (Runtime[Context]):
            The runtime environment.

        call_num (int):
            The sequential number of the model call in the current execution.

        tool_name (str | None):
            The name of the tool the model decided to call, or None if no tool call.

        query (str | None):
            The query argument for tool calls, if applicable.

    Returns:
        None
    """
    _log_entry(runtime, "│\n")
    _log_entry(runtime, f"├─ Model Call #{call_num}\n")

    # Log the tool call and query if applicable, otherwise log that
    # no tool was called
    if tool_name:
        _log_entry(runtime, f"│  ├─ Tool: {tool_name}\n")
        if query:
            _log_wrapped(
                runtime,
                prefix_first='│  ├─ Query: "',
                prefix_next="│             ",
                content=f'{query}"',
            )
    else:
        _log_entry(runtime, "│  └─ Tool: None\n")


def _log_tool_result(
    runtime: Runtime[Context],
    result_content: str,
) -> None:
    """Log tool execution result with automatic wrapping.

    This function logs the result of a tool execution, ensuring that long results are wrapped
    properly in the logs for readability.

    Args:
        runtime (Runtime[Context]):
            The runtime environment.

        result_content (str):
            The result returned by the tool.

    Returns:
        None
    """
    _log_wrapped(
        runtime,
        prefix_first="│  └─ Result: ",
        prefix_next="│             ",
        content=result_content,
    )


def _log_answer(
    runtime: Runtime[Context],
    value: float,
    source: str,
    explanation: str,
) -> None:
    """Log answer with value, source, and explanation.

    This function logs the answer generated by the model, including the
    computed value, the source of the information, and an explanation.
    It uses the wrapped logging function to ensure that all content is formatted
    properly in the logs.

    Args:
        runtime (Runtime[Context]):
            The runtime environment.

        value (float):
            The computed carbon footprint value.

        source (str):
            Source of the value.

        explanation (str):
            Explanation of how the value was obtained.
    """
    _log_entry(runtime, "│\n")
    _log_entry(runtime, "├─ Answer\n")

    _log_wrapped(
        runtime,
        prefix_first="│  ├─ PCF (kg CO2e): ",
        prefix_next="│                     ",
        content=str(value),
    )
    _log_wrapped(
        runtime,
        prefix_first="│  ├─ Source: ",
        prefix_next="│             ",
        content=str(source),
    )
    _log_wrapped(
        runtime,
        prefix_first="│  └─ Explanation: ",
        prefix_next="│                  ",
        content=str(explanation),
    )

    _log_entry(runtime, "│\n")
    _log_entry(runtime, "End\n")


async def call_model(
    state: State,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Call the LLM to generate model responses and tool calls.

    This function initializes the model with available tools, manages
    message history, and logs model invocations with their tool decisions.

    Args:
        state (State):
            The current agent execution state.

        runtime (Runtime[Context]):
            The runtime environment.

    Returns:
        dict[str, Any]:
            State updates.
    """
    # Initialize log file on first call
    if runtime.context.enable_logging and state.search_web_count == 0:
        _log_initialization(state, runtime)

    # Calculate remaining search web calls
    search_web_remaining_calls = (
        runtime.context.search_web_max_calls - state.search_web_count
    )

    # Format the system prompt
    system_prompt = SYSTEM_PROMPT.format(
        product_data=state.product_data,
        search_web_remaining_calls=search_web_remaining_calls,
        search_web_max_calls=runtime.context.search_web_max_calls,
    )

    # Initialize model with tools
    model = init_chat_model(
        model=runtime.context.model,
        temperature=runtime.context.model_temperature,
    ).bind_tools(TOOLS)

    # Prepare the messages list
    if not state.messages:
        # First execution: use system instructions directly
        # as the initial message
        initial_msg = HumanMessage(content=system_prompt)
        messages = [initial_msg]
    else:
        # Subsequent executions: always start with the initial message
        if state.initial_message:
            messages = [state.initial_message, *list(state.messages)]
        else:
            messages = list(state.messages)

        # If the last message is a ToolMessage, add a continuation
        # prompt for the model
        if messages and isinstance(messages[-1], ToolMessage):
            messages.append(
                HumanMessage(content=AFTER_TOOL_MESSAGE_PROMPT),
            )

    # Get the model's response
    response = cast(
        "AIMessage",
        await model.ainvoke(messages),
    )

    # Log the model call and tool decision (if any)
    call_num = state.search_web_count + 1
    if response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] == SEARCH_WEB_TOOL:
                query = tc["args"]["query"]
                _log_model_call(
                    runtime,
                    call_num=call_num,
                    tool_name=tc["name"],
                    query=query,
                )
    else:
        _log_model_call(
            runtime,
            call_num=call_num,
            tool_name=None,
            query=None,
        )

    # Increment the search web counter if the model
    # decided to call it
    new_search_web_count = state.search_web_count
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == SEARCH_WEB_TOOL:
                new_search_web_count += 1
                break

    # Update the state
    return_dict = {
        "messages": [response],
        "search_web_count": new_search_web_count,
    }

    # If this is the first call, save the initial message
    if not state.messages and len(messages) > 0:
        return_dict["initial_message"] = messages[0]

    return return_dict


async def answer(
    state: State,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Generate and return the agent's answer in JSON format.

    This function calls the model without tools to generate a structured
    JSON response containing the carbon footprint value, source, and explanation.

    Args:
        state (State):
            The final agent execution state.

        runtime (Runtime[Context]):
            The runtime environment.

    Returns:
        dict[str, Any]:
            State updates.
    """
    # Format the system prompt
    system_prompt = SYSTEM_PROMPT.format(
        product_data=state.product_data,
        search_web_remaining_calls=(
            runtime.context.search_web_max_calls
            - runtime.context.search_web_max_calls
        ),
        search_web_max_calls=runtime.context.search_web_max_calls,
    )

    # Initialize model without tools
    model = init_chat_model(
        model=runtime.context.model,
        temperature=runtime.context.model_temperature,
    )

    # Prepare messages with the conversation history and
    # output format instructions
    if state.initial_message:
        messages = [
            SystemMessage(content=system_prompt),
            state.initial_message,
            *state.messages,
            HumanMessage(content=OUTPUT_FORMAT_INSTRUCTIONS),
        ]
    else:
        messages = [
            SystemMessage(content=system_prompt),
            *state.messages,
            HumanMessage(content=OUTPUT_FORMAT_INSTRUCTIONS),
        ]

    # Get the model's response
    response = await model.ainvoke(messages)

    try:
        # Clean the response content and parse it as JSON
        co2e_kg = json.loads(
            str(response.content)
            .replace("```json", "")
            .replace("```", "")
            .strip(),
        )
    except Exception:
        # Fallback in case of JSON parsing errors: keep the raw answer as
        # explanation and mark the value as missing
        co2e_kg = {
            "value": None,
            "source": None,
            "explanation": str(response.content),
        }

    # Fill any field missing from the parsed answer
    for key in ("value", "source", "explanation"):
        co2e_kg.setdefault(key, None)

    # Log answer
    _log_answer(
        runtime,
        value=co2e_kg["value"],
        source=co2e_kg["source"],
        explanation=co2e_kg["explanation"],
    )

    return {"co2e_kg": co2e_kg}


def route_model_output(
    state: State,
    runtime: Runtime[Context],
) -> str:
    """Route to next graph node based on model output and search budget.

    This function implements conditional logic to determine whether to
    continue tool calling or move to final answer generation based on:
    - Whether search_web budget has been exhausted
    - Whether the model decided to make tool calls

    Args:
        state (State):
            The current agent execution state.

        runtime (Runtime[Context]):
            The runtime environment.

    Returns:
        str:
            Name of next node.
    """
    # Get the last message from the state
    last_message = state.messages[-1]

    # If max search web calls reached, provide the answer
    if state.search_web_count >= runtime.context.search_web_max_calls:
        return ANSWER_GRAPH_NODE

    # If there is no tool call, provide the answer
    if not last_message.tool_calls:
        return ANSWER_GRAPH_NODE

    # Otherwise, execute the requested actions
    return TOOLS_GRAPH_NODE


async def tool_wrapper(
    state: State,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Execute tools and return results back to agent state.

    This function wraps LangGraph's ToolNode to handle tool execution
    and log the results with proper formatting.

    Args:
        state (State):
            The current agent state.

        runtime (Runtime[Context]):
            The runtime environment.

    Returns:
        dict[str, Any]:
            State updates.
    """
    # Create and execute ToolNode
    tool_node = ToolNode(TOOLS)

    # Execute the tool node and get the result
    result = await tool_node.ainvoke(state)

    # Log tool execution
    if runtime.context.enable_logging and result["messages"]:
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                result_content = str(msg.content)
                _log_tool_result(runtime, result_content)

    return result


# ========================================================================= #
#            ReAct Agent Construction with LangGraph - LangChain            #
# ========================================================================= #
#                                                                           #
#                          +--------------+                                 #
#                          |  __start__   |                                 #
#                          +--------------+                                 #
#                                 |                                         #
#                                 v                                         #
#                          +--------------+                                 #
#                          |  call_model  | <----------+                    #
#                          +--------------+            |                    #
#                                 |                    |                    #
#                        (route_model_output)          |                    #
#                                 |                    |                    #
#                       +---------+---------+          |                    #
#                       |                   |          |                    #
#                       v                   v          |                    #
#                +--------------+    +--------------+  |                    #
#                |    answer    |    | tool_wrapper |--+                    #
#                +--------------+    +--------------+                       #
#                       |                                                   #
#                       v                                                   #
#                +--------------+                                           #
#                |   __end__    |                                           #
#                +--------------+                                           #
#                                                                           #
# ========================================================================= #
# Load environment variables
load_dotenv()

# Graph
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Nodes
builder.add_node(CALL_MODEL_GRAPH_NODE, call_model)
builder.add_node(TOOLS_GRAPH_NODE, tool_wrapper)
builder.add_node(ANSWER_GRAPH_NODE, answer)

# Entry point
builder.add_edge(START_GRAPH_NODE, CALL_MODEL_GRAPH_NODE)

# Conditional edges
builder.add_conditional_edges(
    CALL_MODEL_GRAPH_NODE,
    route_model_output,
)

# Edges
builder.add_edge(TOOLS_GRAPH_NODE, CALL_MODEL_GRAPH_NODE)
builder.add_edge(ANSWER_GRAPH_NODE, END_GRAPH_NODE)

# Compile the graph
graph = builder.compile(name=GRAPH_NAME)

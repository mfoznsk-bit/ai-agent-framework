import os
import json
import requests
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from tavily import TavilyClient

load_dotenv()

# =========================================================
# HOW THIS DIFFERS FROM YOUR MANUAL AGENT
# =========================================================
#
# OLD WAY (your groq_react_agent_fixed.py):
#   - You wrote the loop: while tool_calls → run tool → append → repeat
#   - You manually managed the messages list
#   - You manually handled errors, retries, max steps
#   - Everything was in one big function
#
# LANGGRAPH WAY:
#   - You define NODES (what each step does)
#   - You define EDGES (which node runs next)
#   - LangGraph runs the loop for you
#   - State is managed automatically
#   - The graph IS the agent — structure is explicit, not hidden in a loop
#
# ARCHITECTURE:
#
#   [START]
#      │
#      ▼
#   [agent] ← calls the LLM, decides what to do next
#      │
#      ├── "needs tools" ──► [tools] → runs the tool → back to [agent]
#      │
#      └── "done" ──────────► [END]

# =========================================================
# SETUP
# =========================================================

llm    = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# =========================================================
# TOOLS — defined with @tool decorator
# =========================================================
# The @tool decorator does 3 things automatically:
#   1. Turns the function into a LangChain tool object
#   2. Uses the docstring as the tool description (what the LLM reads)
#   3. Extracts the function signature as the parameter schema
#
# This replaces your manual tools = [...] JSON list entirely.
# The description the LLM sees is YOUR DOCSTRING — write it carefully.

@tool
def calculator(operation: str, num1: float, num2: float) -> str:
    """Perform ONE arithmetic operation on two numbers.
    Use for any math, price calculation, discount, total, or percentage.
    Operations: add, subtract, multiply, divide.
    IMPORTANT: Call once per operation. For multi-step math, call multiple times.
    Never calculate mentally — always use this tool for math."""
    if operation == "add":
        result = num1 + num2
    elif operation == "subtract":
        result = num1 - num2
    elif operation == "multiply":
        result = num1 * num2
    elif operation == "divide":
        if num2 == 0:
            return "Error: Cannot divide by zero"
        result = num1 / num2
    else:
        return f"Error: Unknown operation '{operation}'"
    return str(round(result, 4))


@tool
def get_datetime(format: str) -> str:
    """Get the current date and time in Algeria (UTC+1).
    Use for any question about current time, today's date, day of week.
    Format options: 'full', 'date_only', 'time_only', 'day_only'.
    Never guess the date — always use this tool."""
    algeria_time = datetime.now(timezone.utc) + timedelta(hours=1)
    formats = {
        "full":      "%A %d %B %Y, %H:%M:%S",
        "date_only": "%d %B %Y",
        "time_only": "%H:%M:%S",
        "day_only":  "%A"
    }
    return algeria_time.strftime(formats.get(format, formats["full"]))


@tool
def get_weather(city: str) -> str:
    """Get today's current weather for ONE city.
    Use for temperature and weather conditions in a specific city.
    Only accepts a city name — not a country name, not multiple cities.
    For multiple cities, call this tool once per city separately.
    Never pass 'Algeria' — that is a country, not a city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY not set"
    try:
        url    = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        resp   = requests.get(url, params=params, timeout=10)
        data   = resp.json()
        if data.get("cod") != 200:
            return f"City '{city}' not found: {data.get('message', 'unknown error')}"
        temp      = data["main"]["temp"]
        feels     = data["main"]["feels_like"]
        condition = data["weather"][0]["description"]
        humidity  = data["main"]["humidity"]
        wind      = data["wind"]["speed"]
        return f"{city}: {temp}°C (feels {feels}°C), {condition}, humidity {humidity}%, wind {wind} m/s"
    except requests.exceptions.Timeout:
        return f"Timeout fetching weather for '{city}'"
    except Exception as e:
        return f"Weather error: {e}"


@tool
def web_search(query: str) -> str:
    """Search the internet for current, real-time information.
    Use for: recent news, prices, exchange rates, product comparisons,
    local info about Algeria, anything requiring up-to-date data.
    Do NOT use for math (use calculator), time (use get_datetime),
    or weather (use get_weather)."""
    try:
        response = tavily.search(query=query, max_results=3)
        results  = response.get("results", [])
        if not results:
            return f"No results found for: '{query}'"
        formatted = f"Search: '{query}'\n" + "="*40 + "\n"
        for i, r in enumerate(results, 1):
            title   = r.get("title",   "No title")
            url     = r.get("url",     "")
            content = r.get("content", "")[:300]
            formatted += f"\n[{i}] {title}\n{url}\n{content}\n" + "-"*30 + "\n"
        return formatted
    except Exception as e:
        return f"Search failed: {e}"


# =========================================================
# GRAPH STATE
# =========================================================
# TypedDict defines the "memory" of the graph — what it carries
# between nodes. Every node reads from and writes to this state.
#
# add_messages is a special reducer:
# instead of replacing messages on each step, it APPENDS them.
# This is how the conversation history accumulates automatically.
# You never manually do messages.append(...) anymore.

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# =========================================================
# BIND TOOLS TO LLM
# =========================================================
# This tells the LLM what tools exist and how to call them.
# Equivalent to passing tools=[...] in every API call.
# Now you pass it once here and LangGraph handles it.

tools_list = [calculator, get_datetime, get_weather, web_search]
llm_with_tools = llm.bind_tools(tools_list)

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """You are a ReAct agent for a store in Oran, Algeria.

STRICT RULES:
1. Call ONLY ONE tool per step — never multiple tools at once.
2. After each tool result, stop and think before calling another.
3. For complex multi-step problems: ONE step at a time.
4. Use calculator for math, get_datetime for time/date,
   get_weather for weather, web_search for current info.
5. Never guess. One tool. One step. Wait for result. Repeat.
6. Reply in the same language the user uses."""


# =========================================================
# NODE 1: AGENT
# =========================================================
# This node calls the LLM. It receives the current state
# (all messages so far), sends them to the LLM, and returns
# the LLM's response to be appended to state.messages.
#
# The LLM's response will either:
#   A) Contain tool_calls → graph routes to [tools] node
#   B) Contain only text → graph routes to [END]

def agent_node(state: AgentState):
    """
    The brain of the agent. Calls the LLM with the full
    conversation history and gets back a decision:
    either call a tool or give the final answer.
    """
    print(f"\n{'─'*60}")
    print(f"AGENT NODE — {len(state['messages'])} messages in state")

    # Always prepend system prompt — it guides every decision
    messages_with_system = [
        SystemMessage(content=SYSTEM_PROMPT)
    ] + state["messages"]

    response = llm_with_tools.invoke(messages_with_system)

    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"  TOOL CALL → {tc['name']}({tc['args']})")
    else:
        preview = (response.content or "")[:100]
        print(f"  FINAL ANSWER → {preview}...")

    # Returning a dict with "messages" key causes add_messages
    # to append response to state["messages"] automatically
    return {"messages": [response]}


# =========================================================
# NODE 2: TOOLS
# =========================================================
# ToolNode is a prebuilt LangGraph node that:
#   1. Reads tool_calls from the last message
#   2. Executes each tool call using your @tool functions
#   3. Appends ToolMessage results to state.messages
#
# This replaces your entire run_tool() router + for loop.
# You just pass it your tools list and it handles everything.

tools_node = ToolNode(tools_list)


# =========================================================
# ROUTING FUNCTION
# =========================================================
# This decides which node runs AFTER the agent node.
# It reads the last message and checks if it has tool_calls.
#
# Returns a string: either "tools" or END
# LangGraph uses this string to look up the next node in the graph.

def should_continue(state: AgentState) -> str:
    """
    Router: decides whether to call tools or end the conversation.
    Called after every agent node execution.
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"  ROUTING → tools node")
        return "tools"

    print(f"  ROUTING → END")
    return END


# =========================================================
# BUILD THE GRAPH
# =========================================================
# StateGraph is LangGraph's core class.
# You declare what the state looks like, then add nodes and edges.
#
# ADD NODE:  graph.add_node("name", function)
# ADD EDGE:  graph.add_edge("from", "to")       — always goes here
# ADD COND:  graph.add_conditional_edges(...)    — routing function decides

graph_builder = StateGraph(AgentState)

# Register nodes
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tools_node)

# Entry point — where the graph starts
graph_builder.set_entry_point("agent")

# Conditional edge from agent: routes to tools or END
graph_builder.add_conditional_edges(
    "agent",           # from this node
    should_continue,   # call this function to decide
    {
        "tools": "tools",  # if function returns "tools" → go here
        END: END           # if function returns END → stop
    }
)

# Unconditional edge: after tools always go back to agent
# This creates the ReAct loop: agent → tools → agent → tools → ... → END
graph_builder.add_edge("tools", "agent")

# Compile — turns the graph definition into a runnable object
graph = graph_builder.compile()


# =========================================================
# MEMORY (conversation history)
# =========================================================
# In your old agent, you managed conversation_history manually.
# In LangGraph, you pass a thread_id config to persist state
# between calls. Here we manage it simply with a list.
#
# For production: use SqliteSaver or PostgresSaver from langgraph.

conversation_history = []


def chat(user_input: str) -> str:
    """
    Sends a user message through the graph and returns the final answer.
    Maintains full conversation history across turns.
    """
    conversation_history.append(HumanMessage(content=user_input))

    print(f"\n{'='*60}")
    print(f"USER: {user_input}")
    print(f"HISTORY: {len(conversation_history)} messages")
    print(f"{'='*60}")

    # invoke() runs the graph until it reaches END
    # It takes the current state and returns the final state
    final_state = graph.invoke(
        {"messages": conversation_history},
        config={"recursion_limit": 30}
        # recursion_limit = max node visits = your max_steps equivalent
    )

    # The last message in final state is the assistant's answer
    final_message = final_state["messages"][-1]
    answer = final_message.content

    # Add assistant response to history for next turn
    conversation_history.append(final_message)

    return answer


# =========================================================
# MAIN LOOP
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  LANGGRAPH REACT AGENT")
    print("  Model: llama-3.3-70b-versatile via Groq")
    print("  Tools: calculator, get_datetime, get_weather, web_search")
    print("=" * 60)
    print("Commands: 'quit' to exit, 'clear' to reset memory\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation_history.clear()
            print("Memory cleared.\n")
            continue

        if not user_input:
            continue

        answer = chat(user_input)
        print(f"\nAGENT: {answer}\n")
        print("-" * 60)
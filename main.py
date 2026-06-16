'''
# pip install langgraph langchain langchain-openai langchain-groq langchain-community langchain-tavily psycopg[binary] psycopg_pool python-dotenv tavily-python pip install requests streamlit

# install PostgresSql and create database
CREATE DATABASE langgraph_memory;  ( or open pgadmin4 and create database there )
'''
# LangGraph Multi-Agent Travel Booking System with Long-Term Memory

# main.py

import os
import re
from typing import Any, TypedDict, Annotated
import operator

import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig

try:
    from langgraph.store.base import BaseStore
    from langgraph.store.postgres import PostgresStore
except Exception:
    BaseStore = Any  # type: ignore[misc,assignment]
    PostgresStore = None

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Add it to your environment or .env file.")

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# State
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    memory_context: str
    llm_calls: int


def _get_user_id(config: RunnableConfig) -> str:
    configurable = config.get("configurable", {})
    return str(configurable.get("user_id", "anonymous_user"))


def _merge_profile(existing: dict[str, Any], query: str) -> dict[str, Any]:
    updated = dict(existing)

    name_match = re.search(r"\bmy name is\s+([A-Za-z][A-Za-z\s'-]{1,40})", query, flags=re.IGNORECASE)
    if name_match:
        updated["name"] = name_match.group(1).strip().title()

    preference_patterns = [
        r"\bi prefer\s+([^.,;]+)",
        r"\bi like\s+([^.,;]+)",
        r"\bmy budget is\s+([^.,;]+)",
        r"\bi usually travel with\s+([^.,;]+)",
    ]

    preferences = list(updated.get("preferences", []))
    for pattern in preference_patterns:
        for match in re.findall(pattern, query, flags=re.IGNORECASE):
            value = match.strip()
            if value and value not in preferences:
                preferences.append(value)

    updated["preferences"] = preferences
    updated["last_query"] = query
    return updated


def memory_recall_agent(state: TravelState, config: RunnableConfig, *, store: BaseStore):
    user_id = _get_user_id(config)
    item = store.get(("users",), user_id)

    if item and isinstance(item.value, dict):
        profile = item.value
        name = profile.get("name", "")
        preferences = profile.get("preferences", [])

        profile_parts: list[str] = []
        if name:
            profile_parts.append(f"User name: {name}")
        if preferences:
            profile_parts.append("Known preferences: " + ", ".join(preferences))

        memory_context = "\n".join(profile_parts)
        message_text = "Loaded long-term memory for this user."
    else:
        memory_context = ""
        message_text = "No long-term memory found for this user yet."

    return {
        "memory_context": memory_context,
        "messages": [AIMessage(content=message_text)],
        "llm_calls": state.get("llm_calls", 0),
    }

# Flight Agent
def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)
    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content=f"Flight results fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Hotel Agent
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Itinerary Agent
def itinerary_agent(state: TravelState):

    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}

    Long-Term User Memory:
    {state.get('memory_context', '') or 'No stored user preferences yet.'}
    """

    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel planner"
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Final Response Agent
def final_agent(state: TravelState):

    final_prompt = f"""
    Generate final travel response.

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}
    """

    response = llm.invoke([
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def memory_save_agent(state: TravelState, config: RunnableConfig, *, store: BaseStore):
    user_id = _get_user_id(config)
    existing_item = store.get(("users",), user_id)
    existing_profile: dict[str, Any] = {}

    if existing_item and isinstance(existing_item.value, dict):
        existing_profile = existing_item.value

    updated_profile = _merge_profile(existing_profile, state["user_query"])
    store.put(("users",), user_id, updated_profile)

    return {
        "messages": [AIMessage(content="Updated long-term memory for this user.")],
        "llm_calls": state.get("llm_calls", 0),
    }


graph = StateGraph(TravelState)

graph.add_node("memory_recall_agent", memory_recall_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)
graph.add_node("memory_save_agent", memory_save_agent)

graph.add_edge(START, "memory_recall_agent")
graph.add_edge("memory_recall_agent", "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", "memory_save_agent")
graph.add_edge("memory_save_agent", END)


# Run one-time checkpoint migrations in autocommit mode because
# LangGraph uses CREATE INDEX CONCURRENTLY in its Postgres setup.
with psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    prepare_threshold=0,
    row_factory=dict_row,
) as _setup_conn:
    PostgresSaver(_setup_conn).setup()  # type: ignore[arg-type]

# Persistent runtime connection so both CLI and Streamlit can share the compiled app
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    prepare_threshold=0,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)  # type: ignore[arg-type]

if PostgresStore is None:
    raise ImportError(
        "PostgresStore is unavailable. Install/update LangGraph store support (for example, langgraph-store-postgres)."
    )

store = PostgresStore(_conn)  # type: ignore[arg-type]
store.setup()

app = graph.compile(checkpointer=checkpointer, store=store)


if __name__ == "__main__":
    user_id = input("Enter user id (long-term memory scope) [bharath]: ").strip() or "bharath"
    thread_id = input("Enter thread id (chat window/session) [chat_1]: ").strip() or "chat_1"

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        }
    }

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "memory_context": "",
            "llm_calls": 0
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)

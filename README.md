# AI Travel Planning System using LangGraph

This project is a real-world multi-agent AI system built using LangGraph.

The system now uses 6 agents (including memory recall/save) to plan trips with both short-term and long-term memory.

## Features

- 🧠 Memory Recall Agent
- ✈️ Flight Search Agent
- 🏨 Hotel Search Agent
- 🗓️ Itinerary Planning Agent
- 🤖 Final Response Agent
- 💾 Memory Save Agent
- 🧠 Persistent Short-Term Memory (thread checkpoints) using PostgreSQL
- 🧠 True Long-Term Memory (cross-thread user profile) using PostgreSQL Store
- 🌐 Real-time API Integration
- 💻 Streamlit Web Interface

---

# Tech Stack

- LangGraph
- LangChain
- Groq
- Llama 3.3 70B
- PostgreSQL
- Streamlit
- Tavily API
- AviationStack API

---

# Step 1: Create Python Environment

Open the terminal inside the project folder and run:

		python -m venv venv


Now activate the environment:

#### Windows

		.\venv\Scripts\activate

---

# Step 2: Install Dependencies

Run the following command:

		pip install langgraph langchain langchain-openai langchain-groq langchain-community langchain-tavily psycopg[binary] psycopg_pool python-dotenv tavily-python requests streamlit

		pip install -U "psycopg[binary,pool]" langgraph-checkpoint-postgres langgraph-store-postgres

---

# Step 3: Install PostgreSQL

Download and install PostgreSQL: https://www.postgresql.org/download/

⚠️ Important:
While installing PostgreSQL, remember:
- PostgreSQL Password
- Port Number

You will need them later while creating the database connection string.

---

# Step 4: Create Database

Open PostgreSQL and run:

CREATE DATABASE langgraph_memory_demo;


---

# Step 5: Setup `.env` File

Create a `.env` file inside the project folder.

Add the following keys:

GROQ_API_KEY=your_groq_api_key

TAVILY_API_KEY=your_tavily_api_key

AVIATIONSTACK_API_KEY=your_aviationstack_api_key

DATABASE_URL=postgresql://postgres:postgres@localhost:5433/langgraph_memory_demo


---

# Step 6: Get API Keys

## Get Groq API Key

https://console.groq.com

---

## Get Tavily API Key

https://tavily.com
  
---

## Get AviationStack API Key

https://aviationstack.com

---

# Step 7: Run the Application

#### Run Multi-Agent System in Terminal

		python main.py


This will test the multi-agent system through the terminal.

---

#### Run Streamlit Web App


		streamlit run frontend.py


This will launch the Multi-Agent AI web application.

---

# How To Use Memory Correctly

## Memory Keys

- `user_id` = long-term memory identity
- `thread_id` = one chat window/session

## Rule

- Keep the same `user_id` to reuse long-term memory.
- Change `thread_id` to simulate a new chat/session.

## Example Flow

1. Run with `user_id=bharath`, `thread_id=chat_1`
2. Ask: "My name is Bharath and I prefer budget hotels"
3. Run again with `user_id=bharath`, `thread_id=chat_2`
4. Ask for a new trip plan
5. App recalls saved profile from long-term memory

## In Streamlit

- Enter **User ID** in the sidebar (long-term identity)
- Enter **Thread ID** in the sidebar (current chat)
- Keep User ID same, change Thread ID to test cross-thread recall

---

#### Example Prompt

Plan a complete 7 days Japan trip including flights, hotels and sightseeing under 2 lakhs.


---

# Project Workflow

1. Memory Recall Agent loads long-term memory using `user_id`
2. Flight Agent searches flights
3. Hotel Agent searches hotels
4. Itinerary Agent creates travel plan with recalled memory context
5. Final Agent combines everything together
6. Memory Save Agent updates long-term profile (`name`, `preferences`, `last_query`)
7. PostgreSQL checkpointer stores thread-level chat state (short-term memory via `thread_id`)
8. PostgreSQL store saves user profile/preferences across different threads (long-term memory via `user_id`)

---

# Memory Clarification

- **Short-Term Memory (Thread Level):** `PostgresSaver` with `thread_id`.
- **Long-Term Memory (Cross Thread):** `PostgresStore` with `user_id`.

If you open a new chat thread (`thread_id`) but keep the same `user_id`, the app can still recall saved preferences.

---

# Latest Changes

- Added true long-term memory read/write nodes in the graph (`memory_recall_agent`, `memory_save_agent`)
- Added separate `user_id` and `thread_id` inputs in Streamlit sidebar
- Memory Save card now displays saved profile fields in UI
- Switched to separate PostgreSQL runtime connections for checkpointer and store for better runtime stability


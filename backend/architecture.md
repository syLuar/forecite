# Project: Legal Research Assistant Backend Architecture

## 1. System Overview

### 1.1. Objective
To design and build a robust, Python-based backend for a legal research and drafting application. The backend will expose a RESTful API to a React frontend. Its core logic is powered by **dynamic, self-correcting agentic workflows** built with LangGraph, which reason over a Neo4j knowledge graph.

### 1.2. Core Architecture
-   **API Framework:** FastAPI for high-performance, asynchronous API endpoints.
-   **Agentic Workflows:** LangGraph, utilizing conditional edges to model iterative research and drafting processes with self-critique loops.
-   **Knowledge Base:** Neo4j, storing legal documents, entities, and relationships, with a vector index for semantic search.
-   **LLM Provider:** OpenAI API, integrated via LangChain models.
-   **Data Validation:** Pydantic for strict type enforcement in API requests/responses and internal state management.

### 1.3. Key Principles
-   **Stateless API:** The backend API is stateless. The React frontend is responsible for managing UI state, such as which precedents are selected in the user's "Case File."
-   **Dynamic Reasoning:** Workflows are not fixed sequences. They use conditional logic to assess intermediate results, refine queries, and critique strategies, mimicking an expert's iterative process.
-   **Modularity:** The system is divided into clear layers: API (FastAPI), State/Graphs (LangGraph), Tools (LangChain), and Database (Neo4j), promoting maintainability and separation of concerns.

---

## 2. Directory Structure (Backend)

```
legal_assistant_backend/
├── .env                  # Stores API keys and database credentials
├── .gitignore
├── main.py               # FastAPI application entry point, defines API routes
├── app/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py     # Configuration management (reads from .env)
│   ├── graphs/
│   │   ├── __init__.py
│   │   ├── research_graph.py  # Definition of the research agent team and graph
│   │   ├── drafting_graph.py  # Definition of the drafting agent team and graph
│   |   └── state.py           # TypedDict state definitions for the LangGraph graphs
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py    # Pydantic models for API request/response validation
│   └── tools/
│       ├── __init__.py
│       └── neo4j_tools.py # LangChain Tool definitions for Neo4j interaction
├── scripts/
│   └── ingest_graph.py   # Offline script to build the knowledge graph
├── requirements.txt
└── architecture.md
```

---

## 3. Data Ingestion & Knowledge Graph

*(This process is an offline, prerequisite step before running the main application.)*

### 3.1. Script: `scripts/ingest_graph.py`
This script processes raw PDF documents from `data/raw_docs/` and populates the Neo4j database. It performs entity extraction and summarization using LLM calls to build a rich, interconnected graph.

### 3.2. Neo4j Graph Schema
-   **Nodes:**
    -   `(:Document {source: string})`
    -   `(:Chunk {text: string, summary: string, embedding: vector})`
    -   `(:Concept {name: string, type: string})` - Type can be 'Doctrine', 'Case', etc.
-   **Relationships:**
    -   `(:Chunk) -[:PART_OF]-> (:Document)`
    -   `(:Chunk) -[:APPLIES_CONCEPT]-> (:Concept)`
-   **Indexes:** A vector index named `chunk_embeddings` is created on the `Chunk.embedding` property for efficient semantic search. Uniqueness constraints are applied to `Document.source` and `Concept.name`.

---

## 4. API Layer (FastAPI)

### 4.1. File: `main.py`
This file defines the FastAPI application and its endpoints. It serves as the entry point for all frontend requests.

### 4.2. API Endpoints

#### `POST /api/v1/research/query`
-   **Description:** Initiates the Research Graph to find relevant precedents.
-   **Request Body:** `schemas.ResearchQueryRequest` (defined in Pydantic).
-   **Response Body:** `schemas.ResearchQueryResponse`.
-   **Workflow:**
    1.  Receives request and validates it against `ResearchQueryRequest`.
    2.  Compiles the `research_graph` from `app.graphs.research_graph`.
    3.  Initializes the graph's state: `initial_state = {"query_text": request.query_text, "refinement_count": 0}`.
    4.  Asynchronously invokes the graph: `final_state = await research_graph.ainvoke(initial_state)`.
    5.  Formats the `retrieved_docs` from the `final_state` into the `ResearchQueryResponse` model and returns it.

#### `POST /api/v1/generation/draft-argument`
-   **Description:** Initiates the Drafting Graph to generate a legal argument.
-   **Request Body:** `schemas.ArgumentDraftRequest`.
-   **Response Body:** `schemas.ArgumentDraftResponse`.
-   **Workflow:**
    1.  Receives request and validates it.
    2.  Compiles the `drafting_graph` from `app.graphs.drafting_graph`.
    3.  Initializes the state: `initial_state = {"user_facts": request.user_facts, "case_file": request.case_file.dict()}`.
    4.  Asynchronously invokes the graph: `final_state = await drafting_graph.ainvoke(initial_state)`.
    5.  Constructs the `ArgumentDraftResponse` from the `strategy` and `drafted_argument` fields of the `final_state` and returns it.

### 4.3. Pydantic Schemas (`app/models/schemas.py`)
This file defines the strict data contracts for the API, ensuring type safety and clear communication with the frontend. It includes models like `ResearchQueryRequest`, `ResearchQueryResponse`, `ArgumentDraftRequest`, and `ArgumentDraftResponse`.

---

## 5. Agentic Layer (LangGraph)

This is the core reasoning engine of the application, defined in the `app/graphs/` directory.

### 5.1. File: `app/graphs/state.py`
Defines the `TypedDict` state objects (`ResearchState`, `DraftingState`) that are passed between nodes in the graphs. This ensures that all nodes have a consistent view of the current workflow status.

### 5.2. File: `app/graphs/research_graph.py` - The "Refinement Loop"
This graph finds relevant precedents, with the ability to refine its own search query if initial results are poor.

-   **Mermaid Diagram:**
    ```mermaid
    graph TD
        A[Start] --> B(QueryPlannerNode);
        B -- initial_plan --> C(RetrievalNode);
        C -- retrieved_docs --> D{AssessRetrievalNode};
        D --o|Results Sufficient| E[End];
        D --o|Refine Query| F(QueryRefinerNode);
        F -- refined_plan --> C;
    ```
-   **Nodes & Logic:**
    -   **`QueryPlannerNode`:** Generates an initial search plan.
    -   **`RetrievalNode`:** Executes the plan using tools from `app/tools/neo4j_tools.py`.
    -   **`AssessRetrievalNode` (Conditional Edge):** A router that checks the quality and quantity of `retrieved_docs`. If insufficient, it routes to the `QueryRefinerNode`. Otherwise, it ends the workflow.
    -   **`QueryRefinerNode`:** Takes the original query and poor results as input to formulate a better search plan, then sends the flow back to the `RetrievalNode`.

### 5.3. File: `app/graphs/drafting_graph.py` - The "Critique Loop"
This graph devises a strategy, critiques it, and then drafts an argument.

-   **Mermaid Diagram:**
    ```mermaid
    graph TD
        A[Start] --> B(StrategistNode);
        B -- proposed_strategy --> C{CritiqueNode};
        C --o|Strategy Approved| D(DraftingTeamNode);
        D -- drafted_argument --> E[End];
        C --o|Needs Revision| B;
    ```
-   **Nodes & Logic:**
    -   **`StrategistNode`:** Proposes an initial (or revised) argumentative strategy. It receives feedback from the `CritiqueNode` on subsequent loops.
    -   **`CritiqueNode` (Conditional Edge):** A "red team" agent. It programmatically assesses the proposed strategy for logical flaws or weaknesses. If the strategy is weak, it provides feedback and routes back to the `StrategistNode`. If strong, it proceeds.
    -   **`DraftingTeamNode`:** A sub-graph or single node that executes the approved strategy by gathering evidence and writing the final text.

---

## 6. Tool Layer

### 6.1. File: `app/tools/neo4j_tools.py`
This module defines the `LangChain` tools that agents will use to interact with the database.

-   **`@tool def vector_search(...)`:** Takes a semantic query string as input. Internally, it generates an embedding and runs a `CALL db.index.vector.queryNodes(...)` Cypher query against Neo4j.
-   **`@tool def graph_expand(...)`:** Takes Neo4j node IDs as input and runs a `MATCH` query to find connected nodes and their properties (e.g., retrieving the parent `Document` for a given `Chunk`).

These tools abstract the database logic, allowing the agents in the graph to simply declare their intent (e.g., "find similar ideas") without needing to know the underlying Cypher query syntax.
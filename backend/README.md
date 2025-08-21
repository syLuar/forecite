# Forecite Legal Research Assistant - Backend

A sophisticated AI-powered legal research and argument drafting system built with FastAPI, LangGraph, and Neo4j. This backend provides intelligent agent workflows for comprehensive legal research, document analysis, and argument generation with iterative refinement capabilities.

## 🎯 Overview

The Forecite backend is the core engine powering an AI legal research assistant that helps legal professionals:

- **Conduct Comprehensive Research**: Multi-agent research workflows with automatic query refinement and quality assessment
- **Draft Legal Arguments**: AI-powered argument generation with iterative critique and improvement cycles  
- **Practice Moot Court**: Generate counterarguments and rebuttals for advocacy training
- **Manage Case Files**: Organize research findings, documents, and drafts in structured case files
- **Analyze Precedents**: Deep citation network analysis and precedent strength assessment

## 🏗️ Architecture

### Core Components

```
backend/
├── main.py                    # FastAPI application entry point
├── app/
│   ├── core/                  # Configuration, database, LLM initialization
│   ├── graphs/                # LangGraph agent workflow definitions
│   │   └── v2/               # Current implementation (v2)
│   │       ├── research_graph.py      # Research refinement workflow
│   │       ├── drafting_graph.py      # Argument drafting workflow
│   │       ├── counterargument_graph.py # Moot court practice workflow
│   │       ├── research_agent.py      # ReAct research agent
│   │       └── state.py              # Shared state management
│   ├── models/                # Data models and schemas
│   │   ├── schemas.py         # Pydantic API models
│   │   └── database_models.py # SQLAlchemy database models
│   ├── services/              # Business logic layer
│   │   └── case_file_service.py # Case file and draft management
│   └── tools/                 # Agent tools and utilities
│       ├── neo4j_tools.py     # Knowledge graph operations
│       ├── research_tools.py  # Research-specific tools
│       └── database_tools.py  # Database mutation tools
├── scripts/                   # Utility and setup scripts
│   ├── setup_database.py      # Database initialization
│   ├── ingest_graph.py        # Knowledge graph population
│   ├── scrape_singapore_cases.py # Legal document scraping
│   └── export_graphs_as_mermaid_pngs.py # Graph visualization
└── static/docs/              # Static documentation
```

### Technology Stack

- **Web Framework**: FastAPI with async/await support, automatic OpenAPI documentation
- **AI Orchestration**: LangGraph for complex agent workflows and state management
- **LLM Integration**: Google Gemini, OpenAI GPT with configurable model selection
- **Knowledge Graph**: Neo4j with vector embeddings for semantic search
- **Relational Database**: SQLite (development) / PostgreSQL (production) 
- **Document Processing**: LangChain text splitters, embeddings, and document loaders
- **Type Safety**: Pydantic for request/response validation and structured LLM outputs
- **Caching**: Advanced caching system for embeddings, entities, and knowledge graphs

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (Python 3.13 recommended)
- **Neo4j Database** (hosted or local)
- **Google API Key** (for Gemini LLM)
- **OpenAI API Key** (for OpenAI LLMs)

### Installation

1. **Clone and navigate to backend**:
   ```bash
   git clone https://github.com/syLuar/forecite.git
   cd forecite/backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration (see Configuration section)
   ```

5. **Initialize database**:
   ```bash
   python scripts/setup_database.py
   ```

6. **Start the server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

The API will be available at:
- **Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **API Schema**: `http://localhost:8000/redoc`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```bash
# LLM Configuration
LLM_CONFIG_PATH=./llm_config.yaml
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Neo4j Configuration
NEO4J_URI=bolt://your_neo4j_host:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j
VECTOR_INDEX_NAME=chunk_embeddings

# LangSmith (Optional - for debugging)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=forecite

# Application Settings
ENVIRONMENT=development  # or production
DEBUG=false
```

### LLM Configuration (`llm_config.yaml`)

Configure different models for different workflows:

```yaml
embeddings:
  model: "gemini-embedding-001"
  dimension: 3072
  model_provider: "google"

main:
  research_graph:
    model: "gemini-2.5-flash-lite"
    temperature: 0.2
    model_provider: "google"
  
  drafting_graph:
    model: "gemini-2.5-flash-lite"
    temperature: 0.2
    model_provider: "google"
    # Per-node configuration for specialized tasks
    nodes:
      final_drafter_node:
        model: "gpt-4o"  # Use more powerful model for final drafting
        temperature: 0.5
        model_provider: "openai"
```

## 🤖 Agent Workflows

### 1. Research Graph (`research_graph.py`)

**Purpose**: Comprehensive legal research with automatic refinement

**Workflow**:
```
Query Analysis → Strategy Selection → Search Execution → Quality Assessment
     ↓                                                        ↓
Refinement ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←← (if needed)
```

**Key Features**:
- Multi-strategy search (semantic, fulltext, citation, concept-based)
- Automatic query refinement with quality thresholds
- Jurisdiction and document type filtering
- Maximum 3 refinement cycles to prevent infinite loops

**Example Usage**:
```python
initial_state = {
    "query_text": "negligence duty of care Singapore",
    "jurisdiction_filter": "Singapore",
    "refinement_count": 0
}

final_state = await research_graph.ainvoke(initial_state)
```

### 2. Drafting Graph (`drafting_graph.py`)

**Purpose**: Legal argument generation with iterative improvement

**Workflow**:
```
Fact Analysis → Strategy Development → Argument Drafting → Critique Assessment
     ↓                                         ↓                ↓
Legal Issue Identification            Precedent Integration    Refinement ←←
```

**Key Features**:
- Comprehensive legal strategy development
- Precedent-based argument construction
- Iterative critique and improvement cycles
- Citation validation and authority analysis

**Example Usage**:
```python
initial_state = {
    "user_facts": "Plaintiff entered into a supply contract...",
    "case_file": case_file_documents,
    "legal_question": "Was there a breach of duty?"
}

final_state = await drafting_graph.ainvoke(initial_state)
```

### 3. Research Agent (`research_agent.py`)

**Purpose**: Interactive ReAct agent for guided research

**Capabilities**:
- Natural language research planning
- Tool-based document discovery
- Real-time case file population
- Research note generation

**Available Tools**:
- `semantic_search_legal_content()`: Vector-based document search
- `find_cases_by_fact_pattern()`: Analogical reasoning
- `extract_legal_information()`: Entity extraction from documents
- `add_document_tool()`: Save discoveries to case files

### 4. Counterargument Graph (`counterargument_graph.py`)

**Purpose**: Moot court practice with counterargument generation

**Workflow**:
```
Argument Analysis → Opposition Research → Counterargument Generation → Rebuttal Development
```

**Features**:
- RAG-based opposition research
- Comprehensive counterargument generation
- Strategic rebuttal development
- Practice session management

## 🛠️ Tools & Utilities

### Neo4j Tools (`neo4j_tools.py`)

Comprehensive knowledge graph operations:

```python
# Vector similarity search
results = vector_search(query="contract breach", top_k=10)

# Citation network analysis  
network = find_case_citations("Spandeck Engineering", direction="both")

# Precedent strength assessment
strength = assess_precedent_strength("Spandeck Engineering [2007] SGCA 10")

# Legal concept discovery
concepts = find_legal_concepts(["negligence", "duty of care"])
```

### Research Tools (`research_tools.py`)

Streamlined research operations:

```python
# Semantic search with filtering
results = semantic_search_legal_content(
    query="negligence standard of care",
    legal_tags=["tort", "negligence"],
    jurisdiction="Singapore"
)

# Fact pattern matching
patterns = find_cases_by_fact_pattern(
    fact_pattern="medical negligence misdiagnosis",
    jurisdiction="Singapore"
)
```

### Database Tools (`database_tools.py`)

Real-time database mutations during agent workflows:

```python
# Create case file
case_id = create_case_file(
    title="Contract Dispute Analysis",
    user_facts="Client claims breach of supply agreement",
    party_represented="Plaintiff"
)

# Add discovered documents
add_document_to_case_file(
    case_file_id=case_id,
    document_data=research_result
)
```

## 📋 Scripts & Management

### Database Setup
```bash
# Initialize database tables and constraints
python scripts/setup_database.py

# Migrate database schema
python scripts/db_migration.py
```

### Knowledge Graph Management
```bash
# Populate knowledge graph with legal documents
python scripts/ingest_graph.py --docs-dir data/cases/

# Clear and rebuild graph
python scripts/ingest_graph.py --reset-db

# Filter by legal areas
python scripts/ingest_graph.py --required-tags "contract,tort"
```

### Document Processing
```bash
# Process individual PDF documents
python scripts/process_pdf.py path/to/document.pdf

# Scrape Singapore case law
python scripts/scrape_singapore_cases.py
```

### Visualization
```bash
# Export graph visualizations
python scripts/export_graphs_as_mermaid_pngs.py
```

## 🔌 API Endpoints

### Research Endpoints

- `POST /api/v1/research/query` - Execute research workflow
- `POST /api/v1/research/conduct-research` - ReAct agent research
- `GET /api/v1/research/precedent-analysis/{citation}` - Precedent analysis
- `GET /api/v1/research/citation-network/{citation}` - Citation network

### Drafting Endpoints

- `POST /api/v1/generation/draft-argument` - Generate legal arguments
- `POST /api/v1/drafts/ai-edit` - AI-assisted draft editing
- `GET /api/v1/drafts/{draft_id}` - Retrieve saved drafts

### Case Management Endpoints

- `POST /api/v1/case-files` - Create case files
- `GET /api/v1/case-files/{id}` - Retrieve case file details
- `POST /api/v1/case-files/{id}/documents` - Add documents
- `POST /api/v1/case-files/{id}/notes` - Add research notes

### Moot Court Endpoints

- `POST /api/v1/moot-court/generate-counterarguments` - Generate counterarguments
- `POST /api/v1/moot-court/save-session` - Save practice sessions

## 🔍 Key Features

### Structured Output Validation

All LLM interactions use Pydantic models for guaranteed type safety:

```python
class SearchPlanOutput(BaseModel):
    strategy: str = Field(description="Search strategy")
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=10)

# Structured LLM calls
structured_llm = llm.with_structured_output(SearchPlanOutput)
plan = structured_llm.invoke(prompt)  # Always returns SearchPlanOutput
```

### Streaming Support

Real-time streaming for long-running agent workflows:

```python
# Enable streaming in requests
request = ResearchQueryRequest(
    query_text="negligence analysis",
    stream=True
)

# Streaming response with progress updates
for chunk in stream_research_query(request):
    print(f"Progress: {chunk}")
```

### Caching System

Advanced caching for performance optimization:

```python
# Cached operations during knowledge graph ingestion
cache = LLMCache()
embedding = cache.get_embedding(text)  # Returns cached or generates new
entities = cache.get_entities(chunk)   # Cached entity extraction
summary = cache.get_summary(text)      # Cached summarization
```

### Error Handling & Resilience

- Automatic retry logic for LLM API calls
- Graceful degradation for failed operations
- Comprehensive logging and monitoring
- Circuit breaker patterns for external services

## 🚀 Production Deployment

### Environment Configuration

**Production Settings**:
```bash
ENVIRONMENT=production
DEBUG=false
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

**Performance Optimization**:
```bash
# Use production ASGI server
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker

# Configure Neo4j connection pooling
NEO4J_MAX_CONNECTION_POOL_SIZE=50
NEO4J_CONNECTION_TIMEOUT=30
```

### Monitoring & Observability

- **Health Checks**: `/health` endpoint for service monitoring
- **LangSmith Integration**: Detailed tracing of LLM interactions
- **Structured Logging**: JSON-formatted logs for analysis
- **Performance Metrics**: Request timing and success rates

### Security Considerations

- API key rotation and secure storage
- Input validation and sanitization  
- Rate limiting for resource-intensive operations
- Database connection security and encryption

## 📚 Documentation

- **API Documentation**: Available at `/docs` (Swagger UI)
- **OpenAPI Schema**: Available at `/redoc`
- **User Guide**: Static documentation in `/static/docs/`
- **Architecture Details**: See main project documentation

## 🆘 Troubleshooting

### Common Issues

**Neo4j Connection Errors**:
```bash
# Check Neo4j connectivity
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')); driver.verify_connectivity()"
```

**LLM API Issues**:
- Verify API keys in `.env`
- Check API quotas and rate limits
- Enable LangSmith tracing for detailed debugging

**Performance Issues**:
- Monitor Neo4j query performance
- Check vector index status
- Review caching effectiveness

**Memory Issues**:
- Adjust chunk sizes in configuration
- Monitor embedding cache size
- Configure connection pool limits

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [LangGraph](https://python.langchain.com/docs/langgraph) - Agent orchestration
- [Neo4j](https://neo4j.com/) - Graph database
- [Google Gemini](https://ai.google.dev/) - Large language models
- [LangChain](https://python.langchain.com/) - LLM integration framework

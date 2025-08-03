# Legal Research Assistant - Agent Implementation

This implementation provides the complete agent teams as specified in the architecture documentation. The system uses LangGraph to create dynamic, self-correcting workflows for legal research and argument drafting.

## Implementation Overview

### 1. State Management (`app/graphs/state.py`)
- **ResearchState**: Tracks iterative research refinement with query planning, retrieval results, and assessment metrics
- **DraftingState**: Manages strategy development, critique cycles, and argument generation
- **Supporting Types**: CaseFileDocument, SearchPlan, ArgumentStrategy for structured data flow

### 2. Research Graph (`app/graphs/research_graph.py`)
Implements the "Refinement Loop" workflow with these nodes:

#### Nodes:
- **QueryPlannerNode**: Analyzes user queries and creates search plans with appropriate strategies and filters
- **RetrievalNode**: Executes searches using Neo4j tools (vector search, fulltext, citation analysis, concept search)
- **QueryRefinerNode**: Improves search strategies based on assessment feedback

#### Conditional Logic:
- **AssessRetrievalNode**: Evaluates result quality and quantity, deciding whether to refine or complete
- Maximum 3 refinement cycles to prevent infinite loops
- Quality thresholds: minimum 3 results with average relevance score ≥ 0.6

### 3. Drafting Graph (`app/graphs/drafting_graph.py`)
Implements the "Critique Loop" workflow with these nodes:

#### Nodes:
- **StrategistNode**: Develops legal strategies based on facts and precedents, incorporates critique feedback
- **DraftingTeamNode**: Generates final legal arguments using approved strategies

#### Conditional Logic:
- **CritiqueNode**: "Red team" analysis that evaluates strategy strength, identifies weaknesses, suggests improvements
- Maximum 3 critique cycles with approval threshold ≥ 0.6 quality score
- Routes to revision or final drafting based on assessment

### 4. Neo4j Tools Integration (`app/tools/neo4j_tools.py`)
The graphs leverage these specialized tools:

#### Research Tools:
- `vector_search`: Semantic similarity search with jurisdiction/date filtering
- `fulltext_search`: Traditional text search on documents and chunks
- `find_case_citations`: Citation network analysis for precedent discovery
- `find_legal_concepts`: Concept-based document retrieval

#### Drafting Tools:
- `find_similar_fact_patterns`: Pattern matching for analogical reasoning
- `assess_precedent_strength`: Authority analysis for strategy validation
- `find_legal_tests`: Legal standard identification
- `find_authority_chain`: Precedent hierarchy construction

### 5. API Layer (`main.py`)
FastAPI endpoints that orchestrate the workflows:

#### Core Endpoints:
- `POST /api/v1/research/query`: Executes research graph with refinement
- `POST /api/v1/generation/draft-argument`: Executes drafting graph with critique

#### Analysis Endpoints:
- `GET /api/v1/research/precedent-analysis/{citation}`: Standalone precedent analysis
- `GET /api/v1/research/citation-network/{citation}`: Citation network exploration

## Key Features

### Dynamic Reasoning
- **Conditional Edges**: Workflows adapt based on intermediate results
- **Self-Assessment**: Agents evaluate their own output quality
- **Iterative Refinement**: Automatic improvement through feedback loops

### Structured Output
- **Pydantic Models**: All LLM outputs use structured output with Pydantic models for reliability
- **Type Safety**: Eliminates JSON parsing errors and ensures consistent data structures
- **Validation**: Automatic validation of LLM responses with clear error handling

### Error Handling
- Graceful degradation with fallback strategies
- Comprehensive logging for debugging
- Maximum iteration limits to prevent infinite loops

### Type Safety
- Pydantic models for API contracts
- TypedDict states for workflow consistency
- Full type annotations throughout

## Structured Output Implementation

### Research Graph Structured Models
- **SearchPlanOutput**: Validates search strategy with confidence scoring and rationale
- **RefinementPlanOutput**: Ensures refinement plans have clear improvement rationale

### Drafting Graph Structured Models  
- **StrategyOutput**: Comprehensive legal strategy with typed argument components
- **CritiqueOutput**: Structured critique with assessment scores and specific feedback
- **LegalArgumentComponent**: Individual argument elements with supporting authority

### Benefits of Structured Output
- **Reliability**: Eliminates JSON parsing failures and malformed responses
- **Validation**: Automatic type checking and constraint validation (e.g., scores 0-1)
- **Consistency**: Guaranteed data structure consistency across all agent interactions
- **Debugging**: Clear error messages when LLM outputs don't match expected schema

### Example Usage
```python
# Instead of parsing JSON manually:
response = llm.invoke(prompt)
try:
    plan = json.loads(response.content)  # Error-prone
except JSONDecodeError:
    # Handle parsing error

# Use structured output:
structured_llm = llm.with_structured_output(SearchPlanOutput)
plan = structured_llm.invoke(prompt)  # Always returns valid SearchPlanOutput
```

## Workflow Examples

### Research Workflow:
1. User submits: "negligence duty of care Singapore"
2. QueryPlanner creates semantic search strategy
3. Retrieval finds 8 results with avg score 0.5
4. Assessment determines insufficient quality
5. QueryRefiner broadens to include "tort law" and "reasonable person"
6. Second retrieval finds 12 results with avg score 0.8
7. Assessment approves and workflow completes

### Drafting Workflow:
1. User provides facts about professional negligence case
2. Strategist proposes argument based on Spandeck Engineering framework
3. Critique identifies weak factual analogies and insufficient authority
4. Strategist revises with stronger precedents and improved fact analysis
5. Second critique approves strategy (score 0.85)
6. DraftingTeam generates structured legal argument with citations

## Integration with Knowledge Graph

The agents seamlessly integrate with the Neo4j knowledge graph through:
- **Vector embeddings** for semantic search
- **Citation networks** for authority analysis  
- **Concept relationships** for legal principle tracking
- **Fact pattern matching** for analogical reasoning
- **Temporal precedent analysis** for authority strength

This implementation provides the sophisticated reasoning capabilities described in the architecture while maintaining modularity, type safety, robust error handling, and reliable structured output for all LLM interactions.

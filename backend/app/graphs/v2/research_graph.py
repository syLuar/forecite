"""
Research Graph v2 - Decomposed Legal Precedent Discovery

This module implements an improved research workflow with decomposed nodes
for better LLM performance and reliability. Each node has a single, focused responsibility.
"""

import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph
from langgraph.constants import END
from langgraph.config import get_stream_writer
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json
import time
import uuid
import os

from app.core.config import settings
from app.core.llm import create_llm
from .state import (
    ResearchState, 
    QueryAnalysis, 
    SearchStrategy, 
    SearchFilters,
    QualityAssessment,
    RefinementSuggestion
)
from ...tools.neo4j_tools import (
    vector_search,
    fulltext_search,
    find_case_citations,
    find_legal_concepts,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM using the config for research
task_config = settings.llm_config.get("main", {}).get("research", {})
llm = create_llm(task_config)


# Simplified Pydantic models for structured output
class QueryAnalysisOutput(BaseModel):
    """Simple query analysis with max 5 concepts."""
    legal_area: str = Field(description="Primary legal area (contract, tort, criminal, etc.)")
    key_concepts: List[str] = Field(description="3-5 key legal concepts", max_items=5)
    jurisdiction_hints: Optional[str] = Field(description="Jurisdiction clues if any")


class SearchStrategyOutput(BaseModel):
    """Simple search strategy selection."""
    strategy: str = Field(description="semantic, fulltext, citation, or concept")
    reasoning: str = Field(description="Why this strategy was chosen")
    confidence: float = Field(description="Confidence 0-1", ge=0.0, le=1.0)


class SearchFiltersOutput(BaseModel):
    """Simple search filter configuration."""
    jurisdiction: Optional[str] = Field(description="Jurisdiction filter if applicable")
    # document_type: Optional[str] = Field(description="Document type filter if applicable")
    year_from: Optional[int] = Field(description="Start year filter if applicable")
    year_to: Optional[int] = Field(description="End year filter if applicable")


class QualityAssessmentOutput(BaseModel):
    """Simple quality assessment."""
    sufficient: bool = Field(description="Whether results are sufficient")
    score: float = Field(description="Quality score 0-1", ge=0.0, le=1.0)
    reason: str = Field(description="Brief explanation of assessment")


class RefinementSuggestionOutput(BaseModel):
    """Simple refinement suggestion."""
    new_terms: List[str] = Field(description="Improved search terms (max 5)", max_items=5)
    new_strategy: str = Field(description="Improved search strategy")
    reasoning: str = Field(description="Explanation of improvements")


class QueryPreparationOutput(BaseModel):
    """Prepared search query and parameters."""
    search_query: str = Field(description="Optimized search query text")
    search_parameters_json: str = Field(description="Search parameters and filters (JSON-formatted)")
    execution_strategy: str = Field(description="How to execute the search")

    @property
    def search_parameters(self) -> Dict[str, Any]:
        """
        Convert JSON string to dictionary for easier access.
        """
        if isinstance(self.search_parameters_json, str):
            return json.loads(self.search_parameters_json)
        return self.search_parameters_json

# Decomposed node functions
async def query_analyzer_node(state: ResearchState) -> ResearchState:
    """
    Analyze the user query to extract key legal concepts and areas.
    
    This focused node only does query analysis - no strategy selection.
    """
    step_id = f"query_analyzer_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Analyzing legal query",
        "description": "Analyzing the user's legal query to identify key concepts, legal areas, and jurisdiction hints."
    })
    logger.info("Executing QueryAnalyzerNode")

    query_text = state["query_text"]

    # Create structured LLM for focused task
    structured_llm = llm.with_structured_output(QueryAnalysisOutput)

    system_prompt = """You are a legal research analyst. Your ONLY job is to analyze the user's query and extract:

1. The primary legal area (contract, tort, criminal, constitutional, etc.)
2. 3-5 key legal concepts or terms
3. Any jurisdiction hints (state, federal, specific court mentions)

Be concise and focused. Do NOT suggest search strategies or make plans."""

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Analyze this legal query: {query_text}")
    ])

    try:
        analysis_output = await structured_llm.ainvoke(prompt_template.format_messages())
        
        # Store in state
        state["query_analysis"] = {
            "legal_area": analysis_output.legal_area,
            "key_concepts": analysis_output.key_concepts,
            "jurisdiction_hints": analysis_output.jurisdiction_hints
        }
        
        logger.info(f"Query analysis completed: {analysis_output.legal_area}")
        
        # Stream completion update
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Query analyzed",
            "description": f"Identified legal area: {analysis_output.legal_area}, key concepts: {', '.join(analysis_output.key_concepts[:3])}"
        })
        
    except Exception as e:
        logger.error(f"Error in query analysis: {e}")
        # Fallback to basic analysis
        state["query_analysis"] = {
            "legal_area": "general",
            "key_concepts": [query_text[:50]],  # Use first 50 chars as fallback
            "jurisdiction_hints": None
        }

    return state


async def strategy_selector_node(state: ResearchState) -> ResearchState:
    """
    Select the optimal search strategy based on query analysis.
    
    This focused node only selects strategy - no filter building.
    """
    step_id = f"strategy_selector_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Selecting search strategy",
        "description": "Selecting the optimal search strategy based on the analyzed query characteristics."
    })
    logger.info("Executing StrategySelectorNode")

    query_analysis = state.get("query_analysis", {})
    refinement_count = state.get("refinement_count", 0)

    # Create structured LLM for strategy selection
    structured_llm = llm.with_structured_output(SearchStrategyOutput)

    # Include refinement context if this is a refinement
    refinement_context = ""
    if refinement_count > 0:
        quality_assessment = state.get("quality_assessment", {})
        refinement_context = f"\n\nThis is refinement attempt #{refinement_count}. Previous results were insufficient: {quality_assessment.get('reason', 'Unknown issue')}"

    system_prompt = f"""You are a legal search strategist. Your ONLY job is to select the best search strategy.

Available strategies:
- semantic: Use AI similarity search (best for conceptual queries)
- fulltext: Use keyword matching (best for specific terms/phrases)  
- citation: Use case citation analysis (best when specific cases mentioned)
- concept: Use legal concept mapping (best for broad legal principles)

Legal area: {query_analysis.get('legal_area', 'unknown')}
Key concepts: {query_analysis.get('key_concepts', [])}
Jurisdiction hints: {query_analysis.get('jurisdiction_hints', 'none')}
{refinement_context}

Choose ONE strategy and explain why it's optimal for this query."""

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Select the best search strategy for this legal query.")
    ])

    try:
        strategy_output = await structured_llm.ainvoke(prompt_template.format_messages())
        
        # Store in state
        state["search_strategy"] = {
            "strategy": strategy_output.strategy,
            "reasoning": strategy_output.reasoning,
            "confidence": strategy_output.confidence
        }
        
        logger.info(f"Strategy selected: {strategy_output.strategy} (confidence: {strategy_output.confidence:.2f})")
        
        # Stream completion update
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Strategy selected",
            "description": f"Selected {strategy_output.strategy} search strategy"
        })
        
    except Exception as e:
        logger.error(f"Error in strategy selection: {e}")
        # Fallback to semantic search
        state["search_strategy"] = {
            "strategy": "semantic",
            "reasoning": "Fallback to semantic search due to error",
            "confidence": 0.5
        }

    return state


async def filter_builder_node(state: ResearchState) -> ResearchState:
    """
    Build search filters based on query analysis.
    
    This focused node only builds filters - no strategy or execution.
    """
    step_id = f"filter_builder_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Building search filters",
        "description": "Building search filters for jurisdiction, document type, and date range based on query analysis."
    })
    logger.info("Executing FilterBuilderNode")

    query_analysis = state.get("query_analysis", {})

    # Create structured LLM for filter building
    structured_llm = llm.with_structured_output(SearchFiltersOutput)

    system_prompt = """You are a legal search filter specialist. Your ONLY job is to build appropriate filters.

Based on the query analysis, determine:
1. Jurisdiction filter (if specific state, federal, or court mentioned)
2. Document type filter (case law, statutes, regulations, etc.)
3. Date range filters (if specific time periods mentioned)

Only set filters if clearly indicated in the query. When in doubt, leave as None for broader search."""

    query_text = state["query_text"]
    jurisdiction_hints = query_analysis.get("jurisdiction_hints")

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Original query: {query_text}
Jurisdiction hints: {jurisdiction_hints}
Legal area: {query_analysis.get('legal_area')}

Build appropriate search filters.""")
    ])

    try:
        filters_output = await structured_llm.ainvoke(prompt_template.format_messages())
        
        # Store in state
        state["search_filters"] = {
            "jurisdiction": filters_output.jurisdiction,
            # "document_type": filters_output.document_type,
            "year_from": filters_output.year_from,
            "year_to": filters_output.year_to
        }
        
        logger.info(f"Filters built: jurisdiction={filters_output.jurisdiction}")
        
        # Stream completion update
        active_filters = []
        if filters_output.jurisdiction:
            active_filters.append(f"jurisdiction: {filters_output.jurisdiction}")
        # if filters_output.document_type:
        #     active_filters.append(f"type: {filters_output.document_type}")
        if filters_output.year_from or filters_output.year_to:
            year_range = f"{filters_output.year_from or 'before'}-{filters_output.year_to or 'present'}"
            active_filters.append(f"years: {year_range}")
        
        filter_description = f"Applied filters: {', '.join(active_filters)}" if active_filters else "No specific filters applied - using broad search"
        
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Filters configured",
            "description": filter_description
        })
        
    except Exception as e:
        logger.error(f"Error in filter building: {e}")
        # Fallback to no filters
        state["search_filters"] = {
            "jurisdiction": None,
            # "document_type": None,
            "year_from": None,
            "year_to": None
        }

    return state


async def query_preparation_node(state: ResearchState) -> ResearchState:
    """
    Prepare the search query and parameters using LLM analysis.
    
    This focused node only prepares the query - no actual search execution.
    """
    step_id = f"query_preparation_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Preparing search query",
        "description": "Analyzing strategy and filters to prepare optimized search query and parameters."
    })
    logger.info("Executing QueryPreparationNode")

    # Get strategy and filters
    search_strategy = state.get("search_strategy", {})
    search_filters = state.get("search_filters", {})
    query_analysis = state.get("query_analysis", {})
    
    strategy = search_strategy.get("strategy", "semantic")
    key_concepts = query_analysis.get("key_concepts", [])
    
    # Create structured LLM for query preparation
    structured_llm = llm.with_structured_output(QueryPreparationOutput)
    
    system_prompt = f"""You are a legal search query optimizer. Your ONLY job is to prepare the optimal search query and parameters.

Strategy selected: {strategy}
Key concepts: {key_concepts}
Filters: {search_filters}
Original query: {state["query_text"]}

For {strategy} search strategy, prepare:
1. An optimized search query text that works best with this strategy
2. Search parameters including filters and limits
3. Execution approach specific to this strategy

Focus on query optimization, not execution."""

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Prepare the optimal search query and parameters for this legal research.")
    ])

    try:
        prep_output = await structured_llm.ainvoke(prompt_template.format_messages())
        
        # Store prepared query and parameters
        state["prepared_query"] = prep_output.search_query
        state["search_parameters"] = prep_output.search_parameters
        state["execution_strategy"] = prep_output.execution_strategy
        
        logger.info(f"Query prepared: {prep_output.search_query[:100]}...")
        
        # Stream completion update
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Query prepared",
            "description": f"Optimized query for {strategy} strategy: '{prep_output.search_query[:50]}...'"
        })
        
    except Exception as e:
        logger.error(f"Error in query preparation: {e}")
        # Fallback preparation
        fallback_query = " ".join(key_concepts) if key_concepts else state["query_text"]
        state["prepared_query"] = fallback_query
        state["search_parameters"] = {"limit": 15, "min_score": 0.6}
        state["execution_strategy"] = strategy

    return state


async def search_execution_node(state: ResearchState) -> ResearchState:
    """
    Execute the prepared search query.
    
    This focused node only executes searches - no LLM analysis.
    """
    step_id = f"search_execution_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Executing search",
        "description": "Executing the prepared search query to retrieve relevant legal documents."
    })
    logger.info("Executing SearchExecutionNode")

    # Get prepared query and parameters
    prepared_query = state.get("prepared_query", state["query_text"])
    search_parameters = state.get("search_parameters", {"limit": 15, "min_score": 0.6})
    execution_strategy = state.get("execution_strategy", "semantic")
    search_filters = state.get("search_filters", {})
    
    # Build search kwargs from parameters and filters
    search_kwargs = dict(search_parameters)
    if search_filters.get("year_from"):
        search_kwargs["year_from"] = search_filters["year_from"]
    if search_filters.get("year_to"):
        search_kwargs["year_to"] = search_filters["year_to"]

    retrieved_docs = []

    try:
        if execution_strategy == "semantic":
            results = await vector_search(prepared_query, **search_kwargs)
            retrieved_docs.extend(results)

        elif execution_strategy == "fulltext":
            results = fulltext_search(
                prepared_query,
                "chunks",
                **{k: v for k, v in search_kwargs.items() if k != "min_score"}
            )
            retrieved_docs.extend(results)

        elif execution_strategy == "citation":
            if any(pattern in prepared_query.lower() for pattern in ["v ", " v. ", "[", "]"]):
                results = find_case_citations(prepared_query, "both")
                retrieved_docs.extend(results)
            else:
                # Fallback to semantic if no citations detected
                results = await vector_search(prepared_query, **search_kwargs)
                retrieved_docs.extend(results)

        elif execution_strategy == "concept":
            # Execute concept searches for each term in prepared query
            concepts = prepared_query.split()[:5]  # Limit concepts
            for concept in concepts:
                results = find_legal_concepts(
                    concept,
                    jurisdiction=search_kwargs.get("jurisdiction"),
                    limit=search_kwargs.get("limit", 15) // len(concepts)
                )
                retrieved_docs.extend(results)

        # Remove duplicates
        seen_ids = set()
        unique_docs = []
        for doc in retrieved_docs:
            doc_id = doc.get("chunk_id") or doc.get("document_source") or str(doc)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

        # Sort by relevance score
        unique_docs.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Update state
        state["retrieved_docs"] = unique_docs[:15]
        state["total_results"] = len(unique_docs)

        # Update search history
        search_history = state.get("search_history", [])
        search_history.append({
            "strategy": execution_strategy,
            "query": prepared_query,
            "filters": {k: v for k, v in search_kwargs.items() if k not in ["limit", "min_score"] and v is not None},
            "results_count": len(unique_docs),
            "timestamp": time.time()
        })
        state["search_history"] = search_history

        logger.info(f"Retrieved {len(unique_docs)} documents using {execution_strategy} strategy")
        
        # Stream completion update
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Search completed",
            "description": f"Found {len(unique_docs)} relevant documents using {execution_strategy} search"
        })

    except Exception as e:
        logger.error(f"Error during search execution: {e}")
        state["retrieved_docs"] = []
        state["total_results"] = 0

    return state


async def quality_assessor_node(state: ResearchState) -> ResearchState:
    """
    Assess the quality of retrieved results.
    
    This focused node only does quality assessment.
    """
    step_id = f"quality_assessor_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Assessing result quality",
        "description": "Assessing the quality and sufficiency of the retrieved legal documents."
    })
    logger.info("Executing QualityAssessorNode")

    retrieved_docs = state.get("retrieved_docs", [])
    total_results = state.get("total_results", 0)

    # Simple quality assessment logic
    min_results_threshold = 3
    min_quality_score = 0.6

    # Calculate average relevance score
    quality_score = 0.0
    if retrieved_docs:
        scores = [doc.get("score", 0.5) for doc in retrieved_docs if doc.get("score")]
        if scores:
            quality_score = sum(scores) / len(scores)
        else:
            quality_score = 0.5

    # Determine if results are sufficient
    sufficient_quantity = total_results >= min_results_threshold
    sufficient_quality = quality_score >= min_quality_score
    sufficient = sufficient_quantity and sufficient_quality

    # Generate assessment reason
    if sufficient:
        reason = f"Good results: {total_results} docs, avg relevance {quality_score:.2f}"
    else:
        issues = []
        if not sufficient_quantity:
            issues.append(f"only {total_results} results (need ≥{min_results_threshold})")
        if not sufficient_quality:
            issues.append(f"low relevance {quality_score:.2f} (need ≥{min_quality_score})")
        reason = "Insufficient: " + ", ".join(issues)

    # Store assessment
    state["quality_assessment"] = {
        "sufficient": sufficient,
        "score": quality_score,
        "reason": reason
    }

    logger.info(f"Quality assessment: sufficient={sufficient}, score={quality_score:.2f}")
    
    # Stream completion update
    writer({
        "step_id": step_id,
        "status": "complete",
        "brief_description": "Quality assessed",
        "description": f"Assessment complete: {reason}"
    })
    
    return state


def simple_quality_check(state: ResearchState) -> str:
    """
    Simple conditional edge for quality routing.
    """
    refinement_count = state.get("refinement_count", 0)
    quality_assessment = state.get("quality_assessment", {})
    
    # Maximum refinement attempts
    MAX_REFINEMENTS = 3
    
    if refinement_count >= MAX_REFINEMENTS:
        logger.info("Maximum refinements reached, ending workflow")
        return END
    
    if quality_assessment.get("sufficient", False):
        logger.info("Results sufficient, ending workflow")
        return END
    else:
        logger.info("Results insufficient, refining search")
        return "refine_analyzer"


async def refine_analyzer_node(state: ResearchState) -> ResearchState:
    """
    Analyze what went wrong and suggest refinements, then apply them.
    
    This focused node does LLM analysis and applies the minimal state updates.
    """
    step_id = f"refine_analyzer_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Analyzing refinement needs",
        "description": "Analyzing previous search results to understand issues and suggest improvements."
    })
    logger.info("Executing RefineAnalyzerNode")

    # Increment refinement count
    current_count = state.get("refinement_count", 0)
    state["refinement_count"] = current_count + 1

    query_analysis = state.get("query_analysis", {})
    quality_assessment = state.get("quality_assessment", {})
    search_history = state.get("search_history", [])

    # Create structured LLM for refinement analysis
    structured_llm = llm.with_structured_output(RefinementSuggestionOutput)

    system_prompt = """You are a legal research improvement specialist. Your ONLY job is to suggest refinements.

Previous search failed because: {reason}

Guidelines for improvement:
- If too few results: broaden terms, add synonyms, try related concepts
- If low relevance: use more specific legal terminology, focus on core concepts
- Consider alternative legal areas or related doctrines
- Suggest different search strategy if current one isn't working
- Do not provide more than 5 terms.

Provide specific, actionable improvements."""

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Original query: {state['query_text']}
Current concepts: {query_analysis.get('key_concepts', [])}
Assessment: {quality_assessment.get('reason', 'Unknown issue')}
Search history: {json.dumps(search_history[-2:], indent=2)}

Suggest specific improvements for the next search attempt.""")
    ])

    try:
        refinement_output = await structured_llm.ainvoke(prompt_template.format_messages())
        
        # Store refinement suggestion
        state["refinement_suggestion"] = {
            "new_terms": refinement_output.new_terms,
            "new_strategy": refinement_output.new_strategy,
            "reasoning": refinement_output.reasoning
        }
        
        # Apply refinements immediately
        state["query_analysis"]["key_concepts"] = refinement_output.new_terms
        if refinement_output.new_strategy:
            state["search_strategy"]["strategy"] = refinement_output.new_strategy
            state["search_strategy"]["reasoning"] = f"Refined strategy: {refinement_output.reasoning}"
        
        logger.info(f"Refinement suggested and applied: {refinement_output.reasoning}")
        
        # Stream completion update
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Refinements applied",
            "description": f"Applied refinements for attempt #{state['refinement_count']}: {refinement_output.reasoning}"
        })
        
    except Exception as e:
        logger.error(f"Error in refinement analysis: {e}")
        # Fallback refinement
        current_concepts = query_analysis.get("key_concepts", [])
        fallback_terms = current_concepts + ["legal", "case law"]
        state["refinement_suggestion"] = {
            "new_terms": fallback_terms,
            "new_strategy": "semantic",
            "reasoning": "Fallback refinement due to error"
        }
        state["query_analysis"]["key_concepts"] = fallback_terms

    return state


# Build the improved research graph
def create_research_graph() -> StateGraph:
    """
    Create and compile the improved research workflow graph v2.
    
    Returns:
        Compiled StateGraph with decomposed nodes for better LLM performance
    """
    workflow = StateGraph(ResearchState)

    # Add decomposed nodes
    workflow.add_node("query_analyzer", query_analyzer_node)
    workflow.add_node("strategy_selector", strategy_selector_node)
    workflow.add_node("filter_builder", filter_builder_node)
    workflow.add_node("query_preparation", query_preparation_node)
    workflow.add_node("search_execution", search_execution_node)
    workflow.add_node("quality_assessor", quality_assessor_node)
    workflow.add_node("refine_analyzer", refine_analyzer_node)

    # Linear flow for initial search
    workflow.set_entry_point("query_analyzer")
    workflow.add_edge("query_analyzer", "strategy_selector")
    workflow.add_edge("strategy_selector", "filter_builder")
    workflow.add_edge("filter_builder", "query_preparation")
    workflow.add_edge("query_preparation", "search_execution")
    workflow.add_edge("search_execution", "quality_assessor")

    # Conditional edge for quality assessment
    workflow.add_conditional_edges(
        "quality_assessor",
        simple_quality_check,
        {
            "refine_analyzer": "refine_analyzer",
            END: END
        }
    )

    # Refinement loop back to strategy selection
    workflow.add_edge("refine_analyzer", "strategy_selector")

    compiled_workflow = workflow.compile()

    return compiled_workflow


# Export the compiled graph
research_graph = create_research_graph()
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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json
import time
import uuid
import os

from app.core.config import settings
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

# Initialize LLM
research_attrs = settings.llm_config.get("main", {}).get("research", {})
llm = ChatGoogleGenerativeAI(google_api_key=settings.google_api_key, **research_attrs)


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


async def retrieval_node(state: ResearchState) -> ResearchState:
    """
    Execute the search using the selected strategy and filters.
    
    This node focuses only on retrieval execution.
    """
    step_id = f"retrieval_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Retrieving legal documents",
        "description": "Executing the search strategy to retrieve relevant legal documents and precedents."
    })
    logger.info("Executing RetrievalNode")

    # Get strategy and filters
    search_strategy = state.get("search_strategy", {})
    search_filters = state.get("search_filters", {})
    query_analysis = state.get("query_analysis", {})
    
    strategy = search_strategy.get("strategy", "semantic")
    key_concepts = query_analysis.get("key_concepts", [])
    
    # Build search parameters
    search_kwargs = {"limit": 15, "min_score": 0.6}
    
    # Apply filters
    # if search_filters.get("jurisdiction"):
    #     search_kwargs["jurisdiction"] = search_filters["jurisdiction"]
    # if search_filters.get("document_type"):
    #     search_kwargs["document_type"] = search_filters["document_type"]
    if search_filters.get("year_from"):
        search_kwargs["year_from"] = search_filters["year_from"]
    if search_filters.get("year_to"):
        search_kwargs["year_to"] = search_filters["year_to"]

    retrieved_docs = []

    try:
        if strategy == "semantic":
            # Use vector search with key concepts
            query_text = " ".join(key_concepts) if key_concepts else state["query_text"]
            results = await vector_search(query_text, **search_kwargs)
            retrieved_docs.extend(results)

        elif strategy == "fulltext":
            # Use fulltext search
            query_text = " ".join(key_concepts) if key_concepts else state["query_text"]
            results = fulltext_search(
                query_text,
                "chunks",
                **{k: v for k, v in search_kwargs.items() if k != "min_score"}
            )
            retrieved_docs.extend(results)

        elif strategy == "citation":
            # Use citation analysis
            original_query = state["query_text"]
            if any(pattern in original_query.lower() for pattern in ["v ", " v. ", "[", "]"]):
                results = find_case_citations(original_query, "both")
                retrieved_docs.extend(results)
            else:
                # Fallback to semantic if no citations detected
                results = await vector_search(original_query, **search_kwargs)
                retrieved_docs.extend(results)

        elif strategy == "concept":
            # Use concept-based search
            for concept in key_concepts:
                results = find_legal_concepts(
                    concept,
                    jurisdiction=search_kwargs.get("jurisdiction"),
                    limit=search_kwargs.get("limit", 15)
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
            "strategy": strategy,
            "terms": key_concepts,
            "filters": {k: v for k, v in search_kwargs.items() if k not in ["limit", "min_score"] and v is not None},
            "results_count": len(unique_docs),
            "timestamp": time.time()
        })
        state["search_history"] = search_history

        logger.info(f"Retrieved {len(unique_docs)} documents using {strategy} strategy")
        
        # Stream completion update
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Documents retrieved",
            "description": f"Found {len(unique_docs)} relevant documents using {strategy} search"
        })

    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
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
    Analyze what went wrong and suggest refinements.
    
    This focused node only analyzes and suggests - no strategy selection.
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
        
        # Update query analysis with new terms
        state["query_analysis"]["key_concepts"] = refinement_output.new_terms
        
        logger.info(f"Refinement suggested: {refinement_output.reasoning}")
        
        # Stream completion update
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Suggested refinements",
            "description": f"Refinements suggested: {refinement_output.reasoning}"
        })
        
    except Exception as e:
        logger.error(f"Error in refinement analysis: {e}")
        # Fallback refinement
        current_concepts = query_analysis.get("key_concepts", [])
        state["refinement_suggestion"] = {
            "new_terms": current_concepts + ["legal", "case law"],
            "new_strategy": "semantic",
            "reasoning": "Fallback refinement due to error"
        }

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
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("quality_assessor", quality_assessor_node)
    workflow.add_node("refine_analyzer", refine_analyzer_node)

    # Linear flow for initial search
    workflow.set_entry_point("query_analyzer")
    workflow.add_edge("query_analyzer", "strategy_selector")
    workflow.add_edge("strategy_selector", "filter_builder")
    workflow.add_edge("filter_builder", "retrieval")
    workflow.add_edge("retrieval", "quality_assessor")

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
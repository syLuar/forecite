"""
Research Graph - Legal Precedent Discovery with Refinement Loop

This module implements the research agent team that finds relevant legal precedents
with the ability to refine search queries when initial results are insufficient.
The graph uses conditional edges to model iterative research processes.
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

from google import genai

from app.core.config import settings
from app.core.llm import get_research_llm
from app.core.llm import create_llm

from .state import ResearchState, SearchPlan
from ..tools.neo4j_tools import (
    vector_search,
    fulltext_search,
    find_case_citations,
    find_legal_concepts,
    assess_precedent_strength,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM using the config for research
task_config = settings.llm_config.get("main", {}).get("research", {})
llm = create_llm(task_config)


# Structured output models
class SearchPlanOutput(BaseModel):
    """Structured output for search planning."""

    search_terms: List[str] = Field(description="List of key terms to search for")
    search_strategy: str = Field(
        description="Primary strategy to use: semantic (strongly preferred), fulltext, citation, concept"
    )
    # filters: Dict[str, Any] = Field(
    #     description="Filters to apply (jurisdiction, document_type, date_range)"
    # )
    expected_result_types: List[str] = Field(description="Expected types of results")
    confidence_level: float = Field(
        description="Confidence in this approach (0-1)", ge=0.0, le=1.0
    )
    rationale: Optional[str] = Field(description="Explanation of the search strategy")


class RefinementPlanOutput(BaseModel):
    """Structured output for search refinement."""

    search_terms: List[str] = Field(description="Improved list of search terms")
    search_strategy: str = Field(
        description="Strategy to use: semantic, fulltext, citation, concept"
    )
    # filters: Dict[str, Any] = Field(description="Updated filters object")
    rationale: str = Field(description="Explanation of changes made")
    confidence_level: float = Field(
        description="Confidence in new approach (0-1)", ge=0.0, le=1.0
    )


async def query_planner_node(state: ResearchState) -> ResearchState:
    """
    Generate an initial search plan based on the user's query.

    This node analyzes the user's query to determine the best search strategy,
    extract key terms, and set appropriate filters.
    """
    # Stream custom update
    writer = get_stream_writer()
    writer({
        "brief_description": "Planning legal research",
        "description": "Analyzing user query and generating an optimal search plan for legal precedent discovery."
    })
    logger.info("Executing QueryPlannerNode")

    query_text = state["query_text"]
    refinement_count = state.get("refinement_count", 0)

    # Create structured LLM
    structured_llm = llm.with_structured_output(SearchPlanOutput)

    # Check if this is a refinement iteration
    if refinement_count > 0:
        feedback = state.get("refinement_feedback", "")
        previous_plan = state.get("initial_plan", {})

        system_prompt = """You are a legal research strategist refining a search plan based on previous results.
        
        Previous search plan: {previous_plan}
        Feedback from assessment: {feedback}
        
        Create an improved search plan that addresses the identified issues."""

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Original query: {query_text}\n\nRefine the search plan to get better results."
                ),
            ]
        )

        plan_output = await structured_llm.ainvoke(
            prompt_template.format_messages(
                previous_plan=json.dumps(previous_plan, indent=2), feedback=feedback
            )
        )
    else:
        # Initial planning
        system_prompt = """You are a legal research strategist. Analyze the user's query and create an optimal search plan.

        Consider:
        1. What type of legal issue is this? (contract, tort, criminal, etc.)
        2. What jurisdiction might be relevant?
        3. What search strategy would work best? (semantic, fulltext, citation, concept)
        4. What key terms should be searched?

        Provide a comprehensive search plan with rationale."""

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Create a search plan for: {query_text}"),
            ]
        )

        plan_output = await structured_llm.ainvoke(prompt_template.format_messages())

    # Convert to dict for state storage
    plan = plan_output.model_dump()

    # Update state
    if refinement_count > 0:
        state["refined_plan"] = plan
    else:
        state["initial_plan"] = plan

    # Extract search parameters
    state["search_params"] = {
        "terms": plan["search_terms"],
        "strategy": plan["search_strategy"],
    }

    # Set filters
    # filters = plan["filters"]
    # if "jurisdiction" in filters:
    #     state["jurisdiction_filter"] = filters["jurisdiction"]
    # if "document_type" in filters:
    #     state["document_type_filter"] = filters["document_type"]
    # if "date_range" in filters:
    #     state["date_range"] = filters["date_range"]

    logger.info(f"Generated search plan: {plan['rationale']}")
    return state


async def retrieval_node(state: ResearchState) -> ResearchState:
    """
    Execute the search plan using appropriate tools.

    This node uses the Neo4j tools to retrieve relevant documents
    based on the current search plan.
    """
    # Stream custom update
    writer = get_stream_writer()
    writer({
        "brief_description": "Retrieving legal documents",
        "description": "Retrieving relevant legal documents and precedents using the current search plan and strategy."
    })
    logger.info("Executing RetrievalNode")

    search_params = state.get("search_params", {})
    strategy = search_params.get("strategy", "semantic")
    terms = search_params.get("terms", [])

    # Prepare search arguments
    search_kwargs = {"limit": 15, "min_score": 0.6}

    # Add filters if present
    if "jurisdiction_filter" in state:
        search_kwargs["jurisdiction"] = state["jurisdiction_filter"]
    if "document_type_filter" in state:
        search_kwargs["document_type"] = state["document_type_filter"]
    if "date_range" in state:
        date_range = state["date_range"]
        if "from" in date_range:
            search_kwargs["year_from"] = date_range["from"]
        if "to" in date_range:
            search_kwargs["year_to"] = date_range["to"]

    retrieved_docs = []

    try:
        if strategy == "semantic":
            # Use vector search
            query_text = " ".join(terms) if isinstance(terms, list) else str(terms)
            results = await vector_search(query_text, **search_kwargs)
            retrieved_docs.extend(results)

        elif strategy == "fulltext":
            # Use fulltext search
            query_text = " ".join(terms) if isinstance(terms, list) else str(terms)
            results = fulltext_search(
                query_text,
                "chunks",
                **{k: v for k, v in search_kwargs.items() if k != "min_score"},
            )
            retrieved_docs.extend(results)

        elif strategy == "citation":
            # Use citation analysis for specific cases
            for term in terms:
                if any(pattern in term.lower() for pattern in ["v ", " v. ", "[", "]"]):
                    # Looks like a case citation
                    results = find_case_citations(term, "both")
                    retrieved_docs.extend(results)

        elif strategy == "concept":
            # Use concept-based search
            for term in terms:
                results = find_legal_concepts(
                    term,
                    jurisdiction=search_kwargs.get("jurisdiction"),
                    limit=search_kwargs.get("limit", 15),
                )
                retrieved_docs.extend(results)

        # If primary strategy didn't yield enough results, try semantic as fallback
        if len(retrieved_docs) < 5 and strategy != "semantic":
            logger.info(
                "Primary strategy yielded few results, trying semantic search as fallback"
            )
            query_text = state["query_text"]
            fallback_results = await vector_search(query_text, **search_kwargs)
            retrieved_docs.extend(fallback_results)

    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        retrieved_docs = []

    # Remove duplicates and sort by relevance
    seen_ids = set()
    unique_docs = []
    for doc in retrieved_docs:
        doc_id = doc.get("chunk_id") or doc.get("document_source") or str(doc)
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_docs.append(doc)

    # Sort by score if available
    unique_docs.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Update state
    state["retrieved_docs"] = unique_docs[:15]  # Limit to top 15 results
    state["total_results"] = len(unique_docs)

    # Add to search history
    search_history = state.get("search_history", [])
    search_history.append(
        {
            "strategy": strategy,
            "terms": terms,
            # "filters": {
            #     k: v
            #     for k, v in search_kwargs.items()
            #     if k not in ["limit", "min_score"]
            # },
            "results_count": len(unique_docs),
            "timestamp": time.time(),
        }
    )
    state["search_history"] = search_history

    logger.info(f"Retrieved {len(unique_docs)} documents using {strategy} strategy")
    return state


def assess_retrieval_node(state: ResearchState) -> str:
    """
    Assess the quality of retrieved results and decide next action.

    This is a conditional edge that determines whether to end the workflow
    or refine the search query based on result quality.
    """
    logger.info("Executing AssessRetrievalNode")

    retrieved_docs = state.get("retrieved_docs", [])
    total_results = state.get("total_results", 0)
    refinement_count = state.get("refinement_count", 0)

    # Maximum refinement attempts
    MAX_REFINEMENTS = 3

    # If we've hit the refinement limit, accept current results
    if refinement_count >= MAX_REFINEMENTS:
        state["retrieval_sufficient"] = True
        state["assessment_reason"] = (
            f"Maximum refinement attempts ({MAX_REFINEMENTS}) reached"
        )
        logger.info("Maximum refinements reached, ending workflow")
        return END

    # Quality assessment criteria
    min_results_threshold = 3
    min_quality_score = 0.6

    # Calculate quality metrics
    quality_score = 0.0
    if retrieved_docs:
        # Score based on relevance scores
        scores = [doc.get("score", 0.5) for doc in retrieved_docs if doc.get("score")]
        if scores:
            quality_score = sum(scores) / len(scores)
        else:
            quality_score = 0.5  # Default if no scores available

    # Assess whether results are sufficient
    sufficient_quantity = total_results >= min_results_threshold
    sufficient_quality = quality_score >= min_quality_score

    state["search_quality_score"] = quality_score

    if sufficient_quantity and sufficient_quality:
        state["retrieval_sufficient"] = True
        state["assessment_reason"] = (
            f"Good results: {total_results} docs, avg score {quality_score:.2f}"
        )
        logger.info(
            f"Results sufficient: {total_results} docs with quality score {quality_score:.2f}"
        )
        return END
    else:
        state["retrieval_sufficient"] = False

        # Generate feedback for refinement
        feedback_parts = []
        if not sufficient_quantity:
            feedback_parts.append(
                f"Only found {total_results} results (need at least {min_results_threshold})"
            )
        if not sufficient_quality:
            feedback_parts.append(
                f"Low relevance scores (avg {quality_score:.2f}, need at least {min_quality_score})"
            )

        # Add specific suggestions
        if quality_score < 0.3:
            feedback_parts.append(
                "Consider trying different search terms or broader concepts"
            )
        elif total_results < 2:
            feedback_parts.append(
                "Try expanding search to related legal areas or removing restrictive filters"
            )

        state["assessment_reason"] = "; ".join(feedback_parts)
        state["refinement_feedback"] = "; ".join(feedback_parts)

        logger.info(
            f"Results insufficient, refining search: {state['assessment_reason']}"
        )
        return "refine_query"


async def query_refiner_node(state: ResearchState) -> ResearchState:
    """
    Refine the search query based on assessment feedback.

    This node takes the feedback from assessment and creates an improved
    search strategy to get better results.
    """
    # Stream custom update
    writer = get_stream_writer()
    writer({
        "brief_description": "Refining search strategy",
        "description": "Refining the legal research strategy and search terms based on feedback from previous retrieval attempts."
    })
    logger.info("Executing QueryRefinerNode")

    current_count = state.get("refinement_count", 0)
    state["refinement_count"] = current_count + 1

    query_text = state["query_text"]
    feedback = state.get("refinement_feedback", "")
    previous_results = state.get("retrieved_docs", [])
    search_history = state.get("search_history", [])

    # Create structured LLM
    structured_llm = llm.with_structured_output(RefinementPlanOutput)

    system_prompt = """You are a legal research expert refining a search strategy.

    Previous attempts have not yielded sufficient results. Analyze what went wrong and suggest improvements.

    Guidelines for refinement:
    1. If too few results: broaden search terms, try synonyms, expand to related concepts
    2. If low relevance: make search terms more specific, add legal terminology
    3. If wrong document types: adjust filters or search strategy
    4. Consider alternative legal areas or jurisdictions
    5. Try different search strategies (semantic vs fulltext vs concept-based)

    Provide a clear rationale for your changes."""

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"""
Original query: {query_text}

Assessment feedback: {feedback}

Search history: {json.dumps(search_history, indent=2)}

Number of results from last attempt: {len(previous_results)}

Create an improved search plan that addresses these issues.
        """
            ),
        ]
    )

    refined_plan_output = await structured_llm.ainvoke(prompt_template.format_messages())
    refined_plan = refined_plan_output.model_dump()

    # Update search parameters
    state["search_params"] = {
        "terms": refined_plan["search_terms"],
        "strategy": refined_plan["search_strategy"],
    }

    # Update filters
    # filters = refined_plan["filters"]
    # if "jurisdiction" in filters:
    #     state["jurisdiction_filter"] = filters["jurisdiction"]
    # if "document_type" in filters:
    #     state["document_type_filter"] = filters["document_type"]
    # if "date_range" in filters:
    #     state["date_range"] = filters["date_range"]

    # Store refinement rationale
    state["refined_plan"] = refined_plan

    logger.info(f"Refined search plan: {refined_plan['rationale']}")
    return state


# Build the research graph
def create_research_graph() -> StateGraph:
    """
    Create and compile the research workflow graph.

    Returns:
        Compiled StateGraph for legal research with refinement loop
    """
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("query_planner", query_planner_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("query_refiner", query_refiner_node)

    # Add edges
    workflow.set_entry_point("query_planner")
    workflow.add_edge("query_planner", "retrieval")
    workflow.add_edge("query_refiner", "retrieval")

    # Add conditional edge for assessment
    workflow.add_conditional_edges(
        "retrieval", assess_retrieval_node, {"refine_query": "query_refiner", END: END}
    )

    return workflow.compile()


# Export the compiled graph
research_graph = create_research_graph()

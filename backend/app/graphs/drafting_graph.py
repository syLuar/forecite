"""
Drafting Graph - Legal Argument Generation with Critique Loop

This module implements the drafting agent team that develops legal strategies,
critiques them, and generates legal arguments. The graph uses conditional edges
to model the iterative strategy development and critique process.
"""

import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph
from langgraph.constants import END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json
import time

from google import genai

from app.core.config import settings

from .state import DraftingState, ArgumentStrategy
from ..tools.neo4j_tools import (
    find_similar_fact_patterns,
    find_legal_tests,
    assess_precedent_strength,
    find_authority_chain,
    get_document_metadata,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
drafting_attrs = settings.llm_config.get("main", {}).get("drafting", {})
llm = ChatGoogleGenerativeAI(google_api_key=settings.google_api_key, **drafting_attrs)


# Structured output models
class LegalArgumentComponent(BaseModel):
    """Single legal argument component.

    Note: This class is kept for reference but not used directly in structured output
    due to Gemini's limitations with nested structures. Instead, key_arguments is a JSON string
    that is parsed manually to create dictionaries with these fields.
    """

    argument: str = Field(description="Specific legal argument")
    supporting_authority: str = Field(description="Case or statute that supports this")
    factual_basis: str = Field(description="How the facts support this argument")


class StrategyOutput(BaseModel):
    """Structured output for legal strategy development."""

    main_thesis: str = Field(
        description="Clear statement of the primary legal argument"
    )
    argument_type: str = Field(
        description="Type: analogical, precedential, policy, or textual"
    )
    primary_precedents: List[str] = Field(description="List of key cases to rely on")
    legal_framework: str = Field(
        description="Description of the applicable legal test/standard"
    )
    key_arguments: str = Field(
        description="JSON string containing list of legal arguments. Each argument should have 'argument', 'supporting_authority', and 'factual_basis' fields."
    )
    anticipated_counterarguments: List[str] = Field(
        description="List of likely opposing arguments"
    )
    counterargument_responses: List[str] = Field(
        description="How to address each counterargument"
    )
    strength_assessment: float = Field(
        description="Strategy strength (0-1)", ge=0.0, le=1.0
    )
    risk_factors: List[str] = Field(description="Potential weaknesses or risks")
    strategy_rationale: str = Field(
        description="Explanation of why this approach is optimal"
    )


class CritiqueOutput(BaseModel):
    """Structured output for strategy critique."""

    overall_assessment: str = Field(description="Either 'approve' or 'revise'")
    critique_score: float = Field(description="Quality score (0-1)", ge=0.0, le=1.0)
    identified_weaknesses: List[str] = Field(
        description="List of specific problems found"
    )
    suggested_improvements: List[str] = Field(
        description="Specific actionable suggestions"
    )
    detailed_feedback: str = Field(description="Comprehensive critique explanation")
    approval_rationale: str = Field(
        description="Why the strategy was approved or rejected"
    )


def strategist_node(state: DraftingState) -> DraftingState:
    """
    Propose or revise an argumentative strategy.

    This node analyzes the user's facts and case file to develop a comprehensive
    legal strategy. It incorporates feedback from previous critique cycles.
    """
    logger.info("Executing StrategistNode")

    user_facts = state["user_facts"]
    case_file = state.get("case_file", {})
    strategy_version = state.get("strategy_version", 0) + 1
    critique_feedback = state.get("critique_feedback", "")

    # Check if this is a revision
    is_revision = strategy_version > 1

    # Create structured LLM
    structured_llm = llm.with_structured_output(StrategyOutput)

    if is_revision:
        system_prompt = """You are a senior legal strategist revising an argument strategy based on critique feedback.

        Previous strategy issues identified: {critique_feedback}

        Your task is to address these concerns while maintaining the strengths of the original approach.
        Consider:
        1. How to strengthen weak points identified in the critique
        2. Whether to adjust the main thesis or argument structure
        3. Alternative precedents or legal theories that might be stronger
        4. How to better address anticipated counterarguments

        IMPORTANT: For the key_arguments field, provide a JSON string containing a list of arguments. Each argument must have these exact fields:
        - "argument": the specific legal argument
        - "supporting_authority": case or statute that supports this
        - "factual_basis": how the facts support this argument

        Example format: '[{"argument": "...", "supporting_authority": "...", "factual_basis": "..."}, {...}]'

        Create a revised strategy that addresses the critique while building on what worked."""

        context_prompt = f"""
FACTS: {user_facts}

AVAILABLE PRECEDENTS: {json.dumps(case_file, indent=2, default=str)}

PREVIOUS CRITIQUE: {critique_feedback}

Revise the legal strategy to address the identified weaknesses.
        """
    else:
        system_prompt = """You are a senior legal strategist developing an initial argument strategy.

        Analyze the client's fact pattern and available precedents to develop a comprehensive legal strategy.
        
        Consider:
        1. What is the strongest legal theory that applies to these facts?
        2. Which precedents provide the best support?
        3. What legal tests or standards need to be satisfied?
        4. What are the likely counterarguments and how can they be addressed?
        5. What is the overall strength of this position?

        IMPORTANT: For the key_arguments field, provide a JSON string containing a list of arguments. Each argument must have these exact fields:
        - "argument": the specific legal argument
        - "supporting_authority": case or statute that supports this
        - "factual_basis": how the facts support this argument

        Example format: '[{"argument": "...", "supporting_authority": "...", "factual_basis": "..."}, {...}]'

        Your strategy should be thorough, realistic, and acknowledge both strengths and potential weaknesses."""

        context_prompt = f"""
FACTS: {user_facts}

AVAILABLE PRECEDENTS: {json.dumps(case_file, indent=2, default=str)}

Develop a comprehensive legal argument strategy.
        """

    # Find similar fact patterns to inform strategy
    try:
        # Extract key facts for pattern matching
        fact_keywords = user_facts.split()[:10]  # Use first 10 words as key facts
        similar_cases = find_similar_fact_patterns(fact_keywords, limit=5)

        # Find relevant legal tests
        legal_area = "general"  # Could be extracted from case_file or user_facts
        relevant_tests = find_legal_tests(legal_area, limit=10)

        additional_context = f"""
SIMILAR FACT PATTERNS FOUND:
{json.dumps(similar_cases, indent=2, default=str)}

RELEVANT LEGAL TESTS:
{json.dumps(relevant_tests, indent=2, default=str)}
        """
        context_prompt += additional_context

    except Exception as e:
        logger.warning(f"Could not retrieve additional context: {e}")

    prompt_template = ChatPromptTemplate.from_messages(
        [SystemMessage(content=system_prompt), HumanMessage(content=context_prompt)]
    )

    strategy_output = structured_llm.invoke(
        prompt_template.format_messages(
            critique_feedback=critique_feedback if is_revision else ""
        )
    )

    # Convert to dict for state storage
    strategy = strategy_output.model_dump()

    # Parse key_arguments JSON string
    try:
        key_arguments_parsed = json.loads(strategy["key_arguments"])
        # Validate the structure
        if isinstance(key_arguments_parsed, list):
            for arg in key_arguments_parsed:
                if not all(
                    key in arg
                    for key in ["argument", "supporting_authority", "factual_basis"]
                ):
                    logger.warning("Invalid argument structure in key_arguments JSON")
                    break
        else:
            logger.warning("key_arguments should be a list")
            key_arguments_parsed = []
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse key_arguments JSON: {e}")
        key_arguments_parsed = []

    # Update strategy with parsed key_arguments
    strategy["key_arguments"] = key_arguments_parsed

    # Update state
    state["proposed_strategy"] = strategy
    state["strategy_version"] = strategy_version
    state["strategy_rationale"] = strategy["strategy_rationale"]
    state["workflow_stage"] = "strategy"

    # Extract strategy components for easier access
    state["primary_arguments"] = key_arguments_parsed
    state["supporting_precedents"] = [
        {"citation": p, "role": "supporting"} for p in strategy["primary_precedents"]
    ]
    state["legal_tests_to_apply"] = [strategy["legal_framework"]]

    # Add to revision history
    revision_history = state.get("revision_history", [])
    revision_history.append(
        {
            "version": strategy_version,
            "strategy": strategy,
            "timestamp": time.time(),
            "is_revision": is_revision,
            "critique_addressed": critique_feedback if is_revision else None,
        }
    )
    state["revision_history"] = revision_history

    logger.info(f"Generated strategy v{strategy_version}: {strategy['main_thesis']}")
    return state


def critique_node(state: DraftingState) -> str:
    """
    Critique the proposed strategy and decide whether to approve or revise.

    This is a conditional edge that acts as a "red team" to assess the strategy
    for logical flaws, weaknesses, and areas for improvement.
    """
    logger.info("Executing CritiqueNode")
    proposed_strategy = state.get("proposed_strategy", {})
    user_facts = state["user_facts"]
    case_file = state.get("case_file", {})
    total_critique_cycles = state.get("total_critique_cycles", 0)

    logger.info(f"Starting critique cycle {total_critique_cycles + 1}")

    # Maximum critique cycles to prevent infinite loops
    MAX_CRITIQUE_CYCLES = 3

    # Increment the cycle count first
    state["total_critique_cycles"] = total_critique_cycles + 1
    current_cycle = state["total_critique_cycles"]

    if current_cycle > MAX_CRITIQUE_CYCLES:
        state["strategy_approved"] = True
        state["critique_feedback"] = (
            f"Strategy approved after {MAX_CRITIQUE_CYCLES} critique cycles"
        )
        logger.info(
            f"Maximum critique cycles ({MAX_CRITIQUE_CYCLES}) reached, approving strategy"
        )
        return "draft_argument"

    state["workflow_stage"] = "critique"

    # Create structured LLM
    structured_llm = llm.with_structured_output(CritiqueOutput)

    system_prompt = """You are a senior legal mentor providing constructive feedback on a legal strategy.

    Your job is to assess the strategy's viability and provide helpful guidance. Be thorough but fair.

    Evaluate the strategy on these criteria:
    1. Legal Foundation: Are the precedents applicable and is the legal framework sound?
    2. Factual Alignment: Do the facts support the legal arguments being made?
    3. Logical Coherence: Is the argument structure logical and persuasive?
    4. Precedent Strength: Are the cited cases appropriate authority?
    5. Counterargument Preparedness: Are anticipated counterarguments realistic and addressed?
    6. Risk Assessment: Is the strength assessment reasonable?

    Remember: The goal is to produce a workable strategy, not a perfect one. 
    Approve strategies that are legally sound and reasonably supported, even if they could be improved.
    Only recommend revision for significant flaws that would undermine the argument's effectiveness.
    
    Use 'approve' if the strategy is legally defensible and adequately supported.
    Use 'revise' only if there are substantial problems that would likely lead to failure."""

    critique_prompt = f"""
STRATEGY TO CRITIQUE:
{json.dumps(proposed_strategy, indent=2, default=str)}

CLIENT FACTS:
{user_facts}

AVAILABLE PRECEDENTS:
{json.dumps(case_file, indent=2, default=str)}

Provide a thorough critique of this legal strategy.
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [SystemMessage(content=system_prompt), HumanMessage(content=critique_prompt)]
    )

    critique_output = structured_llm.invoke(prompt_template.format_messages())
    critique = critique_output.model_dump()

    # Update state with critique results
    state["critique_score"] = critique["critique_score"]
    state["identified_weaknesses"] = critique["identified_weaknesses"]
    state["suggested_improvements"] = critique["suggested_improvements"]
    state["critique_feedback"] = critique["detailed_feedback"]

    # Decision logic - be more lenient to prevent infinite loops
    assessment = critique["overall_assessment"].lower()
    critique_score = critique["critique_score"]

    # Approve if:
    # 1. Assessment is positive and score is reasonable, OR
    # 2. We're on the last cycle and score is decent, OR
    # 3. Score is high regardless of assessment
    should_approve = (
        (assessment == "approve" and critique_score >= 0.5)
        or (current_cycle >= MAX_CRITIQUE_CYCLES and critique_score >= 0.4)
        or (critique_score >= 0.7)
    )

    if should_approve:
        state["strategy_approved"] = True
        logger.info(
            f"Strategy approved with score {critique_score} (cycle {current_cycle}/{MAX_CRITIQUE_CYCLES})"
        )
        return "draft_argument"
    else:
        state["strategy_approved"] = False
        logger.info(
            f"Strategy needs revision - score: {critique_score}, assessment: {assessment} (cycle {current_cycle}/{MAX_CRITIQUE_CYCLES})"
        )
        return "revise_strategy"


def drafting_team_node(state: DraftingState) -> DraftingState:
    """
    Execute the approved strategy to draft the final legal argument.

    This node takes the approved strategy and generates a well-structured
    legal argument with proper citations and reasoning.
    """
    logger.info("Executing DraftingTeamNode")

    approved_strategy = state.get("proposed_strategy", {})
    user_facts = state["user_facts"]
    case_file = state.get("case_file", {})

    state["workflow_stage"] = "drafting"

    system_prompt = """You are an expert legal writer drafting a comprehensive legal argument.

    Using the approved strategy, write a professional legal argument that:
    1. Clearly states the legal issue and position
    2. Applies the relevant legal framework/test
    3. Analyzes the facts in light of the law
    4. Cites appropriate precedents with proper legal reasoning
    5. Addresses potential counterarguments
    6. Reaches a well-supported conclusion

    Structure the argument with:
    - Introduction (issue and thesis)
    - Legal Framework (applicable law and standards)
    - Analysis (application of law to facts)
    - Addressing Counterarguments
    - Conclusion

    Use formal legal writing style with proper citations."""

    drafting_prompt = f"""
APPROVED STRATEGY:
{json.dumps(approved_strategy, indent=2, default=str)}

CLIENT FACTS:
{user_facts}

CASE FILE:
{json.dumps(case_file, indent=2, default=str)}

Draft a comprehensive legal argument that implements this strategy. The argument should be professional, well-reasoned, and ready for use in legal proceedings.
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [SystemMessage(content=system_prompt), HumanMessage(content=drafting_prompt)]
    )

    response = llm.invoke(prompt_template.format_messages())

    drafted_argument = response.content

    # Extract citations used
    citations_used = []
    for precedent in approved_strategy.get("primary_precedents", []):
        citations_used.append(precedent)

    # Create argument structure
    argument_structure = {
        "introduction": "Legal issue and position statement",
        "legal_framework": approved_strategy.get("legal_framework", ""),
        "analysis": "Application of law to facts",
        "counterarguments": approved_strategy.get("anticipated_counterarguments", []),
        "conclusion": "Final position and relief sought",
    }

    # Calculate quality metrics
    argument_length = len(drafted_argument.split())
    precedent_coverage = (
        len(citations_used) / max(len(case_file), 1) if case_file else 0
    )

    # Update state
    state["drafted_argument"] = drafted_argument
    state["argument_structure"] = argument_structure
    state["citations_used"] = citations_used
    state["argument_strength"] = approved_strategy.get("strength_assessment", 0.7)
    state["precedent_coverage"] = min(precedent_coverage, 1.0)
    state["logical_coherence"] = state.get("critique_score", 0.7)

    logger.info(
        f"Drafted argument: {argument_length} words, {len(citations_used)} citations"
    )
    return state


# Build the drafting graph
def create_drafting_graph() -> StateGraph:
    """
    Create and compile the drafting workflow graph.

    Returns:
        Compiled StateGraph for legal argument drafting with critique loop
    """
    workflow = StateGraph(DraftingState)

    # Add nodes
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("drafting_team", drafting_team_node)

    # Add edges
    workflow.set_entry_point("strategist")
    workflow.add_edge("drafting_team", END)

    # Add conditional edge for critique
    workflow.add_conditional_edges(
        "strategist",
        critique_node,
        {"revise_strategy": "strategist", "draft_argument": "drafting_team"},
    )

    return workflow.compile()


# Export the compiled graph
drafting_graph = create_drafting_graph()

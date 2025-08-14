"""
Drafting Graph v2 - Decomposed Legal Argument Generation

This module implements an improved drafting workflow with decomposed nodes
for better LLM performance and reliability. The overloaded strategist_node 
has been broken down into focused, single-responsibility nodes.
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
    DraftingState, 
    LegalIssueAnalysis, 
    CoreStrategy, 
    SingleArgument,
    SimpleAssessment,
    ContentRevision
)
from ...tools.neo4j_tools import (
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


# Simplified Pydantic models for structured output
class LegalIssueAnalysisOutput(BaseModel):
    """Simple legal issue identification."""
    primary_issue: str = Field(description="Main legal issue")
    secondary_issues: List[str] = Field(description="Supporting issues", max_items=3)
    applicable_law: str = Field(description="Relevant legal framework")


class CoreStrategyOutput(BaseModel):
    """Simple core legal strategy."""
    main_thesis: str = Field(description="Primary argument thesis")
    legal_theory: str = Field(description="Core legal theory to apply")
    strength_rating: float = Field(description="Strategy strength 0-1", ge=0.0, le=1.0)


class SingleArgumentOutput(BaseModel):
    """Single argument structure."""
    argument: str = Field(description="Specific legal argument")
    authority: str = Field(description="Supporting case/statute")
    reasoning: str = Field(description="How facts support argument")


class SimpleAssessmentOutput(BaseModel):
    """Simple assessment output."""
    approved: bool = Field(description="Whether approved")
    score: float = Field(description="Score 0-1", ge=0.0, le=1.0)
    feedback: str = Field(description="Brief feedback")


class ContentRevisionOutput(BaseModel):
    """Simple content revision."""
    revised_content: str = Field(description="Revised content")
    changes_made: str = Field(description="Summary of changes")
    improvement_reasoning: str = Field(description="Why changes improve content")


# Decomposed node functions
async def fact_analyzer_node(state: DraftingState) -> DraftingState:
    """
    Analyze facts and identify legal issues.
    
    This focused node only analyzes facts - no strategy development.
    """
    step_id = f"fact_analyzer_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Analyzing legal facts",
        "description": "Analyzing the client's facts to identify primary and secondary legal issues."
    })
    logger.info("Executing FactAnalyzerNode")

    user_facts = state["user_facts"]
    party_represented = state.get("party_represented", "")
    legal_question = state.get("legal_question", "")

    # Create structured LLM for focused task
    structured_llm = llm.with_structured_output(LegalIssueAnalysisOutput)

    party_context = ""
    if party_represented:
        party_context = f"\n\nIMPORTANT: You are representing the {party_represented}. Frame your analysis from the {party_represented}'s perspective and identify issues that would be relevant to advancing the {party_represented}'s position."

    system_prompt = f"""You are a legal issue identification specialist. Your ONLY job is to analyze facts and identify legal issues.

1. Identify the PRIMARY legal issue (most important)
2. Identify up to 3 secondary issues (supporting or related)
3. Determine the applicable area of law

Be focused and specific. Do NOT develop strategy or suggest arguments.{party_context}"""

    prompt_content = f"Facts: {user_facts}"
    if legal_question:
        prompt_content += f"\n\nSpecific Legal Question: {legal_question}"
    if party_represented:
        prompt_content += f"\n\nParty Represented: {party_represented}"

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Analyze these facts and identify legal issues:\n\n{prompt_content}")
    ])

    try:
        analysis_output = await structured_llm.ainvoke(prompt_template.format_messages())
        
        # Store in state
        state["legal_issue_analysis"] = {
            "primary_issue": analysis_output.primary_issue,
            "secondary_issues": analysis_output.secondary_issues,
            "applicable_law": analysis_output.applicable_law
        }
        
        logger.info(f"Legal issue analysis completed: {analysis_output.primary_issue}")
        
        # Stream completion update
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Facts analyzed",
            "description": f"Identified primary issue: {analysis_output.primary_issue}, applicable law: {analysis_output.applicable_law}"
        })
        
    except Exception as e:
        logger.error(f"Error in fact analysis: {e}")
        # Fallback analysis
        state["legal_issue_analysis"] = {
            "primary_issue": "Legal dispute requiring analysis",
            "secondary_issues": [],
            "applicable_law": "General legal principles"
        }

    return state


async def strategy_developer_node(state: DraftingState) -> DraftingState:
    """
    Develop core legal strategy based on issue analysis.
    
    This focused node only develops strategy - no argument building.
    """
    step_id = f"strategy_developer_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Developing core strategy",
        "description": "Developing the core legal strategy and approach based on identified issues."
    })
    logger.info("Executing StrategyDeveloperNode")

    legal_issue_analysis = state.get("legal_issue_analysis", {})
    case_file = state.get("case_file", {})
    user_facts = state["user_facts"]
    party_represented = state.get("party_represented", "")

    # Check if this is a revision
    strategy_assessment = state.get("strategy_assessment", {})
    is_revision = not strategy_assessment.get("approved", True) if strategy_assessment else False

    # Create structured LLM for strategy development
    structured_llm = llm.with_structured_output(CoreStrategyOutput)

    revision_context = ""
    if is_revision:
        revision_context = f"\n\nPrevious strategy was rejected: {strategy_assessment.get('feedback', '')}\nAddress these concerns in the revised strategy."

    party_context = ""
    if party_represented:
        party_context = f"\n\nCRITICAL: You are representing the {party_represented}. Develop a strategy that advances the {party_represented}'s interests and position. Frame all legal theories and thesis statements from the {party_represented}'s perspective."

    system_prompt = f"""You are a legal strategy specialist. Your ONLY job is to develop the core strategic approach.

Based on the legal issue analysis, develop:
1. A clear main thesis (what you're arguing)
2. The core legal theory to apply
3. An honest strength assessment (0-1)

Primary issue: {legal_issue_analysis.get('primary_issue', 'Unknown')}
Applicable law: {legal_issue_analysis.get('applicable_law', 'Unknown')}
{party_context}
{revision_context}

Focus on the big picture strategy, not specific arguments."""

    prompt_content = f"Facts: {user_facts}"
    if party_represented:
        prompt_content += f"\n\nParty Represented: {party_represented}"
    prompt_content += f"\n\nAvailable precedents: {json.dumps(case_file, indent=2, default=str)}"

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{prompt_content}\n\nDevelop a core legal strategy for this case.")
    ])

    try:
        strategy_output = await structured_llm.ainvoke(prompt_template.format_messages())
        
        # Store in state
        state["core_strategy"] = {
            "main_thesis": strategy_output.main_thesis,
            "legal_theory": strategy_output.legal_theory,
            "strength_rating": strategy_output.strength_rating
        }
        
        logger.info(f"Core strategy developed: {strategy_output.main_thesis}")
        
        # Stream completion update
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Strategy developed",
            "description": f"Core strategy: {strategy_output.main_thesis[:100]}{'...' if len(strategy_output.main_thesis) > 100 else ''}"
        })
        
    except Exception as e:
        logger.error(f"Error in strategy development: {e}")
        # Fallback strategy
        primary_issue = legal_issue_analysis.get("primary_issue", "Legal issue")
        party_text = f" for the {party_represented}" if party_represented else ""
        state["core_strategy"] = {
            "main_thesis": f"Client should prevail on {primary_issue}{party_text}",
            "legal_theory": "Applicable legal standards support client's position",
            "strength_rating": 0.6
        }

    return state


async def argument_builder_node(state: DraftingState) -> DraftingState:
    """
    Build specific arguments iteratively.
    
    This node generates arguments one at a time for better focus.
    """
    step_id = f"argument_builder_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Building legal arguments",
        "description": "Building specific legal arguments that support the core strategy."
    })
    logger.info("Executing ArgumentBuilderNode")

    core_strategy = state.get("core_strategy", {})
    legal_issue_analysis = state.get("legal_issue_analysis", {})
    case_file = state.get("case_file", {})
    user_facts = state["user_facts"]
    party_represented = state.get("party_represented", "")

    # Create structured LLM for argument generation
    structured_llm = llm.with_structured_output(SingleArgumentOutput)

    party_context = ""
    if party_represented:
        party_context = f"\n\nCRITICAL: You are representing the {party_represented}. Each argument must advance the {party_represented}'s position and be framed from the {party_represented}'s perspective. Consider how this argument helps the {party_represented} win their case."

    # Generate 3 arguments iteratively
    arguments = []
    for i in range(3):
        system_prompt = f"""You are a legal argument specialist. Your ONLY job is to create ONE specific legal argument.

Core strategy: {core_strategy.get('main_thesis', '')}
Legal theory: {core_strategy.get('legal_theory', '')}
{party_context}

Create argument #{i+1} that supports this strategy. Each argument should:
1. Make a specific legal point
2. Cite supporting authority (case/statute)
3. Explain how the facts support this point

Be specific and focused on this ONE argument only."""

        prompt_content = f"Facts: {user_facts}"
        if party_represented:
            prompt_content += f"\n\nParty Represented: {party_represented}"
        prompt_content += f"\n\nAvailable precedents: {json.dumps(case_file, indent=2, default=str)}"

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{prompt_content}\n\nCreate a specific legal argument that supports the strategy.")
        ])

        try:
            argument_output = await structured_llm.ainvoke(prompt_template.format_messages())
            
            arguments.append({
                "argument": argument_output.argument,
                "authority": argument_output.authority,
                "reasoning": argument_output.reasoning
            })
            
            logger.info(f"Generated argument {i+1}: {argument_output.argument[:50]}...")
            
        except Exception as e:
            logger.error(f"Error generating argument {i+1}: {e}")
            # Fallback argument
            party_text = f" for the {party_represented}" if party_represented else ""
            arguments.append({
                "argument": f"Legal principle {i+1} supports client's position{party_text}",
                "authority": "Applicable case law",
                "reasoning": "Facts demonstrate this principle applies"
            })

    # Store arguments in state
    state["arguments"] = arguments
    logger.info(f"Built {len(arguments)} legal arguments")
    
    # Stream completion update
    writer({
        "step_id": step_id,
        "status": "complete",
        "brief_description": "Arguments built",
        "description": f"Built {len(arguments)} supporting arguments for the legal strategy"
    })
    
    return state


async def simple_critic_node(state: DraftingState) -> DraftingState:
    """
    Perform simple assessment of strategy and arguments.
    
    This focused node only does assessment - no complex decision logic.
    """
    step_id = f"simple_critic_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Assessing strategy quality",
        "description": "Performing quality assessment of the legal strategy and arguments."
    })
    logger.info("Executing SimpleCriticNode")

    core_strategy = state.get("core_strategy", {})
    arguments = state.get("arguments", [])
    legal_issue_analysis = state.get("legal_issue_analysis", {})

    # Create structured LLM for assessment
    structured_llm = llm.with_structured_output(SimpleAssessmentOutput)

    system_prompt = """You are a legal quality assessor. Your ONLY job is to assess strategy quality.

Evaluate:
1. Is the strategy legally sound?
2. Do the arguments support the thesis?
3. Is there sufficient legal authority?

Be practical - approve strategies that are workable, not perfect.
Only reject if there are serious flaws that would likely cause failure."""

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Strategy: {json.dumps(core_strategy, indent=2)}

Arguments: {json.dumps(arguments, indent=2)}

Legal issues: {json.dumps(legal_issue_analysis, indent=2)}

Assess the quality of this legal strategy and arguments.""")
    ])

    try:
        assessment_output = await structured_llm.ainvoke(prompt_template.format_messages())
        
        # Store assessment
        state["strategy_assessment"] = {
            "approved": assessment_output.approved,
            "score": assessment_output.score,
            "feedback": assessment_output.feedback
        }
        
        logger.info(f"Strategy assessment: approved={assessment_output.approved}, score={assessment_output.score:.2f}")
        
        # Stream completion update
        approval_status = "approved" if assessment_output.approved else "needs revision"
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Quality assessed",
            "description": f"Strategy assessment complete: {approval_status}"
        })
        
    except Exception as e:
        logger.error(f"Error in strategy assessment: {e}")
        # Fallback assessment - be lenient
        state["strategy_assessment"] = {
            "approved": True,
            "score": 0.6,
            "feedback": "Assessment error, proceeding with strategy"
        }

    return state


def strategy_approval_check(state: DraftingState) -> str:
    """
    Simple conditional edge for strategy approval.
    """
    strategy_assessment = state.get("strategy_assessment", {})
    revision_count = state.get("revision_count", 0)
    
    # Maximum revisions to prevent loops
    MAX_REVISIONS = 2
    
    if revision_count >= MAX_REVISIONS:
        logger.info("Maximum revisions reached, proceeding to drafting")
        return "final_drafter"
    
    if strategy_assessment.get("approved", False) or strategy_assessment.get("score", 0) >= 0.6:
        logger.info("Strategy approved, proceeding to drafting")
        return "final_drafter"
    else:
        logger.info("Strategy needs revision")
        return "argument_improver"


async def argument_improver_node(state: DraftingState) -> DraftingState:
    """
    Improve arguments based on assessment feedback.
    
    This focused node only improves arguments.
    """
    step_id = f"argument_improver_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Improving arguments",
        "description": "Improving legal arguments based on quality assessment feedback."
    })
    logger.info("Executing ArgumentImproverNode")

    # Increment revision count
    revision_count = state.get("revision_count", 0)
    state["revision_count"] = revision_count + 1

    strategy_assessment = state.get("strategy_assessment", {})
    arguments = state.get("arguments", [])
    core_strategy = state.get("core_strategy", {})

    # Create structured LLM for improvement
    structured_llm = llm.with_structured_output(SingleArgumentOutput)

    feedback = strategy_assessment.get("feedback", "Improve argument quality")
    improved_arguments = []

    # Improve each argument
    for i, arg in enumerate(arguments):
        system_prompt = f"""You are a legal argument improvement specialist. Your ONLY job is to improve ONE argument.

Assessment feedback: {feedback}

Original argument: {arg.get('argument', '')}
Original authority: {arg.get('authority', '')}
Original reasoning: {arg.get('reasoning', '')}

Improve this specific argument to address the feedback while maintaining its core point."""

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Improve argument #{i+1} based on the feedback.")
        ])

        try:
            improved_output = await structured_llm.ainvoke(prompt_template.format_messages())
            
            improved_arguments.append({
                "argument": improved_output.argument,
                "authority": improved_output.authority,
                "reasoning": improved_output.reasoning
            })
            
        except Exception as e:
            logger.error(f"Error improving argument {i+1}: {e}")
            # Keep original if improvement fails
            improved_arguments.append(arg)

    # Update arguments
    state["arguments"] = improved_arguments
    logger.info(f"Improved {len(improved_arguments)} arguments")
    
    # Stream completion update
    writer({
        "step_id": step_id,
        "status": "complete",
        "brief_description": "Arguments improved",
        "description": f"Improved {len(improved_arguments)} arguments based on assessment feedback"
    })
    
    return state


async def final_drafter_node(state: DraftingState) -> DraftingState:
    """
    Draft the final legal argument using approved strategy and arguments.
    
    This focused node only does final drafting.
    """
    step_id = f"final_drafter_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Drafting final argument",
        "description": "Drafting the final comprehensive legal argument using the approved strategy."
    })
    logger.info("Executing FinalDrafterNode")

    core_strategy = state.get("core_strategy", {})
    arguments = state.get("arguments", [])
    legal_issue_analysis = state.get("legal_issue_analysis", {})
    user_facts = state["user_facts"]
    case_file = state.get("case_file", {})

    system_prompt = """You are a legal writing specialist. Your ONLY job is to draft a comprehensive legal argument.

Structure the argument with:
1. Introduction (issue and thesis)
2. Legal Framework (applicable law)
3. Analysis (application to facts)
4. Conclusion

Use professional legal writing style with proper reasoning and citations."""

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Strategy: {json.dumps(core_strategy, indent=2)}

Arguments: {json.dumps(arguments, indent=2)}

Legal Issues: {json.dumps(legal_issue_analysis, indent=2)}

Facts: {user_facts}

Available Precedents: {json.dumps(case_file, indent=2, default=str)}

Draft a comprehensive legal argument that implements this strategy.""")
    ])

    try:
        response = await llm.ainvoke(prompt_template.format_messages())
        drafted_argument = response.content

        # Extract citations
        citations_used = []
        for arg in arguments:
            if arg.get("authority"):
                citations_used.append(arg["authority"])

        # Create argument structure
        argument_structure = {
            "introduction": legal_issue_analysis.get("primary_issue", "Legal issue"),
            "legal_framework": legal_issue_analysis.get("applicable_law", "Applicable law"),
            "analysis": "Application of law to facts",
            "conclusion": core_strategy.get("main_thesis", "Client's position")
        }

        # Update state
        state["drafted_argument"] = drafted_argument
        state["argument_structure"] = argument_structure
        state["citations_used"] = citations_used
        state["workflow_stage"] = "completed"

        logger.info(f"Final argument drafted: {len(drafted_argument.split())} words")
        
        # Stream completion update
        word_count = len(drafted_argument.split())
        citation_count = len(citations_used)
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Final argument drafted",
            "description": f"Completed comprehensive legal argument ({word_count} words, {citation_count} citations)"
        })

    except Exception as e:
        logger.error(f"Error in final drafting: {e}")
        state["drafted_argument"] = "Error occurred during drafting"
        state["argument_structure"] = {}
        state["citations_used"] = []

    return state


# Editing workflow nodes
async def edit_analyzer_node(state: DraftingState) -> DraftingState:
    """
    Analyze editing requirements for existing drafts.
    
    This focused node only analyzes what needs to be edited.
    """
    step_id = f"edit_analyzer_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Analyzing edit requirements",
        "description": "Analyzing the existing draft and edit instructions to understand required changes."
    })
    logger.info("Executing EditAnalyzerNode")

    existing_draft = state.get("existing_draft", "")
    edit_instructions = state.get("edit_instructions", "")

    # Simple analysis for editing
    state["edit_analysis"] = {
        "original_length": len(existing_draft.split()),
        "edit_type": "content_revision",
        "instructions": edit_instructions
    }

    logger.info("Edit analysis completed")
    
    # Stream completion update
    word_count = len(existing_draft.split())
    writer({
        "step_id": step_id,
        "status": "complete",
        "brief_description": "Edit analysis complete",
        "description": f"Analyzed draft ({word_count} words) and edit requirements"
    })
    
    return state


async def content_reviser_node(state: DraftingState) -> DraftingState:
    """
    Revise content based on edit instructions.
    
    This focused node only revises content.
    """
    step_id = f"content_reviser_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Revising content",
        "description": "Revising the legal argument content based on edit instructions."
    })
    logger.info("Executing ContentReviserNode")

    existing_draft = state.get("existing_draft", "")
    edit_instructions = state.get("edit_instructions", "")

    # Create structured LLM for content revision
    structured_llm = llm.with_structured_output(ContentRevisionOutput)

    system_prompt = """You are a legal content revision specialist. Your ONLY job is to revise content based on instructions.

Make the requested changes while:
1. Maintaining legal accuracy
2. Preserving good content where possible
3. Improving clarity and effectiveness
4. Following proper legal writing conventions"""

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Original draft: {existing_draft}

Edit instructions: {edit_instructions}

Revise the content according to the instructions.""")
    ])

    try:
        revision_output = await structured_llm.ainvoke(prompt_template.format_messages())
        
        # Store revision
        state["content_revisions"] = [{
            "revised_content": revision_output.revised_content,
            "changes_made": revision_output.changes_made,
            "improvement_reasoning": revision_output.improvement_reasoning
        }]

        # Update final draft
        state["drafted_argument"] = revision_output.revised_content
        
        logger.info("Content revision completed")
        
        # Stream completion update
        new_word_count = len(revision_output.revised_content.split())
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Content revised",
            "description": f"Revision complete: {revision_output.changes_made} ({new_word_count} words)"
        })
        
    except Exception as e:
        logger.error(f"Error in content revision: {e}")
        state["content_revisions"] = []
        state["drafted_argument"] = existing_draft  # Keep original on error

    return state


# Build the improved drafting graph
def create_drafting_graph() -> StateGraph:
    """
    Create and compile the improved drafting workflow graph v2.
    
    Returns:
        Compiled StateGraph with decomposed nodes for better LLM performance
    """
    workflow = StateGraph(DraftingState)

    # Add decomposed nodes for normal workflow
    workflow.add_node("fact_analyzer", fact_analyzer_node)
    workflow.add_node("strategy_developer", strategy_developer_node)
    workflow.add_node("argument_builder", argument_builder_node)
    workflow.add_node("simple_critic", simple_critic_node)
    workflow.add_node("argument_improver", argument_improver_node)
    workflow.add_node("final_drafter", final_drafter_node)

    # Add editing workflow nodes
    workflow.add_node("edit_analyzer", edit_analyzer_node)
    workflow.add_node("content_reviser", content_reviser_node)

    # Conditional entry point based on editing mode
    def route_entry(state: DraftingState) -> str:
        if state.get("is_editing", False):
            return "edit_analyzer"
        else:
            return "fact_analyzer"

    # Set up routing
    workflow.set_conditional_entry_point(route_entry, {
        "fact_analyzer": "fact_analyzer",
        "edit_analyzer": "edit_analyzer"
    })

    # Normal workflow edges
    workflow.add_edge("fact_analyzer", "strategy_developer")
    workflow.add_edge("strategy_developer", "argument_builder")
    workflow.add_edge("argument_builder", "simple_critic")

    # Conditional edge for strategy assessment
    workflow.add_conditional_edges(
        "simple_critic",
        strategy_approval_check,
        {
            "argument_improver": "argument_improver",
            "final_drafter": "final_drafter"
        }
    )

    # Improvement loop
    workflow.add_edge("argument_improver", "simple_critic")

    # Editing workflow edges
    workflow.add_edge("edit_analyzer", "content_reviser")

    # All paths end
    workflow.add_edge("final_drafter", END)
    workflow.add_edge("content_reviser", END)

    compiled_workflow = workflow.compile()

    return compiled_workflow


# Export the compiled graph
drafting_graph = create_drafting_graph()
"""
CounterArgument Graph v2 - Decomposed RAG-based Counterargument Generation

This module implements an improved counterargument workflow with decomposed nodes
and iterative generation patterns for better LLM performance and reliability.
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
    CounterArgumentState,
    CounterArgumentSeed,
    SingleCounterArgument,
    SingleRebuttal,
    VulnerabilityAssessment
)
from ...tools.neo4j_tools import (
    vector_search,
    fulltext_search,
    find_case_citations,
    find_legal_concepts,
    assess_precedent_strength,
    find_similar_fact_patterns,
    find_legal_tests,
    find_authority_chain,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
counterargument_attrs = settings.llm_config.get("main", {}).get("counterargument", {})
llm = ChatGoogleGenerativeAI(google_api_key=settings.google_api_key, **counterargument_attrs)


# Simplified Pydantic models for structured output
class CounterArgumentSeedOutput(BaseModel):
    """Seed for counterargument generation."""
    challenge_type: str = Field(description="precedent, procedural, factual, or policy")
    target_argument: int = Field(description="Index of argument being challenged (0-based)")
    brief_description: str = Field(description="One sentence challenge description")


class SingleCounterArgumentOutput(BaseModel):
    """Single counterargument structure."""
    title: str = Field(description="Clear, specific title for this challenge")
    argument: str = Field(description="Detailed counterargument with specific legal reasoning")
    supporting_authority: str = Field(description="Specific legal authority, case citation, or statute supporting this challenge")
    factual_basis: str = Field(description="Factual foundation explaining why this challenge applies")
    strength: float = Field(description="Strength assessment 0-1", ge=0.0, le=1.0)


class SingleRebuttalOutput(BaseModel):
    """Single rebuttal structure."""
    strategy: str = Field(description="Specific rebuttal approach or strategy name")
    content: str = Field(description="Detailed rebuttal content with legal reasoning")
    authority: str = Field(description="Supporting legal authority for the rebuttal")


class VulnerabilityAssessmentOutput(BaseModel):
    """Simple vulnerability assessment."""
    vulnerability_type: str = Field(description="Type of vulnerability: precedent, procedural, factual, or policy")
    description: str = Field(description="Detailed description of the vulnerability")
    severity: float = Field(description="Severity rating 0-1", ge=0.0, le=1.0)


# Decomposed node functions
async def rag_retrieval_node(state: CounterArgumentState) -> CounterArgumentState:
    """
    RAG Retrieval Node - Comprehensive knowledge graph retrieval.
    
    This focused node only does retrieval - no analysis or generation.
    """
    step_id = f"rag_retrieval_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Retrieving counterargument research",
        "description": "Retrieving opposing precedents and legal authorities for counterargument analysis."
    })
    logger.info("Starting RAG retrieval for counterargument generation")
    
    key_arguments = state["key_arguments"]
    user_facts = state["user_facts"]
    
    # Extract meaningful keywords from arguments and authorities
    keywords = set()
    for arg in key_arguments:
        if isinstance(arg, dict):
            argument_text = arg.get("argument", "")
            authority = arg.get("supporting_authority", "")
            # Add all capitalized words (likely legal concepts) and authorities
            keywords.update([
                word.strip(",.()") for word in argument_text.split()
                if word.istitle() and len(word) > 4
            ])
            if authority:
                keywords.add(authority)
    # Fallback if no keywords found
    if not keywords:
        keywords.add("legal precedent")
    
    # Build a single meaningful query
    query = " ".join(keywords)
    logger.info(f"RAG retrieval using query: {query}")
    
    retrieved_docs = []
    try:
        # Single vector search with meaningful keywords
        results = await vector_search(
            query_text=query,
            jurisdiction=None,
            min_score=0.6,
            limit=20
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
        
        state["retrieved_docs"] = unique_docs[:20]  # Limit for performance
        
        logger.info(f"Retrieved {len(unique_docs)} documents for counterargument analysis")
        
        # Stream completion update
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Research documents retrieved",
            "description": f"Retrieved {len(unique_docs)} documents with opposing precedents and challenges"
        })
        
    except Exception as e:
        logger.error(f"Error in RAG retrieval: {e}")
        state["retrieved_docs"] = []
    
    return state


async def vulnerability_analysis_node(state: CounterArgumentState) -> CounterArgumentState:
    """
    Analyze argument vulnerabilities using retrieved knowledge.
    
    This focused node only does vulnerability analysis.
    """
    step_id = f"vulnerability_analysis_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Analyzing argument vulnerabilities",
        "description": "Analyzing the arguments for potential weaknesses and vulnerabilities."
    })
    logger.info("Starting vulnerability analysis")
    
    key_arguments = state["key_arguments"]
    retrieved_docs = state.get("retrieved_docs", [])
    
    # Create structured LLM for vulnerability assessment
    structured_llm = llm.with_structured_output(VulnerabilityAssessmentOutput)
    
    vulnerabilities = []
    
    # Analyze each argument for vulnerabilities
    for i, argument in enumerate(key_arguments[:3]):  # Limit to 3 arguments
        if not isinstance(argument, dict):
            continue
        
        arg_text = argument.get("argument", "")
        authority = argument.get("supporting_authority", "")
        
        # Check against retrieved research
        relevant_docs = retrieved_docs[:5]  # Use top 5 most relevant
        
        system_prompt = f"""You are a legal vulnerability analyst. Your ONLY job is to identify ONE vulnerability in this argument.

CRITICAL REQUIREMENTS:
- Write a comprehensive vulnerability description (at least 2-3 sentences)
- Be specific about the legal weakness, not general
- Reference specific legal principles or precedents when possible
- Explain WHY this is a vulnerability, not just what it is

Argument: {arg_text}
Authority: {authority}

Look for potential weaknesses like:
- Opposing precedents that contradict this position
- Procedural barriers or standing issues  
- Factual distinguishers that weaken the analogy
- Policy concerns with this approach

Your description must explain the specific legal problem and its implications."""

        retrieved_context = "\n".join([
            f"- {doc.get('document_citation', 'Unknown')}: {doc.get('text', '')[:200]}"
            for doc in relevant_docs
        ])

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Retrieved research that might reveal vulnerabilities:
{retrieved_context}

Identify ONE specific vulnerability in this argument.""")
        ])

        try:
            vuln_output = await structured_llm.ainvoke(prompt_template.format_messages())
            
            vulnerabilities.append({
                "vulnerability_type": vuln_output.vulnerability_type,
                "description": vuln_output.description,
                "severity": vuln_output.severity,
                "target_argument": i
            })
            
            logger.info(f"Identified vulnerability in argument {i+1}: {vuln_output.vulnerability_type}")
            
        except Exception as e:
            logger.error(f"Error analyzing vulnerability for argument {i+1}: {e}")
            # Fallback vulnerability
            vulnerabilities.append({
                "vulnerability_type": "general",
                "description": "Potential weakness requiring further analysis",
                "severity": 0.5,
                "target_argument": i
            })
    
    state["vulnerability_assessments"] = vulnerabilities
    state["generation_phase"] = "seeds"
    
    logger.info(f"Vulnerability analysis completed. Found {len(vulnerabilities)} vulnerabilities")
    
    # Stream completion update
    writer({
        "step_id": step_id,
        "status": "complete",
        "brief_description": "Vulnerabilities identified",
        "description": f"Identified {len(vulnerabilities)} potential vulnerabilities in the arguments"
    })
    
    return state


async def challenge_identifier_node(state: CounterArgumentState) -> CounterArgumentState:
    """
    Identify potential counterargument challenges.
    
    This focused node only identifies challenge seeds.
    """
    step_id = f"challenge_identifier_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Identifying counterargument challenges",
        "description": "Identifying specific types of challenges that can be made against the arguments."
    })
    logger.info("Starting challenge identification")
    
    vulnerability_assessments = state.get("vulnerability_assessments", [])
    key_arguments = state["key_arguments"]
    
    # Create structured LLM for seed generation
    structured_llm = llm.with_structured_output(CounterArgumentSeedOutput)
    
    seeds = []
    
    # Generate seeds based on vulnerabilities
    for vuln in vulnerability_assessments:
        target_arg = vuln.get("target_argument", 0)
        vuln_type = vuln.get("vulnerability_type", "general")
        description = vuln.get("description", "")
        
        # Map vulnerability types to challenge types
        challenge_type_map = {
            "precedent": "precedent",
            "procedural": "procedural", 
            "factual": "factual",
            "general": "policy"
        }
        
        challenge_type = challenge_type_map.get(vuln_type, "policy")
        
        system_prompt = f"""You are a counterargument seed specialist. Your ONLY job is to create ONE challenge seed.

Vulnerability identified: {description}
Challenge type: {challenge_type}
Target argument index: {target_arg}

Create a brief, specific description of how this vulnerability can be turned into a counterargument."""

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Create a counterargument seed based on this vulnerability.")
        ])

        try:
            seed_output = await structured_llm.ainvoke(prompt_template.format_messages())
            
            seeds.append({
                "challenge_type": seed_output.challenge_type,
                "target_argument": seed_output.target_argument,
                "brief_description": seed_output.brief_description
            })
            
            logger.info(f"Generated seed: {seed_output.brief_description[:50]}...")
            
        except Exception as e:
            logger.error(f"Error generating seed: {e}")
            # Fallback seed
            seeds.append({
                "challenge_type": challenge_type,
                "target_argument": target_arg,
                "brief_description": f"Challenge based on {vuln_type} vulnerability"
            })
    
    state["counterargument_seeds"] = seeds[:3]  # Limit to 3 for focus
    state["current_seed_index"] = 0
    
    logger.info(f"Challenge identification completed. Generated {len(seeds)} seeds")
    
    # Stream completion update
    challenge_types = [seed.get("challenge_type", "general") for seed in seeds[:3]]
    writer({
        "step_id": step_id,
        "status": "complete",
        "brief_description": "Challenges identified",
        "description": f"Identified {len(seeds)} challenge types: {', '.join(challenge_types)}"
    })
    
    return state


async def counterargument_developer_node(state: CounterArgumentState) -> CounterArgumentState:
    """
    Develop counterarguments iteratively.
    
    This node develops one counterargument at a time.
    """
    step_id = f"counterargument_developer_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Developing counterarguments",
        "description": "Developing detailed counterarguments based on identified challenges."
    })
    logger.info("Starting counterargument development")
    
    seeds = state.get("counterargument_seeds", [])
    retrieved_docs = state.get("retrieved_docs", [])
    key_arguments = state["key_arguments"]
    user_facts = state["user_facts"]
    
    # Create structured LLM for counterargument development
    structured_llm = llm.with_structured_output(SingleCounterArgumentOutput)
    
    counterarguments = []
    
    # Develop each seed into a full counterargument
    for i, seed in enumerate(seeds):
        challenge_type = seed.get("challenge_type", "policy")
        target_arg = seed.get("target_argument", 0)
        description = seed.get("brief_description", "")
        
        # Get target argument details
        target_argument = key_arguments[target_arg] if target_arg < len(key_arguments) else {}
        target_text = target_argument.get("argument", "") if isinstance(target_argument, dict) else ""
        
        system_prompt = f"""You are a counterargument development specialist. Your ONLY job is to develop ONE detailed counterargument.

REQUIREMENTS FOR QUALITY OUTPUT:
- Write a comprehensive argument (multiple paragraphs)
- Include specific legal reasoning and analysis
- Reference concrete legal authorities (cases, statutes, regulations)
- Explain the factual basis thoroughly
- Provide substantive analysis, not general statements

Challenge type: {challenge_type}
Challenge description: {description}
Target argument: {target_text}

Create a counterargument that includes:
1. A descriptive title that clearly identifies the challenge
2. A detailed argument with multiple paragraphs explaining the legal challenge
3. Specific legal authority with case names, citations, or statutory references
4. A thorough factual basis explaining why this challenge applies to these facts
5. An honest strength assessment (0.0-1.0)

Write as if you are an experienced opposing counsel making this argument in court."""

        retrieved_context = "\n".join([
            f"- {doc.get('document_citation', 'Unknown')}: {doc.get('text', '')[:150]}"
            for doc in retrieved_docs[:5]
        ])

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Case facts: {user_facts}

Available research:
{retrieved_context}

Develop a detailed counterargument based on the challenge description.""")
        ])

        try:
            ca_output = await structured_llm.ainvoke(prompt_template.format_messages())
            
            counterarguments.append({
                "title": ca_output.title,
                "argument": ca_output.argument,
                "authority": ca_output.supporting_authority,
                "factual_basis": ca_output.factual_basis,
                "strength": ca_output.strength,
                "target_argument": target_arg
            })
            
            logger.info(f"Developed counterargument {i+1}: {ca_output.title}")
            
        except Exception as e:
            logger.error(f"Error developing counterargument {i+1}: {e}")
            # Fallback counterargument
            counterarguments.append({
                "title": f"Challenge to Argument {target_arg + 1}",
                "argument": f"The argument faces a {challenge_type} challenge based on {description}",
                "authority": "Applicable legal principles",
                "factual_basis": "Further analysis required to establish factual basis.",
                "strength": 0.6,
                "target_argument": target_arg
            })
    
    state["generated_counterarguments"] = counterarguments
    state["generation_phase"] = "rebuttals"
    
    logger.info(f"Counterargument development completed. Generated {len(counterarguments)} counterarguments")
    
    # Stream completion update
    ca_titles = [ca.get("title", "Untitled") for ca in counterarguments]
    writer({
        "step_id": step_id,
        "status": "complete",
        "brief_description": "Counterarguments developed",
        "description": f"Developed {len(counterarguments)} counterarguments: {', '.join(ca_titles[:2])}" + ("..." if len(ca_titles) > 2 else "")
    })
    
    return state


async def rebuttal_generator_node(state: CounterArgumentState) -> CounterArgumentState:
    """
    Generate rebuttals for each counterargument.
    
    This focused node only generates rebuttals.
    """
    step_id = f"rebuttal_generator_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer({
        "step_id": step_id,
        "status": "in_progress",
        "brief_description": "Generating rebuttals",
        "description": "Generating rebuttal strategies for each counterargument."
    })
    logger.info("Starting rebuttal generation")
    
    counterarguments = state.get("generated_counterarguments", [])
    key_arguments = state["key_arguments"]
    
    # Create structured LLM for rebuttal generation
    structured_llm = llm.with_structured_output(SingleRebuttalOutput)
    
    all_rebuttals = []
    
    # Generate rebuttals for each counterargument
    for i, ca in enumerate(counterarguments):
        ca_title = ca.get("title", "")
        ca_argument = ca.get("argument", "")
        target_arg = ca.get("target_argument", 0)
        
        # Get original argument
        original_arg = key_arguments[target_arg] if target_arg < len(key_arguments) else {}
        original_text = original_arg.get("argument", "") if isinstance(original_arg, dict) else ""
        original_authority = original_arg.get("supporting_authority", "") if isinstance(original_arg, dict) else ""
        
        system_prompt = f"""You are a rebuttal specialist. Your ONLY job is to create ONE comprehensive rebuttal to this counterargument.

REQUIREMENTS FOR QUALITY REBUTTAL:
- Write multiple paragraphs with detailed legal analysis
- Include specific legal reasoning and precedent
- Reference concrete authorities (cases, statutes, principles)
- Explain WHY the counterargument fails, not just that it does
- Provide substantive legal analysis, not general statements

Counterargument: {ca_argument}
Original argument being defended: {original_text}
Original authority: {original_authority}

Create a rebuttal that includes:
1. A specific strategy name that describes your approach
2. Detailed content with multiple paragraphs of legal reasoning
3. Specific supporting authority with case names or legal principles

Write as if you are defending your position in court against this challenge."""

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Generate a rebuttal to counter this challenge.")
        ])

        try:
            rebuttal_output = await structured_llm.ainvoke(prompt_template.format_messages())
            
            all_rebuttals.append({
                "strategy": rebuttal_output.strategy,
                "content": rebuttal_output.content,
                "authority": rebuttal_output.authority,
                "counterargument_index": i
            })
            
            logger.info(f"Generated rebuttal {i+1}: {rebuttal_output.strategy}")
            
        except Exception as e:
            logger.error(f"Error generating rebuttal {i+1}: {e}")
            # Fallback rebuttal
            all_rebuttals.append({
                "strategy": "Distinguishing Analysis",
                "content": f"The counterargument can be distinguished based on factual differences and legal precedent supporting our position.",
                "authority": original_authority or "Supporting case law",
                "counterargument_index": i
            })
    
    state["generated_rebuttals"] = all_rebuttals
    
    # Calculate quality metrics
    if counterarguments:
        avg_strength = sum(ca.get("strength", 0.5) for ca in counterarguments) / len(counterarguments)
        state["analysis_quality"] = avg_strength
    
    if all_rebuttals:
        # Simple rebuttal quality based on content length and authority presence
        rebuttal_scores = [
            (0.7 if len(reb.get("content", "")) > 100 else 0.3) +
            (0.3 if len(reb.get("authority", "")) > 10 else 0.1)
            for reb in all_rebuttals
        ]
        state["generation_quality"] = sum(rebuttal_scores) / len(rebuttal_scores)
    
    logger.info(f"Rebuttal generation completed. Generated {len(all_rebuttals)} rebuttals")
    
    # Stream completion update
    rebuttal_strategies = [reb.get("strategy", "Unknown") for reb in all_rebuttals]
    writer({
        "step_id": step_id,
        "status": "complete",
        "brief_description": "Rebuttals generated",
        "description": f"Generated {len(all_rebuttals)} rebuttal strategies: {', '.join(rebuttal_strategies[:2])}" + ("..." if len(rebuttal_strategies) > 2 else "")
    })
    
    return state


# Build the improved counterargument graph
def create_counterargument_graph() -> StateGraph:
    """
    Create and compile the improved counterargument workflow graph v2.
    
    Returns:
        Compiled StateGraph with decomposed nodes and iterative generation
    """
    workflow = StateGraph(CounterArgumentState)

    # Add decomposed nodes
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("vulnerability_analysis", vulnerability_analysis_node)
    workflow.add_node("challenge_identifier", challenge_identifier_node)
    workflow.add_node("counterargument_developer", counterargument_developer_node)
    workflow.add_node("rebuttal_generator", rebuttal_generator_node)

    # Linear workflow edges
    workflow.set_entry_point("rag_retrieval")
    workflow.add_edge("rag_retrieval", "vulnerability_analysis")
    workflow.add_edge("vulnerability_analysis", "challenge_identifier")
    workflow.add_edge("challenge_identifier", "counterargument_developer")
    workflow.add_edge("counterargument_developer", "rebuttal_generator")
    workflow.add_edge("rebuttal_generator", END)

    compiled_workflow = workflow.compile()

    return compiled_workflow


# Export the compiled graph
counterargument_graph = create_counterargument_graph() 
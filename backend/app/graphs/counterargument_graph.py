"""
CounterArgument Graph - RAG-based Counterargument Generation

This module implements a sophisticated RAG (Retrieval-Augmented Generation) workflow
that leverages the Neo4j knowledge graph to generate comprehensive counterarguments
for moot court practice. The graph retrieves relevant opposing precedents, analyzes
argument vulnerabilities, and generates structured counterarguments with rebuttals.
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

from app.core.config import settings
from .state import CounterArgumentState
from ..tools.neo4j_tools import (
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

# Structured output models for counterargument generation
class CounterArgumentOutput(BaseModel):
    """Structured output for a single counterargument."""
    title: str = Field(description="Clear title summarizing the counterargument")
    argument: str = Field(description="Detailed argument challenging the position")
    supporting_authority: str = Field(description="Legal authority supporting this challenge")
    factual_basis: str = Field(description="Factual foundation for the challenge")
    strength_assessment: float = Field(description="Strength rating 0-1", ge=0.0, le=1.0)


class RebuttalOutput(BaseModel):
    """Structured output for a rebuttal to a counterargument."""
    title: str = Field(description="Title for the rebuttal strategy")
    content: str = Field(description="Detailed response to the counterargument")
    authority: str = Field(description="Legal authority supporting the rebuttal")


class CounterArgumentAnalysisOutput(BaseModel):
    """Structured output for counterargument analysis."""
    counterarguments: List[CounterArgumentOutput] = Field(description="Generated counterarguments")
    rebuttals: List[List[RebuttalOutput]] = Field(description="Rebuttals for each counterargument")
    analysis_summary: str = Field(description="Summary of the counterargument analysis")


async def rag_retrieval_node(state: CounterArgumentState) -> CounterArgumentState:
    """
    RAG Retrieval Node - Comprehensive knowledge graph retrieval for counterarguments.
    
    This node performs multiple types of retrieval to find opposing precedents,
    distinguishing factors, and alternative legal interpretations.
    """
    logger.info("Starting RAG retrieval for counterargument generation")
    
    key_arguments = state["key_arguments"]
    user_facts = state["user_facts"]
    case_file_documents = state["case_file_documents"]
    
    # Extract key legal concepts and precedents from arguments
    primary_concepts = []
    primary_precedents = []
    
    for arg in key_arguments:
        if isinstance(arg, dict):
            # Extract legal concepts from the argument text
            argument_text = arg.get("argument", "")
            authority = arg.get("supporting_authority", "")
            
            primary_concepts.extend([
                concept.strip() for concept in argument_text.split() 
                if len(concept) > 5 and concept[0].isupper()
            ][:3])  # Limit to avoid noise
            
            if authority:
                primary_precedents.append(authority)
    
    # 1. Find opposing precedents using semantic search
    opposing_precedents = []
    for concept in primary_concepts[:5]:  # Limit queries
        try:
            query = f"cases opposing {concept} arguments contrary position"
            results = await vector_search(
                query_text=query,
                jurisdiction=None,  # Don't limit jurisdiction for broader perspective
                min_score=0.6,
                limit=10
            )
            opposing_precedents.extend(results)
        except Exception as e:
            logger.warning(f"Vector search failed for concept '{concept}': {e}")
    
    # 2. Find distinguishing factors for favorable precedents
    distinguishing_factors = []
    for precedent in primary_precedents:
        try:
            # Find cases that distinguish or criticize this precedent
            citation_results = find_case_citations(precedent, direction="cited_by")
            for citation in citation_results[:5]:  # Limit results
                # Look for distinguishing language
                if any(keyword in citation.get("text", "").lower() 
                      for keyword in ["distinguish", "differ", "inapplicable", "not applicable"]):
                    distinguishing_factors.append(citation)
        except Exception as e:
            logger.warning(f"Citation search failed for precedent '{precedent}': {e}")
    
    # 3. Find alternative legal interpretations
    alternative_interpretations = []
    for concept in primary_concepts[:3]:
        try:
            # Search for different interpretations of the same legal concept
            query = f"alternative interpretation {concept} different approach"
            results = fulltext_search(
                search_terms=query,
                search_type="chunks",
                limit=8
            )
            alternative_interpretations.extend(results)
        except Exception as e:
            logger.warning(f"Fulltext search failed for alternative interpretations: {e}")
    
    # 4. Find procedural and evidentiary challenges
    procedural_challenges = []
    try:
        # Search for procedural issues, evidence problems, standing issues
        procedural_queries = [
            "procedural bar statute limitations standing",
            "evidence inadmissible hearsay foundation",
            "burden proof standard review"
        ]
        
        for query in procedural_queries:
            results = await vector_search(query_text=query, limit=5)
            procedural_challenges.extend(results)
    except Exception as e:
        logger.warning(f"Procedural challenge search failed: {e}")
    
    # 5. Find policy-based counterarguments
    policy_arguments = []
    for concept in primary_concepts[:3]:
        try:
            query = f"policy concerns {concept} economic social impact"
            results = await vector_search(query_text=query, limit=5)
            policy_arguments.extend(results)
        except Exception as e:
            logger.warning(f"Policy argument search failed: {e}")
    
    # Store retrieval results
    state["retrieved_counterargument_precedents"] = opposing_precedents[:15]  # Limit for performance
    state["retrieved_distinguishing_factors"] = distinguishing_factors[:10]
    state["retrieved_alternative_interpretations"] = alternative_interpretations[:10]
    state["retrieved_procedural_challenges"] = procedural_challenges[:10]
    state["retrieved_policy_arguments"] = policy_arguments[:10]
    
    # Calculate research comprehensiveness score
    total_retrieved = (
        len(opposing_precedents) + len(distinguishing_factors) + 
        len(alternative_interpretations) + len(procedural_challenges) + 
        len(policy_arguments)
    )
    state["research_comprehensiveness"] = min(total_retrieved / 50.0, 1.0)  # Normalize to 0-1
    
    state["workflow_stage"] = "analysis"
    
    logger.info(f"RAG retrieval completed. Retrieved {total_retrieved} relevant items")
    return state


def vulnerability_analysis_node(state: CounterArgumentState) -> CounterArgumentState:
    """
    Vulnerability Analysis Node - Analyze argument weaknesses using retrieved knowledge.
    
    This node analyzes the user's arguments against the retrieved knowledge to identify
    potential vulnerabilities and weak points.
    """
    logger.info("Starting vulnerability analysis")
    
    key_arguments = state["key_arguments"]
    retrieved_precedents = state.get("retrieved_counterargument_precedents", [])
    distinguishing_factors = state.get("retrieved_distinguishing_factors", [])
    procedural_challenges = state.get("retrieved_procedural_challenges", [])
    
    vulnerabilities = []
    opposing_principles = []
    factual_distinguishers = []
    interpretation_disputes = []
    
    # Analyze each argument for vulnerabilities
    for i, argument in enumerate(key_arguments):
        if not isinstance(argument, dict):
            continue
            
        arg_text = argument.get("argument", "")
        authority = argument.get("supporting_authority", "")
        factual_basis = argument.get("factual_basis", "")
        
        # Check against opposing precedents
        for precedent in retrieved_precedents[:5]:  # Limit analysis
            precedent_text = precedent.get("text", "")
            if any(keyword in precedent_text.lower() 
                  for keyword in ["reject", "deny", "contrary", "opposite"]):
                vulnerabilities.append({
                    "argument_index": i,
                    "vulnerability_type": "opposing_precedent",
                    "description": f"Precedent {precedent.get('document_citation', 'Unknown')} may oppose this argument",
                    "supporting_text": precedent_text[:200]
                })
        
        # Check for distinguishing factors
        for factor in distinguishing_factors[:3]:
            factor_text = factor.get("text", "")
            if any(keyword in factor_text.lower() 
                  for keyword in ["distinguish", "different", "inapplicable"]):
                factual_distinguishers.append(f"Factual difference: {factor_text[:100]}")
        
        # Check for procedural issues
        for challenge in procedural_challenges[:3]:
            challenge_text = challenge.get("text", "")
            if any(keyword in challenge_text.lower() 
                  for keyword in ["bar", "limitation", "inadmissible", "standing"]):
                vulnerabilities.append({
                    "argument_index": i,
                    "vulnerability_type": "procedural_challenge",
                    "description": f"Potential procedural challenge identified",
                    "supporting_text": challenge_text[:200]
                })
    
    # Extract opposing legal principles
    for precedent in retrieved_precedents[:10]:
        holdings = precedent.get("holdings", [])
        for holding in holdings:
            if isinstance(holding, str) and len(holding) > 20:
                opposing_principles.append(holding)
    
    state["argument_vulnerabilities"] = vulnerabilities
    state["opposing_legal_principles"] = opposing_principles[:10]
    state["factual_distinguishers"] = factual_distinguishers[:10]
    state["statutory_interpretation_disputes"] = interpretation_disputes
    
    state["workflow_stage"] = "generation"
    
    logger.info(f"Vulnerability analysis completed. Found {len(vulnerabilities)} vulnerabilities")
    return state


def counterargument_generation_node(state: CounterArgumentState) -> CounterArgumentState:
    """
    CounterArgument Generation Node - Generate structured counterarguments using LLM.
    
    This node uses the retrieved knowledge and vulnerability analysis to generate
    comprehensive counterarguments with supporting rebuttals.
    """
    logger.info("Starting counterargument generation")
    
    key_arguments = state["key_arguments"]
    user_facts = state["user_facts"]
    vulnerabilities = state.get("argument_vulnerabilities", [])
    opposing_principles = state.get("opposing_legal_principles", [])
    retrieved_precedents = state.get("retrieved_counterargument_precedents", [])
    procedural_challenges = state.get("retrieved_procedural_challenges", [])
    
    # Build comprehensive context for LLM
    arguments_context = ""
    for i, arg in enumerate(key_arguments):
        if isinstance(arg, dict):
            arguments_context += f"""
Argument {i + 1}:
- Main Argument: {arg.get('argument', '')}
- Supporting Authority: {arg.get('supporting_authority', '')}
- Factual Basis: {arg.get('factual_basis', '')}
"""
    
    # Build retrieved knowledge context
    opposing_context = ""
    for precedent in retrieved_precedents[:10]:
        citation = precedent.get("document_citation", "Unknown case")
        text = precedent.get("text", "")[:300]
        opposing_context += f"- {citation}: {text}\n"
    
    vulnerability_context = ""
    for vuln in vulnerabilities[:5]:
        vulnerability_context += f"- {vuln.get('description', '')}: {vuln.get('supporting_text', '')[:150]}\n"
    
    procedural_context = ""
    for challenge in procedural_challenges[:5]:
        citation = challenge.get("document_citation", "Unknown")
        text = challenge.get("text", "")[:200]
        procedural_context += f"- {citation}: {text}\n"
    
    system_prompt = """You are an expert legal advocate specializing in generating strong counterarguments for moot court practice. You have access to comprehensive legal research including opposing precedents, distinguishing factors, and procedural challenges. Generate realistic, legally sound counterarguments that test the strength of the given arguments."""
    
    prompt = f"""Based on comprehensive legal research, generate 3 strong counterarguments against the following legal position:

CASE FACTS:
{user_facts}

MY KEY ARGUMENTS:
{arguments_context}

RESEARCH FINDINGS:

Opposing Precedents Found:
{opposing_context}

Identified Vulnerabilities:
{vulnerability_context}

Potential Procedural Challenges:
{procedural_context}

TASK: Generate 3 strong counterarguments that leverage this research. For each counterargument, also provide 3 potential rebuttals.

Focus on:
1. Precedent-based challenges using the opposing cases found
2. Factual distinguishers that weaken analogies
3. Procedural or evidentiary challenges identified
4. Alternative legal interpretations
5. Policy concerns or unintended consequences

For each counterargument, provide:
- A clear, descriptive title
- Detailed argument challenging the position
- Supporting legal authority (using research findings when possible)
- Factual basis for the challenge
- Strength assessment (0.0 to 1.0)

For each rebuttal, provide:
- Title of the rebuttal strategy
- Detailed response to counter the opponent's argument
- Supporting authority for the rebuttal

Generate practical counterarguments that a skilled opposing counsel would actually raise."""
    
    try:
        # Use structured output for reliable parsing
        structured_llm = llm.with_structured_output(CounterArgumentAnalysisOutput)
        
        response = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        
        # Convert structured output to state format
        counterarguments = []
        rebuttals = []
        
        for ca in response.counterarguments:
            counterarguments.append({
                "title": ca.title,
                "argument": ca.argument,
                "supporting_authority": ca.supporting_authority,
                "factual_basis": ca.factual_basis,
                "strength_assessment": ca.strength_assessment
            })
        
        for rebuttal_list in response.rebuttals:
            rebuttal_group = []
            for reb in rebuttal_list:
                rebuttal_group.append({
                    "title": reb.title,
                    "content": reb.content,
                    "authority": reb.authority
                })
            rebuttals.append(rebuttal_group)
        
        state["generated_counterarguments"] = counterarguments
        state["counterargument_rebuttals"] = rebuttals
        
        # Calculate quality metrics
        avg_strength = sum(ca.get("strength_assessment", 0.5) for ca in counterarguments) / len(counterarguments) if counterarguments else 0.5
        state["counterargument_strength"] = avg_strength
        
        # Assess rebuttal quality based on detail and authority citations
        rebuttal_scores = []
        for rebuttal_group in rebuttals:
            group_score = sum(
                0.7 if len(reb.get("content", "")) > 100 else 0.3 + 
                0.3 if len(reb.get("authority", "")) > 10 else 0.1
                for reb in rebuttal_group
            ) / len(rebuttal_group) if rebuttal_group else 0.0
            rebuttal_scores.append(group_score)
        
        state["rebuttal_quality"] = sum(rebuttal_scores) / len(rebuttal_scores) if rebuttal_scores else 0.5
        
        logger.info(f"Generated {len(counterarguments)} counterarguments with {sum(len(r) for r in rebuttals)} rebuttals")
        
    except Exception as e:
        logger.error(f"Error in counterargument generation: {e}")
        
        # Fallback: Generate basic counterarguments based on research
        fallback_counterarguments = []
        fallback_rebuttals = []
        
        # Use opposing precedents for counterarguments
        for i, precedent in enumerate(retrieved_precedents[:3]):
            citation = precedent.get("document_citation", f"Opposing Case {i+1}")
            text = precedent.get("text", "")
            
            fallback_counterarguments.append({
                "title": f"Precedent-Based Challenge {i+1}",
                "argument": f"The position is undermined by {citation}, which establishes a contrary principle.",
                "supporting_authority": citation,
                "factual_basis": text[:200] if text else "Factual analysis from case law",
                "strength_assessment": 0.7
            })
            
            # Basic rebuttals
            fallback_rebuttals.append([
                {
                    "title": "Distinguishing Precedent",
                    "content": f"The facts in {citation} are distinguishable from the present case.",
                    "authority": "Case law on factual distinction"
                },
                {
                    "title": "Subsequent Development",
                    "content": "Legal developments since that decision support our position.",
                    "authority": "Recent case law"
                },
                {
                    "title": "Narrow Interpretation",
                    "content": "The precedent should be interpreted narrowly to its specific facts.",
                    "authority": "Principles of precedential scope"
                }
            ])
        
        # Add procedural challenges if available
        if procedural_challenges:
            fallback_counterarguments.append({
                "title": "Procedural Challenge",
                "argument": "The claim faces significant procedural barriers that prevent relief.",
                "supporting_authority": "Procedural rules and case law",
                "factual_basis": "Analysis of procedural requirements",
                "strength_assessment": 0.6
            })
            
            fallback_rebuttals.append([
                {
                    "title": "Procedural Requirements Met",
                    "content": "All procedural requirements have been satisfied.",
                    "authority": "Procedural compliance analysis"
                },
                {
                    "title": "Equitable Exception",
                    "content": "Equitable principles support waiving technical requirements.",
                    "authority": "Equity jurisprudence"
                },
                {
                    "title": "Substantial Compliance",
                    "content": "Substantial compliance with procedures is sufficient.",
                    "authority": "Substantial compliance doctrine"
                }
            ])
        
        state["generated_counterarguments"] = fallback_counterarguments
        state["counterargument_rebuttals"] = fallback_rebuttals
        state["counterargument_strength"] = 0.6
        state["rebuttal_quality"] = 0.6
    
    return state


# Build the counterargument graph
def create_counterargument_graph() -> StateGraph:
    """
    Create the counterargument generation graph with RAG workflow.
    
    This graph implements a three-stage process:
    1. RAG Retrieval - Comprehensive knowledge graph search
    2. Vulnerability Analysis - Identify argument weaknesses
    3. Generation - Create counterarguments and rebuttals
    """
    graph = StateGraph(CounterArgumentState)
    
    # Add nodes
    graph.add_node("rag_retrieval", rag_retrieval_node)
    graph.add_node("vulnerability_analysis", vulnerability_analysis_node)
    graph.add_node("counterargument_generation", counterargument_generation_node)
    
    # Add edges
    graph.add_edge("rag_retrieval", "vulnerability_analysis")
    graph.add_edge("vulnerability_analysis", "counterargument_generation")
    graph.add_edge("counterargument_generation", END)
    
    # Set entry point
    graph.set_entry_point("rag_retrieval")
    
    return graph.compile()


# Export the compiled graph
counterargument_graph = create_counterargument_graph()
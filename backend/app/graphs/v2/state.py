"""
State definitions for LangGraph workflows v2.

This module defines simplified TypedDict state objects with decomposed node structures
for better LLM performance and reliability.
"""

from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import NotRequired


# Simplified schemas for better LLM performance
class QueryAnalysis(TypedDict):
    """Simple query analysis output."""
    legal_area: str
    key_concepts: List[str]  # Max 5 concepts
    jurisdiction_hints: Optional[str]


class SearchStrategy(TypedDict):
    """Simple search strategy selection."""
    strategy: str  # semantic, fulltext, citation, or concept
    reasoning: str
    confidence: float  # 0-1


class SearchFilters(TypedDict):
    """Simple search filter configuration."""
    jurisdiction: Optional[str]
    document_type: Optional[str]
    year_from: Optional[int]
    year_to: Optional[int]


class QualityAssessment(TypedDict):
    """Simple quality assessment."""
    sufficient: bool
    score: float  # 0-1
    reason: str


class RefinementSuggestion(TypedDict):
    """Simple refinement suggestion."""
    new_terms: List[str]
    new_strategy: str
    reasoning: str


# Drafting graph simplified schemas
class LegalIssueAnalysis(TypedDict):
    """Simple legal issue identification."""
    primary_issue: str
    secondary_issues: List[str]  # Max 3
    applicable_law: str


class CoreStrategy(TypedDict):
    """Simple core legal strategy."""
    main_thesis: str
    legal_theory: str
    strength_rating: float  # 0-1


class SingleArgument(TypedDict):
    """Single argument structure."""
    argument: str
    authority: str
    reasoning: str


class SimpleAssessment(TypedDict):
    """Simple assessment output."""
    approved: bool
    score: float  # 0-1
    feedback: str


class ContentRevision(TypedDict):
    """Simple content revision."""
    revised_content: str
    changes_made: str
    improvement_reasoning: str


# Counterargument graph simplified schemas
class CounterArgumentSeed(TypedDict):
    """Seed for counterargument generation."""
    challenge_type: str  # precedent, procedural, factual, policy
    target_argument: int  # Index of argument being challenged
    brief_description: str


class SingleCounterArgument(TypedDict):
    """Single counterargument structure."""
    title: str
    argument: str
    authority: str
    strength: float  # 0-1


class SingleRebuttal(TypedDict):
    """Single rebuttal structure."""
    strategy: str
    content: str
    authority: str


class VulnerabilityAssessment(TypedDict):
    """Simple vulnerability assessment."""
    vulnerability_type: str
    description: str
    severity: float  # 0-1


# Main state objects with decomposed structure
class ResearchState(TypedDict):
    """
    State object for the Research Graph workflow v2.
    Simplified with decomposed node structure.
    """
    # Input
    query_text: str

    # Decomposed analysis results
    query_analysis: NotRequired[QueryAnalysis]
    search_strategy: NotRequired[SearchStrategy]
    search_filters: NotRequired[SearchFilters]
    
    # Search execution
    search_params: NotRequired[Dict[str, Any]]
    retrieved_docs: NotRequired[List[Dict[str, Any]]]
    total_results: NotRequired[int]
    
    # Assessment
    quality_assessment: NotRequired[QualityAssessment]
    refinement_suggestion: NotRequired[RefinementSuggestion]
    
    # Flow control
    refinement_count: int
    execution_time: NotRequired[float]
    search_history: NotRequired[List[Dict[str, Any]]]


class DraftingState(TypedDict):
    """
    State object for the Drafting Graph workflow v2.
    Decomposed into simpler, focused nodes.
    """
    # Input
    user_facts: str
    case_file: Dict[str, Any]
    party_represented: NotRequired[str]  # Which party the user represents (e.g., "Plaintiff", "Defendant")
    legal_question: NotRequired[str]
    
    # Editing support
    existing_draft: NotRequired[str]
    edit_instructions: NotRequired[str]
    is_editing: NotRequired[bool]
    
    # Decomposed analysis results
    legal_issue_analysis: NotRequired[LegalIssueAnalysis]
    core_strategy: NotRequired[CoreStrategy]
    arguments: NotRequired[List[SingleArgument]]
    
    # Assessment and revision
    strategy_assessment: NotRequired[SimpleAssessment]
    argument_assessments: NotRequired[List[SimpleAssessment]]
    content_revisions: NotRequired[List[ContentRevision]]
    
    # Final output
    drafted_argument: NotRequired[str]
    argument_structure: NotRequired[Dict[str, Any]]
    citations_used: NotRequired[List[str]]
    proposed_strategy: NotRequired[Dict[str, Any]]  # Frontend compatibility - structured strategy with key_arguments
    
    # Quality metrics (for frontend compatibility)
    argument_strength: NotRequired[float]
    precedent_coverage: NotRequired[float]
    logical_coherence: NotRequired[float]
    
    # Flow control
    revision_count: NotRequired[int]
    workflow_stage: NotRequired[str]
    execution_time: NotRequired[float]


class CounterArgumentState(TypedDict):
    """
    State object for the CounterArgument Graph workflow v2.
    Simplified with iterative generation pattern.
    """
    # Input
    case_file_id: int
    draft_id: NotRequired[int]
    user_facts: str
    party_represented: NotRequired[str]  # Which party the user represents (e.g., "Plaintiff", "Defendant")
    key_arguments: List[Dict[str, Any]]
    case_file_documents: List[Dict[str, Any]]
    
    # RAG retrieval
    retrieved_docs: NotRequired[List[Dict[str, Any]]]
    
    # Decomposed analysis
    vulnerability_assessments: NotRequired[List[VulnerabilityAssessment]]
    counterargument_seeds: NotRequired[List[CounterArgumentSeed]]
    
    # Iterative generation
    generated_counterarguments: NotRequired[List[SingleCounterArgument]]
    generated_rebuttals: NotRequired[List[SingleRebuttal]]
    
    # Quality metrics
    analysis_quality: NotRequired[float]
    generation_quality: NotRequired[float]
    
    # Flow control
    generation_phase: NotRequired[str]  # seeds, arguments, rebuttals
    current_seed_index: NotRequired[int]
    execution_time: NotRequired[float]


# Legacy compatibility structures
class CaseFileDocument(TypedDict):
    """Structure for documents in the user's case file."""
    document_id: str
    citation: str
    title: str
    year: int
    jurisdiction: str
    relevance_score_percent: float
    key_holdings: List[str]
    selected_chunks: List[Dict[str, Any]]
    user_notes: NotRequired[str]


class SearchPlan(TypedDict):
    """Structure for search planning (legacy compatibility)."""
    search_terms: List[str]
    search_strategy: str
    filters: Dict[str, Any]
    expected_result_types: List[str]
    confidence_level: float


class ArgumentStrategy(TypedDict):
    """Structure for legal argument strategy (legacy compatibility)."""
    main_thesis: str
    argument_type: str
    primary_precedents: List[str]
    legal_framework: str
    anticipated_counterarguments: List[str]
    strength_assessment: float
    risk_factors: List[str]
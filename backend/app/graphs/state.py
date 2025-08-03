"""
State definitions for LangGraph workflows.

This module defines the TypedDict state objects that are passed between nodes
in the research and drafting graphs, ensuring consistent workflow state management.
"""

from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import NotRequired


class ResearchState(TypedDict):
    """
    State object for the Research Graph workflow.

    This state tracks the iterative refinement process of finding relevant
    legal precedents based on a user query.
    """

    # Input fields
    query_text: str  # Original user query

    # Planning and refinement
    initial_plan: NotRequired[Dict[str, Any]]  # Initial search strategy
    refined_plan: NotRequired[
        Dict[str, Any]
    ]  # Refined search strategy after assessment
    refinement_count: int  # Number of refinement iterations (max 3)
    refinement_feedback: NotRequired[str]  # Feedback from assessment for refinement

    # Search parameters and filters
    search_params: NotRequired[Dict[str, Any]]  # Dynamic search parameters
    jurisdiction_filter: NotRequired[str]  # Jurisdiction preference
    document_type_filter: NotRequired[str]  # Document type preference
    date_range: NotRequired[Dict[str, int]]  # Year range filters

    # Results
    retrieved_docs: NotRequired[List[Dict[str, Any]]]  # Retrieved documents/chunks
    search_quality_score: NotRequired[float]  # Quality assessment of current results
    total_results: NotRequired[int]  # Total number of results found

    # Assessment and routing
    retrieval_sufficient: NotRequired[bool]  # Whether current results are sufficient
    assessment_reason: NotRequired[str]  # Reason for assessment decision

    # Metadata
    search_history: NotRequired[List[Dict[str, Any]]]  # History of search attempts
    execution_time: NotRequired[float]  # Time taken for workflow


class DraftingState(TypedDict):
    """
    State object for the Drafting Graph workflow.

    This state tracks the strategy development and argument drafting process,
    including the critique and revision loop.
    """

    # Input fields
    user_facts: str  # User's fact pattern/legal situation
    case_file: Dict[str, Any]  # Selected precedents and research from frontend
    legal_question: NotRequired[str]  # Specific legal question to address

    # Strategy development
    proposed_strategy: NotRequired[Dict[str, Any]]  # Current argumentative strategy
    strategy_version: NotRequired[
        int
    ]  # Version number of strategy (for iteration tracking)
    strategy_rationale: NotRequired[str]  # Reasoning behind the strategy

    # Strategy components
    primary_arguments: NotRequired[List[Dict[str, Any]]]  # Main legal arguments
    supporting_precedents: NotRequired[
        List[Dict[str, Any]]
    ]  # Cases supporting arguments
    legal_tests_to_apply: NotRequired[List[str]]  # Relevant legal tests/standards
    fact_pattern_analogies: NotRequired[List[Dict[str, Any]]]  # Analogous cases

    # Critique and revision
    critique_feedback: NotRequired[str]  # Feedback from critique node
    critique_score: NotRequired[float]  # Quantitative assessment of strategy (0-1)
    identified_weaknesses: NotRequired[List[str]]  # Specific weaknesses found
    suggested_improvements: NotRequired[List[str]]  # Suggestions for improvement
    strategy_approved: NotRequired[bool]  # Whether strategy passed critique

    # Drafting
    drafted_argument: NotRequired[str]  # Final drafted legal argument
    argument_structure: NotRequired[Dict[str, Any]]  # Structure of the argument
    citations_used: NotRequired[List[str]]  # Cases and authorities cited

    # Quality metrics
    argument_strength: NotRequired[float]  # Overall argument strength assessment
    precedent_coverage: NotRequired[float]  # How well precedents support the argument
    logical_coherence: NotRequired[float]  # Logical flow and coherence score

    # Metadata
    revision_history: NotRequired[List[Dict[str, Any]]]  # History of strategy revisions
    total_critique_cycles: NotRequired[int]  # Number of critique-revision cycles
    workflow_stage: NotRequired[
        str
    ]  # Current stage: 'strategy', 'critique', 'drafting'
    execution_time: NotRequired[float]  # Total workflow execution time


class CaseFileDocument(TypedDict):
    """
    Structure for documents in the user's case file.
    """

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
    """
    Structure for search planning.
    """

    search_terms: List[str]
    search_strategy: str  # 'semantic', 'fulltext', 'citation', 'concept'
    filters: Dict[str, Any]
    expected_result_types: List[str]
    confidence_level: float


class ArgumentStrategy(TypedDict):
    """
    Structure for legal argument strategy.
    """

    main_thesis: str
    argument_type: str  # 'analogical', 'precedential', 'policy', 'textual'
    primary_precedents: List[str]
    legal_framework: str
    anticipated_counterarguments: List[str]
    strength_assessment: float
    risk_factors: List[str]

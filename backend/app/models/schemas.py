"""
Pydantic models for API request/response validation.

This module defines the strict data contracts for the FastAPI endpoints,
ensuring type safety and clear communication with the frontend.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


# Research API Models
class ResearchQueryRequest(BaseModel):
    """Request model for research query endpoint."""

    query_text: str = Field(..., description="The legal research query")
    jurisdiction: Optional[str] = Field(
        None, description="Preferred jurisdiction filter"
    )
    document_type: Optional[str] = Field(
        None, description="Document type filter (Case, Doctrine, etc.)"
    )
    date_range: Optional[Dict[str, int]] = Field(
        None, description="Year range filter with 'from' and 'to' keys"
    )
    max_results: Optional[int] = Field(
        15, description="Maximum number of results to return"
    )
    stream: Optional[bool] = Field(
        False, description="Enable streaming responses"
    )


class RetrievedDocument(BaseModel):
    """Model for a retrieved document/chunk."""

    chunk_id: Optional[str] = None
    text: str
    summary: Optional[str] = None
    document_source: str
    document_citation: Optional[str] = None
    document_year: Optional[int] = None
    jurisdiction: Optional[str] = None
    document_type: Optional[str] = None
    court_level: Optional[str] = None
    score: Optional[float] = None
    statutes: Optional[List[str]] = None
    courts: Optional[List[str]] = None
    cases: Optional[List[str]] = None
    concepts: Optional[List[str]] = None
    judges: Optional[List[str]] = None
    holdings: Optional[List[str]] = None
    facts: Optional[List[str]] = None
    legal_tests: Optional[List[str]] = None


class ResearchQueryResponse(BaseModel):
    """Response model for research query endpoint."""

    retrieved_docs: List[RetrievedDocument]
    total_results: int
    search_quality_score: Optional[float] = None
    refinement_count: int = 0
    assessment_reason: Optional[str] = None
    execution_time: Optional[float] = None
    search_history: Optional[List[Dict[str, Any]]] = None


# Drafting API Models
class CaseFileDocument(BaseModel):
    """Model for documents in the user's case file."""

    document_id: str
    citation: str
    title: str
    year: int
    jurisdiction: str
    relevance_score_percent: float = Field(..., ge=0.0, le=100.0)
    key_holdings: List[str]
    selected_chunks: List[Dict[str, Any]] = []
    user_notes: Optional[str] = None


class CaseFile(BaseModel):
    """Model for the user's case file."""

    documents: List[CaseFileDocument]
    total_documents: int
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None


class ArgumentDraftRequest(BaseModel):
    """Request model for argument drafting endpoint."""

    case_file_id: int = Field(..., description="ID of the case file containing static facts")
    legal_question: Optional[str] = Field(
        None, description="Specific legal question to address"
    )
    additional_drafting_instructions: Optional[str] = Field(
        None, description="Additional instructions for drafting the argument"
    )
    argument_preferences: Optional[Dict[str, Any]] = Field(
        None, description="User preferences for argument style/approach"
    )
    stream: Optional[bool] = Field(
        False, description="Enable streaming responses"
    )


class LegalArgument(BaseModel):
    """Model for a legal argument component."""

    argument: str
    supporting_authority: str
    factual_basis: str
    strength_assessment: Optional[float] = None


class ArgumentStrategy(BaseModel):
    """Model for legal argument strategy."""

    main_thesis: str
    argument_type: str  # analogical, precedential, policy, textual
    primary_precedents: List[str]
    legal_framework: str
    key_arguments: List[LegalArgument]
    anticipated_counterarguments: List[str]
    counterargument_responses: List[str]
    strength_assessment: float = Field(..., ge=0.0, le=1.0)
    risk_factors: List[str]
    strategy_rationale: str


class ArgumentDraftResponse(BaseModel):
    """Response model for argument drafting endpoint."""

    strategy: ArgumentStrategy
    drafted_argument: str
    argument_structure: Dict[str, Any]
    citations_used: List[str]
    argument_strength: float = Field(..., ge=0.0, le=1.0)
    precedent_coverage: float = Field(..., ge=0.0, le=1.0)
    logical_coherence: float = Field(..., ge=0.0, le=1.0)
    total_critique_cycles: int
    revision_history: Optional[List[Dict[str, Any]]] = None
    execution_time: Optional[float] = None


# Utility Models
class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    version: Optional[str] = None


# Additional Analysis Models
class PrecedentAnalysis(BaseModel):
    """Model for precedent strength analysis."""

    case_citation: str
    total_citations: int
    precedent_strength: float
    avg_authority_weight: Optional[float] = None
    citing_jurisdictions: List[str]
    most_recent_citation: Optional[int] = None
    earliest_citation: Optional[int] = None


class ConceptAnalysis(BaseModel):
    """Model for legal concept analysis."""

    concept_name: str
    related_documents: List[Dict[str, Any]]
    evolution_timeline: Optional[List[Dict[str, Any]]] = None
    jurisdiction_coverage: List[str]


class CitationNetwork(BaseModel):
    """Model for citation network analysis."""

    source_case: str
    cited_by: List[Dict[str, Any]]
    cites: List[Dict[str, Any]]
    authority_chain: Optional[List[Dict[str, Any]]] = None


# Case File Management Models
class CreateCaseFileRequest(BaseModel):
    """Request model for creating a new case file."""

    title: str = Field(..., description="Title for the case file")
    description: Optional[str] = Field(None, description="Optional description")
    user_facts: Optional[str] = Field(None, description="Client's fact pattern")
    party_represented: Optional[str] = Field(None, description="Which party the user represents (e.g., 'Plaintiff', 'Defendant', 'Petitioner', 'Respondent', 'Appellant', 'Appellee')")


class UpdateCaseFileRequest(BaseModel):
    """Request model for updating a case file."""

    title: Optional[str] = Field(None, description="Updated title")
    description: Optional[str] = Field(None, description="Updated description")
    user_facts: Optional[str] = Field(None, description="Updated fact pattern")
    party_represented: Optional[str] = Field(None, description="Updated party representation")


class CaseFileResponse(BaseModel):
    """Response model for case file with documents."""

    id: int
    title: str
    description: Optional[str] = None
    user_facts: Optional[str] = None
    party_represented: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    documents: List[Dict[str, Any]] = []
    total_documents: int


class CaseFileListItem(BaseModel):
    """Model for case file list item."""

    id: int
    title: str
    description: Optional[str] = None
    party_represented: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    document_count: int
    draft_count: int


class AddDocumentToCaseFileRequest(BaseModel):
    """Request model for adding a document to a case file."""

    document_id: str
    citation: str
    title: str
    year: Optional[int] = None
    jurisdiction: Optional[str] = None
    relevance_score_percent: Optional[float] = None
    key_holdings: Optional[List[str]] = None
    selected_chunks: Optional[List[Dict[str, Any]]] = None
    user_notes: Optional[str] = None


class SaveDraftRequest(BaseModel):
    """Request model for saving an argument draft."""

    case_file_id: int
    title: Optional[str] = None
    draft_response: ArgumentDraftResponse


class ArgumentDraftListItem(BaseModel):
    """Model for argument draft list item."""

    id: int
    title: str
    created_at: datetime
    argument_strength: Optional[float] = None
    precedent_coverage: Optional[float] = None
    logical_coherence: Optional[float] = None


class SavedArgumentDraft(BaseModel):
    """Model for a saved argument draft."""

    id: int
    case_file_id: int
    title: str
    drafted_argument: str
    strategy: Optional[Dict[str, Any]] = None
    argument_structure: Optional[Dict[str, Any]] = None
    citations_used: Optional[List[str]] = None
    argument_strength: Optional[float] = None
    precedent_coverage: Optional[float] = None
    logical_coherence: Optional[float] = None
    total_critique_cycles: Optional[int] = None
    execution_time: Optional[float] = None
    revision_history: Optional[List[Dict[str, Any]]] = None
    created_at: datetime


class EditDraftRequest(BaseModel):
    """Request model for editing a draft with AI."""
    
    draft_id: int
    edit_instructions: str
    stream: Optional[bool] = Field(
        False, description="Enable streaming responses"
    )


class UpdateDraftRequest(BaseModel):
    """Request model for manually updating a draft."""
    
    drafted_argument: str
    title: Optional[str] = None


# Moot Court Models
class CounterArgument(BaseModel):
    """Model for a single counterargument."""
    
    title: str
    argument: str
    supporting_authority: str
    factual_basis: str
    strength_assessment: Optional[float] = Field(None, ge=0.0, le=1.0)


class CounterArgumentRebuttal(BaseModel):
    """Model for a rebuttal to a counterargument."""
    
    title: str
    content: str
    authority: str


class GenerateCounterArgumentsRequest(BaseModel):
    """Request model for generating counterarguments."""
    
    case_file_id: int = Field(..., description="ID of the case file")
    draft_id: Optional[int] = Field(None, description="ID of specific draft to analyze")
    stream: Optional[bool] = Field(
        False, description="Enable streaming responses"
    )


class GenerateCounterArgumentsResponse(BaseModel):
    """Response model for generated counterarguments."""
    
    counterarguments: List[CounterArgument]
    rebuttals: List[List[CounterArgumentRebuttal]]  # Rebuttals for each counterargument
    execution_time: Optional[float] = None


class SaveMootCourtSessionRequest(BaseModel):
    """Request model for saving a moot court session."""
    
    case_file_id: int = Field(..., description="ID of the case file")
    draft_id: Optional[int] = Field(None, description="ID of the draft used")
    title: str = Field(..., description="Title for the moot court session")
    counterarguments: List[CounterArgument]
    rebuttals: List[List[CounterArgumentRebuttal]]
    source_arguments: Optional[List[Dict[str, Any]]] = Field(None, description="Source arguments analyzed")
    research_context: Optional[Dict[str, Any]] = Field(None, description="RAG retrieval context")
    counterargument_strength: Optional[float] = None
    research_comprehensiveness: Optional[float] = None
    rebuttal_quality: Optional[float] = None
    execution_time: Optional[float] = None


class MootCourtSessionListItem(BaseModel):
    """Model for moot court session list item."""
    
    id: int
    title: str
    created_at: datetime
    draft_title: Optional[str] = None
    counterargument_count: int
    counterargument_strength: Optional[float] = None
    research_comprehensiveness: Optional[float] = None


class SavedMootCourtSession(BaseModel):
    """Model for a saved moot court session."""
    
    id: int
    case_file_id: int
    draft_id: Optional[int] = None
    title: str
    counterarguments: List[CounterArgument]
    rebuttals: List[List[CounterArgumentRebuttal]]
    source_arguments: Optional[List[Dict[str, Any]]] = None
    research_context: Optional[Dict[str, Any]] = None
    counterargument_strength: Optional[float] = None
    research_comprehensiveness: Optional[float] = None
    rebuttal_quality: Optional[float] = None
    execution_time: Optional[float] = None
    created_at: datetime


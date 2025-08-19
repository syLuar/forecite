"""
FastAPI application entry point for Legal Research Assistant Backend.

This file defines the FastAPI application and its endpoints, serving as the
entry point for all frontend requests to the research and drafting workflows.
"""

import logging
from pprint import pprint
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from google import genai
import json
import asyncio

from langchain_core.messages import AIMessage

from langgraph.errors import GraphRecursionError

# from app.core.config import settings  # Uncomment when needed
from app.models.schemas import (
    ResearchQueryRequest,
    ResearchQueryResponse,
    RetrievedDocument,
    ArgumentDraftRequest,
    ArgumentDraftResponse,
    ErrorResponse,
    HealthResponse,
    CreateCaseFileRequest,
    UpdateCaseFileRequest,
    CaseFileResponse,
    CaseFileListItem,
    AddDocumentToCaseFileRequest,
    AddCaseFileNoteRequest,
    UpdateCaseFileNoteRequest,
    SaveDraftRequest,
    ArgumentDraftListItem,
    SavedArgumentDraft,
    EditDraftRequest,
    UpdateDraftRequest,
    GenerateCounterArgumentsRequest,
    GenerateCounterArgumentsResponse,
    CounterArgument,
    CounterArgumentRebuttal,
    SaveMootCourtSessionRequest,
    MootCourtSessionListItem,
    SavedMootCourtSession,
    ConductResearchRequest,
    ConductResearchResponse,
)
from app.graphs import research_graph, drafting_graph, counterargument_graph, research_agent, ResearchState, DraftingState, CounterArgumentState
from app.tools.neo4j_tools import close_neo4j_connection
from app.services.case_file_service import CaseFileService, ArgumentDraftService, MootCourtSessionService, CaseFileNoteService
from app.core.database import create_tables

from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fastapi.staticfiles import StaticFiles
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    # Startup
    logger.info("Starting Legal Research Assistant Backend")
    logger.info(f"Environment: {settings.environment}")
    
    # Log database configuration
    database_url = settings.get_database_url()
    db_type = "PostgreSQL" if database_url.startswith("postgresql") else "SQLite"
    logger.info(f"Using {db_type} database")

    # Initialize database tables
    try:
        create_tables()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        if database_url.startswith("postgresql"):
            logger.error("PostgreSQL connection failed. Please ensure:")
            logger.error("1. PostgreSQL server is running")
            logger.error("2. Database exists and user has CREATE privileges")
            logger.error("3. Connection parameters are correct")
            logger.error("4. Run scripts/postgresql_setup.sql as PostgreSQL superuser if needed")
        raise

    yield
    # Shutdown
    logger.info("Shutting down Legal Research Assistant Backend")
    close_neo4j_connection()

origins = [
    "https://hawkihi.site",
    "http://hawkihi.site",  # You might want to allow non-HTTPS for testing
    "https://forecite.site",
    "http://forecite.site",  # Non-HTTPS for testing
    "http://localhost",
    "http://localhost:3000", # Common for local React development
]

app = FastAPI(
    title="Legal Research Assistant API",
    description="Backend API for AI-powered legal research and argument drafting",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of origins that are allowed to make requests
    allow_credentials=True, # Allow cookies to be included in requests
    allow_methods=["*"],    # Allow all standard methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)

# Serve documentation at /docs path
docs_path = os.path.join(os.path.dirname(__file__), "static", "docs")
if os.path.exists(docs_path):
    app.mount("/docs", StaticFiles(directory=docs_path, html=True), name="docs")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logging.error(f"{request}: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred while processing your request",
        ).model_dump(),
    )


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint for basic health check."""
    from datetime import datetime

    return HealthResponse(status="healthy", timestamp=datetime.now(), version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint."""
    from datetime import datetime

    # TODO: Add checks for Neo4j connection, LLM API availability, etc.
    return HealthResponse(status="healthy", timestamp=datetime.now(), version="1.0.0")


@app.post("/api/v1/research/query")
async def research_query(request: ResearchQueryRequest):
    """
    Execute legal research query using the Research Graph.

    This endpoint initiates the research workflow to find relevant legal precedents
    based on the user's query. The workflow includes automatic refinement if
    initial results are insufficient.
    """
    start_time = time.time()

    try:
        logger.info(f"Starting research query: {request.query_text[:100]}...")

        # Initialize the research graph state
        initial_state: ResearchState = {
            "query_text": request.query_text,
            "refinement_count": 0,
        }

        # Add optional filters from request
        if request.jurisdiction:
            initial_state["jurisdiction_filter"] = request.jurisdiction
        if request.document_type:
            initial_state["document_type_filter"] = request.document_type
        if request.date_range:
            initial_state["date_range"] = request.date_range

        # Define response processor for streaming
        async def process_research_response(final_state):
            # Convert retrieved documents to response models
            retrieved_docs = []
            for doc in final_state.get("retrieved_docs", []):
                retrieved_docs.append(RetrievedDocument(**doc))

            execution_time = time.time() - start_time

            response = ResearchQueryResponse(
                retrieved_docs=retrieved_docs,
                total_results=final_state.get("total_results", len(retrieved_docs)),
                search_quality_score=final_state.get("search_quality_score"),
                refinement_count=final_state.get("refinement_count", 0),
                assessment_reason=final_state.get("assessment_reason"),
                execution_time=execution_time,
                search_history=final_state.get("search_history"),
            )
            return response.model_dump()

        # Handle streaming vs non-streaming execution
        if request.stream:
            logger.info("Starting streaming research query execution")
            return create_streaming_response(
                stream_graph_with_final_response(
                    research_graph, initial_state, process_research_response
                )
            )

        # Execute the research graph
        final_state = await research_graph.ainvoke(initial_state)

        # Use the same processing logic
        response_dict = await process_research_response(final_state)
        response = ResearchQueryResponse(**response_dict)

        logger.info(
            f"Research query completed in {response.execution_time:.2f}s, found {len(response.retrieved_docs)} documents"
        )
        return response

    except Exception as e:
        logger.error(f"Error in research query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research query failed: {str(e)}",
        )


@app.post("/api/v1/research/conduct-research", response_model=ConductResearchResponse)
async def conduct_research(request: ConductResearchRequest):
    """
    Conduct comprehensive legal research using the Research Agent.

    This endpoint uses the Legal Research Agent to systematically research legal issues,
    find relevant authorities, and organize findings in the case file. The agent will
    add discovered documents and research notes to the specified case file.
    """
    start_time = time.time()

    try:
        logger.info(f"Starting research agent for case file ID: {request.case_file_id}")

        # Get the case file to validate it exists and retrieve facts
        case_file_data = CaseFileService.get_case_file(request.case_file_id)
        if not case_file_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case file {request.case_file_id} not found",
            )

        # Extract case facts
        case_facts = case_file_data.get('user_facts', '')
        if not case_facts or not case_facts.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Case file must contain case facts before conducting research",
            )

        party_represented = case_file_data.get('party_represented', '')
        
        # Handle optional legal issues
        legal_issues = request.legal_issues or []
        issue_count_text = f"{len(legal_issues)} legal issues" if legal_issues else "AI-identified legal issues"
        logger.info(f"Conducting research for {issue_count_text}")

        # Define response processor for streaming
        async def process_research_response(response_state):
            execution_time = time.time() - start_time

            messages = response_state.get("messages", [])

            if messages:
                last_message = messages[-1]

                if isinstance(last_message, AIMessage) and last_message.content:
                    CaseFileNoteService.add_note(
                        author_type="ai",
                        author_name="Forecite AI",
                        case_file_id=request.case_file_id,
                        content=last_message.content,
                        note_type="strategy",
                    )

                # Get updated case file to count added documents and notes
                updated_case_file = CaseFileService.get_case_file(request.case_file_id)
                original_doc_count = len(case_file_data.get('documents', []))
                original_note_count = len(case_file_data.get('notes', []))
                new_doc_count = len(updated_case_file.get('documents', []))
                new_note_count = len(updated_case_file.get('notes', []))
                
                documents_added = max(0, new_doc_count - original_doc_count)
                notes_added = max(0, new_note_count - original_note_count)

            response = ConductResearchResponse(
                documents_added=documents_added,
                notes_added=notes_added,
                legal_issues_researched=legal_issues,
                execution_time=execution_time,
                jurisdiction=request.jurisdiction or "Singapore"
            )
            return response.model_dump()

        initial_state = research_agent.get_initial_state(
            case_file_id=request.case_file_id,
            case_facts=case_facts,
            party_represented=party_represented,
            legal_issues=legal_issues if legal_issues else None,
            jurisdiction=request.jurisdiction or "Singapore",
        )

        # Handle streaming vs non-streaming execution
        if request.stream:
            logger.info("Starting streaming research agent execution")
            return create_streaming_response(
                stream_graph_with_final_response(
                    research_agent.get_agent_executor(), initial_state, process_research_response, config={"recursion_limit": 60}
                )
            )

        # Execute the research agent
        final_state = await research_agent.ainvoke(initial_state)

        # Use the same processing logic
        response_dict = await process_research_response(final_state)
        response = ConductResearchResponse(**response_dict)

        logger.info(
            f"Research completed in {response.execution_time:.2f}s"
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in research agent: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research failed: {str(e)}",
        )


@app.post("/api/v1/generation/draft-argument", response_model=ArgumentDraftResponse)
async def draft_argument(request: ArgumentDraftRequest):
    """
    Generate legal argument using the Drafting Graph.

    This endpoint gets case facts from the specified case file and uses the legal question
    and additional drafting instructions from the request to generate arguments.
    """
    start_time = time.time()

    try:
        logger.info(f"Starting argument drafting for case file ID: {request.case_file_id}")

        # Get the case file to retrieve static facts and documents
        case_file_data = CaseFileService.get_case_file(request.case_file_id)
        if not case_file_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case file {request.case_file_id} not found",
            )

        # Extract user facts from case file (static)
        user_facts = case_file_data.get('user_facts', '')
        if not user_facts or not user_facts.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Case file must contain case facts before drafting arguments",
            )

        logger.info(f"Using case facts from case file and {len(case_file_data.get('documents', []))} precedents")

        # Build case file structure for the drafting graph
        case_file_for_graph = {
            "documents": case_file_data.get('documents', []),
            "total_documents": case_file_data.get('total_documents', 0),
            "created_at": case_file_data.get('created_at'),
        }

        # Initialize the drafting graph state
        initial_state: DraftingState = {
            "user_facts": user_facts,
            "case_file": case_file_for_graph,
            "party_represented": case_file_data.get('party_represented'),
        }

        # Add optional legal question and additional instructions
        if request.legal_question:
            initial_state["legal_question"] = request.legal_question
        
        if request.additional_drafting_instructions:
            initial_state["additional_drafting_instructions"] = request.additional_drafting_instructions

        # Define response processor for streaming
        async def process_drafting_response(final_state):
            execution_time = time.time() - start_time

            # Extract strategy from final state
            strategy_data = final_state.get("proposed_strategy", {})

            # Convert to response format
            from app.models.schemas import ArgumentStrategy, LegalArgument
            import json

            # Handle key_arguments - it might be a JSON string or already parsed list
            key_arguments_raw = strategy_data.get("key_arguments", [])
            if isinstance(key_arguments_raw, str):
                try:
                    key_arguments_list = json.loads(key_arguments_raw)
                except (json.JSONDecodeError, TypeError):
                    key_arguments_list = []
            elif isinstance(key_arguments_raw, list):
                key_arguments_list = key_arguments_raw
            else:
                key_arguments_list = []

            # Ensure each argument has the required structure
            parsed_arguments = []
            for arg in key_arguments_list:
                if isinstance(arg, dict) and all(
                    key in arg
                    for key in ["argument", "supporting_authority", "factual_basis"]
                ):
                    parsed_arguments.append(LegalArgument(**arg))
                elif isinstance(arg, str):
                    # Fallback: create a basic argument structure from string
                    parsed_arguments.append(
                        LegalArgument(
                            argument=arg, supporting_authority="", factual_basis=""
                        )
                    )

            strategy = ArgumentStrategy(
                main_thesis=strategy_data.get("main_thesis", ""),
                argument_type=strategy_data.get("argument_type", "precedential"),
                primary_precedents=strategy_data.get("primary_precedents", []),
                legal_framework=strategy_data.get("legal_framework", ""),
                key_arguments=parsed_arguments,
                anticipated_counterarguments=strategy_data.get(
                    "anticipated_counterarguments", []
                ),
                counterargument_responses=strategy_data.get(
                    "counterargument_responses", []
                ),
                strength_assessment=strategy_data.get("strength_assessment", 0.5),
                risk_factors=strategy_data.get("risk_factors", []),
                strategy_rationale=strategy_data.get("strategy_rationale", ""),
            )

            response = ArgumentDraftResponse(
                strategy=strategy,
                drafted_argument=final_state.get("drafted_argument", ""),
                argument_structure=final_state.get("argument_structure", {}),
                citations_used=final_state.get("citations_used", []),
                argument_strength=final_state.get("argument_strength", 0.5),
                precedent_coverage=final_state.get("precedent_coverage", 0.5),
                logical_coherence=final_state.get("logical_coherence", 0.5),
                total_critique_cycles=final_state.get("total_critique_cycles", 0),
                revision_history=final_state.get("revision_history"),
                execution_time=execution_time,
            )
            return response.model_dump()

        # Handle streaming vs non-streaming execution
        if request.stream:
            logger.info("Starting streaming argument drafting execution")
            return create_streaming_response(
                stream_graph_with_final_response(
                    drafting_graph, initial_state, process_drafting_response
                )
            )

        # Execute the drafting graph
        final_state = await drafting_graph.ainvoke(initial_state)

        # Use the same processing logic
        response_dict = await process_drafting_response(final_state)
        response = ArgumentDraftResponse(**response_dict)

        logger.info(f"Argument drafting completed in {response.execution_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error in argument drafting: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Argument drafting failed: {str(e)}",
        )


# Case File Management Endpoints
@app.post("/api/v1/case-files", response_model=Dict[str, int])
async def create_case_file(request: CreateCaseFileRequest):
    """Create a new case file for organizing documents and drafts."""
    try:
        case_file_id = CaseFileService.create_case_file(
            title=request.title,
            description=request.description,
            user_facts=request.user_facts,
            party_represented=request.party_represented,
        )

        logger.info(f"Created new case file with ID: {case_file_id}")
        return {"case_file_id": case_file_id}

    except Exception as e:
        logger.error(f"Error creating case file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create case file: {str(e)}",
        )


@app.get("/api/v1/case-files", response_model=List[CaseFileListItem])
async def list_case_files():
    """List all case files with basic information."""
    try:
        case_files = CaseFileService.list_case_files()
        return case_files

    except Exception as e:
        logger.error(f"Error listing case files: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list case files: {str(e)}",
        )


@app.get("/api/v1/case-files/{case_file_id}", response_model=CaseFileResponse)
async def get_case_file(case_file_id: int):
    """Get a specific case file with all its documents."""
    try:
        case_file = CaseFileService.get_case_file(case_file_id)

        if not case_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case file {case_file_id} not found",
            )

        return case_file

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting case file {case_file_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get case file: {str(e)}",
        )


@app.put("/api/v1/case-files/{case_file_id}")
async def update_case_file(case_file_id: int, request: UpdateCaseFileRequest):
    """Update case file details."""
    try:
        success = CaseFileService.update_case_file(
            case_file_id=case_file_id,
            title=request.title,
            description=request.description,
            user_facts=request.user_facts,
            party_represented=request.party_represented,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case file {case_file_id} not found",
            )

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating case file {case_file_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update case file: {str(e)}",
        )


@app.delete("/api/v1/case-files/{case_file_id}")
async def delete_case_file(case_file_id: int):
    """Delete a case file and all associated data."""
    try:
        success = CaseFileService.delete_case_file(case_file_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case file {case_file_id} not found",
            )

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting case file {case_file_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete case file: {str(e)}",
        )


@app.post("/api/v1/case-files/{case_file_id}/enrich-chunks")
async def enrich_case_file_chunks(case_file_id: int):
    """
    Enrich case file documents with chunk content from Neo4j.

    This endpoint fetches chunk content for documents that may be missing it,
    typically documents that were added before the chunk preservation feature.
    """
    try:
        case_file = CaseFileService.get_case_file(case_file_id)

        if not case_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case file {case_file_id} not found",
            )

        # The get_case_file method already includes enrichment
        logger.info(f"Enriched case file {case_file_id} with chunk content")
        return {"success": True, "message": "Case file enriched with chunk content"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enriching case file {case_file_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enrich case file: {str(e)}",
        )


@app.post("/api/v1/case-files/{case_file_id}/documents")
async def add_document_to_case_file(
    case_file_id: int, request: AddDocumentToCaseFileRequest
):
    """Add a document to a case file."""
    try:
        success = CaseFileService.add_document_to_case_file(
            case_file_id=case_file_id, document_data=request.model_dump()
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case file {case_file_id} not found",
            )

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error adding document to case file {case_file_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add document to case file: {str(e)}",
        )


@app.delete("/api/v1/case-files/{case_file_id}/documents/{document_id}")
async def remove_document_from_case_file(case_file_id: int, document_id: str):
    """Remove a document from a case file."""
    try:
        success = CaseFileService.remove_document_from_case_file(
            case_file_id=case_file_id, document_id=document_id
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case file or document not found",
            )

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error removing document from case file {case_file_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove document from case file: {str(e)}",
        )


@app.delete("/api/v1/case-files/{case_file_id}/documents")
async def remove_all_documents_from_case_file(case_file_id: int):
    """Remove all documents from a case file."""
    try:
        # Check if the case file exists first
        case_file = CaseFileService.get_case_file(case_file_id)
        if not case_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case file {case_file_id} not found",
            )
        
        # Delete all documents
        removed_count = CaseFileService.remove_all_documents_from_case_file(case_file_id)
        
        return {"success": True, "removed_count": removed_count}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error removing all documents from case file {case_file_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove all documents from case file: {str(e)}",
        )


@app.get("/api/v1/case-files/{case_file_id}/documents/{document_id}")
async def get_document_details(case_file_id: int, document_id: str):
    """Get detailed information about a specific document in a case file."""
    try:
        document = CaseFileService.get_document_from_case_file(
            case_file_id=case_file_id, document_id=document_id
        )

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found in case file",
            )

        return document

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting document {document_id} from case file {case_file_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document details: {str(e)}",
        )


# Case File Notes Management Endpoints
@app.post("/api/v1/case-files/{case_file_id}/notes", response_model=Dict[str, int])
async def add_note_to_case_file(
    case_file_id: int, request: AddCaseFileNoteRequest
):
    """Add a note to a case file."""
    try:
        note_id = CaseFileNoteService.add_note(
            case_file_id=case_file_id,
            content=request.content,
            author_type=request.author_type,
            author_name=request.author_name,
            note_type=request.note_type,
            tags=request.tags,
        )

        if not note_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case file not found",
            )

        return {"note_id": note_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding note to case file {case_file_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add note: {str(e)}",
        )


@app.put("/api/v1/case-files/{case_file_id}/notes/{note_id}")
async def update_case_file_note(
    case_file_id: int, note_id: int, request: UpdateCaseFileNoteRequest
):
    """Update a note in a case file."""
    try:
        success = CaseFileNoteService.update_note(
            note_id=note_id,
            content=request.content,
            note_type=request.note_type,
            tags=request.tags,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found",
            )

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating note {note_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update note: {str(e)}",
        )


@app.delete("/api/v1/case-files/{case_file_id}/notes/{note_id}")
async def delete_case_file_note(case_file_id: int, note_id: int):
    """Delete a note from a case file."""
    try:
        success = CaseFileNoteService.delete_note(note_id=note_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found",
            )

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting note {note_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete note: {str(e)}",
        )


# Argument Draft Management Endpoints
@app.post("/api/v1/drafts/save", response_model=Dict[str, int])
async def save_argument_draft(request: SaveDraftRequest):
    """Save an argument draft to a case file."""
    try:
        draft_id = ArgumentDraftService.save_draft(
            case_file_id=request.case_file_id,
            draft_response=request.draft_response,
            title=request.title,
        )

        logger.info(
            f"Saved argument draft with ID: {draft_id} to case file: {request.case_file_id}"
        )
        return {"draft_id": draft_id}

    except Exception as e:
        logger.error(f"Error saving argument draft: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save argument draft: {str(e)}",
        )


@app.get(
    "/api/v1/case-files/{case_file_id}/drafts",
    response_model=List[ArgumentDraftListItem],
)
async def list_drafts_for_case_file(case_file_id: int):
    """List all argument drafts for a case file."""
    try:
        drafts = ArgumentDraftService.list_drafts_for_case_file(case_file_id)
        return drafts

    except Exception as e:
        logger.error(
            f"Error listing drafts for case file {case_file_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list drafts: {str(e)}",
        )


@app.get("/api/v1/drafts/{draft_id}", response_model=SavedArgumentDraft)
async def get_argument_draft(draft_id: int):
    """Get a specific argument draft."""
    try:
        draft = ArgumentDraftService.get_draft(draft_id)

        if not draft:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {draft_id} not found",
            )

        return draft

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting draft {draft_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get draft: {str(e)}",
        )


@app.delete("/api/v1/drafts/{draft_id}")
async def delete_argument_draft(draft_id: int):
    """Delete an argument draft."""
    try:
        success = ArgumentDraftService.delete_draft(draft_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {draft_id} not found",
            )

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting draft {draft_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete draft: {str(e)}",
        )


@app.put("/api/v1/drafts/{draft_id}", response_model=Dict[str, bool])
async def update_draft(draft_id: int, request: UpdateDraftRequest):
    """Update a draft with manual edits."""
    try:
        success = ArgumentDraftService.update_draft(
            draft_id=draft_id,
            drafted_argument=request.drafted_argument,
            title=request.title
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {draft_id} not found",
            )

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating draft {draft_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update draft: {str(e)}",
        )


@app.post("/api/v1/drafts/ai-edit", response_model=ArgumentDraftResponse)
async def ai_edit_draft(request: EditDraftRequest):
    """Edit a draft using AI assistance."""
    start_time = time.time()

    try:
        logger.info(f"Starting AI edit for draft ID: {request.draft_id}")

        # Get the existing draft
        draft = ArgumentDraftService.get_draft(request.draft_id)
        if not draft:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Draft {request.draft_id} not found",
            )

        # Get the case file to provide context
        case_file_data = CaseFileService.get_case_file(draft["case_file_id"])
        if not case_file_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case file {draft['case_file_id']} not found",
            )

        # Extract user facts from case file
        user_facts = case_file_data.get('user_facts', '')
        if not user_facts or not user_facts.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Case file must contain case facts for AI editing",
            )

        # Build case file structure for the editing process
        case_file_for_graph = {
            "documents": case_file_data.get('documents', []),
            "total_documents": case_file_data.get('total_documents', 0),
            "created_at": case_file_data.get('created_at'),
        }

        # Initialize the drafting graph state for editing
        initial_state: DraftingState = {
            "user_facts": user_facts,
            "case_file": case_file_for_graph,
            "existing_draft": draft["drafted_argument"],
            "edit_instructions": request.edit_instructions,
            "is_editing": True,
        }

        # If the draft has strategy information, include it
        if draft.get("strategy"):
            initial_state["proposed_strategy"] = draft["strategy"]

        # Define response processor for streaming
        async def process_ai_edit_response(final_state):
            execution_time = time.time() - start_time

            # Extract strategy from final state
            strategy_data = final_state.get("proposed_strategy", {})

            # Convert to response format
            from app.models.schemas import ArgumentStrategy, LegalArgument
            import json

            # Handle key_arguments - it might be a JSON string or already parsed list
            key_arguments_raw = strategy_data.get("key_arguments", [])
            if isinstance(key_arguments_raw, str):
                try:
                    key_arguments_list = json.loads(key_arguments_raw)
                except (json.JSONDecodeError, TypeError):
                    key_arguments_list = []
            elif isinstance(key_arguments_raw, list):
                key_arguments_list = key_arguments_raw
            else:
                key_arguments_list = []

            # Ensure each argument has the required structure
            parsed_arguments = []
            for arg in key_arguments_list:
                if isinstance(arg, dict) and all(
                    key in arg
                    for key in ["argument", "supporting_authority", "factual_basis"]
                ):
                    parsed_arguments.append(LegalArgument(**arg))
                elif isinstance(arg, str):
                    # Fallback: create a basic argument structure from string
                    parsed_arguments.append(
                        LegalArgument(
                            argument=arg, supporting_authority="", factual_basis=""
                        )
                    )

            strategy = ArgumentStrategy(
                main_thesis=strategy_data.get("main_thesis", ""),
                argument_type=strategy_data.get("argument_type", "precedential"),
                primary_precedents=strategy_data.get("primary_precedents", []),
                legal_framework=strategy_data.get("legal_framework", ""),
                key_arguments=parsed_arguments,
                anticipated_counterarguments=strategy_data.get(
                    "anticipated_counterarguments", []
                ),
                counterargument_responses=strategy_data.get(
                    "counterargument_responses", []
                ),
                strength_assessment=strategy_data.get("strength_assessment", 0.5),
                risk_factors=strategy_data.get("risk_factors", []),
                strategy_rationale=strategy_data.get("strategy_rationale", ""),
            )

            response = ArgumentDraftResponse(
                strategy=strategy,
                drafted_argument=final_state.get("drafted_argument", ""),
                argument_structure=final_state.get("argument_structure", {}),
                citations_used=final_state.get("citations_used", []),
                argument_strength=final_state.get("argument_strength", 0.5),
                precedent_coverage=final_state.get("precedent_coverage", 0.5),
                logical_coherence=final_state.get("logical_coherence", 0.5),
                total_critique_cycles=final_state.get("total_critique_cycles", 0),
                revision_history=final_state.get("revision_history"),
                execution_time=execution_time,
            )

            # Update the draft in the database
            ArgumentDraftService.update_draft_with_response(
                draft_id=request.draft_id,
                draft_response=response
            )

            return response.model_dump()

        # Handle streaming vs non-streaming execution
        if request.stream:
            logger.info("Starting streaming AI draft editing execution")
            return create_streaming_response(
                stream_graph_with_final_response(
                    drafting_graph, initial_state, process_ai_edit_response
                )
            )

        # Execute the drafting graph for editing
        final_state = await drafting_graph.ainvoke(initial_state)

        # Use the same processing logic
        response_dict = await process_ai_edit_response(final_state)
        response = ArgumentDraftResponse(**response_dict)

        logger.info(f"AI draft editing completed in {response.execution_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error in AI draft editing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI draft editing failed: {str(e)}",
        )


# Moot Court Endpoints
@app.post("/api/v1/moot-court/generate-counterarguments", response_model=GenerateCounterArgumentsResponse)
async def generate_counterarguments(request: GenerateCounterArgumentsRequest):
    """
    Generate counterarguments for moot court practice using RAG-based analysis.
    
    This endpoint uses a sophisticated RAG workflow that leverages the Neo4j knowledge graph
    to find opposing precedents, analyze argument vulnerabilities, and generate comprehensive
    counterarguments with supporting rebuttals.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting RAG-based counterargument generation for case file ID: {request.case_file_id}")
        
        # Get the case file to retrieve context and documents
        case_file_data = CaseFileService.get_case_file(request.case_file_id)
        if not case_file_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case file {request.case_file_id} not found",
            )
        
        # Get the specific draft if provided
        draft_data = None
        key_arguments = []
        
        if request.draft_id:
            draft_data = ArgumentDraftService.get_draft(request.draft_id)
            if not draft_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Draft {request.draft_id} not found",
                )
            
            # Extract key arguments from the draft's strategy
            if draft_data.get("strategy") and draft_data["strategy"].get("key_arguments"):
                key_arguments = draft_data["strategy"]["key_arguments"]
        
        if not key_arguments:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No key arguments found in the selected draft",
            )
        
        # Extract user facts from case file for context
        user_facts = case_file_data.get('user_facts', '')
        if not user_facts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Case file must contain case facts for counterargument generation",
            )
        
        # Prepare case file documents for the graph
        case_file_documents = case_file_data.get('documents', [])
        
        # Initialize the counterargument graph state
        initial_state: CounterArgumentState = {
            "case_file_id": request.case_file_id,
            "user_facts": user_facts,
            "party_represented": case_file_data.get('party_represented'),
            "key_arguments": key_arguments,
            "case_file_documents": case_file_documents,
        }
        
        if request.draft_id:
            initial_state["draft_id"] = request.draft_id

        # Define response processor for streaming
        async def process_counterargument_response(final_state):
            # Extract results from the final state
            counterarguments = final_state.get("generated_counterarguments", [])
            
            # Handle both V1 and V2 rebuttal formats
            rebuttals_v1 = final_state.get("counterargument_rebuttals", [])  # V1 format: [[reb1, reb2], [reb3]]
            rebuttals_v2 = final_state.get("generated_rebuttals", [])        # V2 format: [reb1, reb2, reb3] with counterargument_index
            
            # Convert to response format
            response_counterarguments = []
            for ca in counterarguments:
                response_counterarguments.append(CounterArgument(
                    title=ca.get("title", ""),
                    argument=ca.get("argument", ""),
                    supporting_authority=ca.get("supporting_authority", ""),
                    factual_basis=ca.get("factual_basis", ""),
                    strength_assessment=ca.get("strength_assessment")
                ))
            
            response_rebuttals = []
            
            # Handle V1 format (list of lists)
            if rebuttals_v1:
                for rebuttal_group in rebuttals_v1:
                    group = []
                    for reb in rebuttal_group:
                        group.append(CounterArgumentRebuttal(
                            title=reb.get("title", ""),
                            content=reb.get("content", ""),
                            authority=reb.get("authority", "")
                        ))
                    response_rebuttals.append(group)
            
            # Handle V2 format (flat list with counterargument_index)
            elif rebuttals_v2:
                # Group rebuttals by counterargument_index
                rebuttal_groups = {}
                for reb in rebuttals_v2:
                    ca_index = reb.get("counterargument_index", 0)
                    if ca_index not in rebuttal_groups:
                        rebuttal_groups[ca_index] = []
                    
                    # Map V2 fields to V1 format
                    rebuttal_groups[ca_index].append(CounterArgumentRebuttal(
                        title=reb.get("strategy", ""),  # V2 uses "strategy" field
                        content=reb.get("content", ""),
                        authority=reb.get("authority", "")
                    ))
                
                # Convert to ordered list (ensure we have rebuttals for each counterargument)
                for i in range(len(counterarguments)):
                    if i in rebuttal_groups:
                        response_rebuttals.append(rebuttal_groups[i])
                    else:
                        response_rebuttals.append([])  # Empty list if no rebuttals for this counterargument
            
            execution_time = time.time() - start_time
            
            response = GenerateCounterArgumentsResponse(
                counterarguments=response_counterarguments,
                rebuttals=response_rebuttals,
                execution_time=execution_time
            )
            
            # Log RAG retrieval statistics
            research_comprehensiveness = final_state.get("research_comprehensiveness", 0.0)
            counterargument_strength = final_state.get("counterargument_strength", 0.0)
            rebuttal_quality = final_state.get("rebuttal_quality", 0.0)
            
            logger.info(
                f"RAG counterargument generation completed in {execution_time:.2f}s. "
                f"Research comprehensiveness: {research_comprehensiveness:.2f}, "
                f"Counterargument strength: {counterargument_strength:.2f}, "
                f"Rebuttal quality: {rebuttal_quality:.2f}"
            )
            
            return response.model_dump()
        
        # Handle streaming vs non-streaming execution
        if request.stream:
            logger.info("Starting streaming counterargument generation execution")
            return create_streaming_response(
                stream_graph_with_final_response(
                    counterargument_graph, initial_state, process_counterargument_response
                )
            )
        
        # Execute the counterargument graph workflow
        logger.info("Executing RAG-based counterargument generation workflow")
        final_state = await counterargument_graph.ainvoke(initial_state)
        
        # Use the same processing logic
        response_dict = await process_counterargument_response(final_state)
        response = GenerateCounterArgumentsResponse(**response_dict)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in RAG counterargument generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Counterargument generation failed: {str(e)}",
        )


@app.post("/api/v1/moot-court/save-session", response_model=Dict[str, int])
async def save_moot_court_session(request: SaveMootCourtSessionRequest):
    """Save a moot court session to the database."""
    try:
        # Convert Pydantic models to dict format for storage
        counterarguments_data = [ca.model_dump() for ca in request.counterarguments]
        rebuttals_data = [[reb.model_dump() for reb in group] for group in request.rebuttals]
        
        session_id = MootCourtSessionService.save_session(
            case_file_id=request.case_file_id,
            draft_id=request.draft_id,
            title=request.title,
            counterarguments=counterarguments_data,
            rebuttals=rebuttals_data,
            source_arguments=request.source_arguments,
            research_context=request.research_context,
            counterargument_strength=request.counterargument_strength,
            research_comprehensiveness=request.research_comprehensiveness,
            rebuttal_quality=request.rebuttal_quality,
            execution_time=request.execution_time,
        )

        logger.info(f"Saved moot court session with ID: {session_id}")
        return {"session_id": session_id}

    except Exception as e:
        logger.error(f"Error saving moot court session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save moot court session: {str(e)}",
        )


@app.get(
    "/api/v1/case-files/{case_file_id}/moot-court-sessions",
    response_model=List[MootCourtSessionListItem],
)
async def list_moot_court_sessions_for_case_file(case_file_id: int):
    """List all moot court sessions for a case file."""
    try:
        sessions = MootCourtSessionService.list_sessions_for_case_file(case_file_id)
        return sessions

    except Exception as e:
        logger.error(
            f"Error listing moot court sessions for case file {case_file_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list moot court sessions: {str(e)}",
        )


@app.get("/api/v1/moot-court-sessions/{session_id}", response_model=SavedMootCourtSession)
async def get_moot_court_session(session_id: int):
    """Get a specific moot court session."""
    try:
        session = MootCourtSessionService.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Moot court session {session_id} not found",
            )

        # Convert stored data back to Pydantic models
        counterarguments = [CounterArgument(**ca) for ca in session["counterarguments"]]
        rebuttals = [[CounterArgumentRebuttal(**reb) for reb in group] for group in session["rebuttals"]]

        return SavedMootCourtSession(
            id=session["id"],
            case_file_id=session["case_file_id"],
            draft_id=session["draft_id"],
            title=session["title"],
            counterarguments=counterarguments,
            rebuttals=rebuttals,
            source_arguments=session["source_arguments"],
            research_context=session["research_context"],
            counterargument_strength=session["counterargument_strength"],
            research_comprehensiveness=session["research_comprehensiveness"],
            rebuttal_quality=session["rebuttal_quality"],
            execution_time=session["execution_time"],
            created_at=session["created_at"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting moot court session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get moot court session: {str(e)}",
        )


@app.delete("/api/v1/moot-court-sessions/{session_id}")
async def delete_moot_court_session(session_id: int):
    """Delete a moot court session."""
    try:
        success = MootCourtSessionService.delete_session(session_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Moot court session {session_id} not found",
            )

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting moot court session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete moot court session: {str(e)}",
        )


@app.put("/api/v1/moot-court-sessions/{session_id}/title")
async def update_moot_court_session_title(session_id: int, request: Dict[str, str]):
    """Update a moot court session's title."""
    try:
        title = request.get("title")
        if not title:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Title is required",
            )

        success = MootCourtSessionService.update_session_title(session_id, title)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Moot court session {session_id} not found",
            )

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating moot court session title {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update moot court session title: {str(e)}",
        )


# Additional utility endpoints
@app.get("/api/v1/research/precedent-analysis/{case_citation}")
async def analyze_precedent(case_citation: str):
    """Analyze the precedential strength of a specific case."""
    try:
        from app.tools.neo4j_tools import assess_precedent_strength

        result = assess_precedent_strength(case_citation)

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case not found or analysis failed: {result['error']}",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing precedent {case_citation}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Precedent analysis failed: {str(e)}",
        )


@app.get("/api/v1/research/citation-network/{case_citation}")
async def get_citation_network(case_citation: str, direction: str = "both"):
    """Get citation network for a specific case."""
    try:
        from app.tools.neo4j_tools import find_case_citations

        if direction not in ["both", "cited_by", "cites"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Direction must be one of: both, cited_by, cites",
            )

        result = find_case_citations(case_citation, direction)
        return {"citation_network": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting citation network for {case_citation}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Citation network analysis failed: {str(e)}",
        )


# Streaming helper function
async def stream_graph_with_final_response(graph, initial_state, response_processor, chunk_processor=None, stream_mode=["values", "custom"], config=None):
    """
    Stream the execution of a LangGraph and send a final processed response.
    
    Args:
        graph: The LangGraph to execute
        initial_state: Initial state for the graph
        response_processor: Function that takes final_state and returns processed response
        chunk_processor: Optional function to process each chunk before yielding
        stream_mode: Stream mode for the graph execution
    """
    final_state = None

    if config is None:
        config = {}

    if chunk_processor is None:
        async def chunk_processor(chunk, stream_mode):
            def process_chunk(chunk_dict):
                processed_chunk = {}
                for key, value in chunk_dict.items():
                    if isinstance(value, list):
                        processed_chunk[f"{key}_len"] = len(value)
                    elif isinstance(value, dict):
                        processed_chunk[key] = process_chunk(value)
                    else:
                        processed_chunk[key] = value
                return processed_chunk
            processed_chunk = {}
            stream_type = stream_mode
            if isinstance(chunk, tuple):
                stream_type, chunk = chunk
            processed_chunk = process_chunk(chunk)
            if stream_type:
                processed_chunk = {
                    "stream_type": stream_type,
                    "data": processed_chunk
                }
            return processed_chunk
            
    # Stream the graph execution
    try:
        async for chunk in graph.astream(initial_state, config, stream_mode=stream_mode):
            yield f"data: {json.dumps(await chunk_processor(chunk, stream_mode), default=str)}\n\n"
            # Keep track of the final state
            if isinstance(chunk, tuple):
                _stream_type, chunk = chunk
            if isinstance(chunk, dict):
                final_state = chunk
    except GraphRecursionError as e:
        pass
    
    # Process the final response
    if final_state and response_processor:
        try:
            processed_response = await response_processor(final_state)
            final_chunk = {
                "streaming_complete": True,
                "final_response": processed_response
            }
            yield f"data: {json.dumps(final_chunk, default=str)}\n\n"
        except Exception as e:
            raise
            error_chunk = {
                "streaming_complete": True,
                "error": str(e)
            }
            yield f"data: {json.dumps(error_chunk, default=str)}\n\n"

def create_streaming_response(generator):
    """Create a streaming response from an async generator."""
    return StreamingResponse(
        generator,
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


if __name__ == "__main__":
    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")
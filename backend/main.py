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
from fastapi.responses import JSONResponse
import uvicorn
from google import genai

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
    SaveDraftRequest,
    ArgumentDraftListItem,
    SavedArgumentDraft,
)
from app.graphs import research_graph, drafting_graph
from app.graphs.state import ResearchState, DraftingState
from app.tools.neo4j_tools import close_neo4j_connection
from app.services.case_file_service import CaseFileService, ArgumentDraftService
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    # Startup
    logger.info("Starting Legal Research Assistant Backend")

    # Initialize database tables
    try:
        create_tables()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    yield
    # Shutdown
    logger.info("Shutting down Legal Research Assistant Backend")
    close_neo4j_connection()


app = FastAPI(
    title="Legal Research Assistant API",
    description="Backend API for AI-powered legal research and argument drafting",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logging.error(f"{request}: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


# Create FastAPI application


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
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
        ).dict(),
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


@app.post("/api/v1/research/query", response_model=ResearchQueryResponse)
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

        # Execute the research graph
        final_state = await research_graph.ainvoke(initial_state)

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

        logger.info(
            f"Research query completed in {execution_time:.2f}s, found {len(retrieved_docs)} documents"
        )
        return response

    except Exception as e:
        logger.error(f"Error in research query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research query failed: {str(e)}",
        )


@app.post("/api/v1/generation/draft-argument", response_model=ArgumentDraftResponse)
async def draft_argument(request: ArgumentDraftRequest):
    """
    Generate legal argument using the Drafting Graph.

    This endpoint initiates the drafting workflow which develops a legal strategy,
    critiques it through iterative refinement, and generates a final legal argument.
    """
    start_time = time.time()

    try:
        logger.info(f"Received request: {request}")
        logger.info(f"Case file: {request.case_file}")
        logger.info(
            f"Starting argument drafting for case with {len(request.case_file.documents)} precedents"
        )

        # Initialize the drafting graph state
        initial_state: DraftingState = {
            "user_facts": request.user_facts,
            "case_file": request.case_file.model_dump(),
        }

        # Add optional legal question
        if request.legal_question:
            initial_state["legal_question"] = request.legal_question

        # Execute the drafting graph
        final_state = await drafting_graph.ainvoke(initial_state)

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

        logger.info(f"Argument drafting completed in {execution_time:.2f}s")
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
            legal_question=request.legal_question,
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
            legal_question=request.legal_question,
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
            case_file_id=case_file_id, document_data=request.dict()
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


# Modified argument drafting endpoint to optionally save the draft
@app.post("/api/v1/generation/draft-argument", response_model=ArgumentDraftResponse)
async def draft_argument(
    request: ArgumentDraftRequest, save_to_case_file: Optional[int] = None
):
    """
    Generate legal argument using the Drafting Graph.

    Optionally save the generated draft to a specified case file.
    """
    start_time = time.time()

    try:
        logger.info(f"Received request: {request}")
        logger.info(f"Case file: {request.case_file}")
        logger.info(
            f"Starting argument drafting for case with {len(request.case_file.documents)} precedents"
        )

        # Initialize the drafting graph state
        initial_state: DraftingState = {
            "user_facts": request.user_facts,
            "case_file": request.case_file.model_dump(),
        }

        # Add optional legal question
        if request.legal_question:
            initial_state["legal_question"] = request.legal_question

        # Execute the drafting graph
        final_state = await drafting_graph.ainvoke(initial_state)

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

        # Optionally save to case file
        if save_to_case_file:
            try:
                draft_id = ArgumentDraftService.save_draft(
                    case_file_id=save_to_case_file, draft_response=response
                )
                logger.info(
                    f"Saved draft with ID {draft_id} to case file {save_to_case_file}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to save draft to case file {save_to_case_file}: {e}"
                )
                # Don't fail the whole request if saving fails

        logger.info(f"Argument drafting completed in {execution_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error in argument drafting: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Argument drafting failed: {str(e)}",
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


if __name__ == "__main__":
    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")

"""
Database Mutation Tools for ReAct Agents

This module provides tools that allow ReAct agents to mutate the database state
during research and drafting workflows. These tools enable agents to save 
discovered documents, create case files, save drafts, and manage legal research
sessions in real-time.

## Core Database Mutation Tools

### 1. Case File Management
- `create_case_file()`: Create new case files for organizing research
- `update_case_file()`: Update case file details and facts
- `delete_case_file()`: Remove case files and associated data

### 2. Document Management
- `add_document_to_case_file()`: Add discovered documents to case files
- `remove_document_from_case_file()`: Remove documents from case files
- `update_document_notes()`: Add research notes to specific documents

### 3. Draft Management
- `save_argument_draft()`: Save generated legal argument drafts
- `update_argument_draft()`: Update existing drafts with new content
- `delete_argument_draft()`: Remove argument drafts

### 4. Moot Court Session Management
- `save_moot_court_session()`: Save counterargument analysis sessions
- `update_moot_court_session()`: Update session details
- `delete_moot_court_session()`: Remove moot court sessions

## Usage Example

```python
from app.tools.database_tools import LegalDatabaseTools

# Initialize tools
db_tools = LegalDatabaseTools()

# Create a case file during research
case_file_id = db_tools.create_case_file(
    title="Contract Breach Analysis", 
    user_facts="Client entered into supply contract...",
    party_represented="Plaintiff"
)

# Add discovered documents
success = db_tools.add_document_to_case_file(
    case_file_id=case_file_id,
    document_data={
        "document_id": "chunk_123",
        "citation": "[2020] SGHC 45",
        "title": "ABC Corp v XYZ Ltd",
        "relevance_score_percent": 85.5,
        "key_holdings": ["Contract formation requires consideration"]
    }
)

# Save an argument draft
draft_id = db_tools.save_argument_draft(
    case_file_id=case_file_id,
    title="Breach of Contract Argument",
    drafted_argument="The plaintiff has established...",
    strategy_data={...}
)
```
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import uuid
import time
import logging

from pydantic import BaseModel, Field
from langgraph.config import get_stream_writer

from app.services.case_file_service import CaseFileService, ArgumentDraftService, MootCourtSessionService, CaseFileNoteService
from app.models.schemas import (
    ArgumentDraftResponse,
    CounterArgument,
    CounterArgumentRebuttal,
)

logger = logging.getLogger(__name__)


class CaseFileResult(BaseModel):
    """Result model for case file operations."""
    
    case_file_id: int
    title: str
    success: bool = True
    message: Optional[str] = None


class DocumentResult(BaseModel):
    """Result model for document operations."""
    
    chunk_id: str
    case_file_id: int
    success: bool = True
    message: Optional[str] = None


class DraftResult(BaseModel):
    """Result model for draft operations."""
    
    draft_id: int
    case_file_id: int
    title: str
    success: bool = True
    message: Optional[str] = None


class MootCourtResult(BaseModel):
    """Result model for moot court session operations."""
    
    session_id: int
    case_file_id: int
    title: str
    success: bool = True
    message: Optional[str] = None


class NoteResult(BaseModel):
    """Result model for note operations."""
    
    note_id: int
    case_file_id: int
    success: bool = True
    message: Optional[str] = None


class LegalDatabaseTools:
    """
    Database mutation tools for ReAct agents in legal research and drafting workflows.
    
    These tools allow agents to save their research findings, create case files,
    and manage legal documents and drafts during the research process.
    """
    
    def __init__(self):
        """Initialize the database tools."""
        pass
    
    # =================== Case File Management Tools ===================
    
    def create_case_file(
        self,
        title: str,
        description: Optional[str] = None,
        user_facts: Optional[str] = None,
        party_represented: Optional[str] = None,
    ) -> CaseFileResult:
        """
        Create a new case file for organizing legal research and documents.
        
        Args:
            title: Title for the case file
            description: Optional description of the case
            user_facts: Client's factual situation or case background
            party_represented: Which party the user represents (e.g., 'Plaintiff', 'Defendant')
            
        Returns:
            Result with case file ID and status
        """
        step_id = f"create_case_file_{uuid.uuid4().hex[:8]}"
        writer = get_stream_writer()
        writer({
            "step_id": step_id,
            "status": "in_progress",
            "brief_description": "Creating case file",
            "description": f"Creating new case file: {title}"
        })
        
        start_time = time.time()
        
        try:
            logger.info(f"Creating new case file: {title}")
            
            case_file_id = CaseFileService.create_case_file(
                title=title,
                description=description,
                user_facts=user_facts,
                party_represented=party_represented,
            )
            
            logger.info(f"Successfully created case file with ID: {case_file_id}")
            
            execution_time = time.time() - start_time
            writer({
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Case file created",
                "description": f"Created case file '{title}' with ID {case_file_id} in {execution_time:.2f}s"
            })
            
            return CaseFileResult(
                case_file_id=case_file_id,
                title=title,
                success=True,
                message=f"Case file '{title}' created successfully with ID {case_file_id}"
            )
            
        except Exception as e:
            logger.error(f"Error creating case file: {e}", exc_info=True)
            writer({
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Case file creation failed",
                "description": f"Failed to create case file: {str(e)}"
            })
            return CaseFileResult(
                case_file_id=-1,
                title=title,
                success=False,
                message=f"Failed to create case file: {str(e)}"
            )
    
    def update_case_file(
        self,
        case_file_id: int,
        title: Optional[str] = None,
        description: Optional[str] = None,
        user_facts: Optional[str] = None,
        party_represented: Optional[str] = None,
    ) -> CaseFileResult:
        """
        Update an existing case file with new information.
        
        Args:
            case_file_id: ID of the case file to update
            title: Updated title
            description: Updated description
            user_facts: Updated factual situation
            party_represented: Updated party representation
            
        Returns:
            Result with update status
        """
        step_id = f"update_case_file_{uuid.uuid4().hex[:8]}"
        writer = get_stream_writer()
        writer({
            "step_id": step_id,
            "status": "in_progress",
            "brief_description": "Updating case file",
            "description": f"Updating case file ID: {case_file_id}"
        })
        
        start_time = time.time()
        
        try:
            logger.info(f"Updating case file ID: {case_file_id}")
            
            success = CaseFileService.update_case_file(
                case_file_id=case_file_id,
                title=title,
                description=description,
                user_facts=user_facts,
                party_represented=party_represented,
            )
            
            if not success:
                writer({
                    "step_id": step_id,
                    "status": "complete",
                    "brief_description": "Case file not found",
                    "description": f"Case file with ID {case_file_id} not found"
                })
                return CaseFileResult(
                    case_file_id=case_file_id,
                    title=title or "Unknown",
                    success=False,
                    message=f"Case file with ID {case_file_id} not found"
                )
            
            logger.info(f"Successfully updated case file ID: {case_file_id}")
            
            execution_time = time.time() - start_time
            writer({
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Case file updated",
                "description": f"Updated case file {case_file_id} in {execution_time:.2f}s"
            })
            
            return CaseFileResult(
                case_file_id=case_file_id,
                title=title or "Updated",
                success=True,
                message=f"Case file {case_file_id} updated successfully"
            )
            
        except Exception as e:
            logger.error(f"Error updating case file {case_file_id}: {e}", exc_info=True)
            writer({
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Update failed",
                "description": f"Failed to update case file: {str(e)}"
            })
            return CaseFileResult(
                case_file_id=case_file_id,
                title=title or "Unknown",
                success=False,
                message=f"Failed to update case file: {str(e)}"
            )
    
    def delete_case_file(self, case_file_id: int) -> CaseFileResult:
        """
        Delete a case file and all its associated data.
        
        Args:
            case_file_id: ID of the case file to delete
            
        Returns:
            Result with deletion status
        """
        step_id = f"delete_case_file_{uuid.uuid4().hex[:8]}"
        writer = get_stream_writer()
        writer({
            "step_id": step_id,
            "status": "in_progress",
            "brief_description": "Deleting case file",
            "description": f"Deleting case file ID: {case_file_id}"
        })
        
        start_time = time.time()
        
        try:
            logger.info(f"Deleting case file ID: {case_file_id}")
            
            success = CaseFileService.delete_case_file(case_file_id)
            
            if not success:
                writer({
                    "step_id": step_id,
                    "status": "complete",
                    "brief_description": "Case file not found",
                    "description": f"Case file with ID {case_file_id} not found"
                })
                return CaseFileResult(
                    case_file_id=case_file_id,
                    title="Unknown",
                    success=False,
                    message=f"Case file with ID {case_file_id} not found"
                )
            
            logger.info(f"Successfully deleted case file ID: {case_file_id}")
            
            execution_time = time.time() - start_time
            writer({
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Case file deleted",
                "description": f"Deleted case file {case_file_id} in {execution_time:.2f}s"
            })
            
            return CaseFileResult(
                case_file_id=case_file_id,
                title="Deleted",
                success=True,
                message=f"Case file {case_file_id} deleted successfully"
            )
            
        except Exception as e:
            logger.error(f"Error deleting case file {case_file_id}: {e}", exc_info=True)
            return CaseFileResult(
                case_file_id=case_file_id,
                title="Unknown",
                success=False,
                message=f"Failed to delete case file: {str(e)}"
            )
    
    # =================== Document Management Tools ===================
    
    def add_document_to_case_file(
        self,
        case_file_id: int,
        chunk_id: str,
        relevance_score_percent: Optional[float] = None,
        user_notes: Optional[str] = None,
    ) -> DocumentResult:
        """
        Add a discovered document to a case file.
        
        Args:
            case_file_id: ID of the case file
            chunk_id: Unique identifier for the document (chunk ID)
            relevance_score_percent: Optional relevance score from 0-100
            user_notes: Optional notes from the user about the document
            
        Returns:
            Result with addition status
        """
        step_id = f"add_document_{uuid.uuid4().hex[:8]}"
        writer = get_stream_writer()
        writer({
            "step_id": step_id,
            "status": "in_progress",
            "brief_description": "Adding document to case file",
            "description": f"Adding document {chunk_id} to case file {case_file_id}"
        })
        
        start_time = time.time()
        
        try:
            logger.info(f"Adding document {chunk_id} to case file {case_file_id}")

            document_data = {
                "chunk_id": chunk_id,
                "user_notes": user_notes,
                "relevance_score_percent": relevance_score_percent,
            }
            
            success = CaseFileService.add_document_to_case_file(
                case_file_id=case_file_id,
                document_data=document_data
            )
            
            if not success:
                writer({
                    "step_id": step_id,
                    "status": "complete",
                    "brief_description": "Document addition failed",
                    "description": f"Failed to add document to case file {case_file_id} (case file may not exist)"
                })
                return DocumentResult(
                    chunk_id=chunk_id,
                    case_file_id=case_file_id,
                    success=False,
                    message=f"Failed to add document to case file {case_file_id} (case file may not exist)"
                )
            
            logger.info(f"Successfully added document {chunk_id} to case file {case_file_id}")
            
            execution_time = time.time() - start_time
            writer({
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Document added",
                "description": f"Added document '{chunk_id}' to case file {case_file_id} in {execution_time:.2f}s"
            })
            
            return DocumentResult(
                chunk_id=chunk_id,
                case_file_id=case_file_id,
                success=True,
                message=f"Document '{chunk_id}' added to case file {case_file_id}"
            )
            
        except Exception as e:
            logger.error(f"Error adding document to case file: {e}", exc_info=True)
            writer({
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Document addition failed",
                "description": f"Failed to add document: {str(e)}"
            })
            return DocumentResult(
                chunk_id=chunk_id,
                case_file_id=case_file_id,
                success=False,
                message=f"Failed to add document: {str(e)}"
            )
    
    def remove_document_from_case_file(
        self,
        case_file_id: int,
        chunk_id: str
    ) -> DocumentResult:
        """
        Remove a document from a case file.
        
        Args:
            case_file_id: ID of the case file
            document_id: ID of the document to remove
            
        Returns:
            Result with removal status
        """
        try:
            logger.info(f"Removing document {chunk_id} from case file {case_file_id}")
            
            success = CaseFileService.remove_document_from_case_file(
                case_file_id=case_file_id,
                chunk_id=chunk_id
            )
            
            if not success:
                return DocumentResult(
                    chunk_id=chunk_id,
                    case_file_id=case_file_id,
                    success=False,
                    message=f"Document {chunk_id} not found in case file {case_file_id}"
                )
            
            logger.info(f"Successfully removed document {chunk_id} from case file {case_file_id}")
            
            return DocumentResult(
                chunk_id=chunk_id,
                case_file_id=case_file_id,
                success=True,
                message=f"Document {chunk_id} removed from case file {case_file_id}"
            )
            
        except Exception as e:
            logger.error(f"Error removing document from case file: {e}", exc_info=True)
            return DocumentResult(
                chunk_id=chunk_id,
                case_file_id=case_file_id,
                success=False,
                message=f"Failed to remove document: {str(e)}"
            )
    
    # =================== Argument Draft Management Tools ===================
    
    def save_argument_draft(
        self,
        case_file_id: int,
        title: str,
        drafted_argument: str,
        strategy_data: Optional[Dict[str, Any]] = None,
        argument_structure: Optional[Dict[str, Any]] = None,
        citations_used: Optional[List[str]] = None,
        argument_strength: Optional[float] = None,
        precedent_coverage: Optional[float] = None,
        logical_coherence: Optional[float] = None,
        total_critique_cycles: Optional[int] = None,
        execution_time: Optional[float] = None,
        revision_history: Optional[List[Dict[str, Any]]] = None,
    ) -> DraftResult:
        """
        Save a generated argument draft to a case file.
        
        Args:
            case_file_id: ID of the case file
            title: Title for the draft
            drafted_argument: The main argument text
            strategy_data: Legal strategy information
            argument_structure: Structure analysis of the argument
            citations_used: List of citations used in the argument
            argument_strength: Strength score (0-1)
            precedent_coverage: Coverage score (0-1)
            logical_coherence: Coherence score (0-1)
            total_critique_cycles: Number of revision cycles
            execution_time: Time taken to generate
            revision_history: History of revisions made
            
        Returns:
            Result with draft ID and status
        """
        try:
            logger.info(f"Saving argument draft '{title}' to case file {case_file_id}")
            
            # Create a mock ArgumentDraftResponse for compatibility
            draft_response = ArgumentDraftResponse(
                strategy={
                    "main_thesis": strategy_data.get("main_thesis", "") if strategy_data else "",
                    "argument_type": strategy_data.get("argument_type", "precedential") if strategy_data else "precedential",
                    "primary_precedents": strategy_data.get("primary_precedents", []) if strategy_data else [],
                    "legal_framework": strategy_data.get("legal_framework", "") if strategy_data else "",
                    "key_arguments": strategy_data.get("key_arguments", []) if strategy_data else [],
                    "anticipated_counterarguments": strategy_data.get("anticipated_counterarguments", []) if strategy_data else [],
                    "counterargument_responses": strategy_data.get("counterargument_responses", []) if strategy_data else [],
                    "strength_assessment": strategy_data.get("strength_assessment", 0.8) if strategy_data else 0.8,
                    "risk_factors": strategy_data.get("risk_factors", []) if strategy_data else [],
                    "strategy_rationale": strategy_data.get("strategy_rationale", "") if strategy_data else "",
                },
                drafted_argument=drafted_argument,
                argument_structure=argument_structure or {},
                citations_used=citations_used or [],
                argument_strength=argument_strength or 0.8,
                precedent_coverage=precedent_coverage or 0.8,
                logical_coherence=logical_coherence or 0.8,
                total_critique_cycles=total_critique_cycles or 1,
                revision_history=revision_history,
                execution_time=execution_time,
            )
            
            draft_id = ArgumentDraftService.save_draft(
                case_file_id=case_file_id,
                draft_response=draft_response,
                title=title,
            )
            
            logger.info(f"Successfully saved argument draft with ID: {draft_id}")
            
            return DraftResult(
                draft_id=draft_id,
                case_file_id=case_file_id,
                title=title,
                success=True,
                message=f"Argument draft '{title}' saved with ID {draft_id}"
            )
            
        except Exception as e:
            logger.error(f"Error saving argument draft: {e}", exc_info=True)
            return DraftResult(
                draft_id=-1,
                case_file_id=case_file_id,
                title=title,
                success=False,
                message=f"Failed to save argument draft: {str(e)}"
            )
    
    def update_argument_draft(
        self,
        draft_id: int,
        drafted_argument: str,
        title: Optional[str] = None,
    ) -> DraftResult:
        """
        Update an existing argument draft with new content.
        
        Args:
            draft_id: ID of the draft to update
            drafted_argument: Updated argument text
            title: Updated title for the draft
            
        Returns:
            Result with update status
        """
        try:
            logger.info(f"Updating argument draft ID: {draft_id}")
            
            success = ArgumentDraftService.update_draft(
                draft_id=draft_id,
                drafted_argument=drafted_argument,
                title=title,
            )
            
            if not success:
                return DraftResult(
                    draft_id=draft_id,
                    case_file_id=-1,
                    title=title or "Unknown",
                    success=False,
                    message=f"Argument draft with ID {draft_id} not found"
                )
            
            logger.info(f"Successfully updated argument draft ID: {draft_id}")
            
            return DraftResult(
                draft_id=draft_id,
                case_file_id=-1,  # We don't have this info in the update method
                title=title or "Updated",
                success=True,
                message=f"Argument draft {draft_id} updated successfully"
            )
            
        except Exception as e:
            logger.error(f"Error updating argument draft {draft_id}: {e}", exc_info=True)
            return DraftResult(
                draft_id=draft_id,
                case_file_id=-1,
                title=title or "Unknown",
                success=False,
                message=f"Failed to update argument draft: {str(e)}"
            )
    
    def delete_argument_draft(self, draft_id: int) -> DraftResult:
        """
        Delete an argument draft.
        
        Args:
            draft_id: ID of the draft to delete
            
        Returns:
            Result with deletion status
        """
        try:
            logger.info(f"Deleting argument draft ID: {draft_id}")
            
            success = ArgumentDraftService.delete_draft(draft_id)
            
            if not success:
                return DraftResult(
                    draft_id=draft_id,
                    case_file_id=-1,
                    title="Unknown",
                    success=False,
                    message=f"Argument draft with ID {draft_id} not found"
                )
            
            logger.info(f"Successfully deleted argument draft ID: {draft_id}")
            
            return DraftResult(
                draft_id=draft_id,
                case_file_id=-1,
                title="Deleted",
                success=True,
                message=f"Argument draft {draft_id} deleted successfully"
            )
            
        except Exception as e:
            logger.error(f"Error deleting argument draft {draft_id}: {e}", exc_info=True)
            return DraftResult(
                draft_id=draft_id,
                case_file_id=-1,
                title="Unknown",
                success=False,
                message=f"Failed to delete argument draft: {str(e)}"
            )
    
    # =================== Moot Court Session Management Tools ===================
    
    def save_moot_court_session(
        self,
        case_file_id: int,
        title: str,
        counterarguments: List[Dict[str, Any]],
        rebuttals: List[List[Dict[str, Any]]],
        draft_id: Optional[int] = None,
        source_arguments: Optional[List[Dict[str, Any]]] = None,
        research_context: Optional[Dict[str, Any]] = None,
        counterargument_strength: Optional[float] = None,
        research_comprehensiveness: Optional[float] = None,
        rebuttal_quality: Optional[float] = None,
        execution_time: Optional[float] = None,
    ) -> MootCourtResult:
        """
        Save a moot court counterargument analysis session.
        
        Args:
            case_file_id: ID of the case file
            title: Title for the moot court session
            counterarguments: List of counterarguments generated
            rebuttals: List of rebuttals for each counterargument
            draft_id: ID of the draft that was analyzed
            source_arguments: Original arguments that were analyzed
            research_context: RAG retrieval context used
            counterargument_strength: Quality score for counterarguments
            research_comprehensiveness: Comprehensiveness of research
            rebuttal_quality: Quality score for rebuttals
            execution_time: Time taken for analysis
            
        Returns:
            Result with session ID and status
        """
        try:
            logger.info(f"Saving moot court session '{title}' for case file {case_file_id}")
            
            # Convert dict counterarguments to CounterArgument objects
            ca_objects = []
            for ca in counterarguments:
                ca_obj = CounterArgument(
                    title=ca.get("title", ""),
                    argument=ca.get("argument", ""),
                    supporting_authority=ca.get("supporting_authority", ""),
                    factual_basis=ca.get("factual_basis", ""),
                    strength_assessment=ca.get("strength_assessment"),
                )
                ca_objects.append(ca_obj)
            
            # Convert dict rebuttals to CounterArgumentRebuttal objects
            rebuttal_groups = []
            for rebuttal_group in rebuttals:
                rebuttal_objects = []
                for reb in rebuttal_group:
                    reb_obj = CounterArgumentRebuttal(
                        title=reb.get("title", ""),
                        content=reb.get("content", ""),
                        authority=reb.get("authority", ""),
                    )
                    rebuttal_objects.append(reb_obj)
                rebuttal_groups.append(rebuttal_objects)
            
            session_id = MootCourtSessionService.save_session(
                case_file_id=case_file_id,
                draft_id=draft_id,
                title=title,
                counterarguments=ca_objects,
                rebuttals=rebuttal_groups,
                source_arguments=source_arguments,
                research_context=research_context,
                counterargument_strength=counterargument_strength,
                research_comprehensiveness=research_comprehensiveness,
                rebuttal_quality=rebuttal_quality,
                execution_time=execution_time,
            )
            
            logger.info(f"Successfully saved moot court session with ID: {session_id}")
            
            return MootCourtResult(
                session_id=session_id,
                case_file_id=case_file_id,
                title=title,
                success=True,
                message=f"Moot court session '{title}' saved with ID {session_id}"
            )
            
        except Exception as e:
            logger.error(f"Error saving moot court session: {e}", exc_info=True)
            return MootCourtResult(
                session_id=-1,
                case_file_id=case_file_id,
                title=title,
                success=False,
                message=f"Failed to save moot court session: {str(e)}"
            )
    
    def update_moot_court_session_title(
        self,
        session_id: int,
        title: str,
    ) -> MootCourtResult:
        """
        Update the title of a moot court session.
        
        Args:
            session_id: ID of the session to update
            title: New title for the session
            
        Returns:
            Result with update status
        """
        try:
            logger.info(f"Updating moot court session ID: {session_id}")
            
            success = MootCourtSessionService.update_session_title(
                session_id=session_id,
                title=title,
            )
            
            if not success:
                return MootCourtResult(
                    session_id=session_id,
                    case_file_id=-1,
                    title=title,
                    success=False,
                    message=f"Moot court session with ID {session_id} not found"
                )
            
            logger.info(f"Successfully updated moot court session ID: {session_id}")
            
            return MootCourtResult(
                session_id=session_id,
                case_file_id=-1,
                title=title,
                success=True,
                message=f"Moot court session {session_id} title updated successfully"
            )
            
        except Exception as e:
            logger.error(f"Error updating moot court session {session_id}: {e}", exc_info=True)
            return MootCourtResult(
                session_id=session_id,
                case_file_id=-1,
                title=title,
                success=False,
                message=f"Failed to update moot court session: {str(e)}"
            )
    
    def delete_moot_court_session(self, session_id: int) -> MootCourtResult:
        """
        Delete a moot court session.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            Result with deletion status
        """
        try:
            logger.info(f"Deleting moot court session ID: {session_id}")
            
            success = MootCourtSessionService.delete_session(session_id)
            
            if not success:
                return MootCourtResult(
                    session_id=session_id,
                    case_file_id=-1,
                    title="Unknown",
                    success=False,
                    message=f"Moot court session with ID {session_id} not found"
                )
            
            logger.info(f"Successfully deleted moot court session ID: {session_id}")
            
            return MootCourtResult(
                session_id=session_id,
                case_file_id=-1,
                title="Deleted",
                success=True,
                message=f"Moot court session {session_id} deleted successfully"
            )
            
        except Exception as e:
            logger.error(f"Error deleting moot court session {session_id}: {e}", exc_info=True)
            return MootCourtResult(
                session_id=session_id,
                case_file_id=-1,
                title="Unknown",
                success=False,
                message=f"Failed to delete moot court session: {str(e)}"
            )

    # =================== Case File Notes Management Tools ===================
    
    def add_ai_note_to_case_file(
        self,
        case_file_id: int,
        content: str,
        author_name: Optional[str] = None,
        note_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> NoteResult:
        """
        Add an AI-generated note to a case file.
        
        This method is restricted to creating AI notes only, ensuring that
        the database tools cannot interfere with user-created notes.
        
        Args:
            case_file_id: ID of the case file
            content: Content of the note
            author_name: Optional name/identifier for the AI author
            note_type: Type of note ('research', 'strategy', 'fact', 'reminder', etc.)
            tags: Optional tags for organization
            
        Returns:
            Result with note ID and status
        """
        step_id = f"add_ai_note_{uuid.uuid4().hex[:8]}"
        writer = get_stream_writer()
        writer({
            "step_id": step_id,
            "status": "in_progress",
            "brief_description": "Adding AI note",
            "description": f"Adding AI note to case file {case_file_id}"
        })
        
        start_time = time.time()
        
        try:
            logger.info(f"Adding AI note to case file {case_file_id}")
            
            note_id = CaseFileNoteService.add_note(
                case_file_id=case_file_id,
                content=content,
                author_type="ai",  # Hardcoded to 'ai' for security
                author_name=author_name,
                note_type=note_type,
                tags=tags,
            )
            
            if not note_id:
                writer({
                    "step_id": step_id,
                    "status": "complete",
                    "brief_description": "Note addition failed",
                    "description": f"Failed to add note to case file {case_file_id} (case file may not exist)"
                })
                return NoteResult(
                    note_id=-1,
                    case_file_id=case_file_id,
                    success=False,
                    message=f"Failed to add note to case file {case_file_id} (case file may not exist)"
                )
            
            logger.info(f"Successfully added AI note with ID: {note_id}")
            
            execution_time = time.time() - start_time
            writer({
                "step_id": step_id,
                "status": "complete",
                "brief_description": "AI note added",
                "description": f"Added AI note with ID {note_id} to case file {case_file_id} in {execution_time:.2f}s"
            })
            
            return NoteResult(
                note_id=note_id,
                case_file_id=case_file_id,
                success=True,
                message=f"AI note added to case file {case_file_id} with ID {note_id}"
            )
            
        except Exception as e:
            logger.error(f"Error adding AI note to case file: {e}", exc_info=True)
            writer({
                "step_id": step_id,
                "status": "complete",
                "brief_description": "AI note addition failed",
                "description": f"Failed to add AI note: {str(e)}"
            })
            return NoteResult(
                note_id=-1,
                case_file_id=case_file_id,
                success=False,
                message=f"Failed to add AI note: {str(e)}"
            )
    
    def update_ai_note(
        self,
        note_id: int,
        content: str,
        note_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> NoteResult:
        """
        Update an AI-generated note.
        
        This method is restricted to updating AI notes only, ensuring that
        the database tools cannot modify user-created notes.
        
        Args:
            note_id: ID of the note to update
            content: Updated content
            note_type: Updated note type
            tags: Updated tags
            
        Returns:
            Result with update status
        """
        try:
            logger.info(f"Updating AI note ID: {note_id}")
            
            success = CaseFileNoteService.update_note(
                note_id=note_id,
                content=content,
                note_type=note_type,
                tags=tags,
                author_type_restriction="ai",  # Only allow updates to AI notes
            )
            
            if not success:
                return NoteResult(
                    note_id=note_id,
                    case_file_id=-1,
                    success=False,
                    message=f"Note {note_id} not found or is not an AI note"
                )
            
            logger.info(f"Successfully updated AI note ID: {note_id}")
            
            return NoteResult(
                note_id=note_id,
                case_file_id=-1,  # We don't have this info in the update method
                success=True,
                message=f"AI note {note_id} updated successfully"
            )
            
        except Exception as e:
            logger.error(f"Error updating AI note {note_id}: {e}", exc_info=True)
            return NoteResult(
                note_id=note_id,
                case_file_id=-1,
                success=False,
                message=f"Failed to update AI note: {str(e)}"
            )
    
    def delete_ai_note(self, note_id: int) -> NoteResult:
        """
        Delete an AI-generated note.
        
        This method is restricted to deleting AI notes only, ensuring that
        the database tools cannot remove user-created notes.
        
        Args:
            note_id: ID of the note to delete
            
        Returns:
            Result with deletion status
        """
        try:
            logger.info(f"Deleting AI note ID: {note_id}")
            
            success = CaseFileNoteService.delete_note(
                note_id=note_id,
                author_type_restriction="ai",  # Only allow deletion of AI notes
            )
            
            if not success:
                return NoteResult(
                    note_id=note_id,
                    case_file_id=-1,
                    success=False,
                    message=f"Note {note_id} not found or is not an AI note"
                )
            
            logger.info(f"Successfully deleted AI note ID: {note_id}")
            
            return NoteResult(
                note_id=note_id,
                case_file_id=-1,
                success=True,
                message=f"AI note {note_id} deleted successfully"
            )
            
        except Exception as e:
            logger.error(f"Error deleting AI note {note_id}: {e}", exc_info=True)
            return NoteResult(
                note_id=note_id,
                case_file_id=-1,
                success=False,
                message=f"Failed to delete AI note: {str(e)}"
            )


# =================== Simplified Tool Functions for Agent Use ===================

def create_case_file_tool(
    title: str,
    description: Optional[str] = None,
    user_facts: Optional[str] = None,
    party_represented: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Agent tool: Create a new case file for organizing legal research.
    
    This is a simplified wrapper for easy use in ReAct agents.
    
    Args:
        title: Title for the case file
        description: Optional description of the case
        user_facts: Client's factual situation
        party_represented: Which party the user represents
        
    Returns:
        Dictionary with case_file_id and success status
    """
    tools = LegalDatabaseTools()
    result = tools.create_case_file(title, description, user_facts, party_represented)
    return result.model_dump()


def add_document_tool(
    case_file_id: int,
    chunk_id: str,
    relevance_score_percent: Optional[float] = None,
    user_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Agent tool: Add a discovered document to a case file.
    
    This is a simplified wrapper for easy use in ReAct agents.
    
    Args:
        case_file_id: Unique identifier for the case file
        chunk_id: Unique identifier for the document chunk
        user_notes: Optional notes from the user about the document
        
    Returns:
        Dictionary with success status and message
    """
    tools = LegalDatabaseTools()
    result = tools.add_document_to_case_file(
        case_file_id=case_file_id,
        chunk_id=chunk_id,
        relevance_score_percent=relevance_score_percent,
        user_notes=user_notes
    )
    return result.model_dump()


def save_draft_tool(
    case_file_id: int,
    title: str,
    drafted_argument: str,
    strategy_data: Optional[Dict[str, Any]] = None,
    citations_used: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Agent tool: Save a generated argument draft.
    
    This is a simplified wrapper for easy use in ReAct agents.
    
    Args:
        case_file_id: ID of the case file
        title: Title for the draft
        drafted_argument: The argument text
        strategy_data: Legal strategy information
        citations_used: Citations used in the argument
        
    Returns:
        Dictionary with draft_id and success status
    """
    tools = LegalDatabaseTools()
    result = tools.save_argument_draft(
        case_file_id=case_file_id,
        title=title,
        drafted_argument=drafted_argument,
        strategy_data=strategy_data,
        citations_used=citations_used,
    )
    return result.model_dump()


def save_moot_court_tool(
    case_file_id: int,
    title: str,
    counterarguments: List[Dict[str, Any]],
    rebuttals: List[List[Dict[str, Any]]],
    draft_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Agent tool: Save a moot court analysis session.
    
    This is a simplified wrapper for easy use in ReAct agents.
    
    Args:
        case_file_id: ID of the case file
        title: Title for the session
        counterarguments: Generated counterarguments
        rebuttals: Rebuttals for each counterargument
        draft_id: ID of analyzed draft
        
    Returns:
        Dictionary with session_id and success status
    """
    tools = LegalDatabaseTools()
    result = tools.save_moot_court_session(
        case_file_id=case_file_id,
        title=title,
        counterarguments=counterarguments,
        rebuttals=rebuttals,
        draft_id=draft_id,
    )
    return result.model_dump()


def add_ai_note_tool(
    case_file_id: int,
    content: str,
    author_name: Optional[str] = None,
    note_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Agent tool: Add an AI-generated note to a case file.
    
    This is a simplified wrapper for easy use in ReAct agents.
    Restricted to AI notes only for security.
    
    Args:
        case_file_id: ID of the case file
        content: Content of the note
        author_name: Optional name/identifier for the AI author
        note_type: Type of note ('research', 'strategy', 'fact', 'reminder', etc.)
        tags: Optional tags for organization
        
    Returns:
        Dictionary with note_id and success status
    """
    tools = LegalDatabaseTools()
    result = tools.add_ai_note_to_case_file(
        case_file_id=case_file_id,
        content=content,
        author_name=author_name,
        note_type=note_type,
        tags=tags,
    )
    return result.model_dump()

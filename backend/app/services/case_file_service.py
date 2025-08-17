"""
Service layer for managing case files, argument drafts, and moot court sessions.

This module provides high-level operations for creating, retrieving,
and managing case files, their associated legal argument drafts, and
moot court practice sessions.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.core.database import get_db_context
from app.models.database_models import CaseFile, CaseFileDocument, ArgumentDraft, MootCourtSession, CaseFileNote
from app.models.schemas import (
    CaseFile as CaseFileSchema,
    CaseFileDocument as CaseFileDocumentSchema,
    ArgumentDraftResponse,
)
from app.tools.neo4j_tools import get_chunk_by_id
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DocumentService:
    """Service for managing documents."""

    @staticmethod
    def get_document_by_id(document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID using Neo4j.
        
        This method looks for a Chunk node with the given ID and returns
        comprehensive document and chunk information.
        
        Args:
            document_id: The unique ID of the chunk/document
            
        Returns:
            Dictionary containing document and chunk data, or None if not found
        """
        try:
            from app.tools.neo4j_tools import get_session
            from neo4j.exceptions import Neo4jError
            
            with get_session() as session:
                query = """
                MATCH (chunk:Chunk {id: $document_id})-[:PART_OF]->(doc:Document)
                OPTIONAL MATCH (chunk)-[:REFERENCES_CHUNK]->(referenced_chunk:Chunk)
                OPTIONAL MATCH (referencing_chunk:Chunk)-[:REFERENCES_CHUNK]->(chunk)
                RETURN chunk.id as chunk_id,
                       chunk.text as text,
                       chunk.summary as summary,
                       chunk.chunk_index as chunk_index,
                       chunk.statutes as statutes,
                       chunk.courts as courts,
                       chunk.cases as cases,
                       chunk.concepts as concepts,
                       chunk.judges as judges,
                       chunk.holdings as holdings,
                       chunk.facts as facts,
                       chunk.legal_tests as legal_tests,
                       chunk.chunk_references as chunk_references,
                       collect(DISTINCT referenced_chunk.id) as references_outgoing,
                       collect(DISTINCT referencing_chunk.id) as references_incoming,
                       doc.source as document_source,
                       doc.citation as document_citation,
                       doc.parties as parties,
                       doc.year as document_year,
                       doc.jurisdiction as jurisdiction,
                       doc.type as document_type,
                       doc.court_level as court_level
                """
                
                result = session.run(query, {"document_id": document_id})
                record = result.single()
                
                if record:
                    # Structure the return data similar to vector_search results
                    return {
                        "document_id": record["chunk_id"],
                        "chunk_id": record["chunk_id"],
                        "text": record["text"],
                        "summary": record["summary"],
                        "chunk_index": record["chunk_index"],
                        "statutes": record["statutes"] or [],
                        "courts": record["courts"] or [],
                        "cases": record["cases"] or [],
                        "concepts": record["concepts"] or [],
                        "judges": record["judges"] or [],
                        "holdings": record["holdings"] or [],
                        "facts": record["facts"] or [],
                        "legal_tests": record["legal_tests"] or [],
                        "chunk_references": record["chunk_references"] or [],
                        "references_outgoing": [ref for ref in record["references_outgoing"] if ref],
                        "references_incoming": [ref for ref in record["references_incoming"] if ref],
                        "document_source": record["document_source"],
                        "citation": record["document_citation"],
                        "title": record["document_citation"],  # Use citation as title for compatibility
                        "year": record["document_year"],
                        "jurisdiction": record["jurisdiction"],
                        "document_type": record["document_type"],
                        "parties": record["parties"] or [],
                        "court_level": record["court_level"],
                    }
                
                return None
                
        except Neo4jError as e:
            logger.error(f"Neo4j error in get_document_by_id: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get_document_by_id: {e}")
            return None

class CaseFileService:
    """Service for managing case files and their documents."""

    @staticmethod
    def create_case_file(
        title: str,
        description: Optional[str] = None,
        user_facts: Optional[str] = None,
        party_represented: Optional[str] = None,
    ) -> int:
        """Create a new case file and return its ID."""
        with get_db_context() as db:
            case_file = CaseFile(
                title=title,
                description=description,
                user_facts=user_facts,
                party_represented=party_represented,
            )
            db.add(case_file)
            db.flush()  # To get the ID
            return case_file.id

    @staticmethod
    def enrich_case_file_with_chunks(case_file_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich case file documents with chunk content from Neo4j if missing.

        This method helps populate chunk content for documents that were added
        before the chunk content preservation feature was implemented.
        """
        try:
            for document in case_file_data.get("documents", []):
                # If selected_chunks is empty, try to fetch chunk content from Neo4j
                if not document.get("selected_chunks"):
                    document_id = document.get("document_id")
                    if document_id:
                        # Try to get chunk content from Neo4j using the document_id as chunk_id
                        chunk_data = get_chunk_by_id(document_id)
                        if chunk_data:
                            document["selected_chunks"] = [
                                {
                                    "chunk_id": chunk_data.get("chunk_id"),
                                    "text": chunk_data.get("text"),
                                    "summary": chunk_data.get("summary"),
                                    "statutes": chunk_data.get("statutes") or [],
                                    "courts": chunk_data.get("courts") or [],
                                    "cases": chunk_data.get("cases") or [],
                                    "concepts": chunk_data.get("concepts") or [],
                                    "judges": chunk_data.get("judges") or [],
                                    "holdings": chunk_data.get("holdings") or [],
                                    "facts": chunk_data.get("facts") or [],
                                    "legal_tests": chunk_data.get("legal_tests") or [],
                                }
                            ]
                            logger.info(
                                f"Enriched document {document_id} with chunk content"
                            )

            return case_file_data

        except Exception as e:
            logger.warning(f"Failed to enrich case file with chunks: {e}")
            return case_file_data

    @staticmethod
    def get_case_file(case_file_id: int) -> Optional[Dict[str, Any]]:
        """Get a case file by ID with all its documents."""
        with get_db_context() as db:
            case_file = db.query(CaseFile).filter(CaseFile.id == case_file_id).first()
            if not case_file:
                return None

            case_file_data = {
                "id": case_file.id,
                "title": case_file.title,
                "description": case_file.description,
                "user_facts": case_file.user_facts,
                "party_represented": case_file.party_represented,
                "created_at": case_file.created_at,
                "updated_at": case_file.updated_at,
                "documents": [
                    {
                        "id": doc.id,
                        "document_id": doc.document_id,
                        "citation": doc.citation,
                        "title": doc.title,
                        "year": doc.year,
                        "jurisdiction": doc.jurisdiction,
                        "relevance_score_percent": doc.relevance_score_percent,
                        "key_holdings": doc.key_holdings or [],
                        "selected_chunks": doc.selected_chunks or [],
                        "user_notes": doc.user_notes,
                        "added_at": doc.added_at,
                    }
                    for doc in case_file.documents
                ],
                "notes": [
                    {
                        "id": note.id,
                        "content": note.content,
                        "author_type": note.author_type,
                        "author_name": note.author_name,
                        "note_type": note.note_type,
                        "tags": note.tags or [],
                        "created_at": note.created_at,
                        "updated_at": note.updated_at,
                    }
                    for note in case_file.notes
                ],
                "total_documents": len(case_file.documents),
            }

            # Enrich with chunk content if missing
            return CaseFileService.enrich_case_file_with_chunks(case_file_data)

    @staticmethod
    def list_case_files() -> List[Dict[str, Any]]:
        """List all case files with basic info."""
        with get_db_context() as db:
            case_files = db.query(CaseFile).order_by(desc(CaseFile.updated_at)).all()
            return [
                {
                    "id": cf.id,
                    "title": cf.title,
                    "description": cf.description,
                    "party_represented": cf.party_represented,
                    "created_at": cf.created_at,
                    "updated_at": cf.updated_at,
                    "document_count": len(cf.documents),
                    "draft_count": len(cf.drafts),
                }
                for cf in case_files
            ]

    @staticmethod
    def update_case_file(
        case_file_id: int,
        title: Optional[str] = None,
        description: Optional[str] = None,
        user_facts: Optional[str] = None,
        party_represented: Optional[str] = None,
    ) -> bool:
        """Update case file details."""
        with get_db_context() as db:
            case_file = db.query(CaseFile).filter(CaseFile.id == case_file_id).first()
            if not case_file:
                return False

            if title is not None:
                case_file.title = title
            if description is not None:
                case_file.description = description
            if user_facts is not None:
                case_file.user_facts = user_facts
            if party_represented is not None:
                case_file.party_represented = party_represented

            return True

    @staticmethod
    def delete_case_file(case_file_id: int) -> bool:
        """Delete a case file and all associated data."""
        with get_db_context() as db:
            case_file = db.query(CaseFile).filter(CaseFile.id == case_file_id).first()
            if not case_file:
                return False

            db.delete(case_file)
            return True

    @staticmethod
    def add_document_to_case_file(
        case_file_id: int, document_data: Dict[str, Any]
    ) -> bool:
        """Add a document to a case file."""
        with get_db_context() as db:
            # Check if case file exists
            case_file = db.query(CaseFile).filter(CaseFile.id == case_file_id).first()
            if not case_file:
                return False

            # Check if document already exists in this case file
            existing = (
                db.query(CaseFileDocument)
                .filter(
                    CaseFileDocument.case_file_id == case_file_id,
                    CaseFileDocument.document_id == document_data.get("document_id", document_data.get("chunk_id")),
                )
                .first()
            )

            if existing:
                return True  # Document already exists
            
            # Handle case where document_id is not provided but chunk_id is
            if "document_id" not in document_data and "chunk_id" in document_data:
                # Use chunk_id as document_id
                document_data["document_id"] = document_data["chunk_id"]
                
                # Get additional document data from Neo4j
                neo4j_data = DocumentService.get_document_by_id(document_data["document_id"])
                if neo4j_data:
                    # Merge Neo4j data with provided data, prioritizing provided data
                    for key, value in neo4j_data.items():
                        if key not in document_data:
                            document_data[key] = value
                else:
                    logger.warning(f"Could not find document data for chunk_id: {document_data['chunk_id']}")

            citation = document_data["citation"]
            if "parties" in document_data and document_data["parties"]:
                citation = " vs ".join(document_data["parties"])

            document = CaseFileDocument(
                case_file_id=case_file_id,
                document_id=document_data["document_id"],
                citation=citation,
                title=document_data["title"],
                year=document_data.get("year"),
                jurisdiction=document_data.get("jurisdiction"),
                relevance_score_percent=document_data.get("relevance_score_percent"),
                key_holdings=document_data.get("key_holdings", []),
                selected_chunks=document_data.get("selected_chunks", []),
                user_notes=document_data.get("user_notes"),
            )
            db.add(document)
            return True

    @staticmethod
    def remove_document_from_case_file(case_file_id: int, document_id: str) -> bool:
        """Remove a document from a case file."""
        with get_db_context() as db:
            document = (
                db.query(CaseFileDocument)
                .filter(
                    CaseFileDocument.case_file_id == case_file_id,
                    CaseFileDocument.document_id == document_id,
                )
                .first()
            )

            if not document:
                return False

            db.delete(document)
            return True
            
    @staticmethod
    def remove_all_documents_from_case_file(case_file_id: int) -> int:
        """Remove all documents from a case file.
        
        Returns:
            Number of documents removed
        """
        with get_db_context() as db:
            # Query to get the count before deletion
            document_count = db.query(CaseFileDocument).filter(
                CaseFileDocument.case_file_id == case_file_id
            ).count()
            
            # Delete all documents for this case file
            db.query(CaseFileDocument).filter(
                CaseFileDocument.case_file_id == case_file_id
            ).delete()
            
            return document_count

    @staticmethod
    def get_document_from_case_file(
        case_file_id: int, document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a specific document from a case file with all its content."""
        with get_db_context() as db:
            document = (
                db.query(CaseFileDocument)
                .filter(
                    CaseFileDocument.case_file_id == case_file_id,
                    CaseFileDocument.document_id == document_id,
                )
                .first()
            )

            if not document:
                return None

            # Convert to dictionary and enrich with chunks if needed
            document_data = {
                "id": document.id,
                "document_id": document.document_id,
                "citation": document.citation,
                "title": document.title,
                "year": document.year,
                "jurisdiction": document.jurisdiction,
                "relevance_score_percent": document.relevance_score_percent,
                "key_holdings": document.key_holdings or [],
                "selected_chunks": document.selected_chunks or [],
                "user_notes": document.user_notes,
                "added_at": document.added_at,
            }

            # If no chunks, try to enrich from Neo4j
            if not document_data["selected_chunks"]:
                chunk_data = get_chunk_by_id(document_id)
                if chunk_data:
                    document_data["selected_chunks"] = [
                        {
                            "chunk_id": chunk_data.get("chunk_id"),
                            "text": chunk_data.get("text"),
                            "summary": chunk_data.get("summary"),
                            "statutes": chunk_data.get("statutes") or [],
                            "courts": chunk_data.get("courts") or [],
                            "cases": chunk_data.get("cases") or [],
                            "concepts": chunk_data.get("concepts") or [],
                            "judges": chunk_data.get("judges") or [],
                            "holdings": chunk_data.get("holdings") or [],
                            "facts": chunk_data.get("facts") or [],
                            "legal_tests": chunk_data.get("legal_tests") or [],
                        }
                    ]

            return document_data


class ArgumentDraftService:
    """Service for managing argument drafts."""

    @staticmethod
    def save_draft(
        case_file_id: int,
        draft_response: ArgumentDraftResponse,
        title: Optional[str] = None,
    ) -> int:
        """Save an argument draft and return its ID."""
        with get_db_context() as db:
            draft = ArgumentDraft(
                case_file_id=case_file_id,
                title=title
                or f"Draft {db.query(ArgumentDraft).filter(ArgumentDraft.case_file_id == case_file_id).count() + 1}",
                drafted_argument=draft_response.drafted_argument,
                strategy=draft_response.strategy.dict()
                if draft_response.strategy
                else None,
                argument_structure=draft_response.argument_structure,
                citations_used=draft_response.citations_used,
                argument_strength=draft_response.argument_strength,
                precedent_coverage=draft_response.precedent_coverage,
                logical_coherence=draft_response.logical_coherence,
                total_critique_cycles=draft_response.total_critique_cycles,
                execution_time=draft_response.execution_time,
                revision_history=draft_response.revision_history,
            )
            db.add(draft)
            db.flush()
            return draft.id

    @staticmethod
    def get_draft(draft_id: int) -> Optional[Dict[str, Any]]:
        """Get a draft by ID."""
        with get_db_context() as db:
            draft = db.query(ArgumentDraft).filter(ArgumentDraft.id == draft_id).first()
            if not draft:
                return None

            return {
                "id": draft.id,
                "case_file_id": draft.case_file_id,
                "title": draft.title,
                "drafted_argument": draft.drafted_argument,
                "strategy": draft.strategy,
                "argument_structure": draft.argument_structure,
                "citations_used": draft.citations_used,
                "argument_strength": draft.argument_strength,
                "precedent_coverage": draft.precedent_coverage,
                "logical_coherence": draft.logical_coherence,
                "total_critique_cycles": draft.total_critique_cycles,
                "execution_time": draft.execution_time,
                "revision_history": draft.revision_history,
                "created_at": draft.created_at,
            }

    @staticmethod
    def list_drafts_for_case_file(case_file_id: int) -> List[Dict[str, Any]]:
        """List all drafts for a case file."""
        with get_db_context() as db:
            drafts = (
                db.query(ArgumentDraft)
                .filter(ArgumentDraft.case_file_id == case_file_id)
                .order_by(desc(ArgumentDraft.created_at))
                .all()
            )

            return [
                {
                    "id": draft.id,
                    "title": draft.title,
                    "created_at": draft.created_at,
                    "argument_strength": draft.argument_strength,
                    "precedent_coverage": draft.precedent_coverage,
                    "logical_coherence": draft.logical_coherence,
                }
                for draft in drafts
            ]

    @staticmethod
    def delete_draft(draft_id: int) -> bool:
        """Delete a draft."""
        with get_db_context() as db:
            draft = db.query(ArgumentDraft).filter(ArgumentDraft.id == draft_id).first()
            if not draft:
                return False

            db.delete(draft)
            return True

    @staticmethod
    def update_draft(
        draft_id: int,
        drafted_argument: str,
        title: Optional[str] = None,
    ) -> bool:
        """Update a draft with manual edits."""
        with get_db_context() as db:
            draft = db.query(ArgumentDraft).filter(ArgumentDraft.id == draft_id).first()
            if not draft:
                return False

            # Update the fields
            draft.drafted_argument = drafted_argument
            if title is not None:
                draft.title = title

            return True

    @staticmethod
    def update_draft_with_response(
        draft_id: int,
        draft_response: ArgumentDraftResponse,
    ) -> bool:
        """Update a draft with AI editing results."""
        with get_db_context() as db:
            draft = db.query(ArgumentDraft).filter(ArgumentDraft.id == draft_id).first()
            if not draft:
                return False

            # Update with new content from AI editing
            draft.drafted_argument = draft_response.drafted_argument
            draft.strategy = draft_response.strategy.dict() if draft_response.strategy else None
            draft.argument_structure = draft_response.argument_structure
            draft.citations_used = draft_response.citations_used
            draft.argument_strength = draft_response.argument_strength
            draft.precedent_coverage = draft_response.precedent_coverage
            draft.logical_coherence = draft_response.logical_coherence
            
            # Add to revision history if it exists
            if draft.revision_history is None:
                draft.revision_history = []
            
            revision_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "ai_edit",
                "execution_time": draft_response.execution_time,
                "critique_cycles": draft_response.total_critique_cycles
            }
            draft.revision_history.append(revision_entry)

            return True


class MootCourtSessionService:
    """Service for managing moot court practice sessions."""

    @staticmethod
    def save_session(
        case_file_id: int,
        draft_id: Optional[int],
        title: str,
        counterarguments: List[Dict[str, Any]],
        rebuttals: List[List[Dict[str, Any]]],
        source_arguments: Optional[List[Dict[str, Any]]] = None,
        research_context: Optional[Dict[str, Any]] = None,
        counterargument_strength: Optional[float] = None,
        research_comprehensiveness: Optional[float] = None,
        rebuttal_quality: Optional[float] = None,
        execution_time: Optional[float] = None,
    ) -> int:
        """
        Save a moot court session to the database.
        
        Args:
            case_file_id: ID of the case file
            draft_id: ID of the draft used (optional)
            title: Title for the session
            counterarguments: List of generated counterarguments
            rebuttals: List of rebuttal groups
            source_arguments: Source arguments that were analyzed
            research_context: RAG retrieval context
            counterargument_strength: Quality metric
            research_comprehensiveness: Quality metric
            rebuttal_quality: Quality metric
            execution_time: Time taken to generate
            
        Returns:
            ID of the created session
        """
        try:
            with get_db_context() as db:
                session = MootCourtSession(
                    case_file_id=case_file_id,
                    draft_id=draft_id,
                    title=title,
                    counterarguments=counterarguments,
                    rebuttals=rebuttals,
                    source_arguments=source_arguments,
                    research_context=research_context,
                    counterargument_strength=counterargument_strength,
                    research_comprehensiveness=research_comprehensiveness,
                    rebuttal_quality=rebuttal_quality,
                    execution_time=execution_time,
                    created_at=datetime.utcnow(),  # Explicitly set created_at
                )
                
                db.add(session)
                db.flush()  # To get the ID and ensure server defaults are applied
                
                return session.id
                
        except Exception as e:
            logger.error(f"Error saving moot court session: {e}")
            raise

    @staticmethod
    def list_sessions_for_case_file(case_file_id: int) -> List[Dict[str, Any]]:
        """
        List all moot court sessions for a case file.
        
        Args:
            case_file_id: ID of the case file
            
        Returns:
            List of session summary data
        """
        try:
            with get_db_context() as db:
                sessions = (
                    db.query(MootCourtSession)
                    .outerjoin(ArgumentDraft, MootCourtSession.draft_id == ArgumentDraft.id)
                    .filter(MootCourtSession.case_file_id == case_file_id)
                    .order_by(desc(MootCourtSession.created_at))
                    .all()
                )
                
                session_list = []
                for session in sessions:
                    # Get draft title if available
                    draft_title = None
                    if session.draft_id:
                        draft = db.query(ArgumentDraft).filter(ArgumentDraft.id == session.draft_id).first()
                        if draft:
                            draft_title = draft.title
                    
                    counterargument_count = len(session.counterarguments) if session.counterarguments else 0
                    
                    session_list.append({
                        "id": session.id,
                        "title": session.title,
                        "created_at": session.created_at,
                        "draft_title": draft_title,
                        "counterargument_count": counterargument_count,
                        "counterargument_strength": session.counterargument_strength,
                        "research_comprehensiveness": session.research_comprehensiveness,
                    })
                
                return session_list
                
        except Exception as e:
            logger.error(f"Error listing moot court sessions for case file {case_file_id}: {e}")
            return []

    @staticmethod
    def get_session(session_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific moot court session by ID.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Session data or None if not found
        """
        try:
            with get_db_context() as db:
                session = db.query(MootCourtSession).filter(MootCourtSession.id == session_id).first()
                
                if not session:
                    return None
                
                return {
                    "id": session.id,
                    "case_file_id": session.case_file_id,
                    "draft_id": session.draft_id,
                    "title": session.title,
                    "counterarguments": session.counterarguments,
                    "rebuttals": session.rebuttals,
                    "source_arguments": session.source_arguments,
                    "research_context": session.research_context,
                    "counterargument_strength": session.counterargument_strength,
                    "research_comprehensiveness": session.research_comprehensiveness,
                    "rebuttal_quality": session.rebuttal_quality,
                    "execution_time": session.execution_time,
                    "created_at": session.created_at,
                }
                
        except Exception as e:
            logger.error(f"Error getting moot court session {session_id}: {e}")
            return None

    @staticmethod
    def delete_session(session_id: int) -> bool:
        """
        Delete a moot court session.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with get_db_context() as db:
                session = db.query(MootCourtSession).filter(MootCourtSession.id == session_id).first()
                
                if not session:
                    return False
                
                db.delete(session)
                db.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Error deleting moot court session {session_id}: {e}")
            return False

    @staticmethod
    def update_session_title(session_id: int, title: str) -> bool:
        """
        Update a moot court session's title.
        
        Args:
            session_id: ID of the session
            title: New title
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with get_db_context() as db:
                session = db.query(MootCourtSession).filter(MootCourtSession.id == session_id).first()
                
                if not session:
                    return False
                
                session.title = title
                db.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Error updating moot court session {session_id}: {e}")
            return False


class CaseFileNoteService:
    """Service for managing case file notes."""

    @staticmethod
    def add_note(
        case_file_id: int,
        content: str,
        author_type: str,
        author_name: Optional[str] = None,
        note_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[int]:
        """
        Add a note to a case file.
        
        Args:
            case_file_id: ID of the case file
            content: Content of the note
            author_type: Type of author ('user' or 'ai')
            author_name: Optional name/identifier for the author
            note_type: Optional type of note ('research', 'strategy', 'fact', 'reminder', etc.)
            tags: Optional tags for organization
            
        Returns:
            ID of the created note, or None if failed
        """
        try:
            with get_db_context() as db:
                # Check if case file exists
                case_file = db.query(CaseFile).filter(CaseFile.id == case_file_id).first()
                if not case_file:
                    logger.warning(f"Case file {case_file_id} not found")
                    return None

                note = CaseFileNote(
                    case_file_id=case_file_id,
                    content=content,
                    author_type=author_type,
                    author_name=author_name,
                    note_type=note_type,
                    tags=tags or [],
                )
                db.add(note)
                db.flush()
                
                logger.info(f"Added note {note.id} to case file {case_file_id}")
                return note.id

        except Exception as e:
            logger.error(f"Error adding note to case file {case_file_id}: {e}")
            return None

    @staticmethod
    def update_note(
        note_id: int,
        content: str,
        note_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        author_type_restriction: Optional[str] = None,
    ) -> bool:
        """
        Update an existing note.
        
        Args:
            note_id: ID of the note to update
            content: Updated content
            note_type: Updated note type
            tags: Updated tags
            author_type_restriction: If provided, only allow updates to notes with this author_type
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with get_db_context() as db:
                note = db.query(CaseFileNote).filter(CaseFileNote.id == note_id).first()
                
                if not note:
                    logger.warning(f"Note {note_id} not found")
                    return False
                
                # Check author type restriction (for security)
                if author_type_restriction and note.author_type != author_type_restriction:
                    logger.warning(f"Update denied: note {note_id} has author_type '{note.author_type}', but restriction requires '{author_type_restriction}'")
                    return False

                note.content = content
                if note_type is not None:
                    note.note_type = note_type
                if tags is not None:
                    note.tags = tags
                
                logger.info(f"Updated note {note_id}")
                return True

        except Exception as e:
            logger.error(f"Error updating note {note_id}: {e}")
            return False

    @staticmethod
    def delete_note(note_id: int, author_type_restriction: Optional[str] = None) -> bool:
        """
        Delete a note.
        
        Args:
            note_id: ID of the note to delete
            author_type_restriction: If provided, only allow deletion of notes with this author_type
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with get_db_context() as db:
                note = db.query(CaseFileNote).filter(CaseFileNote.id == note_id).first()
                
                if not note:
                    logger.warning(f"Note {note_id} not found")
                    return False
                
                # Check author type restriction (for security)
                if author_type_restriction and note.author_type != author_type_restriction:
                    logger.warning(f"Delete denied: note {note_id} has author_type '{note.author_type}', but restriction requires '{author_type_restriction}'")
                    return False

                db.delete(note)
                
                logger.info(f"Deleted note {note_id}")
                return True

        except Exception as e:
            logger.error(f"Error deleting note {note_id}: {e}")
            return False

    @staticmethod
    def get_notes_for_case_file(case_file_id: int) -> List[Dict[str, Any]]:
        """
        Get all notes for a case file.
        
        Args:
            case_file_id: ID of the case file
            
        Returns:
            List of notes
        """
        try:
            with get_db_context() as db:
                notes = (
                    db.query(CaseFileNote)
                    .filter(CaseFileNote.case_file_id == case_file_id)
                    .order_by(CaseFileNote.created_at.desc())
                    .all()
                )

                return [
                    {
                        "id": note.id,
                        "content": note.content,
                        "author_type": note.author_type,
                        "author_name": note.author_name,
                        "note_type": note.note_type,
                        "tags": note.tags or [],
                        "created_at": note.created_at,
                        "updated_at": note.updated_at,
                    }
                    for note in notes
                ]

        except Exception as e:
            logger.error(f"Error getting notes for case file {case_file_id}: {e}")
            return []

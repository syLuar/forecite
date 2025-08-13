"""
SQLAlchemy models for storing case files and legal argument drafts.

This module defines the database schema for persisting case files,
associated documents, and generated legal argument drafts.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class CaseFile(Base):
    """
    Model for storing case files with basic metadata.
    """

    __tablename__ = "case_files"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    user_facts = Column(Text, nullable=True)  # Static case facts - now belongs to case file
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    documents = relationship(
        "CaseFileDocument", back_populates="case_file", cascade="all, delete-orphan"
    )
    drafts = relationship(
        "ArgumentDraft", back_populates="case_file", cascade="all, delete-orphan"
    )
    moot_court_sessions = relationship(
        "MootCourtSession", back_populates="case_file", cascade="all, delete-orphan"
    )


class CaseFileDocument(Base):
    """
    Model for storing documents associated with a case file.
    """

    __tablename__ = "case_file_documents"

    id = Column(Integer, primary_key=True, index=True)
    case_file_id = Column(Integer, ForeignKey("case_files.id"), nullable=False)
    document_id = Column(String(255), nullable=False)  # From Neo4j
    citation = Column(String(500), nullable=False)
    title = Column(Text, nullable=False)
    year = Column(Integer, nullable=True)
    jurisdiction = Column(String(100), nullable=True)
    relevance_score_percent = Column(Float, nullable=True)
    key_holdings = Column(JSON, nullable=True)  # Store as JSON array
    selected_chunks = Column(JSON, nullable=True)  # Store as JSON array
    user_notes = Column(Text, nullable=True)
    added_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    case_file = relationship("CaseFile", back_populates="documents")


class ArgumentDraft(Base):
    """
    Model for storing generated legal argument drafts.
    """

    __tablename__ = "argument_drafts"

    id = Column(Integer, primary_key=True, index=True)
    case_file_id = Column(Integer, ForeignKey("case_files.id"), nullable=False)
    title = Column(String(255), nullable=True)

    # Argument content
    drafted_argument = Column(Text, nullable=False)

    # Strategy information (stored as JSON)
    strategy = Column(JSON, nullable=True)
    argument_structure = Column(JSON, nullable=True)
    citations_used = Column(JSON, nullable=True)

    # Quality metrics
    argument_strength = Column(Float, nullable=True)
    precedent_coverage = Column(Float, nullable=True)
    logical_coherence = Column(Float, nullable=True)
    total_critique_cycles = Column(Integer, nullable=True)

    # Metadata
    execution_time = Column(Float, nullable=True)
    revision_history = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    case_file = relationship("CaseFile", back_populates="drafts")


class MootCourtSession(Base):
    """
    Model for storing moot court practice sessions with counterarguments and rebuttals.
    """

    __tablename__ = "moot_court_sessions"

    id = Column(Integer, primary_key=True, index=True)
    case_file_id = Column(Integer, ForeignKey("case_files.id"), nullable=False)
    draft_id = Column(Integer, ForeignKey("argument_drafts.id"), nullable=True)
    title = Column(String(255), nullable=False)

    # Session content
    counterarguments = Column(JSON, nullable=False)  # List of counterargument objects
    rebuttals = Column(JSON, nullable=False)  # List of rebuttal groups for each counterargument
    
    # Source arguments that were analyzed
    source_arguments = Column(JSON, nullable=True)  # Key arguments from the selected draft
    
    # RAG retrieval context
    research_context = Column(JSON, nullable=True)  # Summary of retrieved knowledge
    
    # Quality metrics
    counterargument_strength = Column(Float, nullable=True)
    research_comprehensiveness = Column(Float, nullable=True)
    rebuttal_quality = Column(Float, nullable=True)
    
    # Metadata
    execution_time = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    case_file = relationship("CaseFile", back_populates="moot_court_sessions")
    draft = relationship("ArgumentDraft")

"""
Database configuration and session management.

This module provides database connectivity and session management for
storing case files and generated legal argument drafts. Supports both
PostgreSQL (production) and SQLite (development) based on environment settings.
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from .config import settings

logger = logging.getLogger(__name__)

# Get database URL based on environment
DATABASE_URL = settings.get_database_url()
logger.info(f"Using database: {'PostgreSQL' if DATABASE_URL.startswith('postgresql') else 'SQLite'} (Environment: {settings.environment})")

# Create engine with appropriate connection arguments
def get_engine_config():
    """Get engine configuration based on database type."""
    if DATABASE_URL.startswith("sqlite"):
        return {
            "connect_args": {"check_same_thread": False},  # For SQLite
            "echo": settings.debug,
        }
    else:  # PostgreSQL
        return {
            "echo": settings.debug,
            "pool_size": 10,
            "max_overflow": 20,
            "pool_pre_ping": True,  # Validate connections before use
        }

# Create engine
engine = create_engine(DATABASE_URL, **get_engine_config())

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

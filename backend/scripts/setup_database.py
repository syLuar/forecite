#!/usr/bin/env python3
"""
Database setup script for Legal Research Assistant.

This script helps set up the database based on the current environment configuration.
It can create PostgreSQL databases and run initial migrations.
"""

import sys
import os
import logging
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.database import engine, Base
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_postgresql_database():
    """Create PostgreSQL database if it doesn't exist and set up permissions."""
    from sqlalchemy import create_engine

    # Connect to the default postgres database to create our database
    default_url = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/postgres"

    try:
        default_engine = create_engine(default_url)

        with default_engine.connect() as conn:
            # Check if database exists
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                {"db_name": settings.postgres_db},
            )

            if not result.fetchone():
                # Create database
                conn.execute(text("COMMIT"))  # End any transaction
                conn.execute(text(f"CREATE DATABASE {settings.postgres_db}"))
                logger.info(f"Created PostgreSQL database: {settings.postgres_db}")
            else:
                logger.info(
                    f"PostgreSQL database already exists: {settings.postgres_db}"
                )

    except Exception as e:
        logger.error(f"Failed to create PostgreSQL database: {e}")
        logger.error("Please ensure PostgreSQL is running and credentials are correct")
        return False

    # Now connect to the target database and set up permissions
    try:
        target_url = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        target_engine = create_engine(target_url)

        with target_engine.connect() as conn:
            # Grant necessary permissions on public schema
            try:
                conn.execute(
                    text(f"GRANT CREATE ON SCHEMA public TO {settings.postgres_user}")
                )
                conn.execute(
                    text(f"GRANT USAGE ON SCHEMA public TO {settings.postgres_user}")
                )
                conn.execute(text("COMMIT"))
                logger.info(f"Granted permissions to user: {settings.postgres_user}")
            except Exception as perm_error:
                logger.warning(
                    f"Could not grant permissions (may already exist): {perm_error}"
                )

    except Exception as e:
        logger.error(f"Failed to set up database permissions: {e}")
        logger.error(
            "You may need to run the following SQL commands as a PostgreSQL superuser:"
        )
        logger.error(f"GRANT CREATE ON SCHEMA public TO {settings.postgres_user};")
        logger.error(f"GRANT USAGE ON SCHEMA public TO {settings.postgres_user};")
        return False

    return True


def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False


def main():
    """Main setup function."""
    logger.info(f"Setting up database for environment: {settings.environment}")

    database_url = settings.get_database_url()
    logger.info(
        f"Database URL: {database_url.replace(settings.postgres_password, '***') if settings.postgres_password else database_url}"
    )

    # If using PostgreSQL, ensure database exists
    if database_url.startswith("postgresql"):
        logger.info("Setting up PostgreSQL database...")
        if not create_postgresql_database():
            sys.exit(1)
    else:
        logger.info("Using SQLite database (no database creation needed)")

    # Create tables
    logger.info("Creating database tables...")
    if not create_tables():
        sys.exit(1)

    logger.info("Database setup completed successfully!")


if __name__ == "__main__":
    main()

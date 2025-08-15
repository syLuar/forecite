#!/usr/bin/env python3
"""
Knowledge Graph Ingestion Script for Legal Research Assistant

This script processes raw PDF documents from data/raw_docs/ and populates
the Neo4j database with documents, chunks, and their relationships.

Features:
- PDF text extraction
- Document chunking with overlap
- Entity extraction and summarization using LLM
- Embedding generation for semantic search
- Neo4j graph construction with proper schema
- Comprehensive performance profiling and timing

Usage:
    python scripts/ingest_graph.py [--reset] [--docs-dir PATH] [--profile]

    --reset: Clear the database before ingestion
    --docs-dir: Path to directory containing PDF files (default: data/raw_docs)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
import time
import re

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from langchain.schema import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from app.core.config import settings
from app.core.llm import create_llm

from dotenv import load_dotenv

# Import shared helpers
from ingest_graph_helpers import (
    LLMCache, PerformanceTracker, Neo4jConnection, BaseDocumentProcessor,
    LegalEntities, DocumentMetadata,
    extract_citation_from_name, extract_year_from_citation, 
    infer_jurisdiction, infer_court_level, determine_document_type
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global performance tracker
perf_tracker = PerformanceTracker()


class DocumentProcessor(BaseDocumentProcessor):
    """Handles document processing including chunking and embedding generation."""

    def __init__(self, use_embedding_cache: bool = True):
        super().__init__(use_cache=use_embedding_cache, perf_tracker=perf_tracker)

        self.entity_extraction_llm = create_llm(settings.llm_config.get("ingestion", {}).get("entity_extraction", {}))
        self.summary_generation_llm = create_llm(settings.llm_config.get("ingestion", {}).get("summary_generation", {}))
        self.metadata_extraction_llm = create_llm(settings.llm_config.get("ingestion", {}).get("metadata_extraction", {}))

        # Set up structured output parsers and chains
        self.entity_parser = PydanticOutputParser(pydantic_object=LegalEntities)
        self.metadata_parser = PydanticOutputParser(pydantic_object=DocumentMetadata)

        # Create entity extraction chain
        entity_prompt = PromptTemplate(
            template="""Extract the following legal entities from the text:

{format_instructions}

Text: {text}""",
            input_variables=["text"],
            partial_variables={
                "format_instructions": self.entity_parser.get_format_instructions()
            },
        )
        self.entity_extraction_chain = (
            entity_prompt | self.entity_extraction_llm | self.entity_parser
        )

        # Create metadata extraction chain
        metadata_prompt = PromptTemplate(
            template="""Extract metadata from this legal document:

{format_instructions}

Document text (first 2000 chars): 
{text_start}

Document text (last 2000 chars): 
{text_end}

Filename: {filename}""",
            input_variables=["text_start", "text_end", "filename"],
            partial_variables={
                "format_instructions": self.metadata_parser.get_format_instructions()
            },
        )
        self.metadata_extraction_chain = (
            metadata_prompt | self.metadata_extraction_llm | self.metadata_parser
        )

    async def generate_summary(self, text: str) -> str:
        """Generate a summary of the text using LLM with caching support."""
        try:
            # Check cache first
            if self.use_cache and self.cache:
                cached_summary = self.cache.get_summary(text)
                if cached_summary:
                    logger.debug("📦 Summary found in cache")
                    return cached_summary
            
            start_time = time.time()

            prompt = f"""
            Provide a concise summary of the following legal text, from a court case. Focus on:
            1. Key legal concepts and principles
            2. Important case citations or references
            3. Main arguments or holdings

            Remember, do not include any personal opinions or interpretations. Do not include any content that is not directly mentioned in the document.
            
            Text: {text[:2000]}...
            
            Summary:
            """

            response = await self.summary_generation_llm.ainvoke(prompt)
            summary = (
                response.content if hasattr(response, "content") else str(response)
            ).strip()

            summary_time = time.time() - start_time
            perf_tracker.record_metric("summary_generation_time", summary_time)
            perf_tracker.record_metric("llm_time", summary_time)
            perf_tracker.record_metric("llm_calls", 1)

            # Store in cache
            if self.use_cache and self.cache:
                self.cache.store_summary(text, summary)

            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Summary generation failed"

    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities from the text using structured output with caching support."""
        try:
            # Check cache first
            if self.use_cache and self.cache:
                cached_entities = self.cache.get_entities(text)
                if cached_entities:
                    logger.debug("📦 Entities found in cache")
                    return cached_entities
            
            start_time = time.time()

            # Use the structured output chain
            entities = await self.entity_extraction_chain.ainvoke(
                {"text": text[:1500] + "..." if len(text) > 1500 else text}
            )

            entity_time = time.time() - start_time
            perf_tracker.record_metric("entity_extraction_time", entity_time)
            perf_tracker.record_metric("llm_time", entity_time)
            perf_tracker.record_metric("llm_calls", 1)

            # Convert Pydantic model to dictionary with cleaned values
            entities_dict = entities.model_dump()

            # Clean and filter the entities
            for key in entities_dict:
                if isinstance(entities_dict[key], list):
                    entities_dict[key] = [
                        str(item).strip()
                        for item in entities_dict[key]
                        if item and str(item).strip()
                    ]

            # Store in cache
            if self.use_cache and self.cache:
                self.cache.store_entities(text, entities_dict)

            return entities_dict

        except Exception as e:
            logger.error(f"Error extracting entities with structured output: {e}")
            # Return default empty structure
            return {
                "cases": [],
                "statutes": [],
                "concepts": [],
                "courts": [],
                "judges": [],
                "holdings": [],
                "facts": [],
                "legal_tests": [],
            }

    def extract_chunk_references(self, text: str) -> List[str]:
        """
        Extract cross-references to other chunks/paragraphs within the same document.

        Sample pattern to match: see [109] above
        """

        references = set()
        matches = re.finditer(r"see\s*\[(\d+)\] above", text, re.IGNORECASE)
        for match in matches:
            paragraph_num = match.group(1)
            references.add(paragraph_num)

        return list(references)

    def extract_paragraph_numbers(self, text: str) -> List[str]:
        """
        Extract paragraph numbers present in this chunk.

        Pattern to match: paragraph numbers at the start of lines
        """
        paragraph_numbers = set()
        # Look for pattern like ".\n(\d+)\s" where 123 is a paragraph number
        matches = re.finditer(r"\.\n(\d+)\s", text)
        for match in matches:
            paragraph_num = match.group(1)
            paragraph_numbers.add(paragraph_num)

        return list(paragraph_numbers)


class GraphBuilder:
    """Builds and populates the Neo4j knowledge graph."""

    def __init__(self, neo4j_conn: Neo4jConnection, use_embedding_cache: bool = True):
        self.neo4j = neo4j_conn
        self.processor = DocumentProcessor(use_embedding_cache=use_embedding_cache)

    async def setup_database_schema(self):
        """Create constraints, indexes, and vector index in Neo4j."""
        logger.info("Setting up database schema...")

        # Create constraints for unique identification
        constraints = [
            "CREATE CONSTRAINT unique_document_source IF NOT EXISTS FOR (d:Document) REQUIRE d.source IS UNIQUE",
            "CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT unique_case_citation IF NOT EXISTS FOR (case:Case) REQUIRE case.citation IS UNIQUE",
            "CREATE CONSTRAINT unique_statute_reference IF NOT EXISTS FOR (s:Statute) REQUIRE s.reference IS UNIQUE",
            "CREATE CONSTRAINT unique_court_name IF NOT EXISTS FOR (court:Court) REQUIRE court.name IS UNIQUE",
            "CREATE CONSTRAINT unique_legal_concept IF NOT EXISTS FOR (lc:LegalConcept) REQUIRE lc.name IS UNIQUE",
            "CREATE CONSTRAINT unique_judge_name IF NOT EXISTS FOR (j:Judge) REQUIRE j.name IS UNIQUE",
        ]

        for constraint in constraints:
            try:
                await self.neo4j.execute_query(constraint)
                logger.info(f"Created constraint: {constraint}")
            except Exception as e:
                logger.warning(f"Constraint already exists or error: {e}")

        # Create indexes for efficient querying
        indexes = [
            "CREATE INDEX document_type_index IF NOT EXISTS FOR (d:Document) ON (d.type)",
            "CREATE INDEX document_date_index IF NOT EXISTS FOR (d:Document) ON (d.date)",
            "CREATE INDEX case_year_index IF NOT EXISTS FOR (case:Case) ON (case.year)",
            "CREATE INDEX court_jurisdiction_index IF NOT EXISTS FOR (court:Court) ON (court.jurisdiction)",
            "CREATE INDEX chunk_summary_text_index IF NOT EXISTS FOR (c:Chunk) ON (c.summary)",
        ]

        for index in indexes:
            try:
                await self.neo4j.execute_query(index)
                logger.info(f"Created index: {index}")
            except Exception as e:
                logger.warning(f"Index already exists or error: {e}")

        # Create fulltext index with correct syntax
        fulltext_index_query = """
        CREATE FULLTEXT INDEX legal_text_search IF NOT EXISTS 
        FOR (d:Document) ON EACH [d.full_text]
        """

        try:
            await self.neo4j.execute_query(fulltext_index_query)
            logger.info(
                "Created fulltext index: legal_text_search for Document.full_text"
            )
        except Exception as e:
            logger.warning(f"Fulltext index already exists or error: {e}")

        # Create separate fulltext index for chunks
        chunk_fulltext_query = """
        CREATE FULLTEXT INDEX chunk_text_search IF NOT EXISTS 
        FOR (c:Chunk) ON EACH [c.text, c.summary]
        """

        try:
            await self.neo4j.execute_query(chunk_fulltext_query)
            logger.info(
                "Created fulltext index: chunk_text_search for Chunk text and summary"
            )
        except Exception as e:
            logger.warning(f"Chunk fulltext index already exists or error: {e}")

        embedding_dimension = settings.llm_config.get("embeddings", {}).get(
            "dimension", 768
        )
        vector_index_query = f"""
        CREATE VECTOR INDEX {settings.vector_index_name} IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {embedding_dimension},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        await self.neo4j.execute_query(vector_index_query)

    async def reset_database(self):
        """Clear all data from the database."""
        logger.warning("Resetting database - all data will be deleted!")

        # Clear all nodes and relationships first
        try:
            await self.neo4j.execute_query("MATCH (n) DETACH DELETE n")
            logger.info("Successfully cleared all nodes and relationships")
        except Exception as e:
            logger.error(f"Failed to clear nodes and relationships: {e}")

        try:
            # List and drop any existing indexes
            result = await self.neo4j.execute_query("SHOW INDEXES YIELD name, type")
            for record in result:
                index_name = record["name"]
                index_type = record["type"]
                if index_name in [settings.vector_index_name, "legal_text_search", "chunk_text_search"]:
                    try:
                        await self.neo4j.execute_query(f"DROP INDEX `{index_name}`")
                        logger.info(f"Dropped index: {index_name}")
                    except Exception as drop_e:
                        logger.info(f"Could not drop index {index_name}: {drop_e}")
        except Exception as e:
            logger.info(f"Could not list/drop indexes: {e}")

    async def process_document(self, pdf_path: Path) -> bool:
        """Process a single PDF document and add it to the graph."""
        logger.info(f"📚 Processing document: {pdf_path.name}")

        # Extract text
        text = self.processor.extract_text_from_pdf(pdf_path)
        if not text:
            logger.error(f"No text extracted from {pdf_path}")
            return False

        # Extract document metadata
        doc_metadata = self._extract_document_metadata(pdf_path, text)

        # Create document node with enhanced metadata
        doc_params = {
            "source": str(pdf_path.name),
            "type": doc_metadata["type"],
            "full_text": text[:5000],  # Store first 5000 chars
            "file_path": str(pdf_path),
            "word_count": len(text.split()),
            "date": doc_metadata.get("date"),
            "year": doc_metadata.get("year"),
            "jurisdiction": doc_metadata.get("jurisdiction"),
            "court_level": doc_metadata.get("court_level"),
            "case_number": doc_metadata.get("case_number"),
            "citation": doc_metadata.get("citation"),
            "parties": doc_metadata.get("parties", []),
            "legal_areas": doc_metadata.get("legal_areas", []),
        }

        create_doc_query = """
        MERGE (d:Document {source: $source})
        SET d.type = $type,
            d.full_text = $full_text,
            d.file_path = $file_path,
            d.word_count = $word_count,
            d.date = $date,
            d.year = $year,
            d.jurisdiction = $jurisdiction,
            d.court_level = $court_level,
            d.case_number = $case_number,
            d.citation = $citation,
            d.parties = $parties,
            d.legal_areas = $legal_areas,
            d.created_at = datetime()
        RETURN d
        """

        await self.neo4j.execute_write_query(create_doc_query, doc_params)

        # Chunk the document
        chunks = self.processor.chunk_document(text, str(pdf_path.name))
        logger.info(f"📄 Created {len(chunks)} chunks for {pdf_path.name}")

        # Track references and paragraph mappings for later relationship creation
        chunk_references_map = {}  # chunk_id -> list of referenced paragraph numbers
        paragraph_to_chunk_map = {}  # paragraph_number -> chunk_id

        # Process chunks in batches
        batch_size = 5
        tasks = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            logger.info(
                f"⚙️  Processing chunk batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}"
            )
            tasks.append(
                self._process_chunk_batch(
                    batch,
                    str(pdf_path.name),
                    chunk_references_map,
                    paragraph_to_chunk_map,
                )
            )

        await asyncio.gather(*tasks)

        # Create chunk-to-chunk reference relationships after all chunks are processed
        await self._create_all_chunk_references(
            chunk_references_map, paragraph_to_chunk_map, str(pdf_path.name)
        )

        # Create entity nodes and relationships after all chunks are processed
        await self._create_document_entities(str(pdf_path.name), text)

        perf_tracker.record_metric("documents_processed", 1)
        logger.info(f"✅ Successfully processed {pdf_path.name}")
        return True

    async def _process_chunk_batch(
        self,
        chunks: List[Document],
        source: str,
        chunk_references_map: Dict,
        paragraph_to_chunk_map: Dict,
    ):
        """Process a batch of chunks."""
        # Generate embeddings for the batch
        texts = [chunk.page_content for chunk in chunks]
        embeddings = await self.processor.generate_embeddings(texts)

        if len(embeddings) != len(chunks):
            logger.error(f"Embedding count mismatch for {source}")
            return

        # Process each chunk concurrently
        tasks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            tasks.append(
                self._create_chunk_node(
                    chunk,
                    embedding,
                    source,
                    chunk_references_map,
                    paragraph_to_chunk_map,
                )
            )

        await asyncio.gather(*tasks)

    async def _create_chunk_node(
        self,
        chunk: Document,
        embedding: List[float],
        source: str,
        chunk_references_map: Dict,
        paragraph_to_chunk_map: Dict,
    ):
        """Create a chunk node and its relationships."""
        # Generate summary and extract entities
        summary = await self.processor.generate_summary(chunk.page_content)
        entities = await self.processor.extract_entities(chunk.page_content)

        # Extract cross-references to other chunks and paragraph numbers in this chunk
        chunk_references = self.processor.extract_chunk_references(chunk.page_content)
        paragraph_numbers = self.processor.extract_paragraph_numbers(chunk.page_content)

        # Create unique chunk ID
        chunk_id = f"{source}_{chunk.metadata['chunk_index']}"

        # Store references and paragraph mappings for later relationship creation
        if chunk_references:
            chunk_references_map[chunk_id] = chunk_references

        for paragraph_num in paragraph_numbers:
            paragraph_to_chunk_map[paragraph_num] = chunk_id

        chunk_params = {
            "id": chunk_id,
            "text": chunk.page_content,
            "summary": summary,
            "embedding": embedding,
            "chunk_index": chunk.metadata["chunk_index"],
            "source": source,
            "statutes": entities.get("statutes", []),
            "courts": entities.get("courts", []),
            "cases": entities.get("cases", []),
            "concepts": entities.get("concepts", []),
            "judges": entities.get("judges", []),
            "holdings": entities.get("holdings", []),
            "facts": entities.get("facts", []),
            "legal_tests": entities.get("legal_tests", []),
            "chunk_references": chunk_references,
            "paragraph_numbers": paragraph_numbers,
        }

        # Create chunk node and relationship to document
        create_chunk_query = """
        MATCH (d:Document {source: $source})
        CREATE (c:Chunk {
            id: $id,
            text: $text,
            summary: $summary,
            embedding: $embedding,
            chunk_index: $chunk_index,
            statutes: $statutes,
            courts: $courts,
            cases: $cases,
            concepts: $concepts,
            judges: $judges,
            holdings: $holdings,
            facts: $facts,
            legal_tests: $legal_tests,
            chunk_references: $chunk_references,
            paragraph_numbers: $paragraph_numbers,
            created_at: datetime()
        })
        CREATE (c)-[:PART_OF]->(d)
        RETURN c
        """

        await self.neo4j.execute_write_query(create_chunk_query, chunk_params)

        # Create reference relationships based on entities
        await self._create_reference_relationships(chunk_id, entities)

    async def _create_all_chunk_references(
        self, chunk_references_map: Dict, paragraph_to_chunk_map: Dict, source: str
    ):
        """Create all chunk-to-chunk reference relationships after processing all chunks."""
        logger.info(f"Creating chunk reference relationships for {source}")

        relationships_created = 0

        for source_chunk_id, referenced_paragraphs in chunk_references_map.items():
            for paragraph_ref in referenced_paragraphs:
                # Find the chunk that contains this paragraph number
                target_chunk_id = paragraph_to_chunk_map.get(paragraph_ref)

                if target_chunk_id and target_chunk_id != source_chunk_id:
                    # Create the relationship
                    create_ref_query = """
                    MATCH (source_chunk:Chunk {id: $source_chunk_id})
                    MATCH (target_chunk:Chunk {id: $target_chunk_id})
                    MERGE (source_chunk)-[:REFERENCES_CHUNK {paragraph_ref: $paragraph_ref}]->(target_chunk)
                    """

                    try:
                        await self.neo4j.execute_write_query(
                            create_ref_query,
                            {
                                "source_chunk_id": source_chunk_id,
                                "target_chunk_id": target_chunk_id,
                                "paragraph_ref": paragraph_ref,
                            },
                        )
                        relationships_created += 1
                        logger.debug(
                            f"Created reference from {source_chunk_id} to {target_chunk_id} (paragraph {paragraph_ref})"
                        )

                    except Exception as e:
                        logger.warning(
                            f"Failed to create chunk reference relationship: {e}"
                        )
                else:
                    if paragraph_ref not in paragraph_to_chunk_map:
                        logger.debug(
                            f"No chunk found containing paragraph {paragraph_ref} referenced by {source_chunk_id}"
                        )

        logger.info(
            f"Created {relationships_created} chunk reference relationships for {source}"
        )

    async def _create_reference_relationships(
        self, chunk_id: str, entities: Dict[str, List[str]]
    ):
        """Create REFERENCES relationships based on extracted entities."""
        # Find documents that might be referenced by this chunk
        for case in entities.get("cases", []):
            # Simple matching - in production, use more sophisticated entity linking
            find_ref_query = """
            MATCH (c:Chunk {id: $chunk_id})
            MATCH (d:Document) 
            WHERE d.source CONTAINS $case_name OR d.full_text CONTAINS $case_name
            MERGE (c)-[:REFERENCES]->(d)
            """

            try:
                await self.neo4j.execute_write_query(
                    find_ref_query,
                    {
                        "chunk_id": chunk_id,
                        "case_name": case[:50],  # Use first part of case name
                    },
                )
            except Exception as e:
                logger.warning(f"Reference relationship creation failed: {e}")

    def _extract_document_metadata(self, pdf_path: Path, text: str) -> Dict[str, Any]:
        """Extract enhanced metadata from document using structured output."""
        try:
            # Basic metadata from filename
            filename = pdf_path.name
            metadata = {"type": determine_document_type(pdf_path)}

            # Extract year from filename (e.g., "[2019] SGCA 42.pdf")
            year_match = re.search(r"\[(\d{4})\]", filename)
            if year_match:
                metadata["year"] = int(year_match.group(1))
                metadata["date"] = f"{year_match.group(1)}-01-01"  # Default to Jan 1st

            # Extract case citation from filename
            citation_match = re.search(r"\[(\d{4})\]\s*([A-Z]+)\s*(\d+)", filename)
            if citation_match:
                metadata["citation"] = (
                    f"[{citation_match.group(1)}] {citation_match.group(2)} {citation_match.group(3)}"
                )
                metadata["case_number"] = citation_match.group(3)

            # Use structured output for detailed extraction
            start_time = time.time()

            try:
                llm_metadata = self.processor.metadata_extraction_chain.invoke(
                    {
                        "text_start": text[:2000],
                        "text_end": text[-2000:],
                        "filename": filename,
                    }
                )

                metadata_time = time.time() - start_time
                perf_tracker.record_metric("llm_time", metadata_time)
                perf_tracker.record_metric("llm_calls", 1)

                # Convert Pydantic model to dictionary and update metadata
                llm_metadata_dict = llm_metadata.model_dump()
                metadata.update(llm_metadata_dict)

            except Exception as llm_error:
                logger.warning(f"Structured metadata extraction failed: {llm_error}")
                # Fallback to basic extraction with proper structure
                if "SGCA" in filename or "Singapore" in text[:1000]:
                    metadata.update(
                        {
                            "jurisdiction": "Singapore",
                            "court_level": "Court of Appeal"
                            if "SGCA" in filename
                            else "High Court",
                            "parties": [],
                            "legal_areas": [],
                        }
                    )
                else:
                    metadata.update(
                        {
                            "jurisdiction": "Unknown",
                            "court_level": "Unknown",
                            "parties": [],
                            "legal_areas": [],
                        }
                    )

            return metadata
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            return {
                "type": "Document",
                "jurisdiction": "Unknown",
                "court_level": "Unknown",
                "parties": [],
                "legal_areas": [],
            }

    async def _create_document_entities(self, source: str, full_text: str):
        """Create entity nodes and relationships for the entire document."""
        try:
            # Extract entities from full document text (first 3000 chars for efficiency)
            entities = await self.processor.extract_entities(full_text[:3000])

            # Create Case entities and relationships
            for case_name in entities.get("cases", [])[:10]:  # Limit to prevent explosion
                if case_name and len(case_name.strip()) > 5:
                    case_citation = extract_citation_from_name(case_name)
                    year = extract_year_from_citation(case_citation)

                    create_case_query = """
                    MERGE (case:Case {citation: $citation})
                    SET case.name = $name,
                        case.year = $year,
                        case.created_at = datetime()
                    WITH case
                    MATCH (d:Document {source: $source})
                    MERGE (d)-[:CITES]->(case)
                    """

                    await self.neo4j.execute_write_query(
                        create_case_query,
                        {
                            "citation": case_citation,
                            "name": case_name.strip(),
                            "year": year,
                            "source": source,
                        },
                    )

            # Create Statute entities
            for statute_ref in entities.get("statutes", [])[:20]:
                if statute_ref and len(statute_ref.strip()) > 3:
                    create_statute_query = """
                    MERGE (s:Statute {reference: $reference})
                    SET s.created_at = datetime()
                    WITH s
                    MATCH (d:Document {source: $source})
                    MERGE (d)-[:REFERENCES_STATUTE]->(s)
                    """

                    await self.neo4j.execute_write_query(
                        create_statute_query,
                        {"reference": statute_ref.strip(), "source": source},
                    )

            # Create Court entities
            for court_name in entities.get("courts", [])[:5]:
                if court_name and len(court_name.strip()) > 3:
                    jurisdiction = infer_jurisdiction(court_name)
                    level = infer_court_level(court_name)

                    create_court_query = """
                    MERGE (court:Court {name: $name})
                    SET court.jurisdiction = $jurisdiction,
                        court.level = $level,
                        court.created_at = datetime()
                    WITH court
                    MATCH (d:Document {source: $source})
                    MERGE (d)-[:HEARD_IN]->(court)
                    """

                    await self.neo4j.execute_write_query(
                        create_court_query,
                        {
                            "name": court_name.strip(),
                            "jurisdiction": jurisdiction,
                            "level": level,
                            "source": source,
                        },
                    )

            # Create LegalConcept entities
            for concept in entities.get("concepts", [])[:15]:
                if concept and len(concept.strip()) > 5:
                    create_concept_query = """
                    MERGE (lc:LegalConcept {name: $name})
                    SET lc.created_at = datetime()
                    WITH lc
                    MATCH (d:Document {source: $source})
                    MERGE (d)-[:DISCUSSES]->(lc)
                    """

                    await self.neo4j.execute_write_query(
                        create_concept_query,
                        {"name": concept.strip(), "source": source},
                    )

            # Create Judge entities
            for judge_name in entities.get("judges", [])[:5]:
                if judge_name and len(judge_name.strip()) > 3:
                    create_judge_query = """
                    MERGE (j:Judge {name: $name})
                    SET j.created_at = datetime()
                    WITH j
                    MATCH (d:Document {source: $source})
                    MERGE (j)-[:AUTHORED]->(d)
                    """

                    await self.neo4j.execute_write_query(
                        create_judge_query,
                        {"name": judge_name.strip(), "source": source},
                    )

            # Create temporal relationships between cases based on citation patterns
            await self._create_temporal_relationships(source, entities.get("cases", []))

        except Exception as e:
            logger.error(f"Error creating document entities: {e}")

    async def _create_temporal_relationships(
        self, current_source: str, cited_cases: List[str]
    ):
        """Create FOLLOWS and OVERRULES relationships based on temporal analysis."""
        try:
            # Get current document year
            current_doc_query = """
            MATCH (d:Document {source: $source})
            RETURN d.year as year
            """
            result = await self.neo4j.execute_query(
                current_doc_query, {"source": current_source}
            )

            if not result or not result[0].get("year"):
                return

            current_year = result[0]["year"]

            # For each cited case, create appropriate relationships
            for cited_case in cited_cases[:5]:  # Limit to prevent explosion
                citation = extract_citation_from_name(cited_case)
                cited_year = extract_year_from_citation(citation)

                if cited_year and cited_year < current_year:
                    # Current case follows the older case
                    follow_query = """
                    MATCH (current:Document {source: $current_source})
                    MATCH (cited:Case {citation: $cited_citation})
                    MERGE (current)-[:FOLLOWS {weight: 1.0}]->(cited)
                    """

                    await self.neo4j.execute_write_query(
                        follow_query,
                        {"current_source": current_source, "cited_citation": citation},
                    )

        except Exception as e:
            logger.warning(f"Error creating temporal relationships: {e}")


async def main():
    """Main ingestion function."""
    parser = argparse.ArgumentParser(
        description="Ingest legal documents into Neo4j knowledge graph"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset the database before ingestion"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="../data/raw_docs",
        help="Directory containing PDF documents to ingest",
    )
    parser.add_argument(
        "--no-cache", 
        action="store_true", 
        help="Disable embedding cache (regenerate all embeddings)"
    )
    parser.add_argument(
        "--clear-cache", 
        action="store_true", 
        help="Clear the embedding cache before starting"
    )

    args = parser.parse_args()

    logger.info("🚀 Starting legal document ingestion process...")

    # Validate environment
    if not settings.google_api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        sys.exit(1)

    # Set up Neo4j connection
    neo4j_conn = Neo4jConnection(perf_tracker)

    try:
        # Test connection
        await neo4j_conn.execute_query("RETURN 1")
        logger.info("✅ Successfully connected to Neo4j")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Neo4j: {e}")
        sys.exit(1)

    # Initialize graph builder with caching options
    use_cache = not args.no_cache
    graph_builder = GraphBuilder(neo4j_conn, use_embedding_cache=use_cache)
    
    # Handle cache management
    if use_cache and args.clear_cache:
        logger.info("🗑️  Clearing LLM cache...")
        graph_builder.processor.cache.clear_cache()

    # Reset database if requested
    if args.reset:
        await graph_builder.reset_database()

    # Setup schema
    await graph_builder.setup_database_schema()

    # Find PDF files to process
    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        sys.exit(1)

    pdf_files = list(docs_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_dir}")
        return

    logger.info(f"📁 Found {len(pdf_files)} PDF files to process")

    # Process each document
    successful = 0
    failed = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"📋 Processing document {i}/{len(pdf_files)}: {pdf_path.name}")
        try:
            success = await graph_builder.process_document(pdf_path)
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ Failed to process {pdf_path}: {e}")
            failed += 1

        # Print intermediate summary every 5 documents or if it's the last one
        if i % 5 == 0 or i == len(pdf_files):
            logger.info(
                f"📊 Progress: {i}/{len(pdf_files)} processed ({successful} successful, {failed} failed)"
            )

    # Final summary
    logger.info("=" * 60)
    logger.info(f"🎉 Ingestion complete!")
    logger.info(
        f"📈 Results: {successful} successful, {failed} failed out of {len(pdf_files)} total"
    )

    # Print performance summary
    llm_cache = graph_builder.processor.cache if use_cache else None
    perf_tracker.print_summary(llm_cache)

    # Close connection
    await neo4j_conn.close()
    logger.info("🔌 Neo4j connection closed")


if __name__ == "__main__":
    asyncio.run(main())

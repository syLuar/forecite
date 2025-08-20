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
    python scripts/ingest_graph.py [--reset] [--docs-dir PATH] [--profile] [--required-tags TAGS] [--dry-run]

    --reset: Clear the database before ingestion
    --docs-dir: Path to directory containing PDF files (default: data/raw_docs)
    --required-tags: Filter documents to only process those with tags containing this string (case-insensitive)
    --dry-run: Simulate the ingestion process without making any actual operations
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
import time
import re

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from langchain.schema import Document
from langchain_core.prompts import PromptTemplate

from app.core.config import settings
from app.core.llm import create_llm

from dotenv import load_dotenv

# Import shared helpers
from ingest_graph_helpers import (
    LLMCache,
    PerformanceTracker,
    Neo4jConnection,
    BaseDocumentProcessor,
    LegalEntities,
    DocumentMetadata,
    Judge,
    Lawyer,
    LegalSummary,
    extract_citation_from_name,
    extract_year_from_citation,
    infer_jurisdiction,
    infer_court_level,
    determine_document_type,
)

# Load environment variables from .env file
load_dotenv()

# disable langsmith tracing by default
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_TRACING_V2"] = "false"

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

        self.entity_extraction_llm = create_llm(
            settings.llm_config.get("ingestion", {}).get("entity_extraction", {})
        )
        self.summary_generation_llm = create_llm(
            settings.llm_config.get("ingestion", {}).get("summary_generation", {})
        )
        self.metadata_extraction_llm = create_llm(
            settings.llm_config.get("ingestion", {}).get("metadata_extraction", {})
        )

        # Set up LLMs with structured output capabilities
        self.entity_extraction_llm_structured = (
            self.entity_extraction_llm.with_structured_output(LegalEntities)
        )
        self.metadata_extraction_llm_structured = (
            self.metadata_extraction_llm.with_structured_output(DocumentMetadata)
        )
        self.summary_generation_llm_structured = (
            self.summary_generation_llm.with_structured_output(LegalSummary)
        )

        # Create entity extraction chain with structured output
        entity_prompt = PromptTemplate(
            template="""Extract the following legal entities from this chunk of text:

IMPORTANT: Extract only entities that are explicitly mentioned or discussed in this specific chunk. Do not include document-level metadata like court names or judge names.

Focus on:
- Case citations and references
- Statute references and sections 
- Legal concepts and principles
- Legal holdings and reasoning
- Key facts
- Legal tests and standards

Text: {text}""",
            input_variables=["text"],
        )
        self.entity_extraction_chain = (
            entity_prompt | self.entity_extraction_llm_structured
        )

        # Create metadata extraction chain with structured output
        metadata_prompt = PromptTemplate(
            template="""Extract metadata from this legal document including judges, lawyers, and parties:

IMPORTANT: For parties, extract the main parties involved in the case (typically 2-4 parties). These will be used to create citation formats like "(Party 1) vs (Party 2)".

Document text:
{text}

Filename: {filename}""",
            input_variables=["text", "filename"],
        )
        self.metadata_extraction_chain = (
            metadata_prompt | self.metadata_extraction_llm_structured
        )

        # Create summary generation chain with integrated relevance assessment
        summary_prompt = PromptTemplate(
            template="""You are analyzing text from a legal document for potential indexing in a legal research database.

CRITICAL INSTRUCTIONS:
1. ONLY provide a summary if the text contains substantial legal content worth indexing (cases, statutes, legal principles, holdings, reasoning, facts, etc.)
2. DO NOT summarize if the text is just: headers, footers, page numbers, table of contents, procedural notes, administrative content, or other non-substantive material
3. Be STRICT - if in doubt, don't summarize. We want quality over quantity.
4. Your reasoning field should explain your decision clearly.

Text to analyze:
{text}""",
            input_variables=["text"],
        )
        self.summary_generation_chain = (
            summary_prompt | self.summary_generation_llm_structured
        )

    async def generate_summary_with_relevance(
        self, text: str
    ) -> tuple[Optional[str], bool]:
        """Generate a summary of the text with integrated relevance assessment using LLM with structured output and caching support.

        Returns:
            tuple: (summary, is_relevant) where summary is None if not relevant
        """
        try:
            # Check cache first
            if self.use_cache and self.cache:
                cached_summary = self.cache.get_summary(text)
                if cached_summary:
                    logger.debug("📦 Summary found in cache")
                    # If cached summary exists, assume it was relevant
                    return cached_summary, True

            start_time = time.time()

            # Use the structured output chain
            summary_result = await self.summary_generation_chain.ainvoke({"text": text})

            summary_time = time.time() - start_time
            perf_tracker.record_metric("summary_generation_time", summary_time)
            perf_tracker.record_metric("llm_time", summary_time)
            perf_tracker.record_metric("llm_calls", 1)

            # Extract the summary and relevance from the structured result
            summary = summary_result.summary
            is_relevant = summary is not None

            logger.debug(
                f"Chunk relevance: {is_relevant}, reasoning: {summary_result.reasoning}"
            )

            # Store in cache only if relevant
            if is_relevant and self.use_cache and self.cache:
                self.cache.store_summary(text, summary)

            return summary, is_relevant
        except Exception as e:
            logger.error(f"Error generating summary with structured output: {e}")
            return None, False

    async def generate_summary(self, text: str) -> str:
        """Generate a summary of the text using LLM with structured output and caching support.
        This method is kept for backward compatibility.
        """
        summary, _ = await self.generate_summary_with_relevance(text)
        return summary or "Summary generation failed"

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
            entities = await self.entity_extraction_chain.ainvoke({"text": text})

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
            # Return default empty structure matching the updated schema
            return {
                "cases": [],
                "statutes": [],
                "concepts": [],
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

    async def extract_citation_formats(
        self,
        text: str,
        filename: str,
        parties: List[str] = None,
        scraped_metadata: Dict = None,
    ) -> List[str]:
        """Extract the two main citation formats for this case, preferring scraped metadata."""
        try:
            citation_formats_list = []

            # 1. Use scraped metadata if available
            if scraped_metadata:
                # Primary format from scraped parties
                scraped_parties = scraped_metadata.get("parties", [])
                if scraped_parties and len(scraped_parties) >= 2:
                    primary_citation = f"{scraped_parties[0]} vs {scraped_parties[1]}"
                    citation_formats_list.append(primary_citation)

                # Secondary format from scraped citation
                scraped_citation = scraped_metadata.get("citation")
                if scraped_citation:
                    citation_formats_list.append(scraped_citation)

                return citation_formats_list

            # 2. Fallback to LLM-extracted parties
            if parties and len(parties) >= 2:
                primary_citation = f"{parties[0]} vs {parties[1]}"
                citation_formats_list.append(primary_citation)

            # 3. Secondary format: [Year] SGCA [Num] - from filename
            filename_citation = re.search(r"\[(\d{4})\]\s*([A-Z]+)\s*(\d+)", filename)
            if filename_citation:
                secondary_citation = f"[{filename_citation.group(1)}] {filename_citation.group(2)} {filename_citation.group(3)}"
                citation_formats_list.append(secondary_citation)

            return citation_formats_list
        except Exception as e:
            logger.warning(f"Error extracting citation formats: {e}")
            return []

    def load_scraped_metadata(self, cases_dir: Path) -> Dict[str, Dict]:
        """Load metadata from the scraped cases metadata file."""
        metadata_file = cases_dir / "cases_metadata.json"
        if not metadata_file.exists():
            logger.warning(f"No metadata file found at {metadata_file}")
            return {}

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(metadata)} cases")
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata file: {e}")
            return {}

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
        self.scraped_metadata = {}  # Will store the scraped metadata

    def load_scraped_metadata(self, docs_dir: Path):
        """Load scraped metadata for cases."""
        self.scraped_metadata = self.processor.load_scraped_metadata(docs_dir / "cases")

    def get_case_metadata_by_filename(self, filename: str) -> Optional[Dict]:
        """Get scraped metadata for a case by matching filename patterns."""
        # Extract case ID from filename (e.g., "2025_SGHC_163_...")
        case_id_match = re.search(r"(\d{4}_[A-Z]+_\d+)", filename)
        if case_id_match:
            case_id = case_id_match.group(1)
            return self.scraped_metadata.get(case_id)
        return None

    def filter_files_by_required_tags(
        self, pdf_files: List[Path], required_tags: str
    ) -> List[Path]:
        """Filter PDF files to only include those with metadata tags containing the required string."""
        if not required_tags:
            return pdf_files

        required_tags_lower = required_tags.lower()
        filtered_files = []
        no_metadata_count = 0

        for pdf_path in pdf_files:
            case_metadata = self.get_case_metadata_by_filename(pdf_path.name)
            if case_metadata and "tags" in case_metadata:
                # Check if any tag contains the required string (case-insensitive)
                tags = case_metadata.get("tags", [])
                if any(required_tags_lower in tag.lower() for tag in tags):
                    filtered_files.append(pdf_path)
                    logger.debug(
                        f"✅ Including {pdf_path.name} - matches required tag '{required_tags}'"
                    )
                else:
                    logger.debug(f"❌ Excluding {pdf_path.name} - no matching tags")
            else:
                no_metadata_count += 1
                logger.debug(
                    f"❌ Excluding {pdf_path.name} - no metadata or tags found"
                )

        if no_metadata_count > 0:
            logger.info(
                f"📋 Note: {no_metadata_count} files excluded due to missing metadata"
            )

        return filtered_files

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
            "CREATE CONSTRAINT unique_legal_tag_name IF NOT EXISTS FOR (lt:LegalTag) REQUIRE lt.name IS UNIQUE",
            "CREATE CONSTRAINT unique_judge_name IF NOT EXISTS FOR (j:Judge) REQUIRE j.name IS UNIQUE",
            "CREATE CONSTRAINT unique_lawyer_name IF NOT EXISTS FOR (l:Lawyer) REQUIRE l.name IS UNIQUE",
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
            "CREATE INDEX legal_tag_usage_index IF NOT EXISTS FOR (lt:LegalTag) ON (lt.usage_count)",
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
                if index_name in [
                    settings.vector_index_name,
                    "legal_text_search",
                    "chunk_text_search",
                ]:
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

        # Get scraped metadata for this case
        scraped_case_metadata = self.get_case_metadata_by_filename(pdf_path.name)
        if scraped_case_metadata:
            logger.info(f"📋 Found scraped metadata for {pdf_path.name}")
        else:
            logger.info(
                f"📋 No scraped metadata found for {pdf_path.name}, will extract from content"
            )

        # Extract text
        text = self.processor.extract_text_from_pdf(pdf_path)
        if not text:
            logger.error(f"No text extracted from {pdf_path}")
            return False

        # Extract document metadata (combining scraped and extracted)
        doc_metadata = await self._extract_document_metadata(
            pdf_path, text, scraped_case_metadata
        )

        # Create document node with enhanced metadata
        doc_params = {
            "source": str(pdf_path.name),
            "type": doc_metadata["type"],
            "full_text": text,  # Store first 5000 chars
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
            "legal_tags": doc_metadata.get("legal_tags", []),  # From scraped metadata
            "citation_formats": doc_metadata.get("citation_formats", []),
            "court": doc_metadata.get("court"),  # From scraped metadata
            "decision_date": doc_metadata.get("decision_date"),  # From scraped metadata
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
            d.legal_tags = $legal_tags,
            d.citation_formats = $citation_formats,
            d.court = $court,
            d.decision_date = $decision_date,
            d.created_at = datetime()
        RETURN d
        """

        await self.neo4j.execute_write_query(create_doc_query, doc_params)

        # Create judge and lawyer entities
        await self._create_judge_entities(
            str(pdf_path.name), doc_metadata.get("judges", [])
        )
        await self._create_lawyer_entities(
            str(pdf_path.name), doc_metadata.get("lawyers", [])
        )

        # Create legal tag entities and relationships
        await self._create_legal_tag_entities(
            str(pdf_path.name), doc_metadata.get("legal_tags", [])
        )

        # Chunk the document
        chunks = self.processor.chunk_document(text, str(pdf_path.name))
        total_chunks = len(chunks)
        logger.info(f"📄 Created {total_chunks} chunks for {pdf_path.name}")

        # Track references and paragraph mappings for later relationship creation
        chunk_references_map = {}  # chunk_id -> list of referenced paragraph numbers
        paragraph_to_chunk_map = {}  # paragraph_number -> chunk_id

        # Track relevance statistics
        relevant_chunks_count = 0

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

        # Count actual chunks created in database
        count_query = """
        MATCH (c:Chunk {source: $source})
        RETURN count(c) as chunk_count
        """
        result = await self.neo4j.execute_query(
            count_query, {"source": str(pdf_path.name)}
        )
        actual_chunks_count = result[0]["chunk_count"] if result else 0

        logger.info(
            f"📊 Chunk processing summary for {pdf_path.name}: {actual_chunks_count}/{total_chunks} chunks were relevant and stored"
        )

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
        # Check relevance and generate summaries for all chunks
        summary_tasks = []
        for chunk in chunks:
            summary_tasks.append(
                self.processor.generate_summary_with_relevance(chunk.page_content)
            )

        summary_results = await asyncio.gather(*summary_tasks)

        # Filter to only relevant chunks
        relevant_chunks = []
        relevant_texts = []
        relevant_summaries = []

        for i, (chunk, (summary, is_relevant)) in enumerate(
            zip(chunks, summary_results)
        ):
            if is_relevant:
                relevant_chunks.append((i, chunk))
                relevant_texts.append(chunk.page_content)
                relevant_summaries.append(summary)
            else:
                logger.debug(
                    f"Skipping non-relevant chunk {chunk.metadata['chunk_index']} from {source}"
                )

        if not relevant_chunks:
            logger.info(f"⚠️  No relevant chunks found in batch for {source}")
            return

        logger.info(
            f"✅ Found {len(relevant_chunks)}/{len(chunks)} relevant chunks in batch for {source}"
        )

        # Generate embeddings only for relevant chunks
        relevant_embeddings = await self.processor.generate_embeddings(relevant_texts)

        # Create embedding mapping
        embedding_map = {}
        for j, (original_idx, _) in enumerate(relevant_chunks):
            if j < len(relevant_embeddings):
                embedding_map[original_idx] = relevant_embeddings[j]

        # Process only relevant chunks
        tasks = []
        for j, (original_idx, chunk) in enumerate(relevant_chunks):
            embedding = embedding_map.get(original_idx, None)
            summary = (
                relevant_summaries[j]
                if j < len(relevant_summaries)
                else "Summary generation failed"
            )
            tasks.append(
                self._create_chunk_node(
                    chunk,
                    embedding,
                    summary,
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
        summary: str,
        source: str,
        chunk_references_map: Dict,
        paragraph_to_chunk_map: Dict,
    ):
        """Create a chunk node and its relationships for relevant chunks only."""
        # Since we've already filtered for relevance, all chunks reaching here are relevant

        # Extract entities for the relevant chunk
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
            "indexed": True,  # All chunks reaching here are indexed since they're relevant
            "statutes": entities.get("statutes", []),
            "cases": entities.get("cases", []),
            "concepts": entities.get("concepts", []),
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
            source: $source,
            indexed: $indexed,
            statutes: $statutes,
            cases: $cases,
            concepts: $concepts,
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
        await self._create_reference_relationships(chunk_params["id"], entities)

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

    async def _extract_document_metadata(
        self, pdf_path: Path, text: str, scraped_metadata: Dict = None
    ) -> Dict[str, Any]:
        """Extract enhanced metadata from document using structured output and scraped metadata."""
        try:
            filename = pdf_path.name
            metadata = {"type": determine_document_type(pdf_path)}

            # If we have scraped metadata, prioritize it
            if scraped_metadata:
                logger.info(f"Using scraped metadata for {filename}")

                # Map scraped metadata to our schema
                metadata.update(
                    {
                        "title": scraped_metadata.get("title", ""),
                        "parties": scraped_metadata.get("parties", []),
                        "citation": scraped_metadata.get("citation", ""),
                        "case_number": scraped_metadata.get("case_number", ""),
                        "court": scraped_metadata.get("court", ""),
                        "decision_date": scraped_metadata.get("decision_date", ""),
                        "legal_tags": scraped_metadata.get("tags", []),
                        "year": scraped_metadata.get("year"),
                        "jurisdiction": infer_jurisdiction(
                            scraped_metadata.get("court", "")
                        ),
                        "court_level": infer_court_level(
                            scraped_metadata.get("court", "")
                        ),
                    }
                )

                # Convert year to int if it's a string
                if isinstance(metadata.get("year"), str):
                    try:
                        metadata["year"] = int(metadata["year"])
                    except (ValueError, TypeError):
                        metadata["year"] = None

                # Set date from decision_date if available
                if scraped_metadata.get("decision_date"):
                    metadata["date"] = scraped_metadata["decision_date"]
                elif metadata.get("year"):
                    metadata["date"] = f"{metadata['year']}-01-01"

                # Use scraped metadata for citation formats
                citation_formats = await self.processor.extract_citation_formats(
                    text, filename, scraped_metadata=scraped_metadata
                )
                metadata["citation_formats"] = citation_formats

                # Extract judges/lawyers from LLM if not in scraped metadata
                # (scraped metadata doesn't include judge/lawyer info)
                try:
                    if len(text) > 4000:
                        text_input = text[:2000] + "\n...\n" + text[-2000:]
                    else:
                        text_input = text

                    start_time = time.time()
                    llm_metadata = (
                        await self.processor.metadata_extraction_chain.ainvoke(
                            {
                                "text": text_input,
                                "filename": filename,
                            }
                        )
                    )

                    metadata_time = time.time() - start_time
                    perf_tracker.record_metric("llm_time", metadata_time)
                    perf_tracker.record_metric("llm_calls", 1)

                    # Only use judges and lawyers from LLM
                    llm_metadata_dict = llm_metadata.model_dump()
                    metadata["judges"] = llm_metadata_dict.get("judges", [])
                    metadata["lawyers"] = llm_metadata_dict.get("lawyers", [])

                except Exception as llm_error:
                    logger.warning(f"LLM metadata extraction failed: {llm_error}")
                    metadata["judges"] = []
                    metadata["lawyers"] = []

            else:
                # Fallback to original LLM-based extraction when no scraped metadata
                logger.info(f"No scraped metadata for {filename}, using LLM extraction")

                # Basic metadata from filename
                year_match = re.search(r"\[(\d{4})\]", filename)
                if year_match:
                    metadata["year"] = int(year_match.group(1))
                    metadata["date"] = f"{year_match.group(1)}-01-01"

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
                    # Prepare text based on length
                    if len(text) > 4000:
                        text_input = text[:2000] + "\n...\n" + text[-2000:]
                    else:
                        text_input = text

                    llm_metadata = (
                        await self.processor.metadata_extraction_chain.ainvoke(
                            {
                                "text": text_input,
                                "filename": filename,
                            }
                        )
                    )

                    metadata_time = time.time() - start_time
                    perf_tracker.record_metric("llm_time", metadata_time)
                    perf_tracker.record_metric("llm_calls", 1)

                    # Convert Pydantic model to dictionary and update metadata
                    llm_metadata_dict = llm_metadata.model_dump()
                    metadata.update(llm_metadata_dict)

                    # Extract citation formats using the LLM-extracted parties
                    citation_formats = await self.processor.extract_citation_formats(
                        text, filename, parties=llm_metadata_dict.get("parties", [])
                    )
                    metadata["citation_formats"] = citation_formats

                except Exception as llm_error:
                    logger.warning(
                        f"Structured metadata extraction failed: {llm_error}"
                    )
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
                                "legal_tags": [],
                                "judges": [],
                                "lawyers": [],
                            }
                        )
                    else:
                        metadata.update(
                            {
                                "jurisdiction": "Unknown",
                                "court_level": "Unknown",
                                "parties": [],
                                "legal_areas": [],
                                "legal_tags": [],
                                "judges": [],
                                "lawyers": [],
                            }
                        )

                    # Extract citation formats with empty parties (fallback)
                    citation_formats = await self.processor.extract_citation_formats(
                        text, filename, parties=[]
                    )
                    metadata["citation_formats"] = citation_formats

            # Override court_level and jurisdiction for Singapore cases to avoid LLM hallucinations
            if "SGCA" in filename:
                metadata["jurisdiction"] = "Singapore"
                metadata["court_level"] = "Court of Appeal"
            elif "SGHC" in filename:
                metadata["jurisdiction"] = "Singapore"
                metadata["court_level"] = "High Court"

            return metadata
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            return {
                "type": "Document",
                "jurisdiction": "Unknown",
                "court_level": "Unknown",
                "parties": [],
                "legal_areas": [],
                "judges": [],
                "lawyers": [],
                "citation_formats": [],
            }

    async def _create_document_entities(self, source: str, full_text: str):
        """Create entity nodes and relationships for the entire document."""
        try:
            entities = await self.processor.extract_entities(full_text)

            # Create Case entities and relationships
            for case_name in entities.get("cases", [])[
                :10
            ]:  # Limit to prevent explosion
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

            # Create Court entities - hardcode Singapore Court of Appeal to avoid LLM hallucinations
            if "SGCA" in source:
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
                        "name": "Singapore Court of Appeal",
                        "jurisdiction": "Singapore",
                        "level": "Appellate",
                        "source": source,
                    },
                )
            elif "SGHC" in source:
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
                        "name": "Singapore High Court",
                        "jurisdiction": "Singapore",
                        "level": "Superior",
                        "source": source,
                    },
                )
            else:
                # Fallback to LLM extraction for other courts
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

            # Note: Courts and judges are now only created at document level from metadata extraction,
            # not from chunk-level entity extraction, since they are document-level attributes

            # Create temporal relationships between cases based on citation patterns
            await self._create_temporal_relationships(source, entities.get("cases", []))

            # Create case citation relationships based on citation formats
            await self._create_case_citation_relationships(source)

        except Exception as e:
            logger.error(f"Error creating document entities: {e}")

    async def _create_legal_tag_entities(self, source: str, legal_tags: List[str]):
        """Create legal tag entities and relationships to the document."""
        if not legal_tags:
            return

        for tag in legal_tags:
            if not tag or not tag.strip():
                continue

            tag_clean = tag.strip().lower()

            # Create legal tag node and relationship
            create_tag_query = """
            MATCH (d:Document {source: $source})
            MERGE (lt:LegalTag {name: $tag_name})
            SET lt.created_at = coalesce(lt.created_at, datetime()),
                lt.usage_count = coalesce(lt.usage_count, 0) + 1
            MERGE (d)-[:HAS_LEGAL_TAG]->(lt)
            """

            await self.neo4j.execute_write_query(
                create_tag_query, {"source": source, "tag_name": tag_clean}
            )

        logger.info(f"Created {len(legal_tags)} legal tag relationships for {source}")

    async def _create_judge_entities(self, source: str, judges: List[Dict[str, str]]):
        """Create judge entities and relationships to the document."""
        for judge_data in judges:
            if (
                judge_data
                and judge_data.get("name")
                and len(judge_data["name"].strip()) > 2
            ):
                create_judge_query = """
                MERGE (j:Judge {name: $name})
                SET j.role = $role,
                    j.created_at = datetime()
                WITH j
                MATCH (d:Document {source: $source})
                MERGE (j)-[:PRESIDED_OVER]->(d)
                """

                await self.neo4j.execute_write_query(
                    create_judge_query,
                    {
                        "name": judge_data["name"].strip(),
                        "role": judge_data.get("role", "Judge"),
                        "source": source,
                    },
                )

    async def _create_lawyer_entities(self, source: str, lawyers: List[Dict[str, str]]):
        """Create lawyer entities and relationships to the document."""
        for lawyer_data in lawyers:
            if (
                lawyer_data
                and lawyer_data.get("name")
                and len(lawyer_data["name"].strip()) > 2
            ):
                create_lawyer_query = """
                MERGE (l:Lawyer {name: $name})
                SET l.firm = $firm,
                    l.created_at = datetime()
                WITH l
                MATCH (d:Document {source: $source})
                MERGE (l)-[:REPRESENTED_IN {role: $role}]->(d)
                """

                await self.neo4j.execute_write_query(
                    create_lawyer_query,
                    {
                        "name": lawyer_data["name"].strip(),
                        "firm": lawyer_data.get("firm", ""),
                        "role": lawyer_data.get("role", "Counsel"),
                        "source": source,
                    },
                )

    async def _create_case_citation_relationships(self, current_source: str):
        """Create citation relationships between cases based on citation formats."""
        try:
            # Get current document's citation information
            current_doc_query = """
            MATCH (d:Document {source: $source})
            RETURN d.citation_formats as citation_formats, d.citation as citation, d.year as year
            """
            result = await self.neo4j.execute_query(
                current_doc_query, {"source": current_source}
            )

            if not result:
                return

            current_data = result[0]
            current_citation_formats = current_data.get("citation_formats", [])
            current_year = current_data.get("year")

            if not current_citation_formats:
                return

            # Use all citation formats from the list
            all_formats = [f for f in current_citation_formats if f and f.strip()]

            # Search for other documents that cite this case using any of these formats
            for citation_format in all_formats:
                if len(citation_format) > 5:  # Only process meaningful citations
                    cite_query = """
                    MATCH (citing:Document)
                    WHERE citing.source <> $current_source 
                    AND citing.full_text CONTAINS $citation_format
                    MATCH (cited:Document {source: $current_source})
                    MERGE (citing)-[:CITES_CASE {
                        citation_format: $citation_format,
                        created_at: datetime()
                    }]->(cited)
                    """

                    await self.neo4j.execute_write_query(
                        cite_query,
                        {
                            "current_source": current_source,
                            "citation_format": citation_format,
                        },
                    )

            logger.info(f"Created citation relationships for {current_source}")

        except Exception as e:
            logger.warning(f"Error creating case citation relationships: {e}")

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
        help="Disable embedding cache (regenerate all embeddings)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the embedding cache before starting",
    )
    parser.add_argument(
        "--required-tags",
        type=str,
        help="Filter documents to only process those with tags containing this string (case-insensitive)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the ingestion process without making any actual operations (including LLM API calls)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of documents to process (for testing purposes)",
    )

    args = parser.parse_args()

    logger.info("🚀 Starting legal document ingestion process...")

    # Validate environment (skip in dry-run mode since no API calls will be made)
    if not args.dry_run and not settings.google_api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        sys.exit(1)
    elif args.dry_run:
        logger.info("🔍 DRY-RUN: Skipping environment validation...")

    # Set up Neo4j connection
    neo4j_conn = Neo4jConnection(perf_tracker)

    # Skip Neo4j connection test in dry-run mode
    if not args.dry_run:
        try:
            # Test connection
            await neo4j_conn.execute_query("RETURN 1")
            logger.info("✅ Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            sys.exit(1)
    else:
        logger.info("🔍 DRY-RUN: Skipping Neo4j connection test...")

    # Initialize graph builder with caching options
    use_cache = not args.no_cache
    graph_builder = GraphBuilder(neo4j_conn, use_embedding_cache=use_cache)

    # Load scraped metadata
    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        sys.exit(1)

    logger.info("📋 Loading scraped metadata...")
    graph_builder.load_scraped_metadata(docs_dir)

    # Handle cache management
    if use_cache and args.clear_cache and not args.dry_run:
        logger.info("🗑️  Clearing LLM cache...")
        graph_builder.processor.cache.clear_cache()

    # Skip database operations in dry-run mode
    if not args.dry_run:
        # Reset database if requested
        if args.reset:
            await graph_builder.reset_database()

        # Setup schema
        await graph_builder.setup_database_schema()
    else:
        logger.info("🔍 DRY-RUN: Skipping database operations...")

    # Find PDF files to process
    pdf_files = list(docs_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_dir}")
        return

    logger.info(f"📁 Found {len(pdf_files)} PDF files")

    # Filter by required tags if specified
    if args.required_tags:
        logger.info(f"🏷️  Filtering files by required tags: '{args.required_tags}'")
        pdf_files = graph_builder.filter_files_by_required_tags(
            pdf_files, args.required_tags
        )
        logger.info(
            f"📁 After filtering: {len(pdf_files)} PDF files match the required tags"
        )

        if not pdf_files:
            logger.warning(
                f"No PDF files match the required tags: '{args.required_tags}'"
            )
            return

    logger.info(f"📁 Processing {len(pdf_files)} PDF files")

    if args.limit:
        logger.info(f"🔍 Limiting to {args.limit} files for testing")
        pdf_files = pdf_files[: args.limit]

    # Handle dry-run mode
    if args.dry_run:
        logger.info("🔍 DRY-RUN MODE: Simulating ingestion process...")
        logger.info("=" * 60)
        logger.info(f"📊 DRY-RUN SUMMARY:")
        logger.info(f"📁 Total files found: {len(list(docs_dir.rglob('*.pdf')))}")
        if args.required_tags:
            logger.info(f"🏷️  Tag filter applied: '{args.required_tags}'")
        logger.info(f"📋 Files that would be processed: {len(pdf_files)}")

        # Show some example files that would be processed
        if pdf_files:
            logger.info(f"📄 Example files to be processed:")
            for i, pdf_path in enumerate(pdf_files[:5], 1):
                case_metadata = graph_builder.get_case_metadata_by_filename(
                    pdf_path.name
                )
                if case_metadata:
                    tags = case_metadata.get("tags", [])
                    tag_info = f" (tags: {len(tags)} found)" if tags else " (no tags)"
                else:
                    tag_info = " (no metadata)"
                logger.info(f"  {i}. {pdf_path.name}{tag_info}")

            if len(pdf_files) > 5:
                logger.info(f"  ... and {len(pdf_files) - 5} more files")

        logger.info("=" * 60)
        logger.info("🔍 DRY-RUN completed - no actual operations performed")
        await neo4j_conn.close()
        return

    # Process each document
    successful = 0
    failed = 0

    tasks = []

    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"📋 Processing document {i}/{len(pdf_files)}: {pdf_path.name}")
        try:
            task = graph_builder.process_document(pdf_path)
            tasks.append(task)
        except Exception as e:
            logger.error(f"❌ Failed to process {pdf_path}: {e}")
            failed += 1

    # Run all tasks concurrently
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"❌ Task failed: {result}")
                failed += 1
            else:
                successful += 1

            if (successful + failed) % 5 == 0 or successful + failed == len(pdf_files):
                logger.info(
                    f"📊 Progress: {successful + failed}/{len(pdf_files)} processed ({successful} successful, {failed} failed)"
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

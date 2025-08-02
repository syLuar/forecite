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
import functools
from contextlib import contextmanager

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from neo4j import GraphDatabase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from google import genai

from app.core.config import settings

from dotenv import load_dotenv

# Import the new PDF processing module
from process_pdf import process_pdf_document

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track performance metrics during ingestion."""

    def __init__(self):
        self.metrics = {
            "total_start_time": time.time(),
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "llm_calls": 0,
            "db_operations": 0,
            "pdf_extraction_time": 0,
            "chunking_time": 0,
            "embedding_time": 0,
            "llm_time": 0,
            "db_time": 0,
            "entity_extraction_time": 0,
            "summary_generation_time": 0,
        }

    def record_metric(self, key: str, value: float = 1):
        """Record a metric value."""
        if key in self.metrics:
            self.metrics[key] += value
        else:
            self.metrics[key] = value

    def print_summary(self):
        """Print performance summary."""
        total_time = time.time() - self.metrics["total_start_time"]

        logger.info("=" * 60)
        logger.info("🔍 PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total execution time: {total_time:.2f}s")
        logger.info(f"Documents processed: {self.metrics['documents_processed']}")
        logger.info(f"Chunks created: {self.metrics['chunks_created']}")
        logger.info(f"Embeddings generated: {self.metrics['embeddings_generated']}")
        logger.info(f"LLM calls made: {self.metrics['llm_calls']}")
        logger.info(f"Database operations: {self.metrics['db_operations']}")
        logger.info("")
        logger.info("⏱️  Time breakdown:")
        logger.info(f"  PDF extraction: {self.metrics['pdf_extraction_time']:.2f}s")
        logger.info(f"  Document chunking: {self.metrics['chunking_time']:.2f}s")
        logger.info(f"  Embedding generation: {self.metrics['embedding_time']:.2f}s")
        logger.info(f"  LLM operations: {self.metrics['llm_time']:.2f}s")
        logger.info(
            f"    - Entity extraction: {self.metrics['entity_extraction_time']:.2f}s"
        )
        logger.info(
            f"    - Summary generation: {self.metrics['summary_generation_time']:.2f}s"
        )
        logger.info(f"  Database operations: {self.metrics['db_time']:.2f}s")
        logger.info("")

        if self.metrics["documents_processed"] > 0:
            avg_per_doc = total_time / self.metrics["documents_processed"]
            logger.info(f"📊 Average time per document: {avg_per_doc:.2f}s")

        if self.metrics["chunks_created"] > 0:
            avg_per_chunk = total_time / self.metrics["chunks_created"]
            logger.info(f"📊 Average time per chunk: {avg_per_chunk:.2f}s")

        # Identify bottlenecks
        time_operations = [
            ("PDF extraction", self.metrics["pdf_extraction_time"]),
            ("Chunking", self.metrics["chunking_time"]),
            ("Embeddings", self.metrics["embedding_time"]),
            ("LLM operations", self.metrics["llm_time"]),
            ("Database operations", self.metrics["db_time"]),
        ]

        sorted_ops = sorted(time_operations, key=lambda x: x[1], reverse=True)
        logger.info("")
        logger.info("🚨 Potential bottlenecks (sorted by time):")
        for i, (op_name, op_time) in enumerate(sorted_ops[:3], 1):
            percentage = (op_time / total_time) * 100 if total_time > 0 else 0
            logger.info(f"  {i}. {op_name}: {op_time:.2f}s ({percentage:.1f}%)")

        logger.info("=" * 60)


# Global performance tracker
perf_tracker = PerformanceTracker()


class Neo4jConnection:
    """Manages Neo4j database connection and operations."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password)
        )
        self.database = settings.neo4j_database

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()

    def execute_query(
        self, query: str, parameters: Optional[Dict] = None
    ) -> List[Dict]:
        """Execute a Cypher query and return results."""
        start_time = time.time()
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            data = [record.data() for record in result]

        query_time = time.time() - start_time
        perf_tracker.record_metric("db_time", query_time)
        perf_tracker.record_metric("db_operations", 1)

        return data

    def execute_write_query(
        self, query: str, parameters: Optional[Dict] = None
    ) -> List[Dict]:
        """Execute a write Cypher query and return results."""

        def _execute_query(tx):
            result = tx.run(query, parameters or {})
            return [record.data() for record in result]

        start_time = time.time()
        with self.driver.session(database=self.database) as session:
            data = session.execute_write(_execute_query)

        query_time = time.time() - start_time
        perf_tracker.record_metric("db_time", query_time)
        perf_tracker.record_metric("db_operations", 1)

        return data


class DocumentProcessor:
    """Handles document processing including chunking and embedding generation."""

    def __init__(self):
        # Configure Google AI with API key
        client = genai.Client(api_key=settings.google_api_key)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", google_api_key=settings.google_api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            thinking_budget=0,
            temperature=0.2,
        )

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from a PDF file using the enhanced processor."""
        try:
            start_time = time.time()

            # Use the new PDF processor to extract and integrate content
            pages = process_pdf_document(str(pdf_path))

            if not pages:
                logger.error(f"No pages extracted from {pdf_path}")
                return ""

            # Combine all page content into a single text
            full_text = ""
            for page in pages:
                if full_text:
                    full_text += "\n\n"
                full_text += page.get("content", "")

            extraction_time = time.time() - start_time
            perf_tracker.record_metric("pdf_extraction_time", extraction_time)

            logger.info(
                f"📝 Extracted {len(pages)} pages with integrated footnotes from {pdf_path.name}"
            )
            return full_text.strip()

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def chunk_document(self, text: str, source: str) -> List[Document]:
        """Split document text into overlapping chunks."""
        start_time = time.time()
        chunks = self.text_splitter.split_text(text)
        documents = []

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            documents.append(doc)

        chunking_time = time.time() - start_time
        perf_tracker.record_metric("chunking_time", chunking_time)
        perf_tracker.record_metric("chunks_created", len(documents))
        logger.info(f"📝 Created {len(documents)} chunks for {source}")
        return documents

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            start_time = time.time()
            logger.info(f"🔢 Generating embeddings for {len(texts)} texts...")
            embeddings = await self.embeddings.aembed_documents(texts)

            embedding_time = time.time() - start_time
            perf_tracker.record_metric("embedding_time", embedding_time)
            perf_tracker.record_metric("embeddings_generated", len(texts))

            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []

    async def generate_summary(self, text: str) -> str:
        """Generate a summary of the text using LLM."""
        try:
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

            response = await self.llm.ainvoke(prompt)
            summary = (
                response.content if hasattr(response, "content") else str(response)
            )

            summary_time = time.time() - start_time
            perf_tracker.record_metric("summary_generation_time", summary_time)
            perf_tracker.record_metric("llm_time", summary_time)
            perf_tracker.record_metric("llm_calls", 1)

            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Summary generation failed"

    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities from the text using LLM."""
        try:
            start_time = time.time()

            prompt = f"""
            Extract the following legal entities from the text and format as JSON:
            - cases: List of case names and citations (include year if available)
            - statutes: List of statute references with sections
            - concepts: List of key legal concepts and principles
            - courts: List of court names
            - judges: List of judge names mentioned
            - holdings: List of key legal holdings or ratio decidendi
            - facts: List of key factual elements
            - legal_tests: List of legal tests or standards mentioned
            
            Text: {text[:1500]}...
            
            Return only valid JSON with the eight keys above.
            """

            response = await self.llm.ainvoke(prompt)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            entity_time = time.time() - start_time
            perf_tracker.record_metric("entity_extraction_time", entity_time)
            perf_tracker.record_metric("llm_time", entity_time)
            perf_tracker.record_metric("llm_calls", 1)

            # Initialize default entities
            entities = {
                "cases": [],
                "statutes": [],
                "concepts": [],
                "courts": [],
                "judges": [],
                "holdings": [],
                "facts": [],
                "legal_tests": [],
            }

            # Try to parse JSON response first
            import json
            import re

            try:
                # Try to extract JSON from the response
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_entities = json.loads(json_str)

                    # Ensure all values are lists of strings
                    for key in entities.keys():
                        if key in parsed_entities and isinstance(
                            parsed_entities[key], list
                        ):
                            # Clean and filter the entities
                            entities[key] = [
                                str(item).strip()
                                for item in parsed_entities[key]
                                if item and str(item).strip()
                            ]

                    return entities
            except (json.JSONDecodeError, AttributeError):
                logger.warning(
                    "Failed to parse JSON response, falling back to simple extraction"
                )

            # Fallback: Simple entity extraction (replace with more sophisticated NER in production)
            lines = response_text.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if "v." in line or "vs." in line or "[" in line and "]" in line:
                    entities["cases"].append(line)
                elif any(
                    keyword in line.lower()
                    for keyword in ["§", "section", "code", "statute", "act"]
                ):
                    entities["statutes"].append(line)
                elif any(
                    keyword in line.lower()
                    for keyword in ["court", "tribunal", "appeal"]
                ):
                    entities["courts"].append(line)
                elif any(
                    keyword in line.lower() for keyword in ["j.", "judge", "justice"]
                ):
                    entities["judges"].append(line)
                elif any(
                    keyword in line.lower() for keyword in ["held", "holding", "ratio"]
                ):
                    entities["holdings"].append(line)
                elif any(
                    keyword in line.lower()
                    for keyword in ["test", "standard", "criteria"]
                ):
                    entities["legal_tests"].append(line)
                elif len(line) > 10 and not any(
                    c in line for c in [":", "{", "}", "[", "]"]
                ):
                    # Potential legal concept or fact
                    if any(
                        keyword in line.lower()
                        for keyword in ["principle", "doctrine", "rule"]
                    ):
                        entities["concepts"].append(line)
                    else:
                        entities["facts"].append(line)

            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
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


class GraphBuilder:
    """Builds and populates the Neo4j knowledge graph."""

    def __init__(self, neo4j_conn: Neo4jConnection):
        self.neo4j = neo4j_conn
        self.processor = DocumentProcessor()
        self._vector_search_available = None

    def check_vector_search_availability(self) -> bool:
        """Check if vector search procedures are available in the Neo4j instance."""
        if self._vector_search_available is not None:
            return self._vector_search_available

        try:
            # Try to list available procedures to check for vector support
            result = self.neo4j.execute_query(
                "CALL dbms.procedures() YIELD name WHERE name CONTAINS 'vector' RETURN name"
            )
            vector_procedures = [record["name"] for record in result.records]
            self._vector_search_available = any(
                "vector" in proc for proc in vector_procedures
            )

            if self._vector_search_available:
                logger.info("✅ Vector search procedures are available")
            else:
                logger.warning("⚠️  Vector search procedures not available")

            return self._vector_search_available
        except Exception as e:
            logger.warning(f"Could not check vector search availability: {e}")
            self._vector_search_available = False
            return False

    def manual_vector_similarity_search(
        self, query_embedding: List[float], limit: int = 10
    ) -> List[Dict]:
        """
        Perform manual vector similarity search when vector indexes are not available.
        This is slower than native vector search but provides fallback functionality.
        """
        if self._vector_search_available:
            logger.info(
                "Vector search is available - consider using native vector search instead"
            )

        logger.info(f"🔍 Performing manual similarity search for {limit} results")

        # This query computes cosine similarity manually using the stored embeddings
        # Note: This will be slower than native vector search
        manual_search_query = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL
        WITH c, 
             reduce(dot = 0.0, i IN range(0, size($query_embedding)-1) | 
                 dot + ($query_embedding[i] * c.embedding[i])
             ) AS dot_product,
             sqrt(reduce(norm_a = 0.0, i IN range(0, size($query_embedding)-1) | 
                 norm_a + ($query_embedding[i] * $query_embedding[i])
             )) AS norm_a,
             sqrt(reduce(norm_b = 0.0, i IN range(0, size(c.embedding)-1) | 
                 norm_b + (c.embedding[i] * c.embedding[i])
             )) AS norm_b
        WITH c, 
             CASE 
                 WHEN norm_a > 0 AND norm_b > 0 
                 THEN dot_product / (norm_a * norm_b)
                 ELSE 0.0 
             END AS similarity
        WHERE similarity > 0.7
        RETURN c.id as chunk_id, c.text as text, c.summary as summary, 
               c.source as source, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """

        try:
            result = self.neo4j.execute_query(
                manual_search_query,
                {"query_embedding": query_embedding, "limit": limit},
            )
            return [dict(record) for record in result.records]
        except Exception as e:
            logger.error(f"Manual vector search failed: {e}")
            return []

    def setup_database_schema(self):
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
                self.neo4j.execute_query(constraint)
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
                self.neo4j.execute_query(index)
                logger.info(f"Created index: {index}")
            except Exception as e:
                logger.warning(f"Index already exists or error: {e}")

        # Create fulltext index with correct syntax
        fulltext_index_query = """
        CREATE FULLTEXT INDEX legal_text_search IF NOT EXISTS 
        FOR (d:Document) ON EACH [d.full_text]
        """

        try:
            self.neo4j.execute_query(fulltext_index_query)
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
            self.neo4j.execute_query(chunk_fulltext_query)
            logger.info(
                "Created fulltext index: chunk_text_search for Chunk text and summary"
            )
        except Exception as e:
            logger.warning(f"Chunk fulltext index already exists or error: {e}")

        # Create vector index (only available in Neo4j Enterprise or with APOC plugin)
        if self.check_vector_search_availability():
            vector_index_query = f"""
            CALL db.index.vector.createNodeIndex(
                '{settings.vector_index_name}',
                'Chunk',
                'embedding',
                {settings.embedding_dimension},
                'cosine'
            )
            """

            try:
                self.neo4j.execute_query(vector_index_query)
                logger.info(f"Created vector index: {settings.vector_index_name}")
            except Exception as e:
                logger.warning(f"Vector index creation failed: {e}")

    def reset_database(self):
        """Clear all data from the database."""
        logger.warning("Resetting database - all data will be deleted!")

        # Clear all nodes and relationships first
        try:
            self.neo4j.execute_query("MATCH (n) DETACH DELETE n")
            logger.info("Successfully cleared all nodes and relationships")
        except Exception as e:
            logger.error(f"Failed to clear nodes and relationships: {e}")

        # Try to drop vector index if available
        if self.check_vector_search_availability():
            try:
                self.neo4j.execute_query(
                    f"CALL db.index.vector.drop('{settings.vector_index_name}') YIELD name RETURN name"
                )
                logger.info(
                    f"Successfully dropped vector index: {settings.vector_index_name}"
                )
            except Exception as e:
                logger.info(f"Vector index drop failed (might be expected): {e}")
        else:
            logger.info("Skipping vector index drop - procedures not available")

        # Try to drop any existing indexes manually
        try:
            # List and drop any existing indexes
            result = self.neo4j.execute_query("SHOW INDEXES YIELD name, type")
            for record in result.records:
                index_name = record["name"]
                index_type = record["type"]
                if index_name == settings.vector_index_name:
                    try:
                        self.neo4j.execute_query(f"DROP INDEX `{index_name}`")
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

        self.neo4j.execute_write_query(create_doc_query, doc_params)

        # Chunk the document
        chunks = self.processor.chunk_document(text, str(pdf_path.name))
        logger.info(f"📄 Created {len(chunks)} chunks for {pdf_path.name}")

        # Process chunks in batches
        batch_size = 5
        tasks = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            logger.info(
                f"⚙️  Processing chunk batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}"
            )
            tasks.append(self._process_chunk_batch(batch, str(pdf_path.name)))

        await asyncio.gather(*tasks)

        # Create entity nodes and relationships after all chunks are processed
        await self._create_document_entities(str(pdf_path.name), text)

        perf_tracker.record_metric("documents_processed", 1)
        logger.info(f"✅ Successfully processed {pdf_path.name}")
        return True

    async def _process_chunk_batch(self, chunks: List[Document], source: str):
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
            tasks.append(self._create_chunk_node(chunk, embedding, source))

        await asyncio.gather(*tasks)

    async def _create_chunk_node(
        self, chunk: Document, embedding: List[float], source: str
    ):
        """Create a chunk node and its relationships."""
        # Generate summary and extract entities
        summary = await self.processor.generate_summary(chunk.page_content)
        entities = await self.processor.extract_entities(chunk.page_content)

        # Create unique chunk ID
        chunk_id = f"{source}_{chunk.metadata['chunk_index']}"

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
            created_at: datetime()
        })
        CREATE (c)-[:PART_OF]->(d)
        RETURN c
        """

        self.neo4j.execute_write_query(create_chunk_query, chunk_params)

        # Create reference relationships based on entities
        await self._create_reference_relationships(chunk_id, entities)

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
                self.neo4j.execute_write_query(
                    find_ref_query,
                    {
                        "chunk_id": chunk_id,
                        "case_name": case[:50],  # Use first part of case name
                    },
                )
            except Exception as e:
                logger.warning(f"Reference relationship creation failed: {e}")

    def _extract_document_metadata(self, pdf_path: Path, text: str) -> Dict[str, Any]:
        """Extract enhanced metadata from document using LLM analysis."""
        try:
            # Basic metadata from filename
            filename = pdf_path.name
            metadata = {"type": self._determine_document_type(pdf_path, text)}

            # Extract year from filename (e.g., "[2019] SGCA 42.pdf")
            import re

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

            # Use LLM for more detailed extraction
            start_time = time.time()
            prompt = f"""
            Extract metadata from this legal document and return as JSON:
            - jurisdiction: Singapore, UK, US, etc.
            - court_level: Supreme Court, High Court, District Court, etc.
            - parties: List of party names (plaintiff, defendant, appellant, respondent)
            - legal_areas: List of areas of law (contract, tort, criminal, etc.)
            
            Document text (first 2000 chars): {text[:2000]}
            Filename: {filename}
            
            Return only valid JSON.
            """

            response = self.processor.llm.invoke(prompt)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            metadata_time = time.time() - start_time
            perf_tracker.record_metric("llm_time", metadata_time)
            perf_tracker.record_metric("llm_calls", 1)

            try:
                import json

                llm_metadata = json.loads(
                    re.search(r"\{.*\}", response_text, re.DOTALL).group(0)
                )
                metadata.update(llm_metadata)
            except:
                # Fallback to basic extraction
                if "SGCA" in filename or "Singapore" in text[:1000]:
                    metadata["jurisdiction"] = "Singapore"
                    metadata["court_level"] = (
                        "Court of Appeal" if "SGCA" in filename else "High Court"
                    )

            return metadata
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            return {"type": "Document"}

    async def _create_document_entities(self, source: str, full_text: str):
        """Create entity nodes and relationships for the entire document."""
        try:
            # Extract entities from full document text (first 3000 chars for efficiency)
            entities = await self.processor.extract_entities(full_text[:3000])

            # Create Case entities and relationships
            for case_name in entities.get("cases", [])[
                :10
            ]:  # Limit to prevent explosion
                if case_name and len(case_name.strip()) > 5:
                    case_citation = self._extract_citation_from_name(case_name)
                    year = self._extract_year_from_citation(case_citation)

                    create_case_query = """
                    MERGE (case:Case {citation: $citation})
                    SET case.name = $name,
                        case.year = $year,
                        case.created_at = datetime()
                    WITH case
                    MATCH (d:Document {source: $source})
                    MERGE (d)-[:CITES]->(case)
                    """

                    self.neo4j.execute_write_query(
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

                    self.neo4j.execute_write_query(
                        create_statute_query,
                        {"reference": statute_ref.strip(), "source": source},
                    )

            # Create Court entities
            for court_name in entities.get("courts", [])[:5]:
                if court_name and len(court_name.strip()) > 3:
                    jurisdiction = self._infer_jurisdiction(court_name)
                    level = self._infer_court_level(court_name)

                    create_court_query = """
                    MERGE (court:Court {name: $name})
                    SET court.jurisdiction = $jurisdiction,
                        court.level = $level,
                        court.created_at = datetime()
                    WITH court
                    MATCH (d:Document {source: $source})
                    MERGE (d)-[:HEARD_IN]->(court)
                    """

                    self.neo4j.execute_write_query(
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

                    self.neo4j.execute_write_query(
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

                    self.neo4j.execute_write_query(
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
            result = self.neo4j.execute_query(
                current_doc_query, {"source": current_source}
            )

            if not result or not result[0].get("year"):
                return

            current_year = result[0]["year"]

            # For each cited case, create appropriate relationships
            for cited_case in cited_cases[:5]:  # Limit to prevent explosion
                citation = self._extract_citation_from_name(cited_case)
                cited_year = self._extract_year_from_citation(citation)

                if cited_year and cited_year < current_year:
                    # Current case follows the older case
                    follow_query = """
                    MATCH (current:Document {source: $current_source})
                    MATCH (cited:Case {citation: $cited_citation})
                    MERGE (current)-[:FOLLOWS {weight: 1.0}]->(cited)
                    """

                    self.neo4j.execute_write_query(
                        follow_query,
                        {"current_source": current_source, "cited_citation": citation},
                    )

        except Exception as e:
            logger.warning(f"Error creating temporal relationships: {e}")

    def _extract_citation_from_name(self, case_name: str) -> str:
        """Extract formal citation from case name."""
        import re

        # Look for patterns like [2019] SGCA 42
        citation_match = re.search(r"\[(\d{4})\]\s*([A-Z]+)\s*(\d+)", case_name)
        if citation_match:
            return f"[{citation_match.group(1)}] {citation_match.group(2)} {citation_match.group(3)}"
        return case_name.strip()[:100]  # Fallback to truncated name

    def _extract_year_from_citation(self, citation: str) -> Optional[int]:
        """Extract year from citation."""
        import re

        year_match = re.search(r"\[?(\d{4})\]?", citation)
        if year_match:
            return int(year_match.group(1))
        return None

    def _infer_jurisdiction(self, court_name: str) -> str:
        """Infer jurisdiction from court name."""
        court_lower = court_name.lower()
        if any(keyword in court_lower for keyword in ["singapore", "sgca", "sghc"]):
            return "Singapore"
        elif any(
            keyword in court_lower
            for keyword in ["england", "uk", "house of lords", "uksc"]
        ):
            return "United Kingdom"
        elif any(keyword in court_lower for keyword in ["australia", "hca"]):
            return "Australia"
        elif any(keyword in court_lower for keyword in ["canada", "scc"]):
            return "Canada"
        return "Unknown"

    def _infer_court_level(self, court_name: str) -> str:
        """Infer court level from court name."""
        court_lower = court_name.lower()
        if any(
            keyword in court_lower
            for keyword in ["supreme", "court of appeal", "sgca", "hca"]
        ):
            return "Appellate"
        elif any(keyword in court_lower for keyword in ["high court", "sghc"]):
            return "Superior"
        elif any(keyword in court_lower for keyword in ["district", "magistrate"]):
            return "Lower"
        return "Unknown"

    def _determine_document_type(self, pdf_path: Path, text: str) -> str:
        """Determine document type based on filename and content."""
        filename = pdf_path.name.lower()
        text_lower = text.lower()

        if any(keyword in filename for keyword in ["case", "judgment", "opinion"]):
            return "Case"
        elif any(
            keyword in filename for keyword in ["doctrine", "treatise", "commentary"]
        ):
            return "Doctrine"
        elif any(
            keyword in text_lower
            for keyword in ["plaintiff", "defendant", "court held"]
        ):
            return "Case"
        elif any(
            keyword in text_lower
            for keyword in ["legal principle", "doctrine", "commentary"]
        ):
            return "Doctrine"
        else:
            return "Document"


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
        default="data/raw_docs",
        help="Directory containing PDF documents to ingest",
    )

    args = parser.parse_args()

    logger.info("🚀 Starting legal document ingestion process...")

    # Validate environment
    if not settings.google_api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        sys.exit(1)

    # Set up Neo4j connection
    neo4j_conn = Neo4jConnection()

    try:
        # Test connection
        neo4j_conn.execute_query("RETURN 1")
        logger.info("✅ Successfully connected to Neo4j")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Neo4j: {e}")
        sys.exit(1)

    # Initialize graph builder
    graph_builder = GraphBuilder(neo4j_conn)

    # Reset database if requested
    if args.reset:
        graph_builder.reset_database()

    # Setup schema
    graph_builder.setup_database_schema()

    # Find PDF files to process
    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        sys.exit(1)

    pdf_files = list(docs_dir.glob("*.pdf"))
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
    perf_tracker.print_summary()

    # Close connection
    neo4j_conn.close()
    logger.info("🔌 Neo4j connection closed")


if __name__ == "__main__":
    asyncio.run(main())

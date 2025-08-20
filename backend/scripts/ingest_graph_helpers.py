#!/usr/bin/env python3
"""
Shared helpers for knowledge graph ingestion scripts.

This module contains common classes and utilities used by both ingest_graph.py
and ingest_graph_v2.py to avoid code duplication.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import hashlib
import pickle
import json
import asyncio

# Add the backend directory to the Python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from neo4j import GraphDatabase, AsyncGraphDatabase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from google import genai
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm import create_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMCache:
    """
    Caches LLM results including embeddings, summaries, and entity extractions
    to avoid regenerating them during development.
    Uses SHA-256 hash of text content as cache keys for reliability.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the LLM cache.

        Args:
            cache_dir: Directory to store cache files. Defaults to cache/ in project root.
        """
        if cache_dir is None:
            # Use cache directory in project root
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / "cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Separate files for different types of cached data
        self.embeddings_file = self.cache_dir / "embeddings_cache.pkl"
        self.summaries_file = self.cache_dir / "summaries_cache.pkl"
        self.entities_file = self.cache_dir / "entities_cache.pkl"
        self.knowledge_graphs_file = self.cache_dir / "knowledge_graphs_cache.pkl"
        self.metadata_file = self.cache_dir / "cache_metadata.json"

        # Load existing caches
        self.embeddings_cache = self._load_cache_file(
            self.embeddings_file, "embeddings"
        )
        self.summaries_cache = self._load_cache_file(self.summaries_file, "summaries")
        self.entities_cache = self._load_cache_file(self.entities_file, "entities")
        self.knowledge_graphs_cache = self._load_cache_file(
            self.knowledge_graphs_file, "knowledge_graphs"
        )
        self.cache_metadata = self._load_cache_metadata()

        # Track cache stats for different operations
        self.cache_stats = {
            "embeddings": {"hits": 0, "misses": 0},
            "summaries": {"hits": 0, "misses": 0},
            "entities": {"hits": 0, "misses": 0},
            "knowledge_graphs": {"hits": 0, "misses": 0},
        }

        logger.info(f"📦 LLM cache initialized at {self.cache_dir}")
        logger.info(f"📊 Cache contains:")
        logger.info(f"   - {len(self.embeddings_cache)} embeddings")
        logger.info(f"   - {len(self.summaries_cache)} summaries")
        logger.info(f"   - {len(self.entities_cache)} entity extractions")
        logger.info(f"   - {len(self.knowledge_graphs_cache)} knowledge graphs")

    def _generate_cache_key(self, text: str) -> str:
        """Generate a SHA-256 hash key for the given text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_cache_file(self, file_path: Path, cache_type: str) -> Dict[str, Any]:
        """Load a specific cache file from disk."""
        if file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {cache_type} cache: {e}")
                return {}
        return {}

    def _load_cache_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                return {"version": "2.0", "created_at": time.time()}
        return {"version": "2.0", "created_at": time.time()}

    def _save_cache(self):
        """Save all caches to disk."""
        try:
            # Save embeddings
            with open(self.embeddings_file, "wb") as f:
                pickle.dump(self.embeddings_cache, f)

            # Save summaries
            with open(self.summaries_file, "wb") as f:
                pickle.dump(self.summaries_cache, f)

            # Save entities
            with open(self.entities_file, "wb") as f:
                pickle.dump(self.entities_cache, f)

            # Save knowledge graphs
            with open(self.knowledge_graphs_file, "wb") as f:
                pickle.dump(self.knowledge_graphs_cache, f)

            # Update and save metadata
            total_hits = sum(stats["hits"] for stats in self.cache_stats.values())
            total_misses = sum(stats["misses"] for stats in self.cache_stats.values())

            self.cache_metadata.update(
                {
                    "last_updated": time.time(),
                    "total_embeddings": len(self.embeddings_cache),
                    "total_summaries": len(self.summaries_cache),
                    "total_entities": len(self.entities_cache),
                    "total_knowledge_graphs": len(self.knowledge_graphs_cache),
                    "cache_stats": self.cache_stats,
                    "total_hits": total_hits,
                    "total_misses": total_misses,
                }
            )

            with open(self.metadata_file, "w") as f:
                json.dump(self.cache_metadata, f, indent=2)

            logger.debug(
                f"💾 Cache saved with {len(self.embeddings_cache)} embeddings, {len(self.summaries_cache)} summaries, {len(self.entities_cache)} entities, {len(self.knowledge_graphs_cache)} knowledge graphs"
            )
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    # Embedding methods
    def get_embeddings(self, texts: List[str]) -> tuple[List[List[float]], List[str]]:
        """
        Get embeddings for texts, using cache when available.

        Returns:
            tuple: (embeddings_list, uncached_texts)
                - embeddings_list: List of embeddings (None for uncached texts)
                - uncached_texts: List of texts that need to be embedded
        """
        embeddings = []
        uncached_texts = []

        for text in texts:
            cache_key = self._generate_cache_key(text)

            if cache_key in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[cache_key])
                self.cache_stats["embeddings"]["hits"] += 1
            else:
                embeddings.append(None)  # Placeholder for uncached embedding
                uncached_texts.append(text)
                self.cache_stats["embeddings"]["misses"] += 1

        return embeddings, uncached_texts

    def store_embeddings(self, texts: List[str], embeddings: List[List[float]]):
        """Store embeddings in cache."""
        if len(texts) != len(embeddings):
            logger.error(
                f"Mismatch: {len(texts)} texts vs {len(embeddings)} embeddings"
            )
            return

        for text, embedding in zip(texts, embeddings):
            cache_key = self._generate_cache_key(text)
            self.embeddings_cache[cache_key] = embedding

        # Save to disk
        self._save_cache()

    # Summary methods
    def get_summary(self, text: str) -> Optional[str]:
        """Get cached summary for text, or None if not cached."""
        cache_key = self._generate_cache_key(text)

        if cache_key in self.summaries_cache:
            self.cache_stats["summaries"]["hits"] += 1
            return self.summaries_cache[cache_key]
        else:
            self.cache_stats["summaries"]["misses"] += 1
            return None

    def store_summary(self, text: str, summary: str):
        """Store summary in cache."""
        cache_key = self._generate_cache_key(text)
        self.summaries_cache[cache_key] = summary
        self._save_cache()

    # Entity extraction methods
    def get_entities(self, text: str) -> Optional[Dict[str, List[str]]]:
        """Get cached entity extraction for text, or None if not cached."""
        cache_key = self._generate_cache_key(text)

        if cache_key in self.entities_cache:
            self.cache_stats["entities"]["hits"] += 1
            return self.entities_cache[cache_key]
        else:
            self.cache_stats["entities"]["misses"] += 1
            return None

    def store_entities(self, text: str, entities: Dict[str, List[str]]):
        """Store entity extraction results in cache."""
        cache_key = self._generate_cache_key(text)
        self.entities_cache[cache_key] = entities
        self._save_cache()

    # Knowledge graph methods (for V2)
    def get_knowledge_graph(self, text: str) -> Optional[Any]:
        """Get cached knowledge graph for text, or None if not cached."""
        cache_key = self._generate_cache_key(text)

        if cache_key in self.knowledge_graphs_cache:
            self.cache_stats["knowledge_graphs"]["hits"] += 1
            return self.knowledge_graphs_cache[cache_key]
        else:
            self.cache_stats["knowledge_graphs"]["misses"] += 1
            return None

    def store_knowledge_graph(self, text: str, graph_data: Any):
        """Store knowledge graph extraction results in cache."""
        cache_key = self._generate_cache_key(text)
        self.knowledge_graphs_cache[cache_key] = graph_data
        self._save_cache()

    def clear_cache(self):
        """Clear all cached data."""
        self.embeddings_cache.clear()
        self.summaries_cache.clear()
        self.entities_cache.clear()
        self.knowledge_graphs_cache.clear()
        self.cache_stats = {
            "embeddings": {"hits": 0, "misses": 0},
            "summaries": {"hits": 0, "misses": 0},
            "entities": {"hits": 0, "misses": 0},
            "knowledge_graphs": {"hits": 0, "misses": 0},
        }
        self._save_cache()
        logger.info("🗑️  Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_hits = sum(stats["hits"] for stats in self.cache_stats.values())
        total_misses = sum(stats["misses"] for stats in self.cache_stats.values())
        total_requests = total_hits + total_misses
        overall_hit_rate = (
            (total_hits / total_requests * 100) if total_requests > 0 else 0
        )

        # Calculate individual hit rates
        individual_rates = {}
        for operation, stats in self.cache_stats.items():
            operation_total = stats["hits"] + stats["misses"]
            individual_rates[f"{operation}_hit_rate"] = (
                (stats["hits"] / operation_total * 100) if operation_total > 0 else 0
            )

        # Calculate cache file sizes
        cache_sizes = {}
        for cache_type, file_path in [
            ("embeddings", self.embeddings_file),
            ("summaries", self.summaries_file),
            ("entities", self.entities_file),
            ("knowledge_graphs", self.knowledge_graphs_file),
        ]:
            cache_sizes[f"{cache_type}_size_mb"] = (
                round(file_path.stat().st_size / (1024 * 1024), 2)
                if file_path.exists()
                else 0
            )

        return {
            "total_cached_embeddings": len(self.embeddings_cache),
            "total_cached_summaries": len(self.summaries_cache),
            "total_cached_entities": len(self.entities_cache),
            "total_cached_knowledge_graphs": len(self.knowledge_graphs_cache),
            "cache_stats": self.cache_stats,
            "overall_hit_rate_percent": round(overall_hit_rate, 2),
            **individual_rates,
            **cache_sizes,
        }

    def print_cache_stats(self):
        """Print cache statistics."""
        stats = self.get_cache_stats()
        logger.info("📈 LLM CACHE STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total cached embeddings: {stats['total_cached_embeddings']}")
        logger.info(f"Total cached summaries: {stats['total_cached_summaries']}")
        logger.info(f"Total cached entities: {stats['total_cached_entities']}")
        logger.info(
            f"Total cached knowledge graphs: {stats['total_cached_knowledge_graphs']}"
        )
        logger.info("")
        logger.info("Hit rates by operation:")
        logger.info(
            f"  Embeddings: {stats['embeddings_hit_rate']:.1f}% ({stats['cache_stats']['embeddings']['hits']}/{stats['cache_stats']['embeddings']['hits'] + stats['cache_stats']['embeddings']['misses']})"
        )
        logger.info(
            f"  Summaries: {stats['summaries_hit_rate']:.1f}% ({stats['cache_stats']['summaries']['hits']}/{stats['cache_stats']['summaries']['hits'] + stats['cache_stats']['summaries']['misses']})"
        )
        logger.info(
            f"  Entities: {stats['entities_hit_rate']:.1f}% ({stats['cache_stats']['entities']['hits']}/{stats['cache_stats']['entities']['hits'] + stats['cache_stats']['entities']['misses']})"
        )
        logger.info(
            f"  Knowledge Graphs: {stats['knowledge_graphs_hit_rate']:.1f}% ({stats['cache_stats']['knowledge_graphs']['hits']}/{stats['cache_stats']['knowledge_graphs']['hits'] + stats['cache_stats']['knowledge_graphs']['misses']})"
        )
        logger.info(f"  Overall: {stats['overall_hit_rate_percent']:.1f}%")
        logger.info("")
        logger.info("Cache file sizes:")
        logger.info(f"  Embeddings: {stats['embeddings_size_mb']} MB")
        logger.info(f"  Summaries: {stats['summaries_size_mb']} MB")
        logger.info(f"  Entities: {stats['entities_size_mb']} MB")
        logger.info(f"  Knowledge Graphs: {stats['knowledge_graphs_size_mb']} MB")
        logger.info("=" * 50)


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

    def print_summary(self, llm_cache: Optional[LLMCache] = None):
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

        # Print LLM cache statistics if available
        if llm_cache:
            logger.info("")
            llm_cache.print_cache_stats()

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


class Neo4jConnection:
    """Manages Neo4j database connection and operations."""

    def __init__(self, perf_tracker: Optional[PerformanceTracker] = None):
        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password)
        )
        self.database = settings.neo4j_database
        self.perf_tracker = perf_tracker

    async def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            await self.driver.close()

    async def execute_query(
        self, query: str, parameters: Optional[Dict] = None
    ) -> List[Dict]:
        """Execute a Cypher query and return results."""
        start_time = time.time()
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, parameters or {})
            data = [record.data() async for record in result]

        if self.perf_tracker:
            query_time = time.time() - start_time
            self.perf_tracker.record_metric("db_time", query_time)
            self.perf_tracker.record_metric("db_operations", 1)

        return data

    async def execute_write_query(
        self, query: str, parameters: Optional[Dict] = None
    ) -> List[Dict]:
        """Execute a write Cypher query and return results."""

        async def _execute_query(tx):
            result = await tx.run(query, parameters or {})
            return [record.data() async for record in result]

        start_time = time.time()
        async with self.driver.session(database=self.database) as session:
            data = await session.write_transaction(_execute_query)

        if self.perf_tracker:
            query_time = time.time() - start_time
            self.perf_tracker.record_metric("db_time", query_time)
            self.perf_tracker.record_metric("db_operations", 1)

        return data


class BaseDocumentProcessor:
    """Base class for document processing functionality shared between V1 and V2."""

    def __init__(
        self, use_cache: bool = True, perf_tracker: Optional[PerformanceTracker] = None
    ):
        # Configure Google AI with API key
        _client = genai.Client(api_key=settings.google_api_key)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=getattr(settings, "chunk_size", 1000),
            chunk_overlap=getattr(settings, "chunk_overlap", 200),
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        embeddings_attrs = settings.llm_config.get("embeddings", {})
        self.embeddings = GoogleGenerativeAIEmbeddings(**embeddings_attrs)

        # Initialize cache
        self.use_cache = use_cache
        self.cache = LLMCache() if use_cache else None
        self.perf_tracker = perf_tracker

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

        if self.perf_tracker:
            chunking_time = time.time() - start_time
            self.perf_tracker.record_metric("chunking_time", chunking_time)
            self.perf_tracker.record_metric("chunks_created", len(documents))

        logger.info(f"📝 Created {len(documents)} chunks for {source}")
        return documents

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts with caching support."""
        try:
            start_time = time.time()
            logger.info(f"🔢 Generating embeddings for {len(texts)} texts...")

            if self.use_cache and self.cache:
                # Check cache first
                cached_embeddings, uncached_texts = self.cache.get_embeddings(texts)

                if uncached_texts:
                    logger.info(
                        f"📦 Cache hit for {len(texts) - len(uncached_texts)}/{len(texts)} texts"
                    )
                    logger.info(
                        f"🔢 Generating {len(uncached_texts)} new embeddings..."
                    )

                    # Generate embeddings only for uncached texts
                    new_embeddings = await self.embeddings.aembed_documents(
                        uncached_texts
                    )

                    # Store new embeddings in cache
                    self.cache.store_embeddings(uncached_texts, new_embeddings)

                    # Merge cached and new embeddings
                    embeddings = []
                    new_embedding_iter = iter(new_embeddings)

                    for cached_embedding in cached_embeddings:
                        if cached_embedding is None:
                            embeddings.append(next(new_embedding_iter))
                        else:
                            embeddings.append(cached_embedding)
                else:
                    logger.info(f"📦 All embeddings found in cache!")
                    embeddings = [emb for emb in cached_embeddings if emb is not None]
            else:
                # No caching - generate all embeddings
                embeddings = await self.embeddings.aembed_documents(texts)

            if self.perf_tracker:
                embedding_time = time.time() - start_time
                self.perf_tracker.record_metric("embedding_time", embedding_time)
                self.perf_tracker.record_metric("embeddings_generated", len(texts))

            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from a PDF file using the enhanced processor."""
        try:
            # Import the PDF processing module
            from process_pdf import process_pdf_document

            start_time = time.time()

            # Use the PDF processor to extract and integrate content
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

            if self.perf_tracker:
                extraction_time = time.time() - start_time
                self.perf_tracker.record_metric("pdf_extraction_time", extraction_time)

            logger.info(
                f"📝 Extracted {len(pages)} pages with integrated footnotes from {pdf_path.name}"
            )
            return full_text.strip()

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""


# Pydantic schemas for structured output (V1)
class LegalEntities(BaseModel):
    """Schema for extracting legal entities from chunk-level text."""

    cases: List[str] = Field(
        description="List of case names and citations referenced in this chunk (include year if available)"
    )
    statutes: List[str] = Field(
        description="List of statute references with sections mentioned in this chunk"
    )
    concepts: List[str] = Field(
        description="List of key legal concepts and principles discussed in this chunk"
    )
    holdings: List[str] = Field(
        description="List of key legal holdings or ratio decidendi mentioned in this chunk"
    )
    facts: List[str] = Field(
        description="List of key factual elements presented in this chunk"
    )
    legal_tests: List[str] = Field(
        description="List of legal tests or standards mentioned in this chunk"
    )


class Judge(BaseModel):
    """Schema for a judge."""

    name: str = Field(description="Full name of the judge")
    role: str = Field(description="Role such as Chief Justice, Justice, etc.")


class Lawyer(BaseModel):
    """Schema for a lawyer."""

    name: str = Field(description="Full name of the lawyer")
    role: str = Field(
        description="Role such as counsel for plaintiff, counsel for defendant, etc."
    )
    firm: str = Field(description="Law firm name", default="")


class CitationFormat(BaseModel):
    """Schema for citation formats."""

    primary: str = Field(description="Primary citation format")
    alternative: List[str] = Field(description="Alternative citation formats")
    neutral: str = Field(description="Neutral citation if available", default="")


class ChunkRelevance(BaseModel):
    """Schema for chunk relevance assessment."""

    is_relevant: bool = Field(
        description="Whether the chunk contains substantial legal content worth indexing"
    )
    reasoning: str = Field(
        description="Brief explanation of why the chunk is or isn't relevant"
    )


class LegalSummary(BaseModel):
    """Schema for legal text summary with integrated relevance assessment."""

    summary: Optional[str] = Field(
        description="Concise summary focusing on key legal concepts, principles, citations, and holdings. ONLY provide a summary if the text contains substantial legal content worth indexing for legal research. Return None if the text is just headers, footers, page numbers, table of contents, or administrative content with no legal substance."
    )
    key_concepts: List[str] = Field(
        description="List of key legal concepts mentioned (empty if not relevant)"
    )
    case_references: List[str] = Field(
        description="List of case citations or references found (empty if not relevant)"
    )
    reasoning: str = Field(
        description="Brief explanation of why content was or wasn't summarized"
    )


class DocumentMetadata(BaseModel):
    """Schema for extracting document metadata."""

    jurisdiction: str = Field(
        description="Jurisdiction such as Singapore, UK, US, etc."
    )
    court_level: str = Field(
        description="Court level such as Supreme Court, High Court, District Court, etc."
    )
    parties: List[str] = Field(
        description="List of party names (plaintiff, defendant, appellant, respondent)"
    )
    legal_areas: List[str] = Field(
        description="List of areas of law (contract, tort, criminal, etc.)"
    )
    judges: List[Judge] = Field(description="List of judges with their roles")
    lawyers: List[Lawyer] = Field(
        description="List of lawyers with their roles and firms"
    )
    citation_formats: List[str] = Field(
        description="List of all possible citation formats for this case"
    )


# Utility functions
def extract_citation_from_name(case_name: str) -> str:
    """Extract formal citation from case name."""
    import re

    # Look for patterns like [2019] SGCA 42
    citation_match = re.search(r"\[(\d{4})\]\s*([A-Z]+)\s*(\d+)", case_name)
    if citation_match:
        return f"[{citation_match.group(1)}] {citation_match.group(2)} {citation_match.group(3)}"
    return case_name.strip()[:100]  # Fallback to truncated name


def extract_year_from_citation(citation: str) -> Optional[int]:
    """Extract year from citation."""
    import re

    year_match = re.search(r"\[?(\d{4})\]?", citation)
    if year_match:
        return int(year_match.group(1))
    return None


def infer_jurisdiction(court_name: str) -> str:
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


def infer_court_level(court_name: str) -> str:
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


def determine_document_type(pdf_path: Path) -> str:
    """Determine document type based on filename and content."""
    # if the folder of the file is named "cases", then it is a case document
    if "cases" in str(pdf_path.parent).lower():
        return "Case"
    elif "doctrines" in str(pdf_path.parent).lower():
        return "Doctrine"
    return "Document"

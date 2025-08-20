"""
Streamlined ReAct Agent Tools for Legal Research and Argument Drafting

This module provides essential tools for building a ReAct agent focused on
generating legal research and argument drafts using the Forecite knowledge graph.

## Core Tools for Research & Argument Drafting

### 1. Content Discovery
- `semantic_search_chunks()`: Vector similarity search across document chunks
- `search_cases_by_citation()`: Find cases by citation patterns
- `search_cases_by_legal_tags()`: Find cases by legal areas/tags

### 2. Legal Analysis
- `extract_legal_holdings()`: Extract and analyze legal holdings from cases
- `search_statute_references()`: Find cases referencing specific statutes
- `trace_legal_precedent()`: Trace development of legal principles

### 3. Research Synthesis
- `generate_legal_research_memo()`: Generate comprehensive research memos
- `find_cases_by_fact_pattern()`: Find cases with similar factual scenarios
- `validate_legal_proposition()`: Validate legal statements against authorities

### 4. Citation Analysis
- `analyze_citation_network()`: Analyze citation relationships and influence

## Simplified Wrapper Functions

- `semantic_search_legal_content()`: Semantic content search
- `extract_legal_information()`: Extract holdings and legal information
- `generate_research_memo()`: Generate research memos
- `validate_legal_statement()`: Validate legal propositions

## Usage Example

```python
from app.tools.react_tools import semantic_search_legal_content, generate_research_memo

# Search for relevant content
content = semantic_search_legal_content("breach of contract remedies", top_k=5)

# Generate research memo
memo = generate_research_memo(
    "What are the requirements for contract formation?",
    jurisdiction="Singapore"
)
```
"""

import re
import json
import uuid
import time
import logging
import asyncio
import threading
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.config import get_stream_writer

# Load environment variables
load_dotenv()

# Import settings
from app.core.config import settings

# Global embeddings instance
_embeddings_instance = None
_embeddings_lock = threading.Lock()


def get_embeddings_instance():
    """
    Get or create a thread-safe embeddings instance.
    This handles the async event loop issue by ensuring embeddings
    are initialized in a proper context.
    """
    global _embeddings_instance

    if _embeddings_instance is None:
        with _embeddings_lock:
            if _embeddings_instance is None:
                try:
                    # Try to get the current event loop
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # No event loop in current thread, create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Initialize embeddings with the event loop context
                embeddings_attrs = settings.llm_config.get("embeddings", {})
                _embeddings_instance = GoogleGenerativeAIEmbeddings(**embeddings_attrs)

    return _embeddings_instance


class SearchResult(BaseModel):
    document_citation: str
    parties: List[str] = Field(default_factory=list)
    court: str = ""
    court_level: str = ""
    jurisdiction: str = ""
    year: Optional[int] = None
    legal_tags: List[str] = Field(default_factory=list)
    summary: str = ""


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    summary: str = ""
    similarity_score: float
    chunk_index: int
    concepts: List[str] = Field(default_factory=list)
    holdings: List[str] = Field(default_factory=list)
    facts: List[str] = Field(default_factory=list)
    source_document: SearchResult


class CitationNetworkResult(BaseModel):
    case_citation: str
    cited_by: List[str] = Field(default_factory=list)
    cites: List[str] = Field(default_factory=list)
    influence_score: float = 0.0
    temporal_position: str = "intermediate"


class LegalHolding(BaseModel):
    holding_text: str
    source_case: str
    chunk_id: str
    legal_concepts: List[str] = Field(default_factory=list)
    context: str = ""


class FactPattern(BaseModel):
    case_citation: str
    key_facts: List[str] = Field(default_factory=list)
    legal_context: List[str] = Field(default_factory=list)
    similarity_score: float


class LegalResearchMemo(BaseModel):
    research_question: str
    executive_summary: str
    legal_analysis: str
    key_authorities: List[str] = Field(default_factory=list)
    recommendations: str
    gaps_identified: List[str] = Field(default_factory=list)


class StreamlinedLegalResearchTools:
    """
    Streamlined ReAct agent tools for legal research and argument drafting using Neo4j knowledge graph.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password)
        )
        # Use thread-safe embeddings initialization
        self.embeddings = None  # Lazy initialization

    @property
    def embeddings_client(self):
        """Lazy initialization of embeddings client."""
        if self.embeddings is None:
            self.embeddings = get_embeddings_instance()
        return self.embeddings

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()

    # =================== Core Content Discovery Tools ===================

    def search_cases_by_citation(
        self, citation: str, exact_match: bool = False
    ) -> List[SearchResult]:
        """
        Find cases by specific citation patterns.

        Args:
            citation: Citation pattern to search for (e.g., "[2013] SGHC 100", "2013 SGHC 100")
            exact_match: Whether to require exact citation match

        Returns:
            List of case documents with metadata
        """
        with self.driver.session() as session:
            if exact_match:
                query = """
                MATCH (d:Document)
                WHERE d.citation = $citation 
                   OR $citation IN d.citation_formats
                RETURN d.citation as citation, d.parties as parties, d.court as court,
                       d.court_level as court_level, d.jurisdiction as jurisdiction,
                       d.year as year, d.legal_tags as legal_tags
                """
                params = {"citation": citation}
            else:
                # Fuzzy matching for non-exact searches
                query = """
                MATCH (d:Document)
                WHERE d.citation CONTAINS $citation 
                   OR ANY(format IN d.citation_formats WHERE format CONTAINS $citation)
                RETURN d.citation as citation, d.parties as parties, d.court as court,
                       d.court_level as court_level, d.jurisdiction as jurisdiction,
                       d.year as year, d.legal_tags as legal_tags
                ORDER BY d.year DESC
                """
                params = {"citation": citation}

            results = session.run(query, params)
            return [
                SearchResult(
                    document_citation=record["citation"],
                    parties=record["parties"] or [],
                    court=record["court"] or "",
                    court_level=record["court_level"] or "",
                    jurisdiction=record["jurisdiction"] or "",
                    year=record["year"],
                    legal_tags=record["legal_tags"] or [],
                )
                for record in results
            ]

    def search_cases_by_legal_tags(
        self, legal_tags: List[str], combine_mode: str = "ANY"
    ) -> List[SearchResult]:
        """
        Find cases by legal tags (areas of law).

        Args:
            legal_tags: Legal tag patterns to search
            combine_mode: "ANY" or "ALL" for multiple tags

        Returns:
            List of case documents categorized by legal tags
        """
        with self.driver.session() as session:
            if combine_mode.upper() == "ALL":
                # All tags must be present
                conditions = []
                for i, tag in enumerate(legal_tags):
                    conditions.append(
                        f"ANY(existing_tag IN d.legal_tags WHERE existing_tag CONTAINS $tag_{i})"
                    )
                where_clause = " AND ".join(conditions)
            else:
                # Any tag can be present (default)
                conditions = []
                for i, tag in enumerate(legal_tags):
                    conditions.append(
                        f"ANY(existing_tag IN d.legal_tags WHERE existing_tag CONTAINS $tag_{i})"
                    )
                where_clause = " OR ".join(conditions)

            query = f"""
            MATCH (d:Document)
            WHERE {where_clause}
            RETURN d.citation as citation, d.parties as parties, d.court as court,
                   d.court_level as court_level, d.jurisdiction as jurisdiction,
                   d.year as year, d.legal_tags as legal_tags
            ORDER BY d.year DESC
            """
            params = {f"tag_{i}": tag for i, tag in enumerate(legal_tags)}

            results = session.run(query, params)
            return [
                SearchResult(
                    document_citation=record["citation"],
                    parties=record["parties"] or [],
                    court=record["court"] or "",
                    court_level=record["court_level"] or "",
                    jurisdiction=record["jurisdiction"] or "",
                    year=record["year"],
                    legal_tags=record["legal_tags"] or [],
                )
                for record in results
            ]

    # =================== Semantic Search & Analysis Tools ===================

    def semantic_search_chunks(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        legal_tags: Optional[List[str]] = None,
        date_range: Optional[Dict[str, str]] = None,
        court_levels: Optional[List[str]] = None,
    ) -> List[ChunkResult]:
        """
        Perform vector similarity search across document chunks.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            legal_tags: Filter by legal tags from parent documents
            date_range: Filter by document date range
            court_levels: Filter by court level from parent documents

        Returns:
            List of relevant chunks with similarity scores and source documents
        """
        # Generate query embedding
        query_embedding = self.embeddings_client.embed_query(query)

        with self.driver.session() as session:
            # Build base query with vector search
            query_parts = [
                """
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding) 
                YIELD node as c, score
                MATCH (c:Chunk)-[:PART_OF]->(d:Document)
                WHERE score >= $similarity_threshold
                """
            ]
            params = {
                "index_name": settings.vector_index_name,
                "top_k": top_k,
                "query_embedding": query_embedding,
                "similarity_threshold": similarity_threshold,
            }

            # Add filters
            if legal_tags:
                conditions = []
                for i, tag in enumerate(legal_tags):
                    conditions.append(
                        f"ANY(existing_tag IN d.legal_tags WHERE existing_tag CONTAINS $tag_{i})"
                    )
                    params[f"tag_{i}"] = tag
                where_clause = " OR ".join(conditions)
                query_parts.append(f"AND ({where_clause})")

            if date_range:
                if "start" in date_range:
                    query_parts.append("AND d.year >= $start_year")
                    params["start_year"] = int(date_range["start"][:4])
                if "end" in date_range:
                    query_parts.append("AND d.year <= $end_year")
                    params["end_year"] = int(date_range["end"][:4])

            if court_levels:
                query_parts.append("AND d.court_level IN $court_levels")
                params["court_levels"] = court_levels

            query_parts.append("""
            RETURN c.id as chunk_id, c.text as text, c.summary as summary, 
                   score as similarity_score, c.chunk_index as chunk_index,
                   c.concepts as concepts, c.holdings as holdings, c.facts as facts,
                   d.citation as citation, d.parties as parties, d.court as court,
                   d.court_level as court_level, d.jurisdiction as jurisdiction,
                   d.year as year, d.legal_tags as legal_tags
            ORDER BY score DESC
            """)

            query = " ".join(query_parts)
            results = session.run(query, params)

            return [
                ChunkResult(
                    chunk_id=record["chunk_id"],
                    text=record["text"],
                    summary=record["summary"] or "",
                    similarity_score=record["similarity_score"],
                    chunk_index=record["chunk_index"],
                    concepts=record["concepts"] or [],
                    holdings=record["holdings"] or [],
                    facts=record["facts"] or [],
                    source_document=SearchResult(
                        document_citation=record["citation"],
                        parties=record["parties"] or [],
                        court=record["court"] or "",
                        court_level=record["court_level"] or "",
                        jurisdiction=record["jurisdiction"] or "",
                        year=record["year"],
                        legal_tags=record["legal_tags"] or [],
                    ),
                )
                for record in results
            ]

    # =================== Citation Analysis Tools ===================

    def analyze_citation_network(
        self,
        case_citation: str,
        direction: str = "both",
        max_depth: int = 2,
        min_citation_count: int = 1,
    ) -> CitationNetworkResult:
        """
        Analyze how cases cite each other and identify citation patterns.

        Args:
            case_citation: Starting case citation
            direction: "cited_by", "cites", "both"
            max_depth: Maximum citation depth to explore
            min_citation_count: Minimum citations for inclusion

        Returns:
            Citation network with influence scores and temporal analysis
        """
        with self.driver.session() as session:
            result = CitationNetworkResult(case_citation=case_citation)

            if direction in ["cited_by", "both"]:
                # Find cases that cite this case
                query = """
                MATCH (citing:Document)-[:CITES]->(cited:Document)
                WHERE cited.citation = $case_citation
                RETURN citing.citation as citing_case, citing.year as citing_year
                ORDER BY citing.year
                """
                cited_by_results = session.run(query, {"case_citation": case_citation})
                result.cited_by = [record["citing_case"] for record in cited_by_results]

            if direction in ["cites", "both"]:
                # Find cases that this case cites
                query = """
                MATCH (citing:Document)-[:CITES]->(cited:Document)
                WHERE citing.citation = $case_citation
                RETURN cited.citation as cited_case, cited.year as cited_year
                ORDER BY cited.year
                """
                cites_results = session.run(query, {"case_citation": case_citation})
                result.cites = [record["cited_case"] for record in cites_results]

            # Calculate influence score based on citation count
            total_citations = len(result.cited_by) + len(result.cites)
            result.influence_score = min(
                total_citations / 10.0, 1.0
            )  # Normalize to 0-1

            # Determine temporal position
            if len(result.cited_by) > len(result.cites):
                result.temporal_position = "foundational"
            elif len(result.cites) > len(result.cited_by):
                result.temporal_position = "recent"
            else:
                result.temporal_position = "intermediate"

            return result

    def trace_legal_precedent(
        self,
        legal_principle: str,
        start_date: Optional[str] = None,
        jurisdiction: Optional[str] = None,
    ) -> List[Tuple[SearchResult, str]]:
        """
        Trace the development of a legal principle through case law.

        Args:
            legal_principle: Description of the legal principle
            start_date: Starting date for temporal analysis
            jurisdiction: Filter by jurisdiction

        Returns:
            Chronological development of the principle with key cases
        """
        # Use semantic search to find relevant chunks
        chunks = self.semantic_search_chunks(
            legal_principle, top_k=50, similarity_threshold=0.6
        )

        # Filter by jurisdiction if specified
        if jurisdiction:
            chunks = [
                c
                for c in chunks
                if jurisdiction.lower() in c.source_document.jurisdiction.lower()
            ]

        # Filter by date if specified
        if start_date:
            start_year = int(start_date[:4])
            chunks = [
                c
                for c in chunks
                if c.source_document.year and c.source_document.year >= start_year
            ]

        # Group by document and sort chronologically
        doc_chunks = {}
        for chunk in chunks:
            citation = chunk.source_document.document_citation
            if citation not in doc_chunks:
                doc_chunks[citation] = {
                    "document": chunk.source_document,
                    "relevant_holdings": [],
                }
            doc_chunks[citation]["relevant_holdings"].extend(chunk.holdings)

        # Sort by year and return with principle development
        results = []
        for citation, data in doc_chunks.items():
            doc = data["document"]
            principle_text = "; ".join(data["relevant_holdings"][:3])  # Top 3 holdings
            results.append((doc, principle_text))

        # Sort chronologically
        results.sort(key=lambda x: x[0].year or 0)

        return results

    # =================== Legal Holdings & Statutory Analysis Tools ===================

    def search_statute_references(
        self, statute_reference: str, section: Optional[str] = None
    ) -> List[Tuple[SearchResult, str]]:
        """
        Find cases that reference specific statutes or regulations.

        Args:
            statute_reference: Statute reference to search for
            section: Specific section or subsection if included in reference

        Returns:
            Cases referencing the statute with context and interpretation
        """
        with self.driver.session() as session:
            # Search both statute nodes and chunk statute arrays
            query_parts = [
                """
                MATCH (c:Chunk)-[:PART_OF]->(d:Document)
                WHERE ANY(statute IN c.statutes WHERE statute CONTAINS $statute_reference)
                """
            ]
            params = {"statute_reference": statute_reference}

            if section:
                query_parts[0] += (
                    " AND ANY(statute IN c.statutes WHERE statute CONTAINS $section)"
                )
                params["section"] = section

            query_parts.append("""
            RETURN c.text as context, c.summary as summary,
                   d.citation as citation, d.parties as parties, d.court as court,
                   d.court_level as court_level, d.jurisdiction as jurisdiction,
                   d.year as year, d.legal_tags as legal_tags
            ORDER BY d.year DESC
            """)

            query = " ".join(query_parts)
            results = session.run(query, params)

            statute_cases = []
            for record in results:
                search_result = SearchResult(
                    document_citation=record["citation"],
                    parties=record["parties"] or [],
                    court=record["court"] or "",
                    court_level=record["court_level"] or "",
                    jurisdiction=record["jurisdiction"] or "",
                    year=record["year"],
                    legal_tags=record["legal_tags"] or [],
                    summary=record["summary"] or "",
                )
                context = (
                    record["context"][:500] + "..."
                    if len(record["context"]) > 500
                    else record["context"]
                )
                statute_cases.append((search_result, context))

            return statute_cases

    def extract_legal_holdings(
        self,
        case_citations: Optional[List[str]] = None,
        legal_issue: Optional[str] = None,
    ) -> List[LegalHolding]:
        """
        Extract and analyze legal holdings from cases.

        Args:
            case_citations: Specific cases to analyze
            legal_issue: Filter holdings related to specific issue

        Returns:
            Structured legal holdings with source information
        """
        with self.driver.session() as session:
            query_parts = [
                "MATCH (c:Chunk)-[:PART_OF]->(d:Document)",
                "WHERE size(c.holdings) > 0",
            ]
            params = {}

            if case_citations:
                query_parts.append("AND d.citation IN $case_citations")
                params["case_citations"] = case_citations

            query_parts.append("""
            RETURN c.id as chunk_id, c.holdings as holdings, c.concepts as concepts,
                   c.text as context, d.citation as citation
            """)

            query = " ".join(query_parts)
            results = session.run(query, params)

            holdings = []
            for record in results:
                for holding_text in record["holdings"]:
                    # Filter by legal issue if specified
                    if legal_issue and legal_issue.lower() not in holding_text.lower():
                        continue

                    holding = LegalHolding(
                        holding_text=holding_text,
                        source_case=record["citation"],
                        chunk_id=record["chunk_id"],
                        legal_concepts=record["concepts"] or [],
                        context=record["context"][:300] + "..."
                        if len(record["context"]) > 300
                        else record["context"],
                    )
                    holdings.append(holding)

            return holdings

    # =================== Fact Pattern & Case Similarity Tools ===================

    def find_cases_by_fact_pattern(
        self,
        fact_description: str,
        key_facts: Optional[List[str]] = None,
        legal_context: Optional[str] = None,
        similarity_threshold: float = 0.6,
    ) -> List[FactPattern]:
        """
        Find cases with similar factual scenarios.

        Args:
            fact_description: Description of the factual scenario
            key_facts: Specific key facts to match
            legal_context: Legal context to filter by
            similarity_threshold: Minimum similarity for inclusion

        Returns:
            Cases with similar facts and factual similarity analysis
        """
        # Use semantic search on fact description
        chunks = self.semantic_search_chunks(
            fact_description, top_k=50, similarity_threshold=similarity_threshold
        )

        # Filter by legal context if provided
        if legal_context:
            chunks = [
                c
                for c in chunks
                if any(
                    legal_context.lower() in tag.lower()
                    for tag in c.source_document.legal_tags
                )
            ]

        # Group by document and analyze fact patterns
        fact_patterns = []
        doc_groups = {}

        for chunk in chunks:
            citation = chunk.source_document.document_citation
            if citation not in doc_groups:
                doc_groups[citation] = {
                    "document": chunk.source_document,
                    "facts": set(),
                    "similarity_scores": [],
                }

            doc_groups[citation]["facts"].update(chunk.facts)
            doc_groups[citation]["similarity_scores"].append(chunk.similarity_score)

        # Create fact pattern results
        for citation, data in doc_groups.items():
            avg_similarity = sum(data["similarity_scores"]) / len(
                data["similarity_scores"]
            )

            fact_pattern = FactPattern(
                case_citation=citation,
                key_facts=list(data["facts"]),
                legal_context=data["document"].legal_tags,
                similarity_score=avg_similarity,
            )
            fact_patterns.append(fact_pattern)

        # Sort by similarity score
        fact_patterns.sort(key=lambda x: x.similarity_score, reverse=True)

        return fact_patterns

    # =================== Research Synthesis & Analysis Tools ===================

    def generate_legal_research_memo(
        self,
        research_question: str,
        jurisdiction: Optional[str] = None,
        client_facts: Optional[str] = None,
        memo_sections: Optional[List[str]] = None,
    ) -> LegalResearchMemo:
        """
        Generate a comprehensive research memo on a legal topic.

        Args:
            research_question: The legal research question
            jurisdiction: Relevant jurisdiction
            client_facts: Specific client facts to consider
            memo_sections: Sections to include

        Returns:
            Structured legal research memo with citations
        """
        if memo_sections is None:
            memo_sections = ["summary", "analysis", "authorities", "recommendations"]

        # Gather relevant authorities
        chunks = self.semantic_search_chunks(
            research_question, top_k=30, similarity_threshold=0.6
        )

        if jurisdiction:
            chunks = [
                c
                for c in chunks
                if jurisdiction.lower() in c.source_document.jurisdiction.lower()
            ]

        # Extract key authorities
        key_authorities = list(
            set([chunk.source_document.document_citation for chunk in chunks])
        )

        # Group holdings and concepts
        all_holdings = []
        all_concepts = set()
        for chunk in chunks:
            all_holdings.extend(chunk.holdings)
            all_concepts.update(chunk.concepts)

        # Generate memo sections
        executive_summary = f"Research on: {research_question}\n"
        executive_summary += f"Key authorities found: {len(key_authorities)} cases\n"
        executive_summary += f"Primary jurisdiction: {jurisdiction or 'Multiple'}\n"

        legal_analysis = "Legal Analysis:\n\n"
        legal_analysis += f"The research question involves the following key concepts: {', '.join(list(all_concepts)[:10])}\n\n"

        if all_holdings:
            legal_analysis += "Key legal holdings identified:\n"
            for i, holding in enumerate(all_holdings[:5], 1):
                legal_analysis += f"{i}. {holding}\n"

        recommendations = "Recommendations:\n"
        if client_facts:
            recommendations += (
                f"Based on the client facts provided: {client_facts[:200]}...\n"
            )
        recommendations += (
            "Further research may be needed in areas with limited authorities.\n"
        )

        return LegalResearchMemo(
            research_question=research_question,
            executive_summary=executive_summary,
            legal_analysis=legal_analysis,
            key_authorities=key_authorities,
            recommendations=recommendations,
            gaps_identified=[],  # Simplified - removed gap analysis
        )

    def validate_legal_proposition(
        self,
        legal_proposition: str,
        jurisdiction: Optional[str] = None,
        confidence_level: str = "medium",
    ) -> Dict[str, Any]:
        """
        Validate a legal proposition against available authorities.

        Args:
            legal_proposition: The legal statement to validate
            jurisdiction: Relevant jurisdiction
            confidence_level: Required confidence level

        Returns:
            Validation result with supporting and opposing authorities
        """
        # Search for relevant chunks
        chunks = self.semantic_search_chunks(
            legal_proposition, top_k=20, similarity_threshold=0.7
        )

        if jurisdiction:
            chunks = [
                c
                for c in chunks
                if jurisdiction.lower() in c.source_document.jurisdiction.lower()
            ]

        supporting_authorities = []
        opposing_authorities = []
        neutral_authorities = []

        # Analyze each chunk for support/opposition
        for chunk in chunks:
            chunk_text = chunk.text.lower()

            # Simple keyword analysis (could be enhanced with NLP)
            supporting_keywords = [
                "held",
                "established",
                "confirmed",
                "affirmed",
                "correct",
            ]
            opposing_keywords = [
                "rejected",
                "overruled",
                "distinguished",
                "incorrect",
                "wrong",
            ]

            support_score = sum(
                1 for keyword in supporting_keywords if keyword in chunk_text
            )
            oppose_score = sum(
                1 for keyword in opposing_keywords if keyword in chunk_text
            )

            if support_score > oppose_score:
                supporting_authorities.append(
                    {
                        "case": chunk.source_document.document_citation,
                        "court_level": chunk.source_document.court_level,
                        "excerpt": chunk.text[:200] + "...",
                        "strength": "strong"
                        if chunk.similarity_score > 0.8
                        else "moderate",
                    }
                )
            elif oppose_score > support_score:
                opposing_authorities.append(
                    {
                        "case": chunk.source_document.document_citation,
                        "court_level": chunk.source_document.court_level,
                        "excerpt": chunk.text[:200] + "...",
                        "strength": "strong"
                        if chunk.similarity_score > 0.8
                        else "moderate",
                    }
                )
            else:
                neutral_authorities.append(
                    {
                        "case": chunk.source_document.document_citation,
                        "court_level": chunk.source_document.court_level,
                        "excerpt": chunk.text[:200] + "...",
                    }
                )

        # Calculate confidence score
        total_authorities = len(supporting_authorities) + len(opposing_authorities)
        confidence_score = (
            len(supporting_authorities) / total_authorities
            if total_authorities > 0
            else 0
        )

        # Determine validation result
        if confidence_score >= 0.8:
            validation_result = "strongly_supported"
        elif confidence_score >= 0.6:
            validation_result = "supported"
        elif confidence_score >= 0.4:
            validation_result = "uncertain"
        elif confidence_score >= 0.2:
            validation_result = "likely_incorrect"
        else:
            validation_result = "strongly_opposed"

        return {
            "proposition": legal_proposition,
            "validation_result": validation_result,
            "confidence_score": confidence_score,
            "supporting_authorities": supporting_authorities,
            "opposing_authorities": opposing_authorities,
            "neutral_authorities": neutral_authorities,
            "total_authorities_found": len(chunks),
            "jurisdiction_coverage": jurisdiction or "multiple",
        }


# Convenience functions for easy tool access
def create_legal_research_tools() -> StreamlinedLegalResearchTools:
    """Create and return an instance of the streamlined legal research tools."""
    return StreamlinedLegalResearchTools()


# Streamlined ReAct Tool Wrapper Functions
# These functions provide a simplified interface for ReAct agents focused on research and argument drafting


def semantic_search_legal_content(
    query: str,
    top_k: int = 10,
    similarity_threshold: float = 0.7,
    filters: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic search across legal content for ReAct agents.

    Args:
        query: Natural language search query
        top_k: Number of results to return
        similarity_threshold: Minimum similarity score
        filters: Additional filters (legal_tags, date_range, court_levels)

    Returns:
        List of relevant content chunks with metadata
    """
    step_id = f"semantic_search_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer(
        {
            "step_id": step_id,
            "status": "in_progress",
            "brief_description": "Searching legal content",
            "description": f"Performing semantic search for: {query[:100]}{'...' if len(query) > 100 else ''}",
        }
    )

    start_time = time.time()
    tools = create_legal_research_tools()
    results = []

    try:
        filters = filters or {}
        chunk_results = tools.semantic_search_chunks(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            legal_tags=filters.get("legal_tags"),
            date_range=filters.get("date_range"),
            court_levels=filters.get("court_levels"),
        )

        for chunk in chunk_results:
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "summary": chunk.summary,
                    "similarity_score": chunk.similarity_score,
                    "chunk_index": chunk.chunk_index,
                    "concepts": chunk.concepts,
                    "holdings": chunk.holdings,
                    "facts": chunk.facts,
                    "source_case": {
                        "citation": chunk.source_document.document_citation,
                        "parties": chunk.source_document.parties,
                        "court": chunk.source_document.court,
                        "jurisdiction": chunk.source_document.jurisdiction,
                        "year": chunk.source_document.year,
                        "legal_tags": chunk.source_document.legal_tags,
                    },
                }
            )

        execution_time = time.time() - start_time
        writer(
            {
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Search completed",
                "description": f"Found {len(results)} relevant documents in {execution_time:.2f}s",
            }
        )

    finally:
        tools.close()

    return results


def extract_legal_information(
    extraction_type: str, case_citations: List[str] = None, legal_issue: str = None
) -> List[Dict[str, Any]]:
    """
    Extract specific legal information (holdings, etc.) for ReAct agents.

    Args:
        extraction_type: Type of extraction ("holdings")
        case_citations: Specific cases to analyze
        legal_issue: Filter by legal issue

    Returns:
        List of extracted legal information
    """
    step_id = f"extract_legal_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer(
        {
            "step_id": step_id,
            "status": "in_progress",
            "brief_description": f"Extracting {extraction_type}",
            "description": f"Extracting {extraction_type} from legal authorities",
        }
    )

    start_time = time.time()
    tools = create_legal_research_tools()
    results = []

    try:
        if extraction_type == "holdings":
            holdings = tools.extract_legal_holdings(case_citations, legal_issue)
            for holding in holdings:
                results.append(
                    {
                        "holding_text": holding.holding_text,
                        "source_case": holding.source_case,
                        "chunk_id": holding.chunk_id,
                        "legal_concepts": holding.legal_concepts,
                        "context": holding.context,
                    }
                )

        execution_time = time.time() - start_time
        writer(
            {
                "step_id": step_id,
                "status": "complete",
                "brief_description": f"Extraction completed",
                "description": f"Extracted {len(results)} {extraction_type} in {execution_time:.2f}s",
            }
        )

    finally:
        tools.close()

    return results


def generate_research_memo(
    research_question: str, jurisdiction: str = None, client_facts: str = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive legal research memo for ReAct agents.

    Args:
        research_question: The legal research question
        jurisdiction: Relevant jurisdiction
        client_facts: Specific client facts to consider

    Returns:
        Research memo dictionary
    """
    step_id = f"generate_memo_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer(
        {
            "step_id": step_id,
            "status": "in_progress",
            "brief_description": "Generating research memo",
            "description": f"Generating comprehensive legal research memo for: {research_question[:100]}{'...' if len(research_question) > 100 else ''}",
        }
    )

    start_time = time.time()
    tools = create_legal_research_tools()

    try:
        memo = tools.generate_legal_research_memo(
            research_question=research_question,
            jurisdiction=jurisdiction,
            client_facts=client_facts,
        )

        result = {
            "research_question": memo.research_question,
            "executive_summary": memo.executive_summary,
            "legal_analysis": memo.legal_analysis,
            "key_authorities": memo.key_authorities,
            "recommendations": memo.recommendations,
            "gaps_identified": memo.gaps_identified,
        }

        execution_time = time.time() - start_time
        writer(
            {
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Research memo completed",
                "description": f"Generated comprehensive memo with {len(memo.key_authorities)} authorities in {execution_time:.2f}s",
            }
        )

        return result

    finally:
        tools.close()


def validate_legal_statement(
    legal_proposition: str, jurisdiction: str = None, confidence_level: str = "medium"
) -> Dict[str, Any]:
    """
    Validate a legal proposition against available authorities for ReAct agents.

    Args:
        legal_proposition: The legal statement to validate
        jurisdiction: Relevant jurisdiction
        confidence_level: Required confidence level

    Returns:
        Validation results dictionary
    """
    step_id = f"validate_legal_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer(
        {
            "step_id": step_id,
            "status": "in_progress",
            "brief_description": "Validating legal statement",
            "description": f"Validating: {legal_proposition[:100]}{'...' if len(legal_proposition) > 100 else ''}",
        }
    )

    start_time = time.time()
    tools = create_legal_research_tools()

    try:
        validation = tools.validate_legal_proposition(
            legal_proposition=legal_proposition,
            jurisdiction=jurisdiction,
            confidence_level=confidence_level,
        )

        execution_time = time.time() - start_time
        writer(
            {
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Validation completed",
                "description": f"Legal statement validated in {execution_time:.2f}s",
            }
        )

        return validation

    finally:
        tools.close()


def find_cases_by_fact_pattern(
    fact_description: str,
    key_facts: List[str] = None,
    legal_context: str = None,
    similarity_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Find cases with similar factual scenarios for ReAct agents.

    Args:
        fact_description: Description of the factual scenario
        key_facts: Specific key facts to match
        legal_context: Legal context to filter by
        similarity_threshold: Minimum similarity for inclusion

    Returns:
        Cases with similar facts and factual similarity analysis
    """
    step_id = f"find_fact_patterns_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer(
        {
            "step_id": step_id,
            "status": "in_progress",
            "brief_description": "Finding similar fact patterns",
            "description": f"Searching for cases with similar facts: {fact_description[:100]}{'...' if len(fact_description) > 100 else ''}",
        }
    )

    start_time = time.time()
    tools = create_legal_research_tools()
    results = []

    try:
        fact_patterns = tools.find_cases_by_fact_pattern(
            fact_description=fact_description,
            key_facts=key_facts,
            legal_context=legal_context,
            similarity_threshold=similarity_threshold,
        )

        for pattern in fact_patterns:
            results.append(
                {
                    "case_citation": pattern.case_citation,
                    "key_facts": pattern.key_facts,
                    "legal_context": pattern.legal_context,
                    "similarity_score": pattern.similarity_score,
                }
            )

        execution_time = time.time() - start_time
        writer(
            {
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Fact pattern search completed",
                "description": f"Found {len(results)} cases with similar facts in {execution_time:.2f}s",
            }
        )

    finally:
        tools.close()

    return results


def search_statute_references(
    statute_reference: str, section: str = None
) -> List[Dict[str, Any]]:
    """
    Find cases that reference specific statutes for ReAct agents.

    Args:
        statute_reference: Statute reference to search for
        section: Specific section or subsection

    Returns:
        Cases referencing the statute with context
    """
    step_id = f"search_statute_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer(
        {
            "step_id": step_id,
            "status": "in_progress",
            "brief_description": "Searching statute references",
            "description": f"Finding cases that reference: {statute_reference}",
        }
    )

    start_time = time.time()
    tools = create_legal_research_tools()
    results = []

    try:
        statute_cases = tools.search_statute_references(
            statute_reference=statute_reference, section=section
        )

        for case, context in statute_cases:
            results.append(
                {
                    "case": {
                        "citation": case.document_citation,
                        "parties": case.parties,
                        "court": case.court,
                        "jurisdiction": case.jurisdiction,
                        "year": case.year,
                        "legal_tags": case.legal_tags,
                    },
                    "context": context,
                }
            )

        execution_time = time.time() - start_time
        writer(
            {
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Statute search completed",
                "description": f"Found {len(results)} cases referencing {statute_reference} in {execution_time:.2f}s",
            }
        )

    finally:
        tools.close()

    return results


def analyze_citation_network(
    case_citation: str, direction: str = "both"
) -> Dict[str, Any]:
    """
    Analyze citation relationships for ReAct agents.

    Args:
        case_citation: The case to analyze
        direction: "cited_by", "cites", "both"

    Returns:
        Citation network analysis results
    """
    step_id = f"analyze_citations_{uuid.uuid4().hex[:8]}"
    writer = get_stream_writer()
    writer(
        {
            "step_id": step_id,
            "status": "in_progress",
            "brief_description": "Analyzing citation network",
            "description": f"Analyzing citation relationships for: {case_citation}",
        }
    )

    start_time = time.time()
    tools = create_legal_research_tools()

    try:
        network = tools.analyze_citation_network(
            case_citation=case_citation, direction=direction
        )

        result = {
            "case_citation": network.case_citation,
            "cited_by": network.cited_by,
            "cites": network.cites,
            "influence_score": network.influence_score,
            "temporal_position": network.temporal_position,
        }

        execution_time = time.time() - start_time
        writer(
            {
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Citation analysis completed",
                "description": f"Analyzed {len(network.cited_by)} citing cases and {len(network.cites)} cited cases in {execution_time:.2f}s",
            }
        )

        return result

    finally:
        tools.close()


# Tool registry for ReAct agents - streamlined for research and argument drafting
REACT_LEGAL_TOOLS = {
    "semantic_search_legal_content": semantic_search_legal_content,
    "extract_legal_information": extract_legal_information,
    "generate_research_memo": generate_research_memo,
    "validate_legal_statement": validate_legal_statement,
    "find_cases_by_fact_pattern": find_cases_by_fact_pattern,
    "search_statute_references": search_statute_references,
    "analyze_citation_network": analyze_citation_network,
}


# Example usage and testing functions
def test_tools():
    """Test the essential legal research tools with sample queries."""
    print("Testing Streamlined ReAct Legal Research Tools")
    print("=" * 50)

    # Test semantic search
    print("1. Testing semantic search...")
    content = semantic_search_legal_content("contract formation", top_k=3)
    print(f"   Found {len(content)} relevant chunks about contract formation")

    # Test legal holdings extraction
    print("2. Testing legal information extraction...")
    holdings = extract_legal_information("holdings", legal_issue="contract")
    print(f"   Extracted {len(holdings)} holdings related to contract law")

    # Test fact pattern finder
    print("3. Testing fact pattern finder...")
    fact_patterns = find_cases_by_fact_pattern("breach of contract")
    print(f"   Found {len(fact_patterns)} cases with similar facts")

    print("\nAll essential ReAct tools tested successfully!")


if __name__ == "__main__":
    test_tools()

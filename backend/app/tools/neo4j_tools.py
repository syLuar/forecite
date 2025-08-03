"""
Neo4j Tools for Legal Research Assistant

This module provides LangChain tools for interacting with the Neo4j knowledge graph.
These tools are designed to support both research and drafting agent workflows.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.tools import tool
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
import json
from datetime import datetime
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Neo4j connection configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Initialize Neo4j driver with better error handling
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # Test the connection
    with driver.session() as session:
        session.run("RETURN 1")
    logger.info(f"Successfully connected to Neo4j at {NEO4J_URI}")
except Exception as e:
    logger.error(f"Failed to connect to Neo4j: {e}")
    driver = None


def get_session():
    """Get a Neo4j session."""
    if driver is None:
        raise Neo4jError("Neo4j driver not initialized. Check connection credentials.")
    return driver.session()


def get_chunk_by_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific chunk by its ID from Neo4j.

    Args:
        chunk_id: The unique ID of the chunk

    Returns:
        Dictionary containing chunk data or None if not found
    """
    try:
        with get_session() as session:
            query = """
            MATCH (chunk:Chunk {id: $chunk_id})-[:PART_OF]->(doc:Document)
            RETURN chunk.id as chunk_id,
                   chunk.text as text,
                   chunk.summary as summary,
                   chunk.statutes as statutes,
                   chunk.courts as courts,
                   chunk.cases as cases,
                   chunk.concepts as concepts,
                   chunk.judges as judges,
                   chunk.holdings as holdings,
                   chunk.facts as facts,
                   chunk.legal_tests as legal_tests,
                   doc.source as document_source,
                   doc.citation as document_citation,
                   doc.year as document_year,
                   doc.jurisdiction as jurisdiction,
                   doc.type as document_type,
                   doc.court_level as court_level
            """

            result = session.run(query, {"chunk_id": chunk_id})
            record = result.single()

            if record:
                return dict(record)
            return None

    except Neo4jError as e:
        logger.error(f"Neo4j error in get_chunk_by_id: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_chunk_by_id: {e}")
        return None


async def vector_search(
    query_text: str,
    limit: int = 10,
    min_score: float = 0.7,
    jurisdiction: Optional[str] = None,
    document_type: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Perform semantic vector search on legal document chunks.

    Args:
        query_text: The search query text
        limit: Maximum number of results to return
        min_score: Minimum similarity score threshold
        jurisdiction: Filter by jurisdiction (e.g., 'Singapore', 'UK')
        document_type: Filter by document type (e.g., 'Case', 'Doctrine')
        year_from: Filter documents from this year onwards
        year_to: Filter documents up to this year

    Returns:
        List of relevant chunks with metadata and similarity scores
    """
    try:
        embeddings_attrs = settings.llm_config.get("embeddings", {})
        embeddings = GoogleGenerativeAIEmbeddings(**embeddings_attrs)
        query_embedding = await embeddings.aembed_query(query_text)

        with get_session() as session:
            # Build dynamic WHERE clause for filtering
            where_conditions = ["score >= $min_score"]
            params = {
                "query_embedding": query_embedding,
                "limit": limit,
                "min_score": min_score,
            }

            if jurisdiction:
                where_conditions.append("doc.jurisdiction = $jurisdiction")
                params["jurisdiction"] = jurisdiction

            if document_type:
                where_conditions.append("doc.type = $document_type")
                params["document_type"] = document_type

            if year_from:
                where_conditions.append("doc.year >= $year_from")
                params["year_from"] = year_from

            if year_to:
                where_conditions.append("doc.year <= $year_to")
                params["year_to"] = year_to

            where_clause = " AND ".join(where_conditions)

            query = f"""
            CALL db.index.vector.queryNodes('chunk_embeddings', $limit, $query_embedding) 
            YIELD node, score
            MATCH (node)-[:PART_OF]->(doc:Document)
            WHERE {where_clause}
            OPTIONAL MATCH (node)-[:REFERENCES_CHUNK]->(referenced_chunk:Chunk)
            OPTIONAL MATCH (referencing_chunk:Chunk)-[:REFERENCES_CHUNK]->(node)
            RETURN node.id as chunk_id,
                   node.text as text,
                   node.summary as summary,
                   node.chunk_index as chunk_index,
                   node.statutes as statutes,
                   node.courts as courts,
                   node.cases as cases,
                   node.concepts as concepts,
                   node.judges as judges,
                   node.holdings as holdings,
                   node.facts as facts,
                   node.legal_tests as legal_tests,
                   node.chunk_references as chunk_references,
                   collect(DISTINCT referenced_chunk.id) as references_outgoing,
                   collect(DISTINCT referencing_chunk.id) as references_incoming,
                   doc.source as document_source,
                   doc.citation as document_citation,
                   doc.year as document_year,
                   doc.jurisdiction as jurisdiction,
                   doc.type as document_type,
                   doc.court_level as court_level,
                   score * 100 as score
            ORDER BY score DESC
            """

            result = session.run(query, params)
            return [dict(record) for record in result]

    except Neo4jError as e:
        logger.error(f"Neo4j error in vector_search: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in vector_search: {e}")
        return []


def fulltext_search(
    search_terms: str,
    search_type: str = "chunks",
    limit: int = 10,
    jurisdiction: Optional[str] = None,
    document_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform fulltext search on legal documents or chunks.

    Args:
        search_terms: Text to search for
        search_type: 'chunks' or 'documents'
        limit: Maximum number of results
        jurisdiction: Filter by jurisdiction
        document_type: Filter by document type

    Returns:
        List of matching results with relevance scores
    """
    try:
        with get_session() as session:
            if search_type == "chunks":
                base_query = """
                CALL db.index.fulltext.queryNodes('chunk_text_search', $search_terms) 
                YIELD node, score
                MATCH (node)-[:PART_OF]->(doc:Document)
                OPTIONAL MATCH (node)-[:REFERENCES_CHUNK]->(referenced_chunk:Chunk)
                OPTIONAL MATCH (referencing_chunk:Chunk)-[:REFERENCES_CHUNK]->(node)
                """
                return_clause = """
                RETURN node.id as chunk_id,
                       node.text as text,
                       node.summary as summary,
                       node.chunk_index as chunk_index,
                       node.statutes as statutes,
                       node.courts as courts,
                       node.cases as cases,
                       node.concepts as concepts,
                       node.judges as judges,
                       node.holdings as holdings,
                       node.facts as facts,
                       node.legal_tests as legal_tests,
                       node.chunk_references as chunk_references,
                       collect(DISTINCT referenced_chunk.id) as references_outgoing,
                       collect(DISTINCT referencing_chunk.id) as references_incoming,
                       doc.source as document_source,
                       doc.citation as document_citation,
                       doc.year as document_year,
                       doc.jurisdiction as jurisdiction,
                       doc.type as document_type,
                       doc.court_level as court_level,
                       score
                """
            else:  # documents
                base_query = """
                CALL db.index.fulltext.queryNodes('legal_text_search', $search_terms) 
                YIELD node as doc, score
                """
                return_clause = """
                RETURN doc.source as document_source,
                       doc.citation as document_citation,
                       doc.full_text as full_text,
                       doc.year as document_year,
                       doc.jurisdiction as jurisdiction,
                       doc.type as document_type,
                       score
                """

            # Add filtering
            where_conditions = []
            params = {"search_terms": search_terms, "limit": limit}

            if jurisdiction:
                where_conditions.append("doc.jurisdiction = $jurisdiction")
                params["jurisdiction"] = jurisdiction

            if document_type:
                where_conditions.append("doc.type = $document_type")
                params["document_type"] = document_type

            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)

            query = f"""
            {base_query}
            {where_clause}
            {return_clause}
            ORDER BY score DESC
            LIMIT $limit
            """

            result = session.run(query, params)
            return [dict(record) for record in result]

    except Neo4jError as e:
        logger.error(f"Neo4j error in fulltext_search: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in fulltext_search: {e}")
        return []


def find_case_citations(
    case_citation: str, direction: str = "both"
) -> List[Dict[str, Any]]:
    """
    Find citation relationships for a specific case.

    Args:
        case_citation: The case citation to search for
        direction: 'cited_by', 'cites', or 'both'

    Returns:
        List of citation relationships with document metadata
    """
    try:
        with get_session() as session:
            if direction == "cited_by":
                query = """
                MATCH (case:Case {citation: $case_citation})<-[:CITES]-(doc:Document)
                RETURN doc.citation as citing_document,
                       doc.year as citing_year,
                       doc.jurisdiction as jurisdiction,
                       doc.court_level as court_level,
                       doc.type as document_type,
                       'cited_by' as relationship_type
                ORDER BY doc.year DESC
                """
            elif direction == "cites":
                query = """
                MATCH (doc:Document {citation: $case_citation})-[:CITES]->(case:Case)
                RETURN case.citation as cited_case,
                       case.year as cited_year,
                       'cites' as relationship_type
                ORDER BY case.year
                """
            else:  # both
                query = """
                MATCH (case:Case {citation: $case_citation})<-[:CITES]-(citing_doc:Document)
                RETURN citing_doc.citation as related_document,
                       citing_doc.year as related_year,
                       citing_doc.jurisdiction as jurisdiction,
                       'cited_by' as relationship_type
                UNION
                MATCH (doc:Document {citation: $case_citation})-[:CITES]->(cited_case:Case)
                RETURN cited_case.citation as related_document,
                       cited_case.year as related_year,
                       null as jurisdiction,
                       'cites' as relationship_type
                ORDER BY related_year DESC
                """

            result = session.run(query, {"case_citation": case_citation})
            return [dict(record) for record in result]

    except Neo4jError as e:
        logger.error(f"Neo4j error in find_case_citations: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in find_case_citations: {e}")
        return []


def assess_precedent_strength(case_citation: str) -> Dict[str, Any]:
    """
    Assess the precedential strength of a case based on citation analysis.

    Args:
        case_citation: The case citation to analyze

    Returns:
        Dictionary with precedent strength metrics
    """
    try:
        with get_session() as session:
            query = """
            MATCH (case:Case {citation: $case_citation})<-[:CITES]-(citing_doc:Document)
            MATCH (citing_doc)-[:HEARD_IN]->(court:Court)
            WITH case, citing_doc, court,
                 CASE court.level 
                   WHEN 'Appellate' THEN 3
                   WHEN 'Superior' THEN 2 
                   ELSE 1 
                 END as authority_weight
            RETURN case.citation as case_citation,
                   case.year as case_year,
                   count(citing_doc) as total_citations,
                   avg(authority_weight) as avg_authority_weight,
                   sum(authority_weight) as total_authority_score,
                   collect(DISTINCT citing_doc.jurisdiction) as citing_jurisdictions,
                   max(citing_doc.year) as most_recent_citation,
                   min(citing_doc.year) as earliest_citation
            """

            result = session.run(query, {"case_citation": case_citation})
            record = result.single()

            if record:
                data = dict(record)
                # Calculate composite precedent strength score
                total_citations = data.get("total_citations", 0)
                avg_authority = data.get("avg_authority_weight", 0)

                data["precedent_strength"] = total_citations * avg_authority
                data["jurisdictional_reach"] = len(data.get("citing_jurisdictions", []))

                return data
            else:
                return {
                    "case_citation": case_citation,
                    "total_citations": 0,
                    "precedent_strength": 0,
                    "message": "No citation data found for this case",
                }

    except Neo4jError as e:
        logger.error(f"Neo4j error in assess_precedent_strength: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in assess_precedent_strength: {e}")
        return {"error": str(e)}


def find_legal_concepts(
    concept_name: str, jurisdiction: Optional[str] = None, limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Find documents that discuss a specific legal concept.

    Args:
        concept_name: Name of the legal concept
        jurisdiction: Filter by jurisdiction
        limit: Maximum number of results

    Returns:
        List of documents discussing the concept with relevant chunks
    """
    try:
        with get_session() as session:
            where_conditions = []
            params = {"concept_name": concept_name, "limit": limit}

            if jurisdiction:
                where_conditions.append("doc.jurisdiction = $jurisdiction")
                params["jurisdiction"] = jurisdiction

            where_clause = ""
            if where_conditions:
                where_clause = "AND " + " AND ".join(where_conditions)

            query = f"""
            MATCH (concept:LegalConcept {{name: $concept_name}})
            MATCH (doc:Document)-[:DISCUSSES]->(concept)
            MATCH (chunk:Chunk)-[:PART_OF]->(doc)
            WHERE any(c in chunk.concepts WHERE c CONTAINS $concept_name)
            {where_clause}
            RETURN doc.citation as document_citation,
                   doc.year as document_year,
                   doc.jurisdiction as jurisdiction,
                   doc.court_level as court_level,
                   chunk.summary as chunk_summary,
                   chunk.holdings as holdings,
                   chunk.legal_tests as legal_tests
            ORDER BY doc.year DESC
            LIMIT $limit
            """

            result = session.run(query, params)
            return [dict(record) for record in result]

    except Neo4jError as e:
        logger.error(f"Neo4j error in find_legal_concepts: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in find_legal_concepts: {e}")
        return []


def find_similar_fact_patterns(
    key_facts: List[str], jurisdiction: Optional[str] = None, limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Find cases with similar fact patterns.

    Args:
        key_facts: List of key facts to match
        jurisdiction: Filter by jurisdiction
        limit: Maximum number of results

    Returns:
        List of cases with matching fact patterns
    """
    try:
        with get_session() as session:
            where_conditions = []
            params = {"key_facts": key_facts, "limit": limit}

            # Build fact matching conditions
            fact_conditions = []
            for i, fact in enumerate(key_facts):
                fact_param = f"fact_{i}"
                fact_conditions.append(
                    f"any(f in chunk.facts WHERE f CONTAINS ${fact_param})"
                )
                params[fact_param] = fact

            fact_where = " AND ".join(fact_conditions)

            if jurisdiction:
                where_conditions.append("doc.jurisdiction = $jurisdiction")
                params["jurisdiction"] = jurisdiction

            additional_where = ""
            if where_conditions:
                additional_where = "AND " + " AND ".join(where_conditions)

            query = f"""
            MATCH (chunk:Chunk)-[:PART_OF]->(doc:Document)
            WHERE {fact_where}
            {additional_where}
            WITH doc, chunk, size([f in chunk.facts WHERE any(kf in $key_facts WHERE f CONTAINS kf)]) as fact_matches
            WHERE fact_matches > 0
            RETURN doc.citation as document_citation,
                   doc.year as document_year,
                   doc.jurisdiction as jurisdiction,
                   chunk.facts as facts,
                   chunk.holdings as holdings,
                   fact_matches
            ORDER BY fact_matches DESC, doc.year DESC
            LIMIT $limit
            """

            result = session.run(query, params)
            return [dict(record) for record in result]

    except Neo4jError as e:
        logger.error(f"Neo4j error in find_similar_fact_patterns: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in find_similar_fact_patterns: {e}")
        return []


def find_legal_tests(
    legal_area: str, jurisdiction: Optional[str] = None, limit: int = 15
) -> List[Dict[str, Any]]:
    """
    Find legal tests and standards for a specific area of law.

    Args:
        legal_area: Area of law (e.g., 'negligence', 'contract')
        jurisdiction: Filter by jurisdiction
        limit: Maximum number of results

    Returns:
        List of legal tests with source documents
    """
    try:
        with get_session() as session:
            where_conditions = []
            params = {"legal_area": legal_area, "limit": limit}

            if jurisdiction:
                where_conditions.append("doc.jurisdiction = $jurisdiction")
                params["jurisdiction"] = jurisdiction

            where_clause = ""
            if where_conditions:
                where_clause = "AND " + " AND ".join(where_conditions)

            query = f"""
            MATCH (concept:LegalConcept)
            WHERE concept.name CONTAINS $legal_area
            MATCH (doc:Document)-[:DISCUSSES]->(concept)
            MATCH (chunk:Chunk)-[:PART_OF]->(doc)
            WHERE size(chunk.legal_tests) > 0
            {where_clause}
            UNWIND chunk.legal_tests as test
            RETURN DISTINCT test as legal_test,
                   doc.citation as source_document,
                   doc.year as document_year,
                   doc.jurisdiction as jurisdiction,
                   doc.court_level as court_level
            ORDER BY doc.year DESC
            LIMIT $limit
            """

            result = session.run(query, params)
            return [dict(record) for record in result]

    except Neo4jError as e:
        logger.error(f"Neo4j error in find_legal_tests: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in find_legal_tests: {e}")
        return []


@tool
def find_judge_jurisprudence(
    judge_name: str, legal_concept: Optional[str] = None, limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Find a judge's jurisprudence on specific topics.

    Args:
        judge_name: Name of the judge
        legal_concept: Specific legal concept to focus on
        limit: Maximum number of results

    Returns:
        List of cases and holdings by the judge
    """
    try:
        with get_session() as session:
            where_conditions = []
            params = {"judge_name": judge_name, "limit": limit}

            if legal_concept:
                where_conditions.append(
                    "any(c in chunk.concepts WHERE c CONTAINS $legal_concept)"
                )
                params["legal_concept"] = legal_concept

            where_clause = ""
            if where_conditions:
                where_clause = "AND " + " AND ".join(where_conditions)

            query = f"""
            MATCH (judge:Judge {{name: $judge_name}})-[:AUTHORED]->(doc:Document)
            MATCH (chunk:Chunk)-[:PART_OF]->(doc)
            WHERE size(chunk.holdings) > 0
            {where_clause}
            RETURN doc.citation as document_citation,
                   doc.year as document_year,
                   doc.jurisdiction as jurisdiction,
                   chunk.holdings as holdings,
                   chunk.concepts as concepts
            ORDER BY doc.year DESC
            LIMIT $limit
            """

            result = session.run(query, params)
            return [dict(record) for record in result]

    except Neo4jError as e:
        logger.error(f"Neo4j error in find_judge_jurisprudence: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in find_judge_jurisprudence: {e}")
        return []


def find_authority_chain(start_case: str, max_depth: int = 3) -> List[Dict[str, Any]]:
    """
    Find authority chains - cases that cite each other in sequence.

    Args:
        start_case: Starting case citation
        max_depth: Maximum depth of citation chain

    Returns:
        List of authority chains with precedent relationships
    """
    try:
        with get_session() as session:
            query = """
            MATCH path = (start:Case {citation: $start_case})<-[:CITES*1..$max_depth]-(end:Document)
            WITH path, length(path) as chain_length
            UNWIND nodes(path) as node
            WITH path, chain_length, 
                 [n in nodes(path) WHERE n:Case | n.citation] as case_chain,
                 [n in nodes(path) WHERE n:Document | {citation: n.citation, year: n.year}] as doc_chain
            RETURN case_chain,
                   doc_chain,
                   chain_length
            ORDER BY chain_length DESC
            """

            result = session.run(
                query, {"start_case": start_case, "max_depth": max_depth}
            )
            return [dict(record) for record in result]

    except Neo4jError as e:
        logger.error(f"Neo4j error in find_authority_chain: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in find_authority_chain: {e}")
        return []


def get_document_metadata(document_source: str) -> Dict[str, Any]:
    """
    Get comprehensive metadata for a specific document.

    Args:
        document_source: Source filename of the document

    Returns:
        Dictionary with document metadata and statistics
    """
    try:
        with get_session() as session:
            query = """
            MATCH (doc:Document {source: $document_source})
            OPTIONAL MATCH (doc)-[:CITES]->(cited_case:Case)
            OPTIONAL MATCH (doc)<-[:CITES]-(citing_doc:Document)
            OPTIONAL MATCH (doc)-[:DISCUSSES]->(concept:LegalConcept)
            OPTIONAL MATCH (chunk:Chunk)-[:PART_OF]->(doc)
            RETURN doc.source as source,
                   doc.citation as citation,
                   doc.type as document_type,
                   doc.year as year,
                   doc.jurisdiction as jurisdiction,
                   doc.court_level as court_level,
                   doc.parties as parties,
                   doc.legal_areas as legal_areas,
                   doc.word_count as word_count,
                   collect(DISTINCT cited_case.citation) as cites_cases,
                   collect(DISTINCT citing_doc.citation) as cited_by_documents,
                   collect(DISTINCT concept.name) as discusses_concepts,
                   count(DISTINCT chunk) as total_chunks
            """

            result = session.run(query, {"document_source": document_source})
            record = result.single()

            if record:
                return dict(record)
            else:
                return {"error": f"Document not found: {document_source}"}

    except Neo4jError as e:
        logger.error(f"Neo4j error in get_document_metadata: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in get_document_metadata: {e}")
        return {"error": str(e)}


@tool
def expand_graph_context(
    node_ids: List[str],
    relationship_types: Optional[List[str]] = None,
    max_depth: int = 2,
) -> List[Dict[str, Any]]:
    """
    Expand context around specific nodes in the knowledge graph.

    Args:
        node_ids: List of node IDs to expand from
        relationship_types: Specific relationship types to follow
        max_depth: Maximum depth of expansion

    Returns:
        List of connected nodes with relationships
    """
    try:
        with get_session() as session:
            # Build relationship filter
            rel_filter = ""
            if relationship_types:
                rel_types = "|".join(relationship_types)
                rel_filter = f":{rel_types}"

            query = f"""
            UNWIND $node_ids as node_id
            MATCH (start) WHERE id(start) = toInteger(node_id)
            MATCH path = (start)-[{rel_filter}*1..$max_depth]-(connected)
            RETURN start.id as start_node_id,
                   id(connected) as connected_node_id,
                   labels(connected) as node_labels,
                   properties(connected) as node_properties,
                   [r in relationships(path) | type(r)] as relationship_path,
                   length(path) as path_length
            ORDER BY path_length
            """

            result = session.run(query, {"node_ids": node_ids, "max_depth": max_depth})

            return [dict(record) for record in result]

    except Neo4jError as e:
        logger.error(f"Neo4j error in expand_graph_context: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in expand_graph_context: {e}")
        return []


@tool
def find_chunk_references(
    chunk_id: str, direction: str = "both", max_depth: int = 2
) -> List[Dict[str, Any]]:
    """
    Find chunk reference relationships - chunks that reference each other within documents.

    Args:
        chunk_id: The chunk ID to find references for
        direction: 'outgoing', 'incoming', or 'both'
        max_depth: Maximum depth of reference chains to follow

    Returns:
        List of related chunks with reference relationships
    """
    try:
        with get_session() as session:
            if direction == "outgoing":
                query = """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH path = (c)-[:REFERENCES_CHUNK*1..$max_depth]->(referenced)
                MATCH (referenced)-[:PART_OF]->(doc:Document)
                RETURN referenced.id as referenced_chunk_id,
                       referenced.text as text,
                       referenced.summary as summary,
                       referenced.chunk_index as chunk_index,
                       doc.source as document_source,
                       doc.citation as document_citation,
                       [r in relationships(path) | r.paragraph_ref] as paragraph_refs,
                       length(path) as reference_depth
                ORDER BY reference_depth, referenced.chunk_index
                """
            elif direction == "incoming":
                query = """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH path = (referencing)-[:REFERENCES_CHUNK*1..$max_depth]->(c)
                MATCH (referencing)-[:PART_OF]->(doc:Document)
                RETURN referencing.id as referencing_chunk_id,
                       referencing.text as text,
                       referencing.summary as summary,
                       referencing.chunk_index as chunk_index,
                       doc.source as document_source,
                       doc.citation as document_citation,
                       [r in relationships(path) | r.paragraph_ref] as paragraph_refs,
                       length(path) as reference_depth
                ORDER BY reference_depth, referencing.chunk_index
                """
            else:  # both
                query = """
                MATCH (c:Chunk {id: $chunk_id})
                OPTIONAL MATCH outgoing_path = (c)-[:REFERENCES_CHUNK*1..$max_depth]->(referenced)
                OPTIONAL MATCH incoming_path = (referencing)-[:REFERENCES_CHUNK*1..$max_depth]->(c)
                OPTIONAL MATCH (referenced)-[:PART_OF]->(ref_doc:Document)
                OPTIONAL MATCH (referencing)-[:PART_OF]->(ref_doc2:Document)
                WITH c, 
                     collect(DISTINCT {
                         type: 'outgoing',
                         chunk_id: referenced.id,
                         text: referenced.text,
                         summary: referenced.summary,
                         chunk_index: referenced.chunk_index,
                         document_source: ref_doc.source,
                         document_citation: ref_doc.citation,
                         paragraph_refs: [r in relationships(outgoing_path) | r.paragraph_ref],
                         reference_depth: length(outgoing_path)
                     }) as outgoing_refs,
                     collect(DISTINCT {
                         type: 'incoming',
                         chunk_id: referencing.id,
                         text: referencing.text,
                         summary: referencing.summary,
                         chunk_index: referencing.chunk_index,
                         document_source: ref_doc2.source,
                         document_citation: ref_doc2.citation,
                         paragraph_refs: [r in relationships(incoming_path) | r.paragraph_ref],
                         reference_depth: length(incoming_path)
                     }) as incoming_refs
                RETURN outgoing_refs + incoming_refs as all_references
                """

            result = session.run(query, {"chunk_id": chunk_id, "max_depth": max_depth})

            if direction == "both":
                records = list(result)
                if records:
                    all_refs = records[0]["all_references"]
                    # Filter out null entries
                    return [ref for ref in all_refs if ref.get("chunk_id")]
                return []
            else:
                return [dict(record) for record in result]

    except Neo4jError as e:
        logger.error(f"Neo4j error in find_chunk_references: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in find_chunk_references: {e}")
        return []


@tool
def get_chunk_context(
    chunk_id: str, context_size: int = 2, include_references: bool = True
) -> Dict[str, Any]:
    """
    Get contextual information around a specific chunk including surrounding chunks and references.

    Args:
        chunk_id: The chunk ID to get context for
        context_size: Number of surrounding chunks to include before and after
        include_references: Whether to include chunk references

    Returns:
        Dictionary with chunk context and related information
    """
    try:
        with get_session() as session:
            # Get the main chunk with surrounding context
            context_query = """
            MATCH (target:Chunk {id: $chunk_id})-[:PART_OF]->(doc:Document)
            MATCH (context_chunk:Chunk)-[:PART_OF]->(doc)
            WHERE context_chunk.chunk_index >= target.chunk_index - $context_size
              AND context_chunk.chunk_index <= target.chunk_index + $context_size
            RETURN target.id as target_chunk_id,
                   target.text as target_text,
                   target.summary as target_summary,
                   target.chunk_index as target_index,
                   doc.source as document_source,
                   doc.citation as document_citation,
                   collect({
                       chunk_id: context_chunk.id,
                       text: context_chunk.text,
                       summary: context_chunk.summary,
                       chunk_index: context_chunk.chunk_index,
                       is_target: context_chunk.id = target.id
                   }) as context_chunks
            """

            result = session.run(
                context_query, {"chunk_id": chunk_id, "context_size": context_size}
            )

            record = result.single()
            if not record:
                return {"error": "Chunk not found"}

            context_data = {
                "target_chunk_id": record["target_chunk_id"],
                "target_text": record["target_text"],
                "target_summary": record["target_summary"],
                "target_index": record["target_index"],
                "document_source": record["document_source"],
                "document_citation": record["document_citation"],
                "context_chunks": sorted(
                    record["context_chunks"], key=lambda x: x["chunk_index"]
                ),
            }

            # Add chunk references if requested
            if include_references:
                context_data["chunk_references"] = find_chunk_references(
                    chunk_id, "both", 1
                )

            return context_data

    except Neo4jError as e:
        logger.error(f"Neo4j error in get_chunk_context: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in get_chunk_context: {e}")
        return {"error": str(e)}


@tool
def trace_argument_flow(
    starting_chunk_id: str, max_depth: int = 5
) -> List[Dict[str, Any]]:
    """
    Trace the flow of legal arguments by following chunk references.

    Args:
        starting_chunk_id: The chunk to start tracing from
        max_depth: Maximum depth of argument chains to follow

    Returns:
        List representing the argument flow with chunks and their connections
    """
    try:
        with get_session() as session:
            # Find argument flow by following REFERENCES_CHUNK relationships
            flow_query = """
            MATCH (start:Chunk {id: $starting_chunk_id})
            MATCH path = (start)-[:REFERENCES_CHUNK*0..$max_depth]-(connected)
            MATCH (connected)-[:PART_OF]->(doc:Document)
            WITH path, connected, doc,
                 [rel in relationships(path) | {
                     type: type(rel),
                     paragraph_ref: rel.paragraph_ref,
                     direction: CASE 
                         WHEN startNode(rel).id = $starting_chunk_id THEN 'outgoing'
                         WHEN endNode(rel).id = $starting_chunk_id THEN 'incoming'
                         ELSE 'indirect'
                     END
                 }] as relationship_chain
            RETURN connected.id as chunk_id,
                   connected.text as text,
                   connected.summary as summary,
                   connected.chunk_index as chunk_index,
                   connected.holdings as holdings,
                   doc.source as document_source,
                   doc.citation as document_citation,
                   length(path) as distance_from_start,
                   relationship_chain
            ORDER BY distance_from_start, connected.chunk_index
            """

            result = session.run(
                flow_query,
                {"starting_chunk_id": starting_chunk_id, "max_depth": max_depth},
            )

            return [dict(record) for record in result]

    except Neo4jError as e:
        logger.error(f"Neo4j error in trace_argument_flow: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in trace_argument_flow: {e}")
        return []


# Cleanup function to close the driver
def close_neo4j_connection():
    """Close the Neo4j driver connection."""
    if driver:
        driver.close()


# List of all available tools for easy import
__all__ = [
    "vector_search",
    "fulltext_search",
    "find_case_citations",
    "assess_precedent_strength",
    "find_legal_concepts",
    "find_similar_fact_patterns",
    "find_legal_tests",
    "find_judge_jurisprudence",
    "find_authority_chain",
    "get_document_metadata",
    "expand_graph_context",
    "find_chunk_references",
    "get_chunk_context",
    "trace_argument_flow",
    "close_neo4j_connection",
]

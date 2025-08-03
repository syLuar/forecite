# Legal Knowledge Graph: Schema, Construction & Usage

## Overview

The Legal Knowledge Graph is a sophisticated Neo4j-based system designed to support legal research, case discovery, and argument generation. It transforms legal documents (PDFs) into a rich, interconnected graph of entities and relationships that enables AI-powered legal analysis.

## 1. Knowledge Graph Schema

### 1.1 Node Types

#### Docu### 3.4 Enhanced Research Workflows

#### Multi-Stage Research with Cross-References
```cypher
// Stage 1: Find directly relevant cases
CALL db.index.fulltext.queryNodes('chunk_text_search', $search_terms) 
YIELD node, score
MATCH (node)-[:PART_OF]->(doc:Document)
WHERE score > 0.8 AND doc.type = 'Case'

// Stage 2: Follow chunk references to find supporting arguments
MATCH (node)-[:REFERENCES_CHUNK*1..2]->(referenced:Chunk)
MATCH (referenced)-[:PART_OF]->(ref_doc:Document)

// Stage 3: Find cases that cite the relevant cases
MATCH (doc)-[:CITES]->(case:Case)<-[:CITES]-(citing_doc:Document)
WHERE citing_doc.year > doc.year

RETURN DISTINCT ref_doc.citation, ref_doc.year, 
       collect(DISTINCT referenced.summary) as supporting_arguments
ORDER BY ref_doc.year DESC
```

#### Contextual Chunk Retrieval
Enhanced search that includes surrounding context and references:

```cypher
MATCH (target:Chunk)-[:PART_OF]->(doc:Document)
WHERE target.id = $chunk_id

// Get surrounding chunks for context
MATCH (context:Chunk)-[:PART_OF]->(doc)
WHERE context.chunk_index >= target.chunk_index - 2
  AND context.chunk_index <= target.chunk_index + 2

// Get referenced chunks
OPTIONAL MATCH (target)-[:REFERENCES_CHUNK]->(referenced:Chunk)
OPTIONAL MATCH (referencing:Chunk)-[:REFERENCES_CHUNK]->(target)

RETURN target.text as main_text,
       target.summary as main_summary,
       collect(DISTINCT {
           chunk_index: context.chunk_index,
           text: context.text,
           is_target: context.id = target.id
       }) as context_chunks,
       collect(DISTINCT referenced.summary) as referenced_summaries,
       collect(DISTINCT referencing.summary) as referencing_summaries
```ent Nodes
Primary containers for legal documents with comprehensive metadata:

```cypher
(:Document {
  source: string,           // Filename of the PDF
  type: string,            // "Case", "Doctrine", "Document"
  full_text: string,       // First 5000 characters
  file_path: string,       // Absolute file path
  word_count: integer,     // Total word count
  date: string,            // Document date (YYYY-MM-DD)
  year: integer,           // Document year
  jurisdiction: string,    // "Singapore", "UK", "Australia", etc.
  court_level: string,     // "Appellate", "Superior", "Lower"
  case_number: string,     // Case number from citation
  citation: string,        // Formal legal citation
  parties: [string],       // List of case parties
  legal_areas: [string],   // Areas of law covered
  created_at: datetime     // Ingestion timestamp
})
```

#### Chunk Nodes
Text segments with semantic embeddings and extracted entities:

```cypher
(:Chunk {
  id: string,              // Unique identifier: "{source}_{chunk_index}"
  text: string,            // Chunk content
  summary: string,         // AI-generated summary
  embedding: [float],      // 768-dimensional vector embedding
  chunk_index: integer,    // Position in document
  statutes: [string],      // Referenced statutes
  courts: [string],        // Mentioned courts
  cases: [string],         // Referenced cases
  concepts: [string],      // Legal concepts
  judges: [string],        // Judge names
  holdings: [string],      // Legal holdings
  facts: [string],         // Key facts
  legal_tests: [string],   // Legal tests/standards
  created_at: datetime     // Creation timestamp
})
```

#### Entity Nodes

**Case Entities:**
```cypher
(:Case {
  citation: string,        // Formal citation (unique)
  name: string,           // Case name
  year: integer,          // Year decided
  created_at: datetime
})
```

**Statute Entities:**
```cypher
(:Statute {
  reference: string,       // Statute reference (unique)
  created_at: datetime
})
```

**Court Entities:**
```cypher
(:Court {
  name: string,           // Court name (unique)
  jurisdiction: string,   // Legal jurisdiction
  level: string,          // Court hierarchy level
  created_at: datetime
})
```

**Legal Concept Entities:**
```cypher
(:LegalConcept {
  name: string,           // Concept name (unique)
  created_at: datetime
})
```

**Judge Entities:**
```cypher
(:Judge {
  name: string,           // Judge name (unique)
  created_at: datetime
})
```

### 1.2 Relationship Types

#### Hierarchical Relationships
- `(:Chunk)-[:PART_OF]->(:Document)` - Chunk belongs to document

#### Citation & Reference Relationships
- `(:Document)-[:CITES]->(:Case)` - Document cites case
- `(:Document)-[:REFERENCES_STATUTE]->(:Statute)` - Document references statute
- `(:Chunk)-[:REFERENCES]->(:Document)` - Chunk references another document
- `(:Chunk)-[:REFERENCES_CHUNK {paragraph_ref: string}]->(:Chunk)` - Chunk references another chunk within same document

#### Institutional Relationships
- `(:Document)-[:HEARD_IN]->(:Court)` - Case heard in court
- `(:Judge)-[:AUTHORED]->(:Document)` - Judge authored document
- `(:Document)-[:DISCUSSES]->(:LegalConcept)` - Document discusses concept

#### Temporal Precedent Relationships
- `(:Document)-[:FOLLOWS {weight: float}]->(:Case)` - Document follows precedent
- `(:Document)-[:OVERRULES]->(:Case)` - Document overrules precedent
- `(:Document)-[:DISTINGUISHES]->(:Case)` - Document distinguishes precedent

### 1.3 Indexes & Constraints

#### Constraints (Uniqueness)
```cypher
CREATE CONSTRAINT unique_document_source FOR (d:Document) REQUIRE d.source IS UNIQUE
CREATE CONSTRAINT unique_chunk_id FOR (c:Chunk) REQUIRE c.id IS UNIQUE
CREATE CONSTRAINT unique_case_citation FOR (case:Case) REQUIRE case.citation IS UNIQUE
CREATE CONSTRAINT unique_statute_reference FOR (s:Statute) REQUIRE s.reference IS UNIQUE
CREATE CONSTRAINT unique_court_name FOR (court:Court) REQUIRE court.name IS UNIQUE
CREATE CONSTRAINT unique_legal_concept FOR (lc:LegalConcept) REQUIRE lc.name IS UNIQUE
CREATE CONSTRAINT unique_judge_name FOR (j:Judge) REQUIRE j.name IS UNIQUE
```

#### Performance Indexes
```cypher
CREATE INDEX document_type_index FOR (d:Document) ON (d.type)
CREATE INDEX document_date_index FOR (d:Document) ON (d.date)
CREATE INDEX case_year_index FOR (case:Case) ON (case.year)
CREATE INDEX court_jurisdiction_index FOR (court:Court) ON (court.jurisdiction)
CREATE INDEX chunk_summary_text_index FOR (c:Chunk) ON (c.summary)
```

#### Fulltext Search Indexes
```cypher
CREATE FULLTEXT INDEX legal_text_search FOR (d:Document) ON EACH [d.full_text]
CREATE FULLTEXT INDEX chunk_text_search FOR (c:Chunk) ON EACH [c.text, c.summary]
```

#### Vector Index
```cypher
CALL db.index.vector.createNodeIndex(
  'chunk_embeddings',
  'Chunk', 
  'embedding', 
  768, 
  'cosine'
)
```

## 2. Knowledge Graph Construction Process

### 2.1 Document Processing Pipeline

For each PDF file in `data/raw_docs/`, the system follows this pipeline:

#### Step 1: Text Extraction
```python
# Extract text from PDF using PyPDF2
text = extract_text_from_pdf(pdf_path)
```

#### Step 2: Metadata Extraction
Uses LLM analysis and filename parsing to extract:
- **Temporal data:** Year, date from filename patterns like `[2019] SGCA 42.pdf`
- **Citation data:** Formal legal citations
- **Jurisdictional data:** Singapore, UK, Australia, etc.
- **Court information:** Court level and hierarchy
- **Parties:** Plaintiff, defendant, appellant, respondent
- **Legal areas:** Contract law, tort law, criminal law, etc.

#### Step 3: Document Node Creation
```cypher
MERGE (d:Document {source: $source})
SET d.type = $type,
    d.full_text = $full_text,
    d.year = $year,
    d.jurisdiction = $jurisdiction,
    d.citation = $citation,
    // ... other properties
```

#### Step 4: Text Chunking
- **Chunking strategy:** Recursive character splitting
- **Chunk size:** 1000 characters
- **Overlap:** 200 characters
- **Smart separators:** `["\n\n", "\n", ". ", " ", ""]`

#### Step 5: Chunk Processing (Batch Mode)
For each chunk batch:

1. **Embedding Generation**
   ```python
   embeddings = await generate_embeddings(chunk_texts)
   # Uses Google text-embedding-004 (768 dimensions)
   ```

2. **Summarization**
   ```python
   summary = generate_summary(chunk_text)
   # Focuses on: legal concepts, case citations, main arguments
   ```

3. **Entity Extraction**
   ```python
   entities = extract_entities(chunk_text)
   # Extracts: cases, statutes, concepts, courts, judges, holdings, facts, legal_tests
   ```

#### Step 6: Chunk Node Creation
```cypher
CREATE (c:Chunk {
  id: $id,
  text: $text,
  summary: $summary,
  embedding: $embedding,
  statutes: $statutes,
  courts: $courts,
  cases: $cases,
  concepts: $concepts,
  judges: $judges,
  holdings: $holdings,
  facts: $facts,
  legal_tests: $legal_tests
})
CREATE (c)-[:PART_OF]->(d:Document {source: $source})
```

#### Step 7: Entity Node Creation
Creates first-class entity nodes for:
- **Cases:** With citations and years
- **Statutes:** With references
- **Courts:** With jurisdiction and level
- **Legal Concepts:** For principle tracking
- **Judges:** For authorship attribution

#### Step 8: Relationship Creation
- **Citation relationships:** Document → Case
- **Reference relationships:** Chunk → Document
- **Institutional relationships:** Document → Court, Judge → Document
- **Temporal relationships:** Document → Case (FOLLOWS, based on chronological analysis)

### 2.2 Example Construction Flow

Given `[2019] SGCA 42.pdf`:

1. **Extract metadata:**
   ```json
   {
     "year": 2019,
     "citation": "[2019] SGCA 42",
     "jurisdiction": "Singapore",
     "court_level": "Appellate"
   }
   ```

2. **Create document node** with all metadata

3. **Chunk into ~50 segments** (for typical case length)

4. **Process each chunk:**
   - Generate embedding vector
   - Extract entities (e.g., "Spandeck Engineering" case, "duty of care" concept)
   - Create chunk node with all extracted data

5. **Create entity nodes:**
   ```cypher
   MERGE (case:Case {citation: "Spandeck Engineering (S) Pte Ltd v Defence Science & Technology Agency [2007] SGCA 37"})
   MERGE (concept:LegalConcept {name: "duty of care"})
   MERGE (court:Court {name: "Singapore Court of Appeal"})
   ```

6. **Create relationships:**
   ```cypher
   (current_doc)-[:CITES]->(spandeck_case)
   (current_doc)-[:HEARD_IN]->(sgca_court)
   (current_doc)-[:DISCUSSES]->(duty_of_care_concept)
   ```

## 3. Using the Knowledge Graph

### 3.1 Semantic Retrieval

#### Vector Similarity Search
Find semantically similar chunks:

```cypher
CALL db.index.vector.queryNodes(
  'chunk_embeddings', 
  5, 
  $query_embedding
) 
YIELD node, score
MATCH (node)-[:PART_OF]->(doc:Document)
RETURN node.text, node.summary, doc.citation, score
ORDER BY score DESC
```

#### Combined Semantic + Metadata Filtering
```cypher
CALL db.index.vector.queryNodes('chunk_embeddings', 10, $query_embedding) 
YIELD node, score
MATCH (node)-[:PART_OF]->(doc:Document)
WHERE doc.jurisdiction = 'Singapore' 
  AND doc.year >= 2010
  AND score > 0.7
RETURN node.text, doc.citation, doc.year, score
ORDER BY score DESC
LIMIT 5
```

### 3.2 Legal Research Queries

#### Precedent Discovery
Find cases that establish a legal principle:

```cypher
MATCH (doc:Document)-[:DISCUSSES]->(concept:LegalConcept {name: 'negligence'})
MATCH (chunk:Chunk)-[:PART_OF]->(doc)
WHERE any(holding in chunk.holdings WHERE holding CONTAINS 'duty of care')
RETURN doc.citation, doc.year, chunk.holdings
ORDER BY doc.year
```

#### Cross-Reference Analysis
Find chunks that reference each other to trace legal arguments:

```cypher
MATCH (chunk1:Chunk)-[:REFERENCES_CHUNK {paragraph_ref: '109'}]->(chunk2:Chunk)
MATCH (chunk1)-[:PART_OF]->(doc:Document)
RETURN chunk1.text as referencing_text,
       chunk2.text as referenced_text,
       chunk1.chunk_index as referencing_index,
       chunk2.chunk_index as referenced_index,
       doc.citation
```

#### Argument Flow Tracing
Trace the flow of legal reasoning within a document:

```cypher
MATCH path = (start:Chunk)-[:REFERENCES_CHUNK*1..3]->(end:Chunk)
MATCH (start)-[:PART_OF]->(doc:Document)
WHERE start.id = $starting_chunk_id
RETURN [chunk in nodes(path) | {
    id: chunk.id,
    index: chunk.chunk_index,
    summary: chunk.summary,
    holdings: chunk.holdings
}] as argument_chain,
[rel in relationships(path) | rel.paragraph_ref] as paragraph_refs
ORDER BY length(path)
```

#### Citation Network Analysis
Find most cited cases:

```cypher
MATCH (doc:Document)-[:CITES]->(case:Case)
RETURN case.citation, case.name, count(doc) as citation_count
ORDER BY citation_count DESC
LIMIT 10
```

#### Jurisdictional Comparison
Compare approaches across jurisdictions:

```cypher
MATCH (doc:Document)-[:DISCUSSES]->(concept:LegalConcept {name: $legal_concept})
MATCH (chunk:Chunk)-[:PART_OF]->(doc)
WHERE any(test in chunk.legal_tests WHERE test CONTAINS $concept_keywords)
RETURN doc.jurisdiction, collect(chunk.legal_tests) as approaches
```

#### Temporal Precedent Analysis
Track evolution of legal principle:

```cypher
MATCH (doc:Document)-[:DISCUSSES]->(concept:LegalConcept {name: $concept})
MATCH (chunk:Chunk)-[:PART_OF]->(doc)
WHERE any(holding in chunk.holdings WHERE holding CONTAINS $concept)
RETURN doc.year, doc.citation, chunk.holdings
ORDER BY doc.year
```

#### Authority Hierarchy Analysis
Weight cases by court authority:

```cypher
MATCH (doc:Document)-[:HEARD_IN]->(court:Court)
MATCH (doc)-[:DISCUSSES]->(concept:LegalConcept {name: $concept})
RETURN doc.citation, court.name, court.level,
  CASE court.level 
    WHEN 'Appellate' THEN 3
    WHEN 'Superior' THEN 2 
    ELSE 1 
  END as authority_weight
ORDER BY authority_weight DESC, doc.year DESC
```

### 3.3 Research Assistant Workflows

#### Multi-Stage Precedent Research
```cypher
// Stage 1: Find directly relevant cases
CALL db.index.fulltext.queryNodes('chunk_text_search', $search_terms) 
YIELD node, score
MATCH (node)-[:PART_OF]->(doc:Document)
WHERE score > 0.8 AND doc.type = 'Case'

// Stage 2: Find cases that cite the relevant cases
MATCH (doc)-[:CITES]->(case:Case)<-[:CITES]-(citing_doc:Document)
WHERE citing_doc.year > doc.year

// Stage 3: Expand to related concepts
MATCH (doc)-[:DISCUSSES]->(concept:LegalConcept)<-[:DISCUSSES]-(related_doc:Document)

RETURN DISTINCT related_doc.citation, related_doc.year
ORDER BY related_doc.year DESC
```

#### Fact Pattern Matching
```cypher
MATCH (chunk:Chunk)-[:PART_OF]->(doc:Document)
WHERE any(fact in chunk.facts WHERE fact CONTAINS $key_fact_1)
  AND any(fact in chunk.facts WHERE fact CONTAINS $key_fact_2)
  AND doc.jurisdiction = $target_jurisdiction
RETURN doc.citation, chunk.facts, chunk.holdings
ORDER BY doc.year DESC
```

### 3.4 Argument Generation Support

#### Precedent Strength Assessment
```cypher
MATCH (target_case:Case)<-[:CITES]-(citing_doc:Document)
MATCH (citing_doc)-[:HEARD_IN]->(court:Court)
WITH target_case, 
     count(citing_doc) as total_citations,
     avg(
       CASE court.level 
         WHEN 'Appellate' THEN 3
         WHEN 'Superior' THEN 2 
         ELSE 1 
       END
     ) as avg_authority_weight
RETURN target_case.citation, total_citations, avg_authority_weight,
       (total_citations * avg_authority_weight) as precedent_strength
ORDER BY precedent_strength DESC
```

#### Authority Chain Construction
```cypher
MATCH path = (start:Case)<-[:CITES*1..3]-(end:Document)
WHERE start.year < 2000 AND end.year > 2020
RETURN path, 
       [case in nodes(path) WHERE case:Case | case.citation] as precedent_chain,
       length(path) as chain_length
ORDER BY chain_length
```

#### Counterargument Discovery
```cypher
MATCH (main_concept:LegalConcept {name: $argument_concept})
MATCH (doc:Document)-[:DISCUSSES]->(main_concept)
MATCH (doc)-[:DISCUSSES]->(related_concept:LegalConcept)
WHERE related_concept.name CONTAINS 'exception' 
   OR related_concept.name CONTAINS 'limitation'
   OR related_concept.name CONTAINS 'defence'
MATCH (counter_doc:Document)-[:DISCUSSES]->(related_concept)
MATCH (chunk:Chunk)-[:PART_OF]->(counter_doc)
WHERE any(holding in chunk.holdings WHERE holding CONTAINS related_concept.name)
RETURN counter_doc.citation, related_concept.name, chunk.holdings
```

#### Legal Test Identification
```cypher
MATCH (doc:Document)-[:DISCUSSES]->(concept:LegalConcept {name: $legal_area})
MATCH (chunk:Chunk)-[:PART_OF]->(doc)
WHERE size(chunk.legal_tests) > 0
RETURN DISTINCT chunk.legal_tests, doc.citation, doc.year
ORDER BY doc.year DESC
```

### 3.5 Advanced Analytics

#### Judge-Specific Jurisprudence
```cypher
MATCH (judge:Judge)-[:AUTHORED]->(doc:Document)
MATCH (doc)-[:DISCUSSES]->(concept:LegalConcept)
MATCH (chunk:Chunk)-[:PART_OF]->(doc)
WHERE any(holding in chunk.holdings WHERE holding CONTAINS concept.name)
RETURN judge.name, concept.name, collect(chunk.holdings) as judicial_approach
```

#### Concept Evolution Tracking
```cypher
MATCH (concept:LegalConcept {name: $concept})
MATCH (doc:Document)-[:DISCUSSES]->(concept)
MATCH (chunk:Chunk)-[:PART_OF]->(doc)
WHERE any(holding in chunk.holdings WHERE holding CONTAINS concept.name)
WITH doc.year as year, chunk.holdings as holdings
ORDER BY year
RETURN year, 
       collect(holdings) as evolution_points,
       count(*) as cases_per_year
```

#### Cross-Jurisdictional Influence
```cypher
MATCH (sg_doc:Document {jurisdiction: 'Singapore'})-[:CITES]->(case:Case)
MATCH (other_doc:Document)-[:CITES]->(case)
WHERE other_doc.jurisdiction <> 'Singapore'
RETURN case.citation, 
       count(DISTINCT sg_doc) as singapore_citations,
       count(DISTINCT other_doc) as international_citations,
       collect(DISTINCT other_doc.jurisdiction) as citing_jurisdictions
ORDER BY (singapore_citations + international_citations) DESC
```

## 4. Integration with AI Agents

### 4.1 Research Graph Integration
The knowledge graph supports LangGraph research workflows:

```python
@tool
def vector_search(query: str, jurisdiction: str = None, min_score: float = 0.7) -> List[Dict]:
    """Semantic search for relevant legal precedents."""
    # Generate embedding for query
    # Execute vector similarity search with filters
    # Return ranked results with metadata

@tool  
def expand_citations(case_citations: List[str]) -> List[Dict]:
    """Find documents that cite the given cases."""
    # Query citation network
    # Return citing documents with context
```

### 4.2 Argument Generation Integration
The graph provides structured data for argument construction:

```python
@tool
def assess_precedent_strength(case_citation: str) -> Dict:
    """Assess the precedential strength of a case."""
    # Calculate citation frequency
    # Weight by court authority
    # Check for subsequent treatment (overruling, distinguishing)
    
@tool
def find_supporting_authority(legal_principle: str, jurisdiction: str) -> List[Dict]:
    """Find cases that support a legal principle."""
    # Search by legal concept
    # Filter by jurisdiction and authority level
    # Return with holding summaries
```

## 5. Performance Considerations

### 5.1 Query Optimization
- **Use indexes** for filtered searches
- **Limit result sets** to prevent memory issues
- **Batch vector searches** for multiple queries
- **Cache frequent queries** at application level

### 5.2 Scaling Strategies
- **Incremental ingestion** for new documents
- **Temporal partitioning** for large datasets
- **Federation** across multiple Neo4j instances
- **Read replicas** for query-heavy workloads

### 5.3 Maintenance
- **Regular index maintenance** (`CALL db.index.fulltext.listAvailableAnalyzers()`)
- **Graph statistics updates** (`CALL apoc.stats.degrees()`)
- **Embedding index optimization** (periodic recreation)
- **Entity deduplication** (merge similar entities)

This knowledge graph architecture provides a robust foundation for sophisticated legal research and AI-powered argument generation, enabling the system to understand legal relationships, precedent hierarchies, and conceptual evolution across legal documents.

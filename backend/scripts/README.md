# Knowledge Graph Ingestion Script

This script processes raw PDF documents and populates the Neo4j knowledge graph for the Legal Research Assistant.

## Core Components

### `process_pdf.py`
Enhanced PDF processing module that:
- Extracts text using pdfplumber for better accuracy
- Separates footnotes from main content
- Removes headers and cleans formatting
- Integrates footnotes inline with main text
- Provides structured output for downstream processing

### `ingest_graph.py`
Main ingestion script that:
- Uses `process_pdf.py` for document extraction
- Creates document chunks with overlap
- Generates embeddings for semantic search
- Extracts entities and creates graph relationships
- Populates Neo4j database with structured data

## Prerequisites

1. **Neo4j Database**: Ensure Neo4j is running and accessible
2. **Google API Key**: Set up your Google Generative AI credentials
3. **Python Dependencies**: Install required packages

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

3. Ensure Neo4j is running:
```bash
# For local Neo4j installation
neo4j start
```

## Usage

### Basic Usage
```bash
python scripts/ingest_graph.py
```

### Reset Database and Ingest
```bash
python scripts/ingest_graph.py --reset
```

### Custom Documents Directory
```bash
python scripts/ingest_graph.py --docs-dir /path/to/your/documents
```

### Direct PDF Processing (Testing)
```bash
python scripts/process_pdf.py
```

## Document Structure

Place your PDF documents in the `data/raw_docs/` directory (or specify a custom path). The script will:

1. Extract text from each PDF
2. Split documents into overlapping chunks
3. Generate embeddings for semantic search
4. Extract legal entities (cases, statutes, concepts)
5. Create summaries using LLM
6. Build relationships between documents and chunks

## Neo4j Schema

The script creates the following schema:

### Nodes
- `Document`: Represents a legal document
  - Properties: `source`, `type`, `full_text`, `file_path`, `word_count`, `created_at`
- `Chunk`: Represents a text chunk from a document
  - Properties: `id`, `text`, `summary`, `embedding`, `chunk_index`, `entities`, `created_at`

### Relationships
- `(Chunk)-[:PART_OF]->(Document)`: Links chunks to their parent document
- `(Chunk)-[:REFERENCES]->(Document)`: Links chunks that reference other documents

### Indexes
- Vector index on `Chunk.embedding` for semantic search
- Unique constraints on `Document.source` and `Chunk.id`

## Document Types

The script automatically categorizes documents based on filename and content:
- **Case**: Legal cases, judgments, opinions
- **Doctrine**: Legal treatises, commentary, doctrinal texts
- **Document**: General legal documents

## Error Handling

- Failed PDF extractions are logged and skipped
- Embedding generation failures are retried
- Database connection issues halt the process
- Individual document failures don't stop the entire ingestion

## Performance Considerations

- Documents are processed sequentially to avoid API rate limits
- Embeddings are generated in batches for efficiency
- Large documents are chunked to fit within token limits
- Progress is logged for monitoring

## Monitoring

The script provides detailed logging:
- Document processing progress
- Chunk creation statistics
- Error reporting
- Final ingestion summary

Check the console output for real-time progress and any issues that occur during ingestion.

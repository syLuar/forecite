# Forecite Legal Research Assistant

A sophisticated AI-powered legal research assistant that combines advanced LLM capabilities with a comprehensive knowledge graph of legal documents and case law. Built with LangGraph agent workflows, Neo4j knowledge graphs, and modern React frontend, Forecite provides intelligent legal research, argument drafting, and moot court simulation capabilities.

## 🚀 Quick Start

The fastest way to get both the backend and frontend servers running:

```bash
./quickstart.sh
```

This script will:
- ✅ Check all prerequisites (Python 3.11+, Node.js, npm)
- ✅ Automatically install uv if not present
- ✅ Set up Python virtual environment using uv in project root (.venv)
- ✅ Install all backend dependencies with uv
- ✅ Install all frontend dependencies
- ✅ Start the FastAPI backend server on port 8000
- ✅ Start the React frontend development server on port 3000
- ✅ Provide helpful status information and URLs

### What you'll get:
- **Frontend Application**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Stopping the servers:
Press `Ctrl+C` in the terminal where the script is running, or manually kill the processes using the PIDs displayed.

## 📋 Prerequisites

Before running the quickstart script, ensure you have:

- **Python 3.11 or higher** installed
- **Node.js 16 or higher** installed
- **npm** package manager
- **Git** for cloning the repository

Note: `uv` will be automatically installed by the quickstart script if not present.

## ⚙️ Manual Setup

If you prefer to set up the components manually or need more control:

### Backend Setup
```bash
cd backend
uv venv ../.venv  # Create venv in project root
source ../.venv/bin/activate  # On Windows: ..\.venv\Scripts\activate
uv pip install -r requirements.txt
cp .env.example .env  # Configure your API keys and database settings
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 🏗️ Architecture

This project consists of:

- **Backend** (`/backend`): FastAPI-based server with LangGraph agent workflows for legal research and argument generation
- **Frontend** (`/frontend`): React TypeScript application with modern UI built using Create React App
- **Documentation** (`/docusaurus`): Comprehensive project documentation built with Docusaurus v3
- **Knowledge Graph**: Neo4j database (hosted on VPS) storing legal documents, cases, and relationships
- **AI/LLM**: Google Gemini and OpenAI integration for legal research, drafting, and analysis
- **Vector Search**: Embedded document chunks for semantic similarity search
- **Caching System**: Optimized caching for embeddings, entities, knowledge graphs, and summaries

### Key Technologies
- **Backend**: FastAPI, LangGraph, LangChain, Pydantic, Uvicorn
- **Frontend**: React 18, TypeScript, Tailwind CSS, Vite build system
- **Database**: Neo4j (graph database), Vector embeddings
- **AI/ML**: Google Gemini, OpenAI GPT, LangSmith for tracing
- **Documentation**: Docusaurus v3, MDX, TypeScript
- **DevOps**: Python virtual environments, npm/Node.js package management

## ✨ Key Features

### 🔍 Legal Research
- **Intelligent Case Law Search**: Semantic search through legal documents using vector embeddings
- **Knowledge Graph Navigation**: Explore relationships between cases, statutes, and legal concepts
- **Citation Analysis**: Automatic extraction and analysis of legal citations and precedents
- **Document Processing**: Advanced PDF processing and text extraction for legal documents

### 📝 Argument Drafting
- **AI-Powered Legal Writing**: Generate legal arguments based on research findings
- **Template-Based Generation**: Structured legal document creation with customizable templates
- **Citation Integration**: Automatic integration of relevant case law and statutes
- **Multi-Format Export**: Support for various legal document formats

### 🎯 Moot Court Simulation
- **Interactive Practice**: Simulate moot court proceedings with AI-generated scenarios
- **Argument Evaluation**: Get feedback on legal arguments and presentation
- **Case Strategy Development**: Plan and refine legal strategies through simulation

### 🤖 LangGraph Agent Workflows
- **Research Agent**: Autonomous legal research with self-correction capabilities
- **Drafting Agent**: Intelligent legal document generation and refinement
- **Analysis Agent**: Deep analysis of legal precedents and case law
- **Coordination Agent**: Orchestrates multi-agent workflows for complex tasks

## 📚 Documentation

- **[Backend Documentation](./backend/README.md)** - Detailed installation guide, API documentation, and LangGraph agent architecture
- **[Frontend Documentation](./frontend/README.md)** - React app setup, development guide, and component structure
- **[Interactive Documentation Site](./docusaurus/README.md)** - Comprehensive project documentation with user guides and features

### Running Documentation Site

The Docusaurus documentation site is **separate** from the quickstart script and requires manual setup:

```bash
# Install documentation dependencies
cd docusaurus
npm install

# Start documentation development server
npm run start
# Opens at http://localhost:3001

# Or build and serve for production
npm run build
npm run serve
```

**Note**: The quickstart script (`./quickstart.sh`) only starts the backend and frontend servers. Documentation must be served separately.

The documentation includes:
- 📖 **User Guide**: Complete walkthrough of features and workflows
- 🔧 **API Reference**: Backend endpoints and integration examples  
- 🎯 **Features Overview**: Detailed explanation of legal research capabilities
- 🚀 **Getting Started**: Quick setup and configuration guide

## 🛠️ Development

For development work, see the individual README files in the `backend` and `frontend` directories for detailed setup instructions, testing, and deployment information.

### Project Structure
```
forecite/
├── backend/                 # FastAPI backend with LangGraph agents
│   ├── app/                # Application modules
│   │   ├── core/          # Core business logic
│   │   ├── graphs/        # LangGraph agent definitions
│   │   ├── models/        # Data models and schemas
│   │   ├── services/      # Service layer implementations
│   │   └── tools/         # Agent tools and utilities
│   ├── scripts/           # Database and utility scripts
│   ├── static/docs/       # Static documentation files
│   └── requirements.txt   # Python dependencies
├── frontend/               # React TypeScript frontend
│   ├── src/               # Source code
│   ├── public/            # Static assets
│   └── build/             # Production build output
├── docusaurus/            # Documentation website
│   ├── docs/              # Documentation content
│   ├── src/               # Custom components and pages
│   └── static/            # Static assets
├── demo/                  # Demo and testing scripts
├── data/                  # Sample legal documents and cases
├── cache/                 # Application caches (embeddings, entities, etc.)
├── docusaurus/            # Documentation website (served separately)
└── package.json           # Root package dependencies
```

### Development Workflow
1. **Backend Development**: Use `uvicorn main:app --reload` for hot reload during development
2. **Frontend Development**: Use `npm start` for React development server with hot reload
3. **Documentation Updates**: Use `npm run start` in `/docusaurus` for live documentation editing
4. **Database Setup**: Run scripts in `/backend/scripts/` for database initialization and migration
5. **Testing**: Use demo scripts in `/demo/` for end-to-end testing

### Available Scripts & Tools
- **Database Setup**: `python backend/scripts/setup_database.py` - **Essential first step** for initializing the Neo4j database
- **Database Migration**: `python backend/scripts/db_migration.py` - Migrate database schema and data
- **Graph Ingestion**: `python backend/scripts/ingest_graph.py` - Populate knowledge graph with legal documents
- **PDF Processing**: `python backend/scripts/process_pdf.py` - Extract and process legal documents
- **Graph Export**: `python backend/scripts/export_graphs_as_mermaid_pngs.py` - Export graph visualizations
- **Case Scraping**: `python backend/scripts/scrape_singapore_cases.py` - Scrape legal cases from sources
- **Demo Testing**: `python demo/fast_demo.py` - Run end-to-end system tests

## 🔧 Configuration

The quickstart script will copy `.env.example` to `.env` in the backend directory. Make sure to configure:

### Required Environment Variables
- `GOOGLE_API_KEY`: Your Google Gemini API key for LLM operations
- `OPENAI_API_KEY`: OpenAI API key for additional LLM capabilities
- `NEO4J_URI`: Your VPS Neo4j database connection (bolt://your-server:7687)
- `NEO4J_USERNAME` and `NEO4J_PASSWORD`: Neo4j credentials
- `NEO4J_DATABASE`: Database name (default: neo4j)

### Optional Configuration
- `LANGSMITH_TRACING`: Enable LangSmith tracing for debugging (default: true)
- `LANGSMITH_API_KEY`: LangSmith API key for workflow monitoring
- `CHUNK_SIZE`: Document chunk size for embeddings (default: 2500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `VECTOR_INDEX_NAME`: Neo4j vector index name (default: chunk_embeddings)

### Demo Configuration
A separate demo configuration is available in `/demo` with its own setup script and requirements for testing purposes.

## 📊 Data & Caching

### Knowledge Base
The knowledge graph is populated with:
- Legal case precedents and citations
- Statutory references and regulations
- Legal concept relationships and hierarchies
- Document metadata and classifications

## 🐛 Troubleshooting

If the quickstart script fails:

1. **Port conflicts**: The script automatically cleans up ports 3000 and 8000
2. **Missing dependencies**: Ensure Python 3.11+ and Node.js 16+ are installed
3. **Permission issues**: Make sure the script is executable: `chmod +x quickstart.sh`
4. **Environment issues**: Check that `.env` file is properly configured

For detailed troubleshooting, refer to the [Backend README](./backend/README.md).

## 🚀 Production Deployment

### Backend Deployment
- Use `uvicorn main:app --host 0.0.0.0 --port 8000` for production
- Configure environment variables for production database and API keys
- Consider using process managers like PM2 or systemd for service management
- Set up reverse proxy (nginx) for SSL and load balancing

### Frontend Deployment
- Build production bundle: `npm run build` in `/frontend`
- Serve static files through nginx or similar web server
- Configure environment variables for production API endpoints

### Documentation Deployment
- Build documentation: `npm run build` in `/docusaurus`
- Deploy static files to web hosting platform
- Configure custom domain and SSL if needed

### Database Considerations
- Ensure Neo4j database is properly secured and backed up
- Configure appropriate indexes for optimal query performance
- Monitor database health and storage usage
- Set up regular backups of knowledge graph data

## 📧 Support & Contributing

For issues, feature requests, or contributions, please refer to the project repository and documentation.

### Repository Information
- **Repository**: [forecite](https://github.com/syLuar/forecite)
- **Current Branch**: main
- **License**: Check repository for license information
- **Documentation**: Available in `/docusaurus` and component READMEs

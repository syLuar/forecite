# Forecite Legal Research Assistant

A sophisticated AI-powered legal research assistant that combines advanced LLM capabilities with a comprehensive knowledge graph of legal documents and case law.

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

- **Backend** (`/backend`): FastAPI-based server with LangGraph agent workflows
- **Frontend** (`/frontend`): React TypeScript application
- **Knowledge Graph**: Neo4j database (hosted on VPS)
- **AI/LLM**: Google Gemini integration for legal research and drafting

## 📚 Documentation

- [Backend Documentation](./backend/README.md) - Detailed installation guide and API documentation
- [Frontend Documentation](./frontend/README.md) - React app setup and development guide

## 🛠️ Development

For development work, see the individual README files in the `backend` and `frontend` directories for detailed setup instructions, testing, and deployment information.

## 🔧 Configuration

The quickstart script will copy `.env.example` to `.env` in the backend directory. Make sure to configure:

- `GOOGLE_API_KEY`: Your Google Gemini API key
- `NEO4J_URI`: Your VPS Neo4j database connection
- `NEO4J_USERNAME` and `NEO4J_PASSWORD`: Neo4j credentials

## 🐛 Troubleshooting

If the quickstart script fails:

1. **Port conflicts**: The script automatically cleans up ports 3000 and 8000
2. **Missing dependencies**: Ensure Python 3.11+ and Node.js 16+ are installed
3. **Permission issues**: Make sure the script is executable: `chmod +x quickstart.sh`
4. **Environment issues**: Check that `.env` file is properly configured

For detailed troubleshooting, refer to the [Backend README](./backend/README.md).

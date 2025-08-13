#!/bin/bash

# Forecite Legal Research Assistant - Quick Start Script
# This script starts both the backend FastAPI server and frontend React development server

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect operating system
detect_os() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "macos";;
        CYGWIN*|MINGW*|MSYS*) echo "windows";;
        *)          echo "unknown";;
    esac
}

# Function to check if a port is in use
port_in_use() {
    local port=$1
    local os=$(detect_os)
    
    if [ "$os" = "windows" ]; then
        netstat -an | grep ":$port " | grep "LISTENING" >/dev/null 2>&1
    else
        lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1
    fi
}

# Function to kill processes on specific ports
cleanup_ports() {
    print_status "Cleaning up any existing processes..."
    local os=$(detect_os)
    
    if port_in_use 8000; then
        print_warning "Port 8000 is in use, attempting to free it..."
        if [ "$os" = "windows" ]; then
            netstat -ano | grep ":8000 " | grep "LISTENING" | awk '{print $5}' | xargs taskkill /PID /F 2>/dev/null || true
        else
            lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        fi
    fi
    
    if port_in_use 3000; then
        print_warning "Port 3000 is in use, attempting to free it..."
        if [ "$os" = "windows" ]; then
            netstat -ano | grep ":3000 " | grep "LISTENING" | awk '{print $5}' | xargs taskkill /PID /F 2>/dev/null || true
        else
            lsof -ti:3000 | xargs kill -9 2>/dev/null || true
        fi
    fi
    
    sleep 2
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    local os=$(detect_os)
    
    # Check if Python is installed
    if [ "$os" = "windows" ]; then
        if ! command_exists python; then
            print_error "Python is not installed. Please install Python 3.11 or higher."
            exit 1
        fi
        
        # Check Python version on Windows
        python_version=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    else
        if ! command_exists python3; then
            print_error "Python 3 is not installed. Please install Python 3.11 or higher."
            exit 1
        fi
        
        # Check Python version on Unix
        python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    fi
    
    required_version="3.11"
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        print_error "Python $required_version or higher is required. Found: $python_version"
        exit 1
    fi
    
    # Check if Node.js is installed
    if ! command_exists node; then
        print_error "Node.js is not installed. Please install Node.js 16 or higher."
        exit 1
    fi
    
    # Check if npm is installed
    if ! command_exists npm; then
        print_error "npm is not installed. Please install npm."
        exit 1
    fi
    
    # Check if uv is installed
    if ! command_exists uv; then
        print_warning "uv is not installed. Installing uv automatically..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
        # Source the shell profile to make uv available in current session
        if [ -f "$HOME/.cargo/env" ]; then
            source "$HOME/.cargo/env"
        fi
        
        # Check if uv is now available
        if ! command_exists uv; then
            print_error "Failed to install uv. Please install manually: pip install uv"
            exit 1
        fi
        
        print_success "uv installed successfully!"
    fi
    
    print_success "All prerequisites met!"
}

# Function to setup backend
setup_backend() {
    print_status "Setting up backend..."
    local os=$(detect_os)
    
    # Check if virtual environment exists in project root
    if [ ! -d "backend/.venv" ]; then
        print_status "Creating Python virtual environment with uv in project root..."
        if [ "$os" = "windows" ]; then
            uv venv backend/.venv --python python
        else
            uv venv backend/.venv --python python3
        fi
    fi
    
    # Activate virtual environment based on OS
    print_status "Activating virtual environment..."
    if [ "$os" = "windows" ]; then
        source backend/.venv/Scripts/activate
    else
        source backend/.venv/bin/activate
    fi
    
    # Install dependencies using uv
    print_status "Installing backend dependencies with uv..."
    cd backend
    uv pip install -r requirements.txt

    # Run idempotent DB migration before starting services
    if [ -f "scripts/db_migration.py" ]; then
        print_status "Running database migration..."
        if [ "$os" = "windows" ]; then
            python scripts/db_migration.py || print_warning "Migration script encountered an issue; proceeding."
        else
            python3 scripts/db_migration.py || print_warning "Migration script encountered an issue; proceeding."
        fi
    fi

    # Check if .env file exists
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_warning ".env file not found. Copying from .env.example..."
            cp .env.example .env
            print_warning "Please configure your .env file with proper API keys and database credentials!"
        else
            print_error ".env.example file not found. Please create a .env file manually."
            exit 1
        fi
    fi
    
    cd ..
    print_success "Backend setup complete!"
}

# Function to setup frontend
setup_frontend() {
    print_status "Setting up frontend..."
    
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        print_status "Installing frontend dependencies..."
        npm install
    else
        print_status "Frontend dependencies already installed, checking for updates..."
        npm install
    fi
    
    cd ..
    print_success "Frontend setup complete!"
}

# Function to start backend server
start_backend() {
    print_status "Starting backend server on port 8000..."
    local os=$(detect_os)
    
    # Activate virtual environment from project root
    if [ "$os" = "windows" ]; then
        source backend/.venv/Scripts/activate
    else
        source backend/.venv/bin/activate
    fi
    
    cd backend
    # Start FastAPI server in background
    if [ "$os" = "windows" ]; then
        python main.py &
    else
        python3 main.py &
    fi
    BACKEND_PID=$!
    
    # Wait a bit and check if the backend started successfully
    sleep 3
    if kill -0 $BACKEND_PID 2>/dev/null; then
        print_success "Backend server started successfully! (PID: $BACKEND_PID)"
        print_status "Backend API available at: http://localhost:8000"
        print_status "API Documentation available at: http://localhost:8000/docs"
    else
        print_error "Failed to start backend server!"
        exit 1
    fi
    
    cd ..
}

# Function to start frontend server
start_frontend() {
    print_status "Starting frontend development server on port 3000..."
    
    cd frontend
    
    # Start React development server in background
    npm start &
    FRONTEND_PID=$!
    
    # Wait a bit and check if the frontend started successfully
    sleep 5
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        print_success "Frontend server started successfully! (PID: $FRONTEND_PID)"
        print_status "Frontend application available at: http://localhost:3000"
    else
        print_error "Failed to start frontend server!"
        exit 1
    fi
    
    cd ..
}

# Function to display final status
show_status() {
    echo ""
    echo "======================================"
    print_success "🚀 Forecite Legal Research Assistant is now running!"
    echo "======================================"
    echo ""
    print_status "📱 Frontend:     http://localhost:3000"
    print_status "🔧 Backend API:  http://localhost:8000"
    print_status "📚 API Docs:     http://localhost:8000/docs"
    echo ""
    print_status "Backend PID: $BACKEND_PID"
    print_status "Frontend PID: $FRONTEND_PID"
    echo ""
    print_warning "To stop the servers, press Ctrl+C or run:"
    print_warning "kill $BACKEND_PID $FRONTEND_PID"
    echo ""
}

# Function to handle cleanup on exit
cleanup() {
    echo ""
    print_status "Shutting down servers..."
    
    if [ ! -z "$BACKEND_PID" ] && kill -0 $BACKEND_PID 2>/dev/null; then
        kill $BACKEND_PID
        print_status "Backend server stopped"
    fi
    
    if [ ! -z "$FRONTEND_PID" ] && kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID
        print_status "Frontend server stopped"
    fi
    
    print_success "Cleanup complete!"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    echo "======================================"
    echo "🏛️  Forecite Legal Research Assistant"
    echo "======================================"
    echo ""
    
    # Check if we're in the right directory
    if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
        print_error "This script must be run from the root directory of the Forecite project!"
        print_error "Expected to find 'backend' and 'frontend' directories."
        exit 1
    fi
    
    check_prerequisites
    cleanup_ports
    setup_backend
    setup_frontend
    start_backend
    start_frontend
    show_status
    
    # Keep script running and wait for user interrupt
    print_status "Press Ctrl+C to stop both servers..."
    wait
}

# Run main function
main "$@"

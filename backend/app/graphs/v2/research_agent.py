"""
Legal Research ReAct Agent

This module implements a ReAct (Reasoning + Acting) agent specifically designed for
conducting detailed legal research, retrieving relevant documents, and organizing
findings in case files. The agent focuses on research and document collection, 
leaving argument drafting to specialized downstream agents.

Uses LangGraph's create_react_agent for reliable and modern agent implementation.

## Agent Capabilities

### 1. Legal Research
- Semantic search across legal document chunks
- Case law analysis by citation patterns
- Statutory reference research
- Fact pattern matching
- Legal holdings extraction

### 2. Document Management
- Add discovered documents to case files
- Organize research findings with relevance scores
- Attach research notes and key holdings
- Track citation networks and precedent chains
- Create and manage AI-generated research notes

### 3. Research Strategy
- Multi-step research planning
- Evidence gap identification
- Research quality assessment
- Strategic research note generation
- Document research insights and findings

## Usage Example

```python
from app.graphs.v2.research_agent import LegalResearchAgent

# Initialize agent
agent = LegalResearchAgent()

# Conduct research with AI-identified legal issues
research_notes = agent.research_case(
    case_file_id=123,
    case_facts="Client's contract was breached...",
    party_represented="Plaintiff"
    # legal_issues will be automatically identified by the AI
)

# Or provide specific legal issues if desired
research_notes = agent.research_case(
    case_file_id=123,
    case_facts="Client's contract was breached...",
    party_represented="Plaintiff",
    legal_issues=["breach of contract", "damages", "mitigation"]
)

print(research_notes)

# The agent will automatically:
# 1. Identify legal issues from case facts (if not provided)
# 2. Find relevant legal authorities and add them to the case file
# 3. Create research notes documenting insights and findings
# 4. Organize research with proper tagging and categorization
```

## Technical Implementation

This agent uses LangGraph's `create_react_agent` function which provides:
- Built-in ReAct prompting pattern
- Proper tool calling integration
- State management
- Error handling
- Streaming support

The agent expects messages in the format:
```python
{"messages": [{"role": "user", "content": "your prompt"}]}
```

And returns results in the format:
```python
{"messages": [...]}  # List of message objects
```
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid
import re
from pydantic import BaseModel, Field

# ReAct framework imports  
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool

# Local imports
from app.core.llm import create_llm
from app.core.config import settings
from app.tools.research_tools import (
    semantic_search_legal_content,
    extract_legal_information,
    find_cases_by_fact_pattern,
    search_statute_references,
    analyze_citation_network,
    validate_legal_statement
)
from app.tools.database_tools import add_document_tool, add_ai_note_tool
from langgraph.config import get_stream_writer

# Configure logging
logger = logging.getLogger(__name__)


class ResearchPlan(BaseModel):
    """Structure for research planning."""
    legal_issues: List[str] = Field(description="List of legal issues to research")
    search_strategies: List[str] = Field(description="Research strategies to employ")
    priority_order: List[int] = Field(description="Priority order for issues")
    expected_authorities: List[str] = Field(description="Expected types of authorities needed")


class ResearchFindings(BaseModel):
    """Structure for organizing research findings."""
    relevant_cases: List[Dict[str, Any]] = Field(default_factory=list)
    key_holdings: List[str] = Field(default_factory=list)
    statutory_authorities: List[Dict[str, Any]] = Field(default_factory=list)
    research_gaps: List[str] = Field(default_factory=list)
    strategic_notes: str = Field(default="")


# Input schemas for StructuredTool implementations
class IdentifyLegalIssuesInput(BaseModel):
    """Input schema for identifying legal issues."""
    case_facts: str = Field(description="Factual background of the case to analyze for legal issues")


class SemanticSearchInput(BaseModel):
    """Input schema for semantic search."""
    query: str = Field(description="Search query for legal content")
    top_k: int = Field(default=10, description="Maximum number of results to return")


class FactPatternSearchInput(BaseModel):
    """Input schema for fact pattern search."""
    fact_description: str = Field(description="Description of the factual scenario to find similar cases")
    key_facts: Optional[List[str]] = Field(default=None, description="Specific key facts to match")
    legal_context: Optional[str] = Field(default=None, description="Legal context to filter by")
    similarity_threshold: float = Field(default=0.6, description="Minimum similarity for inclusion")


class StatuteSearchInput(BaseModel):
    """Input schema for statute reference search."""
    statute_reference: str = Field(description="Statute or regulation reference to search for")
    section: Optional[str] = Field(default=None, description="Specific section or subsection")


class LegalHoldingsInput(BaseModel):
    """Input schema for extracting legal holdings."""
    legal_issue: str = Field(description="Legal issue to extract holdings for")


class CitationNetworkInput(BaseModel):
    """Input schema for citation network analysis."""
    case_citation: str = Field(description="Case citation to analyze citation network for")
    direction: str = Field(default="both", description="Direction of citation analysis: 'cited_by', 'cites', or 'both'")


class ValidateLegalStatementInput(BaseModel):
    """Input schema for validating legal statements."""
    legal_proposition: str = Field(description="Legal proposition to validate against authorities")
    jurisdiction: Optional[str] = Field(default=None, description="Relevant jurisdiction")
    confidence_level: str = Field(default="medium", description="Required confidence level")


class AddDocumentInput(BaseModel):
    """Input schema for adding documents to case file."""
    case_file_id: int = Field(description="ID of the case file to add document to")
    chunk_id: str = Field(description="Unique identifier for the chunk")
    relevance_score_percent: Optional[float] = Field(default=None, description="Relevance score from 0-100")
    user_notes: Optional[str] = Field(default=None, description="Notes about the document's relevance")


class AddResearchNoteInput(BaseModel):
    """Input schema for adding research notes."""
    case_file_id: int = Field(description="ID of the case file to add note to")
    content: str = Field(description="Content of the research note")
    note_type: str = Field(default="research", description="Type of note: 'research', 'strategy', 'fact', 'analysis', etc.")
    tags: Optional[List[str]] = Field(default=None, description="Tags to categorize the note")


class LegalResearchAgent:
    """
    ReAct agent specialized for legal research and document collection.
    
    This agent conducts systematic legal research, identifies relevant authorities,
    organizes findings in case files, and documents insights through research notes
    for use by downstream drafting agents.
    
    Uses the modern LangChain create_react_agent framework for reliable performance.
    """
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the legal research agent.
        
        Args:
            llm_config: Optional LLM configuration override
        """
        # Initialize LLM
        self.llm = create_llm(settings.llm_config.get("main", {}).get("research_agent", {}))

        # Setup tools for the agent
        self.tools = self._setup_tools()
        
        # Create the ReAct agent using LangGraph
        self.agent_executor = create_react_agent(self.llm, self.tools, pre_model_hook=self._pre_model_hook, post_model_hook=self._post_model_hook)

    def _pre_model_hook(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Pre-model hook to add a custom stream update for the frontend.
        Args:
            messages: List of messages in the conversation
        """
        step_id = "researcher"
        writer = get_stream_writer()
        writer({
            "step_id": step_id,
            "status": "in_progress",
            "brief_description": "Planning research strategy",
            "description": "Developing the research strategy and approach based on identified issues."
        })

        return messages
    
    def _post_model_hook(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Post-model hook to add a custom stream update for the frontend.
        
        Args:
            messages: List of messages in the conversation
        """
        step_id = "researcher"
        writer = get_stream_writer()
        writer({
            "step_id": step_id,
            "status": "complete",
            "brief_description": "Executed research strategy",
            "description": "Executed research based on identified issues."
        })

        return messages

    def _setup_tools(self) -> List[StructuredTool]:
        """Setup tools available to the ReAct agent."""
        return [
            StructuredTool.from_function(
                func=self._identify_legal_issues,
                name="identify_legal_issues",
                description=(
                    "Analyze case facts to identify potential legal issues and claims. "
                    "Use this when legal issues are not explicitly provided to systematically "
                    "extract all potential legal questions from the factual scenario."
                ),
                args_schema=IdentifyLegalIssuesInput
            ),
            
            StructuredTool.from_function(
                func=self._semantic_search_wrapper,
                name="semantic_search_legal_content",
                description=(
                    "Search for legal content using semantic similarity. "
                    "Use this to find cases and documents related to legal concepts, "
                    "fact patterns, or legal questions."
                ),
                args_schema=SemanticSearchInput
            ),
            
            StructuredTool.from_function(
                func=find_cases_by_fact_pattern,
                name="find_cases_by_fact_pattern",
                description=(
                    "Find cases with similar factual scenarios. "
                    "Use this to identify precedents with comparable facts."
                ),
                args_schema=FactPatternSearchInput
            ),
            
            StructuredTool.from_function(
                func=search_statute_references,
                name="search_statute_references",
                description=(
                    "Find cases that reference specific statutes or regulations. "
                    "Use this to research statutory interpretation and application."
                ),
                args_schema=StatuteSearchInput
            ),
            
            StructuredTool.from_function(
                func=self._extract_legal_holdings_wrapper,
                name="extract_legal_holdings",
                description=(
                    "Extract legal holdings and principles from cases. "
                    "Use this to identify key legal principles and rules."
                ),
                args_schema=LegalHoldingsInput
            ),
            
            StructuredTool.from_function(
                func=analyze_citation_network,
                name="analyze_citation_network",
                description=(
                    "Analyze how cases cite each other to understand precedent relationships. "
                    "Use this to trace legal development and case influence."
                ),
                args_schema=CitationNetworkInput
            ),
            
            StructuredTool.from_function(
                func=validate_legal_statement,
                name="validate_legal_statement",
                description=(
                    "Validate a legal proposition against available authorities. "
                    "Use this to verify legal statements and identify supporting authority."
                ),
                args_schema=ValidateLegalStatementInput
            ),
            
            StructuredTool.from_function(
                func=add_document_tool,
                name="add_document_to_case_file",
                description=(
                    "Add a discovered document to the case file. "
                    "Use this to save relevant cases and authorities you find during research. "
                    "Important: The user notes should be concise and focused on the relevance of the document."
                ),
                args_schema=AddDocumentInput
            ),
            
            StructuredTool.from_function(
                func=self._add_research_note_wrapper,
                name="add_research_note",
                description=(
                    "Add a research note to the case file. Use this to save insights, "
                    "observations, research strategies, analysis, or other important findings "
                    "discovered during legal research. "
                    "Important: These research notes should be designed for a HUMAN LAWYER to "
                    "easily reference and utilize in their case work. "
                    "DO NOT include any code here."
                ),
                args_schema=AddResearchNoteInput
            )
        ]
    
    def _semantic_search_wrapper(self, query: str, top_k: int = 10) -> str:
        """Wrapper for semantic search with proper input handling."""
        return semantic_search_legal_content(query, top_k=top_k)
    
    def _extract_legal_holdings_wrapper(self, legal_issue: str) -> str:
        """Wrapper for extracting legal holdings with proper input handling."""
        return extract_legal_information("holdings", legal_issue=legal_issue)
    
    def _add_research_note_wrapper(
        self,
        case_file_id: int,
        content: str,
        note_type: str = "research",
        tags: Optional[List[str]] = None
    ) -> str:
        """Wrapper for adding research notes with proper input handling."""
        return add_ai_note_tool(
            case_file_id=case_file_id,
            content=content,
            author_name="Forecite AI",
            note_type=note_type,
            tags=tags
        )
    
    
    def _identify_legal_issues(self, case_facts: str) -> str:
        """
        Analyze case facts to identify potential legal issues and claims.
        
        Args:
            case_facts: Factual background of the case
            
        Returns:
            JSON string with identified legal issues
        """
        step_id = f"issue_identifier_{uuid.uuid4().hex[:8]}"
        writer = get_stream_writer()
        writer({
            "step_id": step_id,
            "status": "in_progress",
            "brief_description": "Identifying legal issues",
            "description": "Analyzing case facts to extract potential legal issues and claims."
        })
        try:
            # Create a prompt for legal issue identification
            prompt = f"""
            Analyze the following case facts and identify all potential legal issues, claims, and causes of action that may arise:

            Case Facts:
            {case_facts}

            Please identify:
            1. Primary legal issues (main claims and causes of action)
            2. Secondary legal issues (potential defenses, counterclaims, or related matters)
            3. Procedural issues (if any)
            4. Remedy/relief issues (damages, injunctions, etc.)

            Return your analysis as a JSON object with the following structure:
            {{
                "primary_issues": ["list of main legal issues"],
                "secondary_issues": ["list of secondary issues"],
                "procedural_issues": ["list of procedural issues"],
                "remedy_issues": ["list of remedy/relief issues"],
                "analysis": "Brief explanation of the legal landscape"
            }}
            """
            
            # Use the LLM to identify legal issues
            response = self.llm.invoke(prompt)
            
            # Try to extract JSON from the response
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Find JSON in the response (it might be wrapped in markdown or other text)
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)

            writer({
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Legal issues identified",
                "description": "Successfully identified legal issues from case facts."
            })

            if json_match:
                json_str = json_match.group()
                # Validate the JSON
                json.loads(json_str)  # This will raise an exception if invalid
                return json_str
            else:
                # Fallback: return a simple structure with the analysis
                return json.dumps({
                    "primary_issues": ["Legal issues identified from case facts"],
                    "secondary_issues": [],
                    "procedural_issues": [],
                    "remedy_issues": [],
                    "analysis": content
                })
                
        except Exception as e:
            writer({
                "step_id": step_id,
                "status": "complete",
                "brief_description": "Error identifying legal issues",
                "description": f"Failed to identify legal issues: {str(e)}"
            })
            logger.error(f"Error identifying legal issues: {e}")
            return json.dumps({
                "error": f"Failed to identify legal issues: {str(e)}",
                "primary_issues": ["Manual issue identification required"],
                "secondary_issues": [],
                "procedural_issues": [],
                "remedy_issues": [],
                "analysis": "An error occurred during automatic legal issue identification."
            })
    
    def get_initial_state(
        self,
        case_file_id: int,
        case_facts: str,
        party_represented: str,
        legal_issues: Optional[List[str]] = None,
        jurisdiction: str = "Singapore",
    ) -> str:
        """
        Get initial state for the research agent.
        
        Args:
            case_file_id: ID of the case file to organize research in
            case_facts: Factual background of the case
            party_represented: Which party the research supports
            legal_issues: Optional list of legal issues to research. If not provided, 
                         the AI will identify legal issues from the case facts.
            jurisdiction: Relevant jurisdiction
            
        Returns:
            Research notes summarizing findings and strategic insights
        """
        
        # Handle legal issues - either provided or to be identified by AI
        if legal_issues:
            legal_issues_section = f"""
        ## Legal Issues to Research:
        {', '.join(legal_issues)}"""
            step1_instructions = """### Step 1: Research Planning
        - Analyze the provided legal issues and identify key legal concepts for each
        - Plan search strategies (semantic search, fact patterns, statutory references)
        - Prioritize issues based on case strength and client needs
        """
        else:
            legal_issues_section = """
        ## Legal Issues to Research:
        FIRST IDENTIFY the legal issues from the case facts below. Analyze the factual scenario to determine what legal questions and claims arise."""
            step1_instructions = """### Step 1: Issue Identification and Research Planning
        - FIRST: Use the identify_legal_issues tool to systematically analyze the case facts and identify all potential legal issues and claims
        - Document the identified legal issues using add_research_note with note_type='analysis'
        - Then identify key legal concepts for each issue you've identified
        - Plan search strategies (semantic search, fact patterns, statutory references)
        - Prioritize issues based on case strength and client needs
        """
        
        # Create detailed research prompt
        research_prompt = f"""
        You are a expert legal researcher conducting detailed research for a case. Your job is to:
        
        1. IDENTIFY LEGAL ISSUES (if not provided): Analyze case facts to determine legal questions
        2. RESEARCH RELEVANT AUTHORITIES: Use your tools to find cases, and add at most 5 documents and 3 notes to the case file
        3. GENERATE CONCLUSIONS: Provide insights for argument development
        
        ## Case Information:
        - Case File ID: {case_file_id}
        - Party Represented: {party_represented}
        - Jurisdiction: {jurisdiction}
        
        ## Case Facts:
        {case_facts}
        {legal_issues_section}
        
        ## Research Instructions:
        
        {step1_instructions}
        
        ### Step 2: Conduct Systematic Research
        For each legal issue:
        - Search for relevant cases using semantic search
        - Find cases with similar fact patterns
        - Research applicable statutes and regulations
        - Trace important precedent chains
        - Extract key legal holdings
        - Document findings and insights as human-readable research notes. DO NOT INCLUDE CODE OR TOOL USE NOTES IN THE RESEARCH NOTES.
        - Add relevant cases to the case file using add_document_to_case_file
        - ADD AT MOST 5 DOCUMENTS AND 3 NOTES. PLAN YOUR RESEARCH STEPS CAREFULLY.

        ### Step 3: Final Summary and Conclusion
        Once all research steps are complete and all relevant documents and notes have been saved to the case file, you MUST conclude your work.
        Provide a final, comprehensive summary that includes:
        - A brief overview of the research conducted.
        - A summary of the key authorities found (both supporting and adverse).
        - A final assessment of the legal position's strength.
        - Your top 3-5 strategic recommendations for the legal team.
        
        This final summary is your last action. Do not use any more tools after this step.
        
        ## Important Guidelines:
        - Focus on finding and organizing authorities, NOT writing arguments
        - Save relevant documents you find to the case file
        - Document your research process and insights using research notes, without including code or tool use notes
        - Consider both supportive and adverse authorities
        - Provide strategic insights for argument development
        - ADD AT MOST 5 DOCUMENTS AND 3 NOTES. PLAN YOUR RESEARCH STEPS CAREFULLY.
        - DO NOT INCLUDE CODE OR TOOL USE NOTES IN THE RESEARCH NOTES.

        Begin your research now. Remember not to add too many documents or notes, as you will need to conclude your research with a final summary.
        There is a short time limit for this research, so do not research for too long.
        """

        return {"messages": [{"role": "user", "content": research_prompt}]}
    
    def research_case(
        self,
        case_file_id: int,
        case_facts: str,
        party_represented: str,
        legal_issues: Optional[List[str]] = None,
        jurisdiction: str = "Singapore"
    ) -> str:
        """
        Conduct comprehensive legal research for a case.
        
        Args:
            case_file_id: ID of the case file to organize research in
            case_facts: Factual background of the case
            party_represented: Which party the research supports
            legal_issues: Optional list of legal issues to research. If not provided, 
                         the AI will identify legal issues from the case facts.
            jurisdiction: Relevant jurisdiction
            
        Returns:
            Research notes summarizing findings and strategic insights
        """
        try:
            initial_state = self.get_initial_state(
                case_file_id=case_file_id,
                case_facts=case_facts,
                party_represented=party_represented,
                legal_issues=legal_issues,
                jurisdiction=jurisdiction
            )
            
            result = self.agent_executor.invoke(initial_state)
            final_message = result["messages"][-1]
            return final_message.content if hasattr(final_message, 'content') else str(final_message)
            
        except Exception as e:
            logger.error(f"Error during legal research: {e}")
            return f"Research error: {str(e)}"
    
    def targeted_research(
        self,
        case_file_id: int,
        research_query: str,
        research_type: str = "general"
    ) -> str:
        """
        Conduct targeted research on a specific legal question.
        
        Args:
            case_file_id: ID of the case file
            research_query: Specific legal question or topic
            research_type: "precedent", "statutory", "factual", or "general"
            
        Returns:
            Research findings and analysis
        """
        
        research_prompt = f"""
        Conduct targeted legal research on the following question:
        
        ## Research Query:
        {research_query}
        
        ## Research Type: {research_type}
        ## Case File ID: {case_file_id}
        
        ## Instructions:
        1. Use appropriate research tools based on the research type
        2. Find and analyze relevant authorities
        3. Add discovered documents to the case file using add_document_to_case_file
        4. Document insights and findings using add_research_note
        5. Provide analysis and strategic insights
        
        Focus your research strategy based on the research type:
        - precedent: Focus on case law and judicial decisions
        - statutory: Focus on statutes, regulations, and their interpretation
        - factual: Focus on cases with similar fact patterns
        - general: Use comprehensive research across all sources
        
        Save all relevant findings to the case file, document insights with research notes, and provide detailed analysis.
        """
        
        try:
            result = self.agent_executor.invoke({"messages": [{"role": "user", "content": research_prompt}]})
            final_message = result["messages"][-1]
            return final_message.content if hasattr(final_message, 'content') else str(final_message)
            
        except Exception as e:
            logger.error(f"Error during targeted research: {e}")
            return f"Research error: {str(e)}"
    
    def validate_legal_arguments(
        self,
        case_file_id: int,
        proposed_arguments: List[str]
    ) -> str:
        """
        Validate proposed legal arguments against available authorities.
        
        Args:
            case_file_id: ID of the case file
            proposed_arguments: List of legal arguments to validate
            
        Returns:
            Validation results and recommendations
        """
        
        validation_prompt = f"""
        Validate the following proposed legal arguments using research tools:
        
        ## Case File ID: {case_file_id}
        
        ## Proposed Arguments:
        {chr(10).join(f"{i+1}. {arg}" for i, arg in enumerate(proposed_arguments))}
        
        ## Instructions:
        For each argument:
        1. Use validate_legal_statement to check against authorities
        2. Search for supporting case law and precedents
        3. Identify any contrary authorities
        4. Add relevant supporting documents to the case file
        5. Document validation findings and recommendations using add_research_note
        6. Assess argument strength and provide recommendations
        
        Provide a comprehensive validation report with strategic recommendations.
        """
        
        try:
            result = self.agent_executor.invoke({"messages": [{"role": "user", "content": validation_prompt}]})
            final_message = result["messages"][-1]
            return final_message.content if hasattr(final_message, 'content') else str(final_message)
            
        except Exception as e:
            logger.error(f"Error during argument validation: {e}")
            return f"Validation error: {str(e)}"
        
    def get_agent_executor(self):
        """
        Get the agent executor for this ReAct agent.
        
        Returns:
            Instance of the ReAct agent executor
        """
        return self.agent_executor


# Convenience functions for easy use
def conduct_case_research(
    case_file_id: int,
    case_facts: str,
    party_represented: str,
    legal_issues: Optional[List[str]] = None,
    jurisdiction: str = "Singapore"
) -> str:
    """
    Convenience function to conduct case research using the ReAct agent.
    
    Args:
        case_file_id: ID of the case file
        case_facts: Factual background
        party_represented: Party being represented
        legal_issues: Optional legal issues to research. If not provided, 
                     the AI will identify them from case facts.
        jurisdiction: Relevant jurisdiction
        
    Returns:
        Research notes and strategic insights
    """
    agent = LegalResearchAgent()
    return agent.research_case(
        case_file_id=case_file_id,
        case_facts=case_facts,
        party_represented=party_represented,
        legal_issues=legal_issues,
        jurisdiction=jurisdiction
    )


def validate_arguments_with_research(
    case_file_id: int,
    proposed_arguments: List[str]
) -> str:
    """
    Convenience function to validate arguments using research.
    
    Args:
        case_file_id: ID of the case file
        proposed_arguments: Arguments to validate
        
    Returns:
        Validation results
    """
    agent = LegalResearchAgent()
    return agent.validate_legal_arguments(
        case_file_id=case_file_id,
        proposed_arguments=proposed_arguments
    )


def create_research_agent(
    llm_config: Optional[Dict[str, Any]] = None
) -> LegalResearchAgent:
    """
    Factory function to create a new instance of the legal research agent.
    
    Args:
        llm_config: Optional LLM configuration override
        
    Returns:
        Instance of LegalResearchAgent
    """
    return LegalResearchAgent(llm_config=llm_config)


research_agent = create_research_agent(settings.llm_config)
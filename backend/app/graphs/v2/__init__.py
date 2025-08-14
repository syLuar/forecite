"""
LangGraph Workflows v2 - Improved Legal Assistant Graphs

This package contains the improved versions of the legal assistant workflows
with decomposed nodes for better LLM performance and reliability.
"""

from .research_graph import research_graph
from .drafting_graph import drafting_graph
from .counterargument_graph import counterargument_graph
from .state import (
    ResearchState,
    DraftingState, 
    CounterArgumentState,
    # Legacy compatibility
    CaseFileDocument,
    SearchPlan,
    ArgumentStrategy
)

__all__ = [
    "research_graph",
    "drafting_graph", 
    "counterargument_graph",
    "ResearchState",
    "DraftingState",
    "CounterArgumentState",
    "CaseFileDocument",
    "SearchPlan", 
    "ArgumentStrategy"
]
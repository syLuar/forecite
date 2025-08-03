"""
LangGraph workflow definitions for the Legal Research Assistant.

This package contains the agent team definitions and state management
for research and drafting workflows.
"""

from .research_graph import research_graph
from .drafting_graph import drafting_graph
from .state import ResearchState, DraftingState

__all__ = ["research_graph", "drafting_graph", "ResearchState", "DraftingState"]

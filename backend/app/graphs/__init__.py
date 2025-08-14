"""
LangGraph workflow definitions for the Legal Research Assistant.

This package contains the agent team definitions and state management
for research and drafting workflows. Now using v2 improved implementations.
"""

from .v2.research_graph import research_graph
from .v2.drafting_graph import drafting_graph
from .v2.counterargument_graph import counterargument_graph
from .v2.state import ResearchState, DraftingState, CounterArgumentState

__all__ = ["research_graph", "drafting_graph", "counterargument_graph", "ResearchState", "DraftingState", "CounterArgumentState"]

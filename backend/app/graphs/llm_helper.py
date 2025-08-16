"""
LLM Helper for Graph Nodes

This module provides graph-specific LLM initialization functionality.
It handles both node-specific and graph-level LLM configuration modes
while keeping the core llm.py module agnostic of graph specifics.
"""

import logging
from typing import Dict, Any
from app.core.llm import get_llm_config_for_path, create_llm_from_config_path, create_llm

logger = logging.getLogger(__name__)


class GraphLLMHelper:
    """
    Helper class for managing LLM instances within a specific graph.
    
    Supports two configuration modes:
    1. Node-specific config: {task_category}.{graph_name}.nodes.{node_name}
    2. Graph-level config fallback: {task_category}.{graph_name}
    """
    
    def __init__(self, graph_name: str, task_category: str = "main"):
        """
        Initialize the LLM helper for a specific graph.
        
        Args:
            graph_name: Name of the graph (e.g., 'drafting', 'research', 'counterargument')
            task_category: Category in config (default: 'main')
        """
        self.graph_name = graph_name
        self.task_category = task_category
        self._llm_cache: Dict[str, Any] = {}
        
    def get_node_llm(self, node_name: str) -> Any:
        """
        Get LLM instance for a specific node in this graph.
        
        Args:
            node_name: Name of the node (e.g., 'fact_analyzer_node', 'strategy_developer_node')
            
        Returns:
            LLM instance configured for the node
        """
        cache_key = f"{self.task_category}.{self.graph_name}.{node_name}"
        
        if cache_key not in self._llm_cache:
            # Try node-specific config first
            node_config_path = f"{self.task_category}.{self.graph_name}.nodes.{node_name}"
            graph_config_path = f"{self.task_category}.{self.graph_name}"
            
            # Get graph-level config as fallback
            try:
                graph_config = get_llm_config_for_path(graph_config_path)
            except Exception:
                graph_config = None
            
            # Create LLM with node-specific config and graph fallback
            try:
                llm = create_llm_from_config_path(node_config_path, fallback_config=graph_config)
                logger.info(f"Using node-specific LLM config for {self.graph_name}.{node_name}")
            except Exception:
                if graph_config:
                    llm = create_llm(graph_config)
                    logger.info(f"Using graph-level LLM config for {self.graph_name}.{node_name}")
                else:
                    raise RuntimeError(f"No LLM configuration found for {self.graph_name}.{node_name}")
            
            self._llm_cache[cache_key] = llm
        
        return self._llm_cache[cache_key]
    
    def clear_cache(self):
        """Clear the LLM cache for this graph."""
        self._llm_cache.clear()
        logger.info(f"LLM cache cleared for {self.graph_name} graph")


# Convenience function for simple usage
def create_graph_llm_helper(graph_name: str, task_category: str = "main") -> GraphLLMHelper:
    """
    Create a GraphLLMHelper instance for a specific graph.
    
    Args:
        graph_name: Name of the graph
        task_category: Category in config (default: 'main')
        
    Returns:
        GraphLLMHelper instance
    """
    return GraphLLMHelper(graph_name, task_category)

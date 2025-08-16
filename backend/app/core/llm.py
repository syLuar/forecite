"""
LLM Initialization and Configuration Module

This module provides a centralized way to initialize LLMs with different providers
(Google Gemini, OpenAI, etc.) based on configuration. It abstracts away provider-specific
initialization logic and provides a consistent interface.

Supports two modes of LLM configuration:
1. Graph-level: Single LLM config for all nodes in a graph
2. Node-level: Individual LLM configs for specific nodes within a graph
"""

import logging
from typing import Dict, Any, Optional, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from app.core.config import settings

logger = logging.getLogger(__name__)

# Supported LLM providers
SUPPORTED_PROVIDERS = {
    "google": ChatGoogleGenerativeAI,
    "openai": ChatOpenAI,
}

# Default provider if not specified
DEFAULT_PROVIDER = "google"


class LLMError(Exception):
    """Custom exception for LLM initialization errors."""
    pass


def _get_provider_config(provider: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get provider-specific configuration parameters.
    
    Args:
        provider: The LLM provider name (google, openai)
        model_config: Raw model configuration dict
        
    Returns:
        Dict of provider-specific parameters
    """
    config = model_config.copy()
    
    if provider == "google":
        # Google-specific configuration
        if "google_api_key" not in config:
            config["google_api_key"] = settings.google_api_key
        
        # Map common parameters to Google-specific names
        if "thinking_budget" in config:
            # thinking_budget is a Google-specific parameter, keep as-is
            pass
            
    elif provider == "openai":
        # OpenAI-specific configuration
        if "openai_api_key" not in config and "api_key" not in config:
            # Try to get from environment or settings
            import os
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                config["api_key"] = openai_key
            else:
                logger.warning("No OpenAI API key found in environment")
        
        # Map common parameters to OpenAI-specific names
        if "model" in config:
            # OpenAI expects 'model' parameter
            pass
        if "temperature" in config:
            # Both providers use 'temperature'
            pass
        
        # Remove Google-specific parameters
        config.pop("thinking_budget", None)
        config.pop("google_api_key", None)
    
    return config


def create_llm(
    model_config: Dict[str, Any], 
    provider: Optional[str] = None
) -> BaseChatModel:
    """
    Create an LLM instance based on the provider and configuration.
    
    Args:
        model_config: Configuration dict containing model parameters
        provider: LLM provider name. If None, will look for 'model_provider' in config
        
    Returns:
        Initialized LLM instance
        
    Raises:
        LLMError: If provider is not supported or configuration is invalid
    """
    # Determine provider
    if provider is None:
        provider = model_config.get("model_provider", DEFAULT_PROVIDER)
    
    provider = provider.lower()
    
    if provider not in SUPPORTED_PROVIDERS:
        raise LLMError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: {list(SUPPORTED_PROVIDERS.keys())}"
        )
    
    # Get provider class
    llm_class = SUPPORTED_PROVIDERS[provider]
    
    # Get provider-specific configuration
    try:
        provider_config = _get_provider_config(provider, model_config)
        
        # Remove our custom 'model_provider' key from the config
        provider_config.pop("model_provider", None)
        
        # Initialize the LLM
        logger.info(f"Initializing {provider} LLM with model: {provider_config.get('model', 'default')}")
        llm = llm_class(**provider_config)
        
        return llm
        
    except Exception as e:
        raise LLMError(f"Failed to initialize {provider} LLM: {str(e)}") from e


# Convenience function for custom configurations
def create_custom_llm(
    model: str,
    provider: str = DEFAULT_PROVIDER,
    temperature: float = 0.2,
    **kwargs
) -> BaseChatModel:
    """
    Create a custom LLM with specific parameters.
    
    Args:
        model: Model name/identifier
        provider: LLM provider (google, openai)
        temperature: Sampling temperature
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Initialized LLM instance
    """
    config = {
        "model": model,
        "temperature": temperature,
        "model_provider": provider,
        **kwargs
    }
    
    return create_llm(config)


def get_llm_config_for_path(
    config_path: str, 
    fallback_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get LLM configuration for a specific config path, with optional fallback.
    
    Args:
        config_path: Dot-separated path in config (e.g., 'main.drafting', 'main.drafting.nodes.fact_analyzer_node')
        fallback_config: Optional fallback configuration if specific path not found
        
    Returns:
        Dict containing LLM configuration for the specified path
        
    Raises:
        LLMError: If no configuration is found and no fallback provided
    """
    try:
        # Navigate through the config using the path
        config = settings.llm_config
        path_parts = config_path.split('.')
        
        for part in path_parts:
            if part in config:
                config = config[part]
            else:
                if fallback_config:
                    logger.info(f"Config path '{config_path}' not found, using fallback")
                    return fallback_config.copy()
                else:
                    raise LLMError(f"Configuration path '{config_path}' not found")
        
        # Remove nested 'nodes' section if present in final config
        final_config = config.copy() if isinstance(config, dict) else {}
        final_config.pop("nodes", None)
        
        return final_config
        
    except Exception as e:
        if fallback_config:
            logger.warning(f"Error getting config for path '{config_path}': {str(e)}, using fallback")
            return fallback_config.copy()
        else:
            raise LLMError(f"Failed to get LLM config for path '{config_path}': {str(e)}") from e


def create_llm_from_config_path(
    config_path: str, 
    fallback_config: Optional[Dict[str, Any]] = None
) -> BaseChatModel:
    """
    Create an LLM instance from a configuration path.
    
    Args:
        config_path: Dot-separated path in config (e.g., 'main.drafting.nodes.fact_analyzer_node')
        fallback_config: Optional fallback configuration if specific path not found
        
    Returns:
        Initialized LLM instance
        
    Raises:
        LLMError: If configuration is invalid or LLM creation fails
    """
    config = get_llm_config_for_path(config_path, fallback_config)
    return create_llm(config)
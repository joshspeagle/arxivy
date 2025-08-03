#!/usr/bin/env python3
"""
LLM Configuration and Utilities

Handles loading and managing different LLM providers with aliases.
Supports OpenAI, Anthropic, Google, local models, and custom endpoints.

Usage:
    from llm_utils import LLMManager

    manager = LLMManager()
    config = manager.get_model_config("gpt-41-mini")
    client = manager.get_client("claude-sonnet-4")
"""

import yaml
import os
from typing import Dict, Any


class LLMManager:
    """
    Manages LLM providers and configurations with alias support.

    Loads provider configurations from config/llm_providers.yaml and
    provides easy access to different models via aliases.
    """

    def __init__(self, providers_config_path: str = "config/llm.yaml"):
        """
        Initialize the LLM Manager.

        Args:
            providers_config_path: Path to the LLM providers config file
        """
        self.providers_config_path = providers_config_path
        self.providers = {}
        self._load_config()

    def _load_config(self):
        """Load LLM provider configurations from YAML file."""
        try:
            with open(self.providers_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Initialize providers with the loaded config
            self.providers = config

            # Validate configurations
            self._validate_configs()

        except FileNotFoundError:
            print(f"LLM providers config not found at {self.providers_config_path}")
            print("Creating default config...")
            self._create_default_config()
        except Exception as e:
            print(f"Error loading LLM config: {e}")
            raise

    def _create_default_config(self):
        """Create a default LLM providers config file."""
        default_config = {
            "gemini-25-flash-lite": {
                "provider": "google",
                "model": "gemini-2.5-flash-lite",
                "temperature": 0.1,
                "max_tokens": 1000,
                "reasoning": True,
                "description": "Gemini 2.5 Flash Lite - Most cost-effective",
            },
        }

        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.providers_config_path), exist_ok=True)

        with open(self.providers_config_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)

        self.providers = default_config

        print(f"Created default config at {self.providers_config_path}")

    def _validate_configs(self):
        """Validate that all provider configurations have required fields."""
        required_fields = ["provider", "model"]

        for alias, config in self.providers.items():
            for field in required_fields:
                if field not in config:
                    raise ValueError(
                        f"Missing required field '{field}' in provider '{alias}'"
                    )

    def get_model_config(self, alias: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific model alias.

        Args:
            alias: The model alias (e.g., "gpt-41-mini")

        Returns:
            Dictionary containing the model configuration

        Raises:
            KeyError: If the alias is not found
        """
        if alias not in self.providers:
            available = list(self.providers.keys())
            raise KeyError(f"Model alias '{alias}' not found. Available: {available}")

        config = self.providers[alias].copy()

        # Resolve environment variables in API keys
        if "api_key_env" in config:
            api_key = os.getenv(config["api_key_env"])
            if api_key:
                config["api_key"] = api_key
            else:
                print(f"Warning: Environment variable {config['api_key_env']} not set")

        return config

    def list_models(self, filters: Dict[str, Any] = None) -> Dict[str, str]:
        """
        List all available model aliases with descriptions, filtered by config attributes.

        Args:
            filters: Dictionary of config attributes to filter by (e.g., {"provider": "openai", "reasoning": True})

        Returns:
            Dictionary mapping alias -> description
        """
        models = {}
        filters = filters or {}
        for alias, config in self.providers.items():
            match = True
            for key, value in filters.items():
                if config.get(key) != value:
                    match = False
                    break
            if match:
                models[alias] = config.get(
                    "description", f"{config['provider']} {config['model']}"
                )
        return models

    def get_client(self, alias: str):
        """
        Get an API client for the specified model alias.

        Args:
            alias: Model alias

        Returns:
            Configured API client (OpenAI, Anthropic, etc.)

        Note:
            This method requires the appropriate client libraries to be installed
            (openai, anthropic, etc.)
        """
        config = self.get_model_config(alias)
        provider = config["provider"]

        if provider == "openai":
            try:
                import openai

                return openai.OpenAI(api_key=config.get("api_key"))
            except ImportError:
                raise ImportError(
                    "OpenAI client not installed. Run: pip install openai"
                )

        elif provider == "anthropic":
            try:
                import anthropic

                return anthropic.Anthropic(api_key=config.get("api_key"))
            except ImportError:
                raise ImportError(
                    "Anthropic client not installed. Run: pip install anthropic"
                )

        elif provider == "google":
            try:
                import google.generativeai as genai

                genai.configure(api_key=config.get("api_key"))
                return genai
            except ImportError:
                raise ImportError(
                    "Google AI client not installed. Run: pip install google-generativeai"
                )

        elif provider in ["ollama", "lmstudio", "local", "custom"]:
            # Local providers don't need clients - handled directly in scoring_utils
            return None

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def create_prompt_config(
        self, alias: str, system_prompt: str = None
    ) -> Dict[str, Any]:
        """
        Create a standardized prompt configuration for any LLM.

        Args:
            alias: Model alias
            system_prompt: Optional system prompt

        Returns:
            Dictionary with model config and prompt setup
        """
        config = self.get_model_config(alias)

        prompt_config = {
            "model": config["model"],
            "temperature": config.get("temperature", 0.1),
            "max_tokens": config.get("max_tokens", 1000),
        }

        if system_prompt:
            prompt_config["system_prompt"] = system_prompt

        return prompt_config


def load_llm_manager(config_path: str = "config/llm.yaml") -> LLMManager:
    """
    Convenience function to load an LLM manager.

    Args:
        config_path: Path to the LLM providers config

    Returns:
        Configured LLMManager instance
    """
    return LLMManager(config_path)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    manager = LLMManager()

    print("Available models:")
    for alias, description in manager.list_models().items():
        print(f"  {alias}: {description}")

    print("\nReasoning models only:")
    for alias, description in manager.list_models({"reasoning": True}).items():
        print(f"  {alias}: {description}")

    # Get a specific model config
    try:
        config = manager.get_model_config("gpt-41-mini")
        print(f"\nConfig for gpt-41-mini: {config}")
    except KeyError as e:
        print(f"Error: {e}")

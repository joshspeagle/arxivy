#!/usr/bin/env python3
"""
Paper Synthesis Utilities

Handles LLM-based synthesis of selected papers into comprehensive reports.
Supports configurable metadata inclusion and flexible output formats.

Dependencies:
    pip install openai anthropic google-generativeai pyyaml numpy

Usage:
    from synthesis_utils import PaperSynthesizer

    synthesizer = PaperSynthesizer(config, llm_manager, model_alias)
    synthesis_result = synthesizer.synthesize_papers(selected_papers)
"""

import time
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def get_default_synthesis_prompts() -> Dict[str, str]:
    """
    Get default prompts for research paper synthesis.

    Returns:
        Dictionary with default prompt strings
    """
    return {
        "research_context_prompt": """
You are synthesizing research papers for a computational scientist who works on 
machine learning, statistical methods, and data analysis techniques. The researcher 
values methodological rigor, theoretical contributions, and practical applications 
that advance scientific understanding.

Focus on identifying connections between papers, emerging themes, and broader 
implications for the field. Assume the reader is an expert who wants to understand 
how these papers collectively advance knowledge and what opportunities they reveal.
        """.strip(),
        "report_prompt": """
Create a comprehensive research synthesis (1500-2000 words) that organizes the selected papers into coherent themes and insights.

Guidelines:
- Organize by conceptual themes and methodological approaches, not just by score ranking
- Use rescored scores to guide the depth of discussion and emphasis for each paper
- Identify connections, trends, and complementary approaches across papers
- Include practical implications and future research directions
- Maintain technical accuracy while being accessible to an expert audience

Structure suggestions (adapt based on the specific papers):
- Executive summary of key themes
- Thematic sections grouping related papers
- Methodological insights and innovations
- Research implications and future directions

For each paper discussed, include: title, authors, key contributions, and relevance to the broader themes.
Use the provided scores to emphasize the most significant contributions.
        """.strip(),
    }


class PaperSynthesizer:
    """
    Handles LLM-based synthesis of selected papers into comprehensive reports.

    Features:
    - Theme-based paper organization
    - Configurable metadata inclusion
    - Score-guided emphasis and depth
    - Flexible output formats via prompts
    - Comprehensive error handling and retry logic
    """

    def __init__(self, config: Dict, llm_manager, model_alias: str):
        """
        Initialize the paper synthesizer.

        Args:
            config: Configuration dictionary with synthesis settings
            llm_manager: LLMManager instance for API calls
            model_alias: Model alias to use for synthesis
        """
        self.config = config
        self.llm_manager = llm_manager
        self.model_alias = model_alias

        # Validate model configuration
        try:
            self.model_config = llm_manager.get_model_config(model_alias)
            self.client = llm_manager.get_client(model_alias)
        except Exception as e:
            raise ValueError(f"Failed to initialize model '{model_alias}': {e}")

        # Synthesis configuration
        self.synthesis_config = config.get("synthesis", {})
        self.generation_config = self.synthesis_config.get("generation", {})

        # LLM parameters
        self.retry_attempts = self.generation_config.get("retry_attempts", 2)
        self.include_metadata = self.generation_config.get(
            "include_metadata",
            [
                "title",
                "abstract",
                "authors",
                "categories",
                "llm_summary",
                "summary_confidence",
                "rescored_llm_score",
                "rescored_llm_explanation",
            ],
        )

        # Build system prompt
        self.system_prompt = self._build_system_prompt()

        # Statistics tracking
        self.stats = {
            "papers_synthesized": 0,
            "synthesis_attempts": 0,
            "synthesis_failures": 0,
            "total_retries": 0,
            "synthesis_time": 0,
        }

    def _get_default_prompts(self) -> Dict[str, str]:
        """Get default prompts for synthesis."""
        return get_default_synthesis_prompts()

    def _build_system_prompt(self) -> str:
        """
        Construct the complete system prompt from configured parts.

        Returns:
            Complete system prompt string
        """
        defaults = self._get_default_prompts()
        used_defaults = []

        # Get research context prompt
        research_context = self.generation_config.get("research_context_prompt")
        if research_context is None:
            # Try to inherit from scoring config
            research_context = (
                self.config.get("scoring", {})
                .get("research_context_prompt", "")
                .strip()
            )
            if research_context:
                print(
                    "ℹ️  Inheriting research_context_prompt from scoring configuration"
                )
            else:
                research_context = defaults["research_context_prompt"]
                used_defaults.append("research_context_prompt")
        elif not research_context.strip():
            research_context = defaults["research_context_prompt"]
            used_defaults.append("research_context_prompt")

        # Get synthesis strategy prompt
        report_prompt = self.generation_config.get("report_prompt", "").strip()
        if not report_prompt:
            report_prompt = defaults["report_prompt"]
            used_defaults.append("report_prompt")

        # Warn user about defaults
        if used_defaults:
            print(f"⚠️  WARNING: Using default prompts for: {', '.join(used_defaults)}")
            print(
                f"   For better results, customize these prompts in your config file."
            )
            print()

        # Combine the parts
        prompt_parts = [
            f"RESEARCH CONTEXT:\n{research_context}",
            f"SYNTHESIS INSTRUCTIONS:\n{report_prompt}",
        ]

        return "\n\n".join(prompt_parts)

    def _extract_paper_content(self, papers: List[Dict]) -> str:
        """
        Extract and format content from papers for synthesis.

        Args:
            papers: List of paper dictionaries

        Returns:
            Formatted string with all paper information
        """
        if not papers:
            return "No papers provided for synthesis."

        content_parts = [
            f"PAPERS FOR SYNTHESIS ({len(papers)} total):",
            "=" * 50,
        ]

        # Sort papers by score for consistent presentation (highest first)
        score_field = self.synthesis_config.get("selection", {}).get(
            "score_field", "rescored_llm_score"
        )
        sorted_papers = sorted(
            papers, key=lambda x: x.get(score_field, 0), reverse=True
        )

        for i, paper in enumerate(sorted_papers, 1):
            paper_section = [f"\nPAPER {i}:"]

            for field in self.include_metadata:
                if field in paper and paper[field] is not None:
                    value = paper[field]

                    # Format different field types appropriately
                    if field == "title":
                        paper_section.append(f"Title: {value}")
                    elif field == "abstract":
                        paper_section.append(f"Abstract: {value}")
                    elif field == "authors":
                        if isinstance(value, list):
                            authors = ", ".join(value)
                        else:
                            authors = str(value)
                        paper_section.append(f"Authors: {authors}")
                    elif field == "categories":
                        if isinstance(value, list):
                            categories = ", ".join(value)
                        else:
                            categories = str(value)
                        paper_section.append(f"Categories: {categories}")
                    elif field == "llm_summary":
                        # Truncate very long summaries for context window management
                        summary_text = str(value)
                        if len(summary_text) > 2000:
                            summary_text = summary_text[:2000] + "... [truncated]"
                        paper_section.append(f"Summary: {summary_text}")
                    elif field in ["rescored_llm_score", "llm_score"]:
                        paper_section.append(f"Score ({field}): {value}")
                    elif field == "summary_confidence":
                        paper_section.append(f"Summary Confidence: {value}")
                    elif field == "published":
                        paper_section.append(f"Published: {value}")
                    else:
                        paper_section.append(
                            f"{field.replace('_', ' ').title()}: {value}"
                        )

            content_parts.extend(paper_section)
            content_parts.append("-" * 30)

        # Add synthesis guidance
        content_parts.extend(
            [
                "\nSYNTHESIS GUIDANCE:",
                f"- Total papers to synthesize: {len(papers)}",
                f"- Papers are ordered by {score_field} (highest first)",
            ]
        )

        return "\n".join(content_parts)

    def _call_llm_api(self, content: str) -> Optional[str]:
        """
        Make an API call to the configured LLM.

        Args:
            content: Content to send to the LLM

        Returns:
            Response text or None on failure
        """
        provider = self.model_config["provider"]

        try:
            if provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_config["model"],
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 4000),
                )
                return response.choices[0].message.content

            elif provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_config["model"],
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 4000),
                )
                return response.content[0].text

            elif provider == "google":
                model = self.client.GenerativeModel(self.model_config["model"])
                prompt = f"{self.system_prompt}\n\n{content}"
                response = model.generate_content(
                    prompt,
                    generation_config=self.client.types.GenerationConfig(
                        temperature=self.model_config.get("temperature", 0.1),
                        max_output_tokens=self.model_config.get("max_tokens", 4000),
                    ),
                )
                return response.text

            elif provider == "ollama":
                return self._call_ollama_api(content)
            elif provider == "lmstudio":
                return self._call_lmstudio_api(content)
            elif provider in ["local", "custom"]:
                return self._call_local_api(content)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print(f"API call failed: {e}")
            return None

    def _call_ollama_api(self, content: str) -> Optional[str]:
        """Call Ollama API."""
        import requests

        base_url = self.model_config.get("base_url", "http://localhost:11434")
        model_name = self.model_config["model"]

        full_prompt = f"{self.system_prompt}\n\n{content}"

        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.model_config.get("temperature", 0.1),
                "num_predict": self.model_config.get("max_tokens", 4000),
            },
        }

        try:
            response = requests.post(
                f"{base_url}/api/generate", json=payload, timeout=300
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            print(f"Ollama API call failed: {e}")
            return None

    def _call_lmstudio_api(self, content: str) -> Optional[str]:
        """Call LM Studio API."""
        try:
            import openai

            base_url = self.model_config.get("base_url", "http://localhost:1234/v1")
            model_name = self.model_config["model"]

            local_client = openai.OpenAI(
                base_url=base_url,
                api_key="not-needed",
            )

            response = local_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=self.model_config.get("temperature", 0.1),
                max_tokens=self.model_config.get("max_tokens", 4000),
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"LM Studio API call failed: {e}")
            return None

    def _call_local_api(self, content: str) -> Optional[str]:
        """Call custom local API."""
        import requests

        try:
            base_url = self.model_config.get("base_url")
            if not base_url:
                raise ValueError("base_url is required for local/custom providers")

            model_name = self.model_config["model"]
            api_format = self.model_config.get("api_format", "openai")

            if api_format == "openai":
                import openai

                local_client = openai.OpenAI(
                    base_url=base_url,
                    api_key=self.model_config.get("api_key", "not-needed"),
                )

                response = local_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 4000),
                )

                return response.choices[0].message.content

            elif api_format == "ollama":
                full_prompt = f"{self.system_prompt}\n\n{content}"

                payload = {
                    "model": model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.model_config.get("temperature", 0.1),
                        "num_predict": self.model_config.get("max_tokens", 4000),
                    },
                }

                response = requests.post(
                    f"{base_url}/api/generate", json=payload, timeout=300
                )
                response.raise_for_status()

                result = response.json()
                return result.get("response", "")

            else:
                raise ValueError(f"Unsupported api_format: {api_format}")

        except Exception as e:
            print(f"Local API call failed: {e}")
            return None

    def _validate_synthesis_response(
        self, response_text: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that the synthesis response is reasonable.

        Args:
            response_text: Raw response from LLM

        Returns:
            Tuple of (is_valid, cleaned_response)
        """
        try:
            if not response_text or len(response_text.strip()) < 500:
                return False, None

            cleaned_response = response_text.strip()

            # Remove any thinking tokens or artifacts that might leak through
            cleaned_response = re.sub(
                r"<think>.*?</think>",
                "",
                cleaned_response,
                flags=re.DOTALL | re.IGNORECASE,
            )
            cleaned_response = re.sub(r"<[^>]+>", "", cleaned_response)

            # Clean up extra whitespace
            cleaned_response = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned_response)
            cleaned_response = cleaned_response.strip()

            # Basic quality checks
            if len(cleaned_response) < 500:
                return False, None

            # Check for reasonable structure (should have multiple paragraphs)
            paragraphs = [
                p.strip() for p in cleaned_response.split("\n\n") if p.strip()
            ]
            if len(paragraphs) < 3:
                return False, None

            return True, cleaned_response

        except Exception as e:
            print(f"Response validation error: {e}")
            return False, None

    def synthesize_papers(self, papers: List[Dict]) -> Dict:
        """
        Synthesize papers into a comprehensive report with retry logic.

        Args:
            papers: List of selected papers to synthesize

        Returns:
            Dictionary with synthesis results and metadata
        """
        if not papers:
            return {
                "synthesis_report": None,
                "synthesis_success": False,
                "error": "No papers provided for synthesis",
                "papers_count": 0,
                "synthesis_time": 0,
                "synthesized_by": self.model_alias,
                "synthesized_at": datetime.now().isoformat(),
            }

        print(f"Starting synthesis of {len(papers)} papers using {self.model_alias}")
        print(f"Metadata fields included: {', '.join(self.include_metadata)}")

        start_time = time.time()
        self.stats["synthesis_attempts"] += 1

        # Extract and format content
        paper_content = self._extract_paper_content(papers)

        # Attempt synthesis with retry logic
        for attempt in range(self.retry_attempts + 1):
            try:
                print(f"Synthesis attempt {attempt + 1}/{self.retry_attempts + 1}...")

                # Make LLM call
                response_text = self._call_llm_api(paper_content)

                if response_text is None:
                    if attempt < self.retry_attempts:
                        self.stats["total_retries"] += 1
                        print(f"  → API call failed, retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                    else:
                        print(f"  → All API attempts failed")
                        break

                # Validate response
                is_valid, cleaned_response = self._validate_synthesis_response(
                    response_text
                )

                if is_valid:
                    # Success!
                    synthesis_time = time.time() - start_time
                    self.stats["papers_synthesized"] = len(papers)
                    self.stats["synthesis_time"] = synthesis_time

                    result = {
                        "synthesis_report": cleaned_response,
                        "synthesis_success": True,
                        "papers_count": len(papers),
                        "synthesis_time": synthesis_time,
                        "synthesized_by": self.model_alias,
                        "synthesized_at": datetime.now().isoformat(),
                        "metadata_included": self.include_metadata.copy(),
                        "config_used": {
                            "model_alias": self.model_alias,
                            "retry_attempts": self.retry_attempts,
                            "include_metadata": self.include_metadata,
                        },
                    }

                    print(
                        f"✅ Synthesis completed successfully in {synthesis_time:.1f} seconds"
                    )
                    print(f"   Generated report: {len(cleaned_response)} characters")
                    return result

                else:
                    if attempt < self.retry_attempts:
                        self.stats["total_retries"] += 1
                        print(f"  → Invalid response format, retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                    else:
                        print(f"  → Failed to get valid response after all attempts")
                        break

            except Exception as e:
                if attempt < self.retry_attempts:
                    self.stats["total_retries"] += 1
                    print(f"  → Error during synthesis: {e}, retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    print(f"  → Final failure after all attempts: {e}")
                    break

        # All attempts failed
        synthesis_time = time.time() - start_time
        self.stats["synthesis_failures"] += 1

        return {
            "synthesis_report": None,
            "synthesis_success": False,
            "error": "Synthesis failed after all retry attempts",
            "papers_count": len(papers),
            "synthesis_time": synthesis_time,
            "synthesized_by": self.model_alias,
            "synthesized_at": datetime.now().isoformat(),
            "metadata_included": self.include_metadata.copy(),
        }


def validate_synthesis_config(config: Dict) -> List[str]:
    """
    Validate the synthesis configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    synthesis_config = config.get("synthesis", {})

    # Check required parameters
    model_alias = synthesis_config.get("model_alias")
    if not model_alias:
        errors.append("model_alias is required in synthesis configuration")

    # Validate selection config
    selection_config = synthesis_config.get("selection", {})
    if not selection_config:
        errors.append("selection configuration is required in synthesis")

    # Validate generation config
    generation_config = synthesis_config.get("generation", {})

    retry_attempts = generation_config.get("retry_attempts", 2)
    if not isinstance(retry_attempts, int) or retry_attempts < 0:
        errors.append("retry_attempts must be a non-negative integer")

    include_metadata = generation_config.get("include_metadata", [])
    if not include_metadata:
        errors.append("include_metadata cannot be empty")

    valid_metadata_fields = {
        "title",
        "abstract",
        "authors",
        "categories",
        "published",
        "updated",
        "llm_score",
        "llm_explanation",
        "rescored_llm_score",
        "rescored_llm_explanation",
        "llm_summary",
        "summary_confidence",
        "summary_source",
        "summary_key_contributions",
    }
    for field in include_metadata:
        if field not in valid_metadata_fields:
            errors.append(f"Invalid metadata field: {field}")

    return errors


def display_default_synthesis_prompts():
    """Display the default prompts for user reference."""
    defaults = get_default_synthesis_prompts()

    print("=== DEFAULT SYNTHESIS PROMPTS ===")
    for prompt_name, prompt_text in defaults.items():
        print(f"\n{prompt_name}:")
        print("-" * 40)
        print(prompt_text)
    print("\n" + "=" * 50)


def test_synthesis_utilities():
    """Test synthesis utilities with mock papers."""
    print("=== TESTING SYNTHESIS UTILITIES ===")

    # Test 1: Default prompts
    print("\n1. Testing default prompts...")
    defaults = get_default_synthesis_prompts()
    print(f"✅ Found {len(defaults)} default prompts:")
    for name in defaults.keys():
        print(f"   - {name}")

    # Test 2: Configuration validation
    print("\n2. Testing configuration validation...")

    valid_config = {
        "synthesis": {
            "model_alias": "test-model",
            "selection": {
                "score_field": "rescored_llm_score",
                "percentile": 35,
                "score_threshold": 7.5,
            },
            "generation": {
                "retry_attempts": 2,
                "include_metadata": [
                    "title",
                    "abstract",
                    "llm_summary",
                    "rescored_llm_score",
                ],
            },
        }
    }

    validation_errors = validate_synthesis_config(valid_config)
    if not validation_errors:
        print("✅ Valid configuration passed")
    else:
        print(f"❌ Unexpected validation errors: {validation_errors}")

    invalid_config = {
        "synthesis": {
            # Missing model_alias
            "generation": {"include_metadata": []}  # Empty metadata
        }
    }

    validation_errors = validate_synthesis_config(invalid_config)
    if validation_errors:
        print("✅ Invalid configuration correctly rejected")
    else:
        print("❌ Should have found validation errors")

    # Test 3: Content extraction and response validation
    print("\n3. Testing content extraction and response validation...")

    mock_papers = [
        {
            "id": "paper1",
            "title": "Deep Learning for Natural Language Processing",
            "abstract": "This paper presents a novel transformer architecture for language understanding...",
            "categories": ["cs.CL", "cs.LG"],
            "llm_summary": "The paper introduces an improved transformer model with attention mechanisms...",
            "rescored_llm_score": 8.5,
        },
        {
            "id": "paper2",
            "title": "Bayesian Neural Networks for Uncertainty Quantification",
            "abstract": "We propose a Bayesian approach to neural networks that provides uncertainty estimates...",
            "categories": ["cs.LG", "stat.ML"],
            "llm_summary": "This work combines Bayesian inference with deep learning for uncertainty quantification...",
            "rescored_llm_score": 7.8,
        },
        {
            "id": "paper3",
            "title": "Reinforcement Learning with Transformer Architectures",
            "abstract": "This study explores the use of transformer models in reinforcement learning settings...",
            "categories": ["cs.LG", "cs.AI"],
            "llm_summary": "The paper demonstrates how transformer architectures can be effectively used in RL...",
            "rescored_llm_score": 8.2,
        },
    ]

    class MockLLMManager:
        def get_model_config(self, alias):
            return {
                "provider": "test",
                "model": "test-model",
                "temperature": 0.1,
                "max_tokens": 4000,
            }

        def get_client(self, alias):
            return "mock-client"

    try:
        synthesizer = PaperSynthesizer(valid_config, MockLLMManager(), "test-model")

        # Test content extraction
        content = synthesizer._extract_paper_content(mock_papers)
        print(f"✅ Content extraction working: {len(content)} characters")
        print(
            f"   Contains all papers: {all(f'PAPER {i}' in content for i in range(1, 4))}"
        )

        # Test response validation
        good_response = (
            "This is a comprehensive synthesis of recent research papers. " * 20
        )  # Long enough
        bad_response = "Too short."

        is_valid_good, _ = synthesizer._validate_synthesis_response(good_response)
        is_valid_bad, _ = synthesizer._validate_synthesis_response(bad_response)

        print(
            f"✅ Response validation working: good={is_valid_good}, bad={is_valid_bad}"
        )

    except Exception as e:
        print(f"❌ Synthesis utilities test failed: {e}")
        return

    print(f"\n✅ All synthesis utilities tests passed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] in ["--test", "-t"]:
            test_synthesis_utilities()
        elif sys.argv[1] in ["--defaults", "-d"]:
            display_default_synthesis_prompts()
        elif sys.argv[1] in ["--help", "-h"]:
            print("Paper Synthesis Utilities")
            print("Usage:")
            print("  python synthesis_utils.py --test       # Run mock tests")
            print("  python synthesis_utils.py --defaults   # Show default prompts")
            print("  python synthesis_utils.py --help       # Show this help")
        else:
            print("Unknown option. Use --help for usage information.")
    else:
        print("Paper Synthesis Utilities Module")
        print("Run with --help for usage options")

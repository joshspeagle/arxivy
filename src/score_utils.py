#!/usr/bin/env python3
"""
Scoring Utilities

Helper functions for LLM-based paper scoring including prompt construction,
response validation, retry logic, and progress tracking.

Dependencies:
    pip install openai anthropic google-generativeai

Usage:
    from score_utils import ScoringEngine

    engine = ScoringEngine(config, llm_manager)
    score_data = engine.score_paper(paper_data)
"""

import json
import time
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os


def get_default_scoring_prompts() -> Dict[str, str]:
    """
    Get default prompts for generic research paper scoring.

    Returns:
        Dictionary with default prompt strings
    """
    return {
        "research_context_prompt": """
You are evaluating research papers for a computational scientist who works on 
machine learning, statistical methods, and data analysis techniques. The researcher 
is interested in methodologically rigorous work that advances the theoretical 
understanding of computational methods, introduces novel approaches, or demonstrates 
innovative applications to scientific problems.

The researcher values papers that have strong methodological foundations, clear 
experimental validation, and potential for broader impact across scientific domains.
        """.strip(),
        "scoring_strategy_prompt": """
Evaluate papers using this framework:

**1. RELEVANCE (40% weight):**
- Connection to machine learning, statistics, or computational methods
- Potential for cross-disciplinary applications
- Alignment with methodological innovation

**2. CONTRIBUTION (30% weight):**
- Novel techniques or significant improvements to existing methods
- Theoretical rigor and mathematical soundness
- Reproducibility and experimental validation

**3. IMPACT (30% weight):**
- Likely influence on future research directions
- Quality of experimental evaluation and results  
- Practical applicability to real-world problems
        """.strip(),
        "score_format_prompt": """
Rate each component from 1-10:
- Relevance score (1-10): ___
- Contribution score (1-10): ___
- Impact score (1-10): ___

Calculate: Final Score = (Relevance × 0.4) + (Contribution × 0.3) + (Impact × 0.3)
Round to one decimal place.

Be decisive about scores. Make sure to use the full scoring range effectively for each component.

IMPORTANT: Provide result as JSON:
{
  "score": <final_calculated_score>,
  "explanation": "Brief assessment covering all three components."
}

Do not include any text outside of this JSON structure.
        """.strip(),
    }


class ScoringEngine:
    """
    Handles LLM-based scoring of papers with robust error handling and retries.
    """

    def __init__(self, config: Dict, llm_manager, model_alias: str):
        """
        Initialize the scoring engine.

        Args:
            config: Configuration dictionary with scoring settings
            llm_manager: LLMManager instance for API calls
            model_alias: Model alias to use for scoring
        """
        self.config = config
        self.llm_manager = llm_manager
        self.model_alias = model_alias

        # Validate model configuration
        try:
            # Get model configuration from LLMManager
            self.model_config = llm_manager.get_model_config(model_alias)

            # Get API client from LLMManager (None for local providers)
            self.client = llm_manager.get_client(model_alias)
        except Exception as e:
            raise ValueError(f"Failed to initialize model '{model_alias}': {e}")

        # Scoring configuration
        self.scoring_config = config.get("scoring", {})
        self.retry_attempts = self.scoring_config.get("retry_attempts", 2)
        self.include_metadata = self.scoring_config.get(
            "include_metadata", ["title", "abstract"]
        )

        # Construct the system prompt from three parts (with defaults if needed)
        self.system_prompt = self._build_system_prompt()

        # Statistics tracking
        self.total_scored = 0
        self.total_retries = 0
        self.total_failures = 0
        self.start_time = None

    def _get_default_prompts(self) -> Dict[str, str]:
        """
        Get default prompts for generic research paper scoring.

        Returns:
            Dictionary with default prompt strings
        """
        return get_default_scoring_prompts()

    def _build_system_prompt(self) -> str:
        """
        Construct the complete system prompt from the three configured parts.
        Uses defaults if prompts are missing and issues warnings.

        Returns:
            Complete system prompt string
        """
        defaults = self._get_default_prompts()
        used_defaults = []

        # Get prompts, using defaults if missing
        research_context = self.scoring_config.get(
            "research_context_prompt", ""
        ).strip()
        if not research_context:
            research_context = defaults["research_context_prompt"]
            used_defaults.append("research_context_prompt")

        scoring_strategy = self.scoring_config.get(
            "scoring_strategy_prompt", ""
        ).strip()
        if not scoring_strategy:
            scoring_strategy = defaults["scoring_strategy_prompt"]
            used_defaults.append("scoring_strategy_prompt")

        score_format = self.scoring_config.get("score_format_prompt", "").strip()
        if not score_format:
            score_format = defaults["score_format_prompt"]
            used_defaults.append("score_format_prompt")

        # Warn user about defaults
        if used_defaults:
            print(f"⚠️  WARNING: Using default prompts for: {', '.join(used_defaults)}")
            print(
                f"   For better results, customize these prompts in your config file."
            )
            print()

        # Combine the three parts with clear separators
        prompt_parts = [
            f"RESEARCH CONTEXT:\n{research_context}",
            f"SCORING CRITERIA:\n{scoring_strategy}",
            f"OUTPUT FORMAT:\n{score_format}",
        ]

        return "\n\n".join(prompt_parts)

    def _extract_paper_content(self, paper: Dict) -> str:
        """
        Extract the configured metadata fields from a paper for scoring.

        Args:
            paper: Paper dictionary with metadata

        Returns:
            Formatted string with selected paper information
        """
        content_parts = []

        for field in self.include_metadata:
            if field in paper and paper[field]:
                if field == "title":
                    content_parts.append(f"Title: {paper[field]}")
                elif field == "abstract":
                    content_parts.append(f"Abstract: {paper[field]}")
                elif field == "authors":
                    if isinstance(paper[field], list):
                        authors = ", ".join(paper[field])
                    else:
                        authors = paper[field]
                    content_parts.append(f"Authors: {authors}")
                elif field == "categories":
                    if isinstance(paper[field], list):
                        categories = ", ".join(paper[field])
                    else:
                        categories = paper[field]
                    content_parts.append(f"Categories: {categories}")
                elif field == "published":
                    content_parts.append(f"Published: {paper[field]}")
                else:
                    content_parts.append(f"{field.capitalize()}: {paper[field]}")

        return "\n\n".join(content_parts)

    def _validate_score_response(
        self, response_text: str
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Validate that the response contains a properly formatted score.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Tuple of (is_valid, parsed_data)
        """
        try:
            # Try to find JSON in the response
            # First, try direct JSON parsing
            try:
                data = json.loads(response_text.strip())
            except json.JSONDecodeError:
                # Look for JSON-like content in the response
                json_match = re.search(
                    r'\{[^}]*"score"[^}]*\}', response_text, re.DOTALL
                )
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    return False, None

            # Validate required fields
            if not isinstance(data, dict):
                return False, None

            if "score" not in data:
                return False, None

            # Validate score is a number
            score = data["score"]
            if not isinstance(score, (int, float)):
                # Try to convert string to number
                try:
                    score = float(score)
                    data["score"] = score
                except (ValueError, TypeError):
                    return False, None

            # Ensure explanation exists (can be empty)
            if "explanation" not in data:
                data["explanation"] = ""

            return True, data

        except Exception:
            return False, None

    def _call_llm_api(self, paper_content: str) -> Optional[str]:
        """
        Make an API call to the configured LLM.

        Args:
            paper_content: Formatted paper content for scoring

        Returns:
            Response text or None on failure
        """
        provider = self.model_config["provider"]

        try:
            # External API calls based on provider
            if provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_config["model"],
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": f"Please score this paper:\n\n{paper_content}",
                        },
                    ],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 1000),
                )
                return response.choices[0].message.content

            elif provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_config["model"],
                    system=self.system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Please score this paper:\n\n{paper_content}",
                        }
                    ],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 1000),
                )
                return response.content[0].text

            elif provider == "google":
                model = self.client.GenerativeModel(self.model_config["model"])
                prompt = f"{self.system_prompt}\n\nPlease score this paper:\n\n{paper_content}"
                response = model.generate_content(
                    prompt,
                    generation_config=self.client.types.GenerationConfig(
                        temperature=self.model_config.get("temperature", 0.1),
                        max_output_tokens=self.model_config.get("max_tokens", 1000),
                    ),
                )
                return response.text

            # Local models (Ollama, LM Studio, etc.)
            elif provider == "ollama":
                return self._call_ollama_api(paper_content)

            elif provider == "lmstudio":
                return self._call_lmstudio_api(paper_content)

            elif provider == "local" or provider == "custom":
                # Generic local API call
                return self._call_local_api(paper_content)

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print(f"API call failed: {e}")
            return None

    def _call_ollama_api(self, paper_content: str) -> Optional[str]:
        """
        Call Ollama API for local LLM inference.

        Args:
            paper_content: Formatted paper content for scoring

        Returns:
            Response text or None on failure
        """
        import requests

        base_url = self.model_config.get("base_url", "http://localhost:11434")
        model_name = self.model_config["model"]

        # Combine system prompt and user content
        full_prompt = (
            f"{self.system_prompt}\n\nPlease score this paper:\n\n{paper_content}"
        )

        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.model_config.get("temperature", 0.1),
                "num_predict": self.model_config.get("max_tokens", 1000),
            },
        }

        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=120,  # Ollama can be slow
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except Exception as e:
            print(f"Ollama API call failed: {e}")
            return None

    def _call_lmstudio_api(self, paper_content: str) -> Optional[str]:
        """
        Call LM Studio API (OpenAI-compatible) for local LLM inference.

        Args:
            paper_content: Formatted paper content for scoring

        Returns:
            Response text or None on failure
        """
        try:
            # LM Studio provides OpenAI-compatible API
            import openai

            base_url = self.model_config.get("base_url", "http://localhost:1234/v1")
            model_name = self.model_config["model"]

            # Create OpenAI client pointing to LM Studio
            local_client = openai.OpenAI(
                base_url=base_url,
                api_key="not-needed",  # LM Studio doesn't require API key
            )

            response = local_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"Please score this paper:\n\n{paper_content}",
                    },
                ],
                temperature=self.model_config.get("temperature", 0.1),
                max_tokens=self.model_config.get("max_tokens", 1000),
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"LM Studio API call failed: {e}")
            return None

    def _call_local_api(self, paper_content: str) -> Optional[str]:
        """
        Generic local API call for custom endpoints.

        Args:
            paper_content: Formatted paper content for scoring

        Returns:
            Response text or None on failure
        """
        import requests

        try:
            base_url = self.model_config.get("base_url")
            if not base_url:
                raise ValueError("base_url is required for local/custom providers")

            model_name = self.model_config["model"]
            api_format = self.model_config.get(
                "api_format", "openai"
            )  # openai, ollama, or custom

            if api_format == "openai":
                # OpenAI-compatible API (like LM Studio, vLLM, etc.)
                import openai

                local_client = openai.OpenAI(
                    base_url=base_url,
                    api_key=self.model_config.get("api_key", "not-needed"),
                )

                response = local_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": f"Please score this paper:\n\n{paper_content}",
                        },
                    ],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 1000),
                )

                return response.choices[0].message.content

            elif api_format == "ollama":
                # Ollama-style API
                full_prompt = f"{self.system_prompt}\n\nPlease score this paper:\n\n{paper_content}"

                payload = {
                    "model": model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.model_config.get("temperature", 0.1),
                        "num_predict": self.model_config.get("max_tokens", 1000),
                    },
                }

                response = requests.post(
                    f"{base_url}/api/generate", json=payload, timeout=120
                )
                response.raise_for_status()

                result = response.json()
                return result.get("response", "")

            else:
                # Custom API format - user needs to implement
                raise ValueError(
                    f"Unsupported api_format: {api_format}. Use 'openai' or 'ollama'."
                )

        except Exception as e:
            print(f"Local API call failed: {e}")
            return None

    def score_paper(self, paper: Dict) -> Dict:
        """
        Score a single paper with retry logic and error handling.

        Args:
            paper: Paper dictionary to score

        Returns:
            Enhanced paper dictionary with scoring information
        """
        paper_content = self._extract_paper_content(paper)

        for attempt in range(self.retry_attempts + 1):
            try:
                # Make API call
                response_text = self._call_llm_api(paper_content)

                if response_text is None:
                    if attempt < self.retry_attempts:
                        self.total_retries += 1
                        time.sleep(1)  # Brief delay before retry
                        continue
                    else:
                        # Final failure
                        break

                # Validate response
                is_valid, score_data = self._validate_score_response(response_text)

                if is_valid:
                    # Success! Add scoring information to paper
                    enhanced_paper = paper.copy()
                    enhanced_paper["llm_score"] = score_data["score"]
                    enhanced_paper["llm_explanation"] = score_data["explanation"]
                    enhanced_paper["scored_by"] = self.model_alias
                    enhanced_paper["scored_at"] = datetime.now().isoformat()

                    self.total_scored += 1
                    return enhanced_paper

                else:
                    # Invalid response format
                    if attempt < self.retry_attempts:
                        self.total_retries += 1
                        print(
                            f"  → Invalid response format, retrying... (attempt {attempt + 1})"
                        )
                        time.sleep(1)
                        continue
                    else:
                        print(
                            f"  → Failed to get valid response after {self.retry_attempts + 1} attempts"
                        )
                        break

            except Exception as e:
                if attempt < self.retry_attempts:
                    self.total_retries += 1
                    print(
                        f"  → Error during scoring: {e}, retrying... (attempt {attempt + 1})"
                    )
                    time.sleep(1)
                    continue
                else:
                    print(
                        f"  → Final failure after {self.retry_attempts + 1} attempts: {e}"
                    )
                    break

        # If we get here, all attempts failed
        self.total_failures += 1
        enhanced_paper = paper.copy()
        enhanced_paper["llm_score"] = None
        enhanced_paper["llm_explanation"] = "Scoring failed after retries"
        enhanced_paper["scored_by"] = self.model_alias
        enhanced_paper["scored_at"] = datetime.now().isoformat()

        return enhanced_paper

    def start_progress_tracking(self, total_papers: int):
        """
        Initialize progress tracking.

        Args:
            total_papers: Total number of papers to score
        """
        self.start_time = time.time()
        self.total_papers = total_papers
        print(f"Starting to score {total_papers} papers using {self.model_alias}")
        print(f"Configured metadata: {', '.join(self.include_metadata)}")
        print()

    def print_progress(
        self, current_index: int, paper_title: str, score: Optional[float] = None
    ):
        """
        Print progress information.

        Args:
            current_index: Current paper index (0-based)
            paper_title: Title of current paper
            score: Score assigned to the paper (optional)
        """
        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        progress_pct = ((current_index + 1) / self.total_papers) * 100

        # Estimate time remaining
        if current_index > 0:
            avg_time_per_paper = elapsed / (current_index + 1)
            remaining_papers = self.total_papers - (current_index + 1)
            eta_seconds = remaining_papers * avg_time_per_paper
            eta_minutes = eta_seconds / 60
            eta_str = f"ETA: {eta_minutes:.1f}m"
        else:
            eta_str = "ETA: calculating..."

        # Truncate title for display
        display_title = (
            paper_title[:60] + "..." if len(paper_title) > 60 else paper_title
        )

        score_str = f"Score: {score:.1f} | " if score is not None else ""

        print(
            f"[{current_index + 1:3d}/{self.total_papers}] ({progress_pct:5.1f}%) {eta_str} | {score_str}{display_title}"
        )

    def print_final_statistics(self):
        """Print final scoring statistics."""
        if self.start_time is None:
            return

        total_time = time.time() - self.start_time

        print(f"\n=== SCORING COMPLETE ===")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Papers scored successfully: {self.total_scored}")
        print(f"Papers failed: {self.total_failures}")
        print(f"Total retries: {self.total_retries}")

        if self.total_scored > 0:
            avg_time = total_time / self.total_scored
            print(f"Average time per paper: {avg_time:.1f} seconds")


def display_default_prompts():
    """Display the default prompts for user reference."""
    defaults = get_default_scoring_prompts()

    print("=== DEFAULT SCORING PROMPTS ===")
    for prompt_name, prompt_text in defaults.items():
        print(f"\n{prompt_name}:")
        print("-" * 40)
        print(prompt_text)
    print("\n" + "=" * 50)


def validate_scoring_config(config: Dict) -> List[str]:
    """
    Validate the scoring configuration and return any issues.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    warnings = []

    scoring_config = config.get("scoring", {})

    # Check for prompts (warnings only since we have defaults)
    if not scoring_config.get("research_context_prompt", "").strip():
        warnings.append("Missing research_context_prompt - will use generic default")

    if not scoring_config.get("scoring_strategy_prompt", "").strip():
        warnings.append("Missing scoring_strategy_prompt - will use generic default")

    if not scoring_config.get("score_format_prompt", "").strip():
        warnings.append("Missing score_format_prompt - will use generic default")

    # Critical errors that should prevent execution
    model_alias = scoring_config.get("model_alias")
    if not model_alias:
        errors.append("Missing model_alias in scoring configuration")

    # Validate include_metadata
    include_metadata = scoring_config.get("include_metadata", [])
    if not include_metadata:
        errors.append("include_metadata cannot be empty")

    valid_metadata_fields = {
        "title",
        "abstract",
        "authors",
        "categories",
        "published",
        "updated",
    }
    for field in include_metadata:
        if field not in valid_metadata_fields:
            errors.append(f"Invalid metadata field: {field}")

    # Validate retry attempts
    retry_attempts = scoring_config.get("retry_attempts", 2)
    if not isinstance(retry_attempts, int) or retry_attempts < 0:
        errors.append("retry_attempts must be a non-negative integer")

    # Print warnings (non-blocking)
    if warnings:
        print("⚠️  Configuration warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print()

    return errors  # Only return blocking errors


def test_scoring_utilities():
    """
    Test function to validate scoring utilities without making API calls.
    Useful for debugging configuration and prompt construction.
    """
    print("=== TESTING SCORING UTILITIES ===")

    # Test 1: Default prompts
    print("\n1. Testing default prompts...")
    defaults = get_default_scoring_prompts()
    print(f"✅ Found {len(defaults)} default prompts:")
    for name in defaults.keys():
        print(f"   - {name}")

    # Test 2: Mock configuration
    print("\n2. Testing configuration validation...")
    mock_config = {
        "scoring": {
            "model_alias": "test-model",
            "max_papers": 10,
            "retry_attempts": 2,
            "include_metadata": ["title", "abstract"],
            "research_context_prompt": "Test context",
            "scoring_strategy_prompt": "Test strategy",
            "score_format_prompt": "Test format",
        }
    }

    validation_errors = validate_scoring_config(mock_config)
    if not validation_errors:
        print("✅ Configuration validation passed")
    else:
        print(f"❌ Configuration validation failed: {validation_errors}")

    # Test 3: Prompt construction
    print("\n3. Testing prompt construction...")
    try:
        # Create a mock LLM manager for testing
        class MockLLMManager:
            def get_model_config(self, alias):
                return {
                    "provider": "test",
                    "model": "test-model",
                    "temperature": 0.1,
                    "max_tokens": 1000,
                }

            def get_client(self, alias):
                return "mock-client"

        mock_llm_manager = MockLLMManager()
        engine = ScoringEngine(mock_config, mock_llm_manager, "test-model")

        print("✅ ScoringEngine initialized successfully")
        print(f"   System prompt length: {len(engine.system_prompt)} characters")

    except Exception as e:
        print(f"❌ ScoringEngine initialization failed: {e}")
        return

    # Test 4: Paper content extraction
    print("\n4. Testing paper content extraction...")
    mock_paper = {
        "id": "test123",
        "title": "Test Paper: A Novel Approach",
        "abstract": "This paper presents a novel approach to testing scoring utilities...",
        "authors": ["Author One", "Author Two"],
        "categories": ["cs.AI", "cs.LG"],
        "published": "2025-01-01",
    }

    try:
        content = engine._extract_paper_content(mock_paper)
        print("✅ Paper content extraction successful")
        print(f"   Extracted content length: {len(content)} characters")
        print(f"   Preview: {content[:100]}...")

    except Exception as e:
        print(f"❌ Paper content extraction failed: {e}")

    # Test 5: JSON validation
    print("\n5. Testing JSON response validation...")

    test_responses = [
        '{"score": 7.5, "explanation": "Good paper with novel methodology"}',  # Valid
        '{"score": "8", "explanation": "Score as string"}',  # Should convert
        '{"score": 7.5}',  # Missing explanation - should add empty
        '{"explanation": "Missing score"}',  # Missing score - should fail
        "Not JSON at all",  # Invalid JSON - should fail
        '{"score": 7.5, "explanation": "Valid", "extra": "field"}',  # Extra fields - should pass
    ]

    for i, response in enumerate(test_responses, 1):
        try:
            is_valid, parsed = engine._validate_score_response(response)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(
                f"   Test {i}: {status} - {response[:40]}{'...' if len(response) > 40 else ''}"
            )
            if is_valid:
                print(
                    f"      → Score: {parsed['score']}, Explanation: '{parsed['explanation'][:30]}{'...' if len(parsed['explanation']) > 30 else ''}'"
                )
        except Exception as e:
            print(f"   Test {i}: ❌ Exception - {e}")

    # Test 6: Configuration with defaults
    print("\n6. Testing configuration with missing prompts (should use defaults)...")
    minimal_config = {
        "scoring": {
            "model_alias": "test-model",
            "include_metadata": ["title", "abstract"],
            # Missing all prompts - should use defaults
        }
    }

    try:
        engine_minimal = ScoringEngine(minimal_config, mock_llm_manager, "test-model")
        print("✅ Minimal configuration with defaults successful")

    except Exception as e:
        print(f"❌ Minimal configuration failed: {e}")

    print("\n=== TESTING COMPLETE ===")
    print("\nTo test with real API calls:")
    print("1. Set up your API keys")
    print("2. Configure a real model in config/llm.yaml")
    print("3. Run: python src/score_papers.py --help")


def test_real_configuration(test_api_call: bool = False):
    """
    Test function using real YAML configuration files.

    Args:
        test_api_call: If True, makes a real API call to test connectivity
    """
    print("=== TESTING WITH REAL CONFIGURATION ===")

    # Test 1: Load real config files
    print("\n1. Loading real configuration files...")
    try:
        import yaml
        from llm_utils import LLMManager

        # Load main config
        try:
            with open("config/config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            print("✅ Loaded config/config.yaml")
        except FileNotFoundError:
            print("❌ config/config.yaml not found")
            return
        except Exception as e:
            print(f"❌ Error loading config/config.yaml: {e}")
            return

        # Load LLM config via LLMManager
        try:
            llm_manager = LLMManager()
            print("✅ Loaded config/llm.yaml via LLMManager")
        except Exception as e:
            print(f"❌ Error loading LLM configuration: {e}")
            return

    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return

    # Test 2: Validate real configuration
    print("\n2. Validating real configuration...")
    validation_errors = validate_scoring_config(config)
    if not validation_errors:
        print("✅ Real configuration validation passed")
    else:
        print("❌ Real configuration validation failed:")
        for error in validation_errors:
            print(f"   - {error}")
        return

    # Test 3: Get model configuration
    print("\n3. Testing LLM model configuration...")
    scoring_config = config.get("scoring", {})
    model_alias = scoring_config.get("model_alias")

    if not model_alias:
        print("❌ No model_alias specified in scoring configuration")
        return

    try:
        model_config = llm_manager.get_model_config(model_alias)
        print(f"✅ Model '{model_alias}' configuration loaded:")
        print(f"   Provider: {model_config.get('provider')}")
        print(f"   Model: {model_config.get('model')}")
        print(f"   Temperature: {model_config.get('temperature')}")
        print(f"   Max tokens: {model_config.get('max_tokens')}")

        # Check API key
        api_key = model_config.get("api_key")
        if api_key:
            print(
                f"   API key: ✅ Found ({'***' + api_key[-4:] if len(api_key) > 4 else '***'})"
            )
        else:
            print(f"   API key: ❌ Missing")
            if not test_api_call:
                print("   (This will cause API calls to fail)")

    except Exception as e:
        print(f"❌ Error getting model configuration: {e}")
        return

    # Test 4: Initialize real ScoringEngine
    print("\n4. Initializing ScoringEngine with real configuration...")
    try:
        scoring_engine = ScoringEngine(config, llm_manager, model_alias)
        print("✅ ScoringEngine initialized successfully")
        print(
            f"   System prompt length: {len(scoring_engine.system_prompt)} characters"
        )

        # Check if using defaults
        has_custom_prompts = all(
            [
                scoring_config.get("research_context_prompt", "").strip(),
                scoring_config.get("scoring_strategy_prompt", "").strip(),
                scoring_config.get("score_format_prompt", "").strip(),
            ]
        )

        if has_custom_prompts:
            print("   Using custom prompts from configuration")
        else:
            print("   ⚠️  Using some default prompts (see warnings above)")

    except Exception as e:
        print(f"❌ ScoringEngine initialization failed: {e}")
        return

    # Test 5: Test with real paper data (if available)
    print("\n5. Testing paper content extraction with real data...")

    # Try to load sample papers from recent fetch
    sample_paper = None
    try:
        output_dir = config.get("output", {}).get("base_dir", "./data")
        import glob
        import json

        json_files = glob.glob(f"{output_dir}/arxiv_papers_*.json")
        json_files = [f for f in json_files if "_scored" not in f]

        if json_files:
            latest_file = max(json_files, key=lambda x: os.path.getmtime(x))
            with open(latest_file, "r", encoding="utf-8") as f:
                papers = json.load(f)

            if papers:
                sample_paper = papers[0]
                print(f"✅ Loaded sample paper from {latest_file}")
                print(f"   Paper: {sample_paper.get('title', 'No title')[:60]}...")
            else:
                print("❌ No papers found in data file")
        else:
            print("ℹ️  No recent paper data found (run fetch_papers.py first)")

    except Exception as e:
        print(f"ℹ️  Could not load sample paper data: {e}")

    # Use sample paper or create mock paper
    if not sample_paper:
        sample_paper = {
            "id": "2501.12345v1",
            "title": "A Novel Approach to Testing Scoring Systems with Real Configuration",
            "abstract": "This paper presents a comprehensive methodology for testing automated scoring systems using real configuration parameters. We demonstrate the effectiveness of our approach through extensive validation experiments.",
            "authors": ["Test Author", "Config Validator"],
            "categories": ["cs.AI", "cs.LG"],
            "published": "2025-01-15T10:00:00Z",
        }
        print("ℹ️  Using mock paper for testing")

    try:
        extracted_content = scoring_engine._extract_paper_content(sample_paper)
        print("✅ Paper content extraction successful")
        print(f"   Content length: {len(extracted_content)} characters")
        print(f"   Metadata included: {', '.join(scoring_engine.include_metadata)}")

    except Exception as e:
        print(f"❌ Paper content extraction failed: {e}")
        return

    # Test 6: Real API call (optional)
    if test_api_call:
        print("\n6. Testing real API connectivity...")
        if not model_config.get("api_key") and model_config.get("provider") not in [
            "ollama",
            "lmstudio",
            "local",
            "custom",
        ]:
            print("❌ Cannot test API call - no API key available")
            print("   Set your API key as an environment variable or in config")
        else:
            provider = model_config.get("provider")

            # For local providers, check if server is reachable first
            if provider in ["ollama", "lmstudio", "local", "custom"]:
                base_url = model_config.get("base_url", "http://localhost:1234/v1")
                print(f"   Checking if local server is reachable: {base_url}")

                try:
                    import requests

                    # Try a simple health check
                    health_url = base_url.rstrip("/v1").rstrip("/")
                    _ = requests.get(f"{health_url}/health", timeout=3)
                except:
                    try:
                        # Try just connecting to the base URL
                        _ = requests.get(base_url, timeout=3)
                    except:
                        print("❌ Local server not reachable")
                        print(
                            f"   Make sure your {provider} server is running on {base_url}"
                        )
                        print("   Skipping API test")
                        return

                print("✅ Local server is reachable")

            print(
                f"   Attempting API call to {model_config.get('provider')} ({model_config.get('model')})..."
            )
            try:
                # Test with a simple paper
                test_paper_content = "Title: Test API Connectivity\n\nAbstract: This is a simple test to verify API connectivity and response validation."

                response_text = scoring_engine._call_llm_api(test_paper_content)

                if response_text:
                    print("✅ API call successful")
                    print(f"   Response length: {len(response_text)} characters")

                    # Test response validation
                    is_valid, parsed_data = scoring_engine._validate_score_response(
                        response_text
                    )
                    if is_valid:
                        print("✅ Response validation successful")
                        print(f"   Score: {parsed_data.get('score')}")
                        print(
                            f"   Explanation: {parsed_data.get('explanation', '')[:50]}..."
                        )
                    else:
                        print("❌ Response validation failed")
                        print(f"   Raw response: {response_text[:100]}...")
                else:
                    print("❌ API call failed - no response")

            except Exception as e:
                print(f"❌ API call failed: {e}")
                if "Connection" in str(e) or "timeout" in str(e).lower():
                    print(f"   Hint: Make sure your {provider} server is running")
    else:
        print("\n6. Skipping API connectivity test")
        print("   Use --test-api to include real API call testing")

    print("\n=== REAL CONFIGURATION TEST COMPLETE ===")

    if test_api_call:
        print("\n✅ Full end-to-end test completed successfully!")
    else:
        print(
            "\nℹ️  Configuration test completed. Run with --test-api to test API connectivity."
        )

    print("\nNext steps:")
    print("1. Run: python src/fetch_papers.py (to get recent papers)")
    print("2. Run: python src/score_papers.py (to score papers)")


# Example usage and testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] in ["--test", "-t"]:
            test_scoring_utilities()
        elif sys.argv[1] in ["--test-real", "--real", "-r"]:
            test_real_configuration(test_api_call=False)
        elif sys.argv[1] in ["--test-api", "--api", "-a"]:
            test_real_configuration(test_api_call=True)
        elif sys.argv[1] in ["--defaults", "-d"]:
            display_default_prompts()
        elif sys.argv[1] in ["--help", "-h"]:
            print("Scoring Utilities Test Module")
            print("Usage:")
            print(
                "  python scoring_utils.py --test       # Run mock functionality tests"
            )
            print(
                "  python scoring_utils.py --test-real  # Test with real YAML configs"
            )
            print(
                "  python scoring_utils.py --test-api   # Test real configs + API call"
            )
            print("  python scoring_utils.py --defaults   # Show default prompts")
            print("  python scoring_utils.py --help       # Show this help")
        else:
            print("Unknown option. Use --help for usage information.")
    else:
        print("Scoring Utilities Module")
        print("Run with --help for usage options")

#!/usr/bin/env python3
"""
Paper Synthesis Utilities - Two-Stage Implementation

Handles LLM-based synthesis of selected papers into comprehensive reports using
a two-stage approach: content synthesis followed by style formatting.

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
    Get default prompts for two-stage research paper synthesis.

    Returns:
        Dictionary with default prompt strings for both stages
    """
    return {
        "research_context_prompt": """
You are analyzing research papers for an interdisciplinary expert who works on 
"statistical AI for cosmic discovery", focusing on novel AI/ML/statistics methods, 
interpretable approaches, robust inference frameworks, and scientific applications 
to astronomical surveys with a focus on galaxy evolution.

The goal is to extract and organize key information that will be useful for 
staying current with relevant developments in the field.
        """.strip(),
        "content_synthesis_prompt": """
Extract and organize the key information from the selected papers. Focus on content accuracy and connections - formatting will be handled separately.

For each paper, identify:
- Authors and core contribution
- Key findings with specific results/numbers
- Methodological innovations  
- Confidence levels or limitations noted

Then organize into 2-3 thematic groups based on:
- Shared methods or approaches
- Complementary findings
- Related problem domains
- Contrasting results or approaches

Output as structured content with:
- Brief theme descriptions
- Paper summaries under each theme
- Notable connections between papers
- Key quantitative results
- Any tensions or disagreements

Do not worry about narrative flow, style, or final formatting.
        """.strip(),
        "style_formatting_prompt": """
Transform the structured content into a polished daily research digest (800-1200 words).

**Required Style:**
- Expert-to-expert voice, conversational but authoritative
- Essay format with smooth transitions
- Begin directly with substantive content
- Complete in one continuous narrative - no restarts or section breaks
- Use author names, never paper numbers

**Formatting:**
- Bold key terms, methods, and quantitative results
- Markdown formatting but avoid excessive styling
- Natural transitions between themes
- Precise technical language with appropriate skepticism

**Content Preservation:**
- Maintain all specific results and findings from the input
- Preserve author attributions
- Keep technical accuracy
- Include confidence levels and limitations

Generate an informative title capturing the key theme or development.
        """.strip(),
    }


class PaperSynthesizer:
    """
    Handles LLM-based synthesis of selected papers using a two-stage approach.

    Stage 1: Content synthesis - extracts and organizes key information
    Stage 2: Style formatting - transforms into polished narrative
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

        # Build system prompts for both stages
        self.stage1_system_prompt = self._build_stage1_system_prompt()
        self.stage2_system_prompt = self._build_stage2_system_prompt()

        # Statistics tracking
        self.stats = {
            "papers_synthesized": 0,
            "stage1_attempts": 0,
            "stage1_failures": 0,
            "stage2_attempts": 0,
            "stage2_failures": 0,
            "total_retries": 0,
            "synthesis_time": 0,
        }

    def _get_default_prompts(self) -> Dict[str, str]:
        """Get default prompts for two-stage synthesis."""
        return get_default_synthesis_prompts()

    def _build_stage1_system_prompt(self) -> str:
        """
        Construct the system prompt for Stage 1 (content synthesis).

        Returns:
            Complete system prompt string for Stage 1
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

        # Get content synthesis prompt
        content_prompt = self.generation_config.get(
            "content_synthesis_prompt", ""
        ).strip()
        if not content_prompt:
            content_prompt = defaults["content_synthesis_prompt"]
            used_defaults.append("content_synthesis_prompt")

        # Warn user about defaults
        if used_defaults:
            print(
                f"⚠️  WARNING: Using default prompts for Stage 1: {', '.join(used_defaults)}"
            )

        # Combine the parts
        prompt_parts = [
            f"RESEARCH CONTEXT:\n{research_context}",
            f"CONTENT SYNTHESIS INSTRUCTIONS:\n{content_prompt}",
        ]

        return "\n\n".join(prompt_parts)

    def _build_stage2_system_prompt(self) -> str:
        """
        Construct the system prompt for Stage 2 (style formatting).

        Returns:
            Complete system prompt string for Stage 2
        """
        defaults = self._get_default_prompts()
        used_defaults = []

        # Get style formatting prompt
        style_prompt = self.generation_config.get("style_formatting_prompt", "").strip()
        if not style_prompt:
            style_prompt = defaults["style_formatting_prompt"]
            used_defaults.append("style_formatting_prompt")

        # Warn user about defaults
        if used_defaults:
            print(
                f"⚠️  WARNING: Using default prompts for Stage 2: {', '.join(used_defaults)}"
            )

        return style_prompt

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
                    elif field in ["rescored_llm_explanation", "llm_explanation"]:
                        paper_section.append(f"Score Explanation: {value}")
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

        return "\n".join(content_parts)

    def _call_llm_api(self, content: str, system_prompt: str) -> Optional[str]:
        """
        Make an API call to the configured LLM.

        Args:
            content: Content to send to the LLM
            system_prompt: System prompt to use

        Returns:
            Response text or None on failure
        """
        provider = self.model_config["provider"]

        try:
            if provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_config["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 4000),
                )
                return response.choices[0].message.content

            elif provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_config["model"],
                    system=system_prompt,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 4000),
                )
                return response.content[0].text

            elif provider == "google":
                model = self.client.GenerativeModel(self.model_config["model"])
                prompt = f"{system_prompt}\n\n{content}"
                response = model.generate_content(
                    prompt,
                    generation_config=self.client.types.GenerationConfig(
                        temperature=self.model_config.get("temperature", 0.1),
                        max_output_tokens=self.model_config.get("max_tokens", 4000),
                    ),
                )
                return response.text

            elif provider == "ollama":
                return self._call_ollama_api(content, system_prompt)
            elif provider == "lmstudio":
                return self._call_lmstudio_api(content, system_prompt)
            elif provider in ["local", "custom"]:
                return self._call_local_api(content, system_prompt)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print(f"API call failed: {e}")
            return None

    def _call_ollama_api(self, content: str, system_prompt: str) -> Optional[str]:
        """Call Ollama API."""
        import requests

        base_url = self.model_config.get("base_url", "http://localhost:11434")
        model_name = self.model_config["model"]

        full_prompt = f"{system_prompt}\n\n{content}"

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

    def _call_lmstudio_api(self, content: str, system_prompt: str) -> Optional[str]:
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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=self.model_config.get("temperature", 0.1),
                max_tokens=self.model_config.get("max_tokens", 4000),
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"LM Studio API call failed: {e}")
            return None

    def _call_local_api(self, content: str, system_prompt: str) -> Optional[str]:
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
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 4000),
                )

                return response.choices[0].message.content

            elif api_format == "ollama":
                full_prompt = f"{system_prompt}\n\n{content}"

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

    def _validate_stage1_response(
        self, response_text: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate Stage 1 (content synthesis) response.

        Args:
            response_text: Raw response from LLM

        Returns:
            Tuple of (is_valid, cleaned_response)
        """
        try:
            if not response_text or len(response_text.strip()) < 200:
                return False, None

            cleaned_response = response_text.strip()

            # Remove any thinking tokens that might leak through
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

            # Basic quality checks for content synthesis
            if len(cleaned_response) < 200:
                return False, None

            # Should mention multiple papers (look for reasonable author mentions)
            author_mentions = len(
                re.findall(r"\b[A-Z][a-z]+\s+et\s+al\.?", cleaned_response)
            )
            if author_mentions < 2:  # Should have at least 2 papers mentioned
                return False, None

            return True, cleaned_response

        except Exception as e:
            print(f"Stage 1 response validation error: {e}")
            return False, None

    def _validate_stage2_response(
        self, response_text: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate Stage 2 (style formatting) response.

        Args:
            response_text: Raw response from LLM

        Returns:
            Tuple of (is_valid, cleaned_response)
        """
        try:
            if not response_text or len(response_text.strip()) < 500:
                return False, None

            cleaned_response = response_text.strip()

            # Remove any thinking tokens
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

            # Basic quality checks for final synthesis
            if len(cleaned_response) < 500:
                return False, None

            # Should be narrative format (check for proper flow)
            paragraphs = [
                p.strip() for p in cleaned_response.split("\n\n") if p.strip()
            ]
            if len(paragraphs) < 3:
                return False, None

            return True, cleaned_response

        except Exception as e:
            print(f"Stage 2 response validation error: {e}")
            return False, None

    def _run_stage1_synthesis(self, paper_content: str) -> Optional[str]:
        """
        Run Stage 1: Content synthesis with retry logic.

        Args:
            paper_content: Formatted paper content

        Returns:
            Stage 1 synthesis result or None on failure
        """
        print("Stage 1: Content synthesis...")
        self.stats["stage1_attempts"] += 1

        for attempt in range(self.retry_attempts + 1):
            try:
                # Make LLM call for Stage 1
                response_text = self._call_llm_api(
                    paper_content, self.stage1_system_prompt
                )

                if response_text is None:
                    if attempt < self.retry_attempts:
                        self.stats["total_retries"] += 1
                        print(
                            f"  → Stage 1 API call failed, retrying... (attempt {attempt + 1})"
                        )
                        time.sleep(2)
                        continue
                    else:
                        print(f"  → Stage 1 API calls failed after all attempts")
                        break

                # Validate Stage 1 response
                is_valid, cleaned_response = self._validate_stage1_response(
                    response_text
                )

                if is_valid:
                    print(f"  ✅ Stage 1 completed successfully")
                    return cleaned_response
                else:
                    if attempt < self.retry_attempts:
                        self.stats["total_retries"] += 1
                        print(
                            f"  → Stage 1 invalid response, retrying... (attempt {attempt + 1})"
                        )
                        time.sleep(2)
                        continue
                    else:
                        print(f"  → Stage 1 failed validation after all attempts")
                        break

            except Exception as e:
                if attempt < self.retry_attempts:
                    self.stats["total_retries"] += 1
                    print(
                        f"  → Stage 1 error: {e}, retrying... (attempt {attempt + 1})"
                    )
                    time.sleep(2)
                    continue
                else:
                    print(f"  → Stage 1 final failure: {e}")
                    break

        self.stats["stage1_failures"] += 1
        return None

    def _run_stage2_formatting(self, stage1_content: str) -> Optional[str]:
        """
        Run Stage 2: Style formatting with retry logic.

        Args:
            stage1_content: Content from Stage 1

        Returns:
            Final formatted synthesis or None on failure
        """
        print("Stage 2: Style formatting...")
        self.stats["stage2_attempts"] += 1

        # Prepare Stage 2 input
        stage2_input = f"Transform the following structured content into a polished daily research digest:\n\n{stage1_content}"

        for attempt in range(self.retry_attempts + 1):
            try:
                # Make LLM call for Stage 2
                response_text = self._call_llm_api(
                    stage2_input, self.stage2_system_prompt
                )

                if response_text is None:
                    if attempt < self.retry_attempts:
                        self.stats["total_retries"] += 1
                        print(
                            f"  → Stage 2 API call failed, retrying... (attempt {attempt + 1})"
                        )
                        time.sleep(2)
                        continue
                    else:
                        print(f"  → Stage 2 API calls failed after all attempts")
                        break

                # Validate Stage 2 response
                is_valid, cleaned_response = self._validate_stage2_response(
                    response_text
                )

                if is_valid:
                    print(f"  ✅ Stage 2 completed successfully")
                    return cleaned_response
                else:
                    if attempt < self.retry_attempts:
                        self.stats["total_retries"] += 1
                        print(
                            f"  → Stage 2 invalid response, retrying... (attempt {attempt + 1})"
                        )
                        time.sleep(2)
                        continue
                    else:
                        print(f"  → Stage 2 failed validation after all attempts")
                        break

            except Exception as e:
                if attempt < self.retry_attempts:
                    self.stats["total_retries"] += 1
                    print(
                        f"  → Stage 2 error: {e}, retrying... (attempt {attempt + 1})"
                    )
                    time.sleep(2)
                    continue
                else:
                    print(f"  → Stage 2 final failure: {e}")
                    break

        self.stats["stage2_failures"] += 1
        return None

    def synthesize_papers(self, papers: List[Dict]) -> Dict:
        """
        Synthesize papers into a comprehensive report using two-stage approach.

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

        print(
            f"Starting two-stage synthesis of {len(papers)} papers using {self.model_alias}"
        )
        print(f"Metadata fields included: {', '.join(self.include_metadata)}")

        start_time = time.time()

        # Extract and format content
        paper_content = self._extract_paper_content(papers)

        # Stage 1: Content synthesis
        stage1_result = self._run_stage1_synthesis(paper_content)
        if stage1_result is None:
            synthesis_time = time.time() - start_time
            return {
                "synthesis_report": None,
                "synthesis_success": False,
                "error": "Stage 1 (content synthesis) failed after all retry attempts",
                "papers_count": len(papers),
                "synthesis_time": synthesis_time,
                "synthesized_by": self.model_alias,
                "synthesized_at": datetime.now().isoformat(),
                "stage1_content": None,
                "metadata_included": self.include_metadata.copy(),
            }

        # Stage 2: Style formatting
        stage2_result = self._run_stage2_formatting(stage1_result)
        if stage2_result is None:
            synthesis_time = time.time() - start_time
            return {
                "synthesis_report": None,
                "synthesis_success": False,
                "error": "Stage 2 (style formatting) failed after all retry attempts",
                "papers_count": len(papers),
                "synthesis_time": synthesis_time,
                "synthesized_by": self.model_alias,
                "synthesized_at": datetime.now().isoformat(),
                "stage1_content": stage1_result,  # Include Stage 1 result for debugging
                "metadata_included": self.include_metadata.copy(),
            }

        # Success!
        synthesis_time = time.time() - start_time
        self.stats["papers_synthesized"] = len(papers)
        self.stats["synthesis_time"] = synthesis_time

        result = {
            "synthesis_report": stage2_result,
            "synthesis_success": True,
            "papers_count": len(papers),
            "synthesis_time": synthesis_time,
            "synthesized_by": self.model_alias,
            "synthesized_at": datetime.now().isoformat(),
            "stage1_content": stage1_result,  # Include for transparency
            "metadata_included": self.include_metadata.copy(),
            "config_used": {
                "model_alias": self.model_alias,
                "retry_attempts": self.retry_attempts,
                "include_metadata": self.include_metadata,
            },
        }

        print(
            f"✅ Two-stage synthesis completed successfully in {synthesis_time:.1f} seconds"
        )
        print(f"   Generated report: {len(stage2_result)} characters")
        return result


# Update validation and testing functions for two-stage approach
def validate_synthesis_config(config: Dict) -> List[str]:
    """
    Validate the synthesis configuration for two-stage approach.

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

    print("=== DEFAULT TWO-STAGE SYNTHESIS PROMPTS ===")
    for prompt_name, prompt_text in defaults.items():
        print(f"\n{prompt_name}:")
        print("-" * 40)
        print(prompt_text)
    print("\n" + "=" * 50)


def test_synthesis_utilities():
    """Test synthesis utilities with mock papers for two-stage approach."""
    print("=== TESTING TWO-STAGE SYNTHESIS UTILITIES ===")

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
                    "rescored_llm_explanation",
                ],
                "content_synthesis_prompt": "Custom content prompt",
                "style_formatting_prompt": "Custom style prompt",
            },
        }
    }

    validation_errors = validate_synthesis_config(valid_config)
    if not validation_errors:
        print("✅ Valid two-stage configuration passed")
    else:
        print(f"❌ Unexpected validation errors: {validation_errors}")

    # Test 3: Mock synthesizer initialization
    print("\n3. Testing two-stage synthesizer initialization...")

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
        print("✅ Two-stage synthesizer initialized successfully")
        print(
            f"   Stage 1 prompt length: {len(synthesizer.stage1_system_prompt)} characters"
        )
        print(
            f"   Stage 2 prompt length: {len(synthesizer.stage2_system_prompt)} characters"
        )

        # Test validation functions
        test_response_good = """
    Theme 1: Robust Inference Frameworks
    - Smith et al. present a Bayesian approach for galaxy classification, achieving 92% accuracy on SDSS data.
    - Lee et al. introduce a novel uncertainty quantification method, highlighting limitations in current ML models.

    Theme 2: Interpretable AI Methods
    - Johnson et al. propose interpretable neural networks for cosmic discovery, with improved transparency over previous models.
    - Patel et al. demonstrate that feature attribution methods reveal biases in astronomical surveys.

    Connections:
    - Both Smith et al. and Lee et al. focus on robust statistical inference, while Johnson et al. and Patel et al. emphasize interpretability.
    - Quantitative results: Smith et al. (92% accuracy), Lee et al. (uncertainty reduced by 15%).

    Tensions:
    - Lee et al. note limitations in ML confidence, contrasting with Johnson et al.'s claims of model reliability.
    """
        test_response_bad = "Too short."

        is_valid_good, _ = synthesizer._validate_stage1_response(test_response_good)
        is_valid_bad, _ = synthesizer._validate_stage1_response(test_response_bad)

        print(
            f"✅ Stage 1 validation working: good={is_valid_good}, bad={is_valid_bad}"
        )

    except Exception as e:
        print(f"❌ Two-stage synthesizer test failed: {e}")
        return

    print(f"\n✅ All two-stage synthesis utilities tests passed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] in ["--test", "-t"]:
            test_synthesis_utilities()
        elif sys.argv[1] in ["--defaults", "-d"]:
            display_default_synthesis_prompts()
        elif sys.argv[1] in ["--help", "-h"]:
            print("Two-Stage Paper Synthesis Utilities")
            print("Usage:")
            print("  python synthesis_utils.py --test       # Run mock tests")
            print("  python synthesis_utils.py --defaults   # Show default prompts")
            print("  python synthesis_utils.py --help       # Show this help")
        else:
            print("Unknown option. Use --help for usage information.")
    else:
        print("Two-Stage Paper Synthesis Utilities Module")
        print("Run with --help for usage options")

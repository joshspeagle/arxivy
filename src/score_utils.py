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
import yaml


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
        "score_calculation_prompt": """
Rate each component from 1-10:
- Relevance score (1-10): ___
- Contribution score (1-10): ___
- Impact score (1-10): ___

Calculate: Final Score = (Relevance × 0.4) + (Contribution × 0.3) + (Impact × 0.3)
Round to one decimal place.

Be decisive about scores. Make sure to use the full scoring range effectively for each component.
        """.strip(),
    }


def get_single_paper_json_format() -> str:
    """Get JSON format instructions for single paper scoring."""
    return """
IMPORTANT: Provide result as JSON:
{
  "score": <final_calculated_score>,
  "explanation": "Brief 1-3 sentence assessment covering all three components."
}

Do not include any text outside of this JSON structure.""".strip()


def get_batch_json_format() -> str:
    """Get JSON format instructions for batch paper scoring."""
    return """
IMPORTANT: Provide result as JSON array with one entry per paper:
[
  {
    "paper_id": "paper_1",
    "score": <final_calculated_score>,
    "explanation": "Brief 1-3 sentence assessment covering all three components."
  },
  {
    "paper_id": "paper_2", 
    "score": <final_calculated_score>,
    "explanation": "Brief 1-3 sentence assessment covering all three components."
  }
]

The paper_id must exactly match the paper identifiers provided in the input (paper_1, paper_2, etc.).
Do not include any text outside of this JSON structure.""".strip()


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

        # Batch configuration - default to single paper mode
        self.batch_size = self.scoring_config.get("batch_size")
        if self.batch_size is None:
            self.batch_size = 1
        elif not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer or null")

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
        Construct the complete system prompt from the three configured parts,
        automatically appending the appropriate JSON format based on batch_size.

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

        # Get user's score format prompt (without JSON format)
        score_calculation = self.scoring_config.get(
            "score_calculation_prompt", ""
        ).strip()
        if not score_calculation:
            score_calculation = defaults["score_calculation_prompt"]
            used_defaults.append("score_calculation_prompt")

        # Append appropriate JSON format based on batch size
        if self.batch_size == 1:
            json_format = get_single_paper_json_format()
        else:
            json_format = get_batch_json_format()

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
            f"SCORE FORMAT:\n{score_calculation}",
            f"OUTPUT FORMAT:\n{json_format}",
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

    def _extract_batch_content(self, papers: List[Dict]) -> str:
        """
        Extract content for multiple papers formatted for batch scoring.

        Args:
            papers: List of paper dictionaries

        Returns:
            Formatted string with all papers labeled for batch processing
        """
        if len(papers) == 1:
            return self._extract_paper_content(papers[0])

        content_parts = []
        for i, paper in enumerate(papers, 1):
            paper_id = f"paper_{i}"
            paper_content = self._extract_paper_content(paper)
            content_parts.append(f"=== {paper_id.upper()} ===\n{paper_content}")

        return "\n\n".join(content_parts)

    def _validate_score_response(
        self, response_text: str, expected_count: int = 1
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Validate that the response contains properly formatted score(s).

        Args:
            response_text: Raw response text from LLM
            expected_count: Expected number of papers (1 for single, >1 for batch)

        Returns:
            Tuple of (is_valid, parsed_data)
            For single papers: parsed_data is a dict
            For batches: parsed_data is a list of dicts
        """
        try:
            # Clean up the response text first
            cleaned_text = response_text.strip()

            # Remove markdown code blocks if present
            if cleaned_text.startswith("```"):
                # Find the start and end of the code block
                lines = cleaned_text.split("\n")
                start_idx = 0
                end_idx = len(lines)

                # Find first line that's not a code block marker
                for i, line in enumerate(lines):
                    if not line.strip().startswith("```"):
                        start_idx = i
                        break

                # Find last line that's not a code block marker
                for i in range(len(lines) - 1, -1, -1):
                    if not lines[i].strip().startswith("```"):
                        end_idx = i + 1
                        break

                cleaned_text = "\n".join(lines[start_idx:end_idx])

            # Try to parse JSON directly
            try:
                data = json.loads(cleaned_text)
            except json.JSONDecodeError:
                # Look for JSON-like content in the response
                if expected_count == 1:
                    json_match = re.search(
                        r'\{[^}]*"score"[^}]*\}', cleaned_text, re.DOTALL
                    )
                else:
                    json_match = re.search(
                        r'\[[^\]]*"score"[^\]]*\]', cleaned_text, re.DOTALL
                    )

                if json_match:
                    data = json.loads(json_match.group())
                else:
                    return False, None

            # Validate based on expected count
            if expected_count == 1:
                return self._validate_single_response(data)
            else:
                return self._validate_batch_response(data, expected_count)

        except Exception as e:
            print(f"   Debug - JSON parsing error: {e}")
            print(f"   Debug - Response preview: {response_text[:200]}...")
            return False, None

    def _validate_single_response(self, data) -> Tuple[bool, Optional[Dict]]:
        """Validate single paper response."""
        if not isinstance(data, dict):
            return False, None

        if "score" not in data:
            return False, None

        # Validate score is a number
        score = data["score"]
        if not isinstance(score, (int, float)):
            try:
                score = float(score)
                data["score"] = score
            except (ValueError, TypeError):
                return False, None

        # Ensure explanation exists (can be empty)
        if "explanation" not in data:
            data["explanation"] = ""

        return True, data

    def _validate_batch_response(
        self, data, expected_count: int
    ) -> Tuple[bool, Optional[List[Dict]]]:
        """Validate batch response."""
        if not isinstance(data, list):
            return False, None

        if len(data) != expected_count:
            return False, None

        # Validate each entry
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                return False, None
            if "paper_id" not in entry or "score" not in entry:
                return False, None

            # Validate paper_id matches expected format
            paper_id = entry["paper_id"]
            expected_formats = [f"paper_{i+1}", f"PAPER_{i+1}", f"Paper_{i+1}"]
            if paper_id not in expected_formats:
                print(
                    f"   Debug - Expected paper_id: {expected_formats}, got: {paper_id}"
                )
                return False, None

            # Validate score
            score = entry["score"]
            if not isinstance(score, (int, float)):
                try:
                    score = float(score)
                    entry["score"] = score
                except (ValueError, TypeError):
                    return False, None

            # Ensure explanation exists
            if "explanation" not in entry:
                entry["explanation"] = ""

        return True, data

    def _call_llm_api(self, paper_content: str) -> Optional[str]:
        """
        Make an API call to the configured LLM.
        Handles both single paper and batch content automatically.

        Args:
            paper_content: Formatted paper content for scoring (single or batch)

        Returns:
            Response text or None on failure
        """
        provider = self.model_config["provider"]

        # Determine if this is a batch request based on content
        is_batch = "=== PAPER_" in paper_content.upper()
        content_type = "batch" if is_batch else "single paper"

        try:
            # External API calls based on provider
            if provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_config["model"],
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": f"Please score {'these papers' if is_batch else 'this paper'}:\n\n{paper_content}",
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
                            "content": f"Please score {'these papers' if is_batch else 'this paper'}:\n\n{paper_content}",
                        }
                    ],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 1000),
                )
                return response.content[0].text

            elif provider == "google":
                model = self.client.GenerativeModel(self.model_config["model"])
                prompt = f"{self.system_prompt}\n\nPlease score {'these papers' if is_batch else 'this paper'}:\n\n{paper_content}"
                response = model.generate_content(
                    prompt,
                    generation_config=self.client.types.GenerationConfig(
                        temperature=self.model_config.get("temperature", 0.1),
                        max_output_tokens=self.model_config.get("max_tokens", 1000),
                    ),
                )
                return response.text

            # Local models (no changes needed - they handle batch content the same way)
            elif provider == "ollama":
                return self._call_ollama_api(paper_content)
            elif provider == "lmstudio":
                return self._call_lmstudio_api(paper_content)
            elif provider == "local" or provider == "custom":
                return self._call_local_api(paper_content)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print(f"API call failed ({content_type}): {e}")
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

    def score_papers_batch(self, papers: List[Dict]) -> List[Dict]:
        """
        Score papers in batches for efficiency.

        Args:
            papers: List of papers to score

        Returns:
            List of scored papers
        """
        scored_papers = []

        # Initialize progress tracking
        self.start_progress_tracking(len(papers))

        # Process in batches
        for i in range(0, len(papers), self.batch_size):
            batch = papers[i : i + self.batch_size]

            if self.batch_size == 1:
                # Use existing single-paper logic
                scored_paper = self.score_paper(batch[0])
                scored_papers.append(scored_paper)

                # Print progress for single paper
                title = scored_paper.get("title", "Untitled")
                score = scored_paper.get("llm_score")
                self.print_progress(i, title, score)

            else:
                # Use batch scoring logic
                batch_results = self._score_paper_batch(batch)
                scored_papers.extend(batch_results)

                # Print progress for batch
                self.print_batch_progress(i, batch_results)

        # Print final statistics
        self.print_final_statistics()

        return scored_papers

    def _score_paper_batch(self, papers: List[Dict]) -> List[Dict]:
        """
        Score a batch of papers in a single API call.

        Args:
            papers: List of papers to score (2+ papers)

        Returns:
            List of scored papers
        """
        batch_content = self._extract_batch_content(papers)

        for attempt in range(self.retry_attempts + 1):
            try:
                # Make API call
                response_text = self._call_llm_api(batch_content)

                if response_text is None:
                    if attempt < self.retry_attempts:
                        self.total_retries += 1
                        time.sleep(1)
                        continue
                    else:
                        break

                # Validate batch response
                is_valid, batch_data = self._validate_score_response(
                    response_text, len(papers)
                )

                if is_valid:
                    # Success! Map results back to papers
                    scored_papers = []
                    for i, (paper, score_data) in enumerate(zip(papers, batch_data)):
                        enhanced_paper = paper.copy()
                        enhanced_paper["llm_score"] = score_data["score"]
                        enhanced_paper["llm_explanation"] = score_data["explanation"]
                        enhanced_paper["scored_by"] = self.model_alias
                        enhanced_paper["scored_at"] = datetime.now().isoformat()
                        enhanced_paper["batch_index"] = i  # Track position in batch
                        scored_papers.append(enhanced_paper)

                    self.total_scored += len(papers)
                    return scored_papers

                else:
                    # Invalid response format
                    if attempt < self.retry_attempts:
                        self.total_retries += 1
                        print(
                            f"  → Invalid batch response format, retrying... (attempt {attempt + 1})"
                        )
                        time.sleep(1)
                        continue
                    else:
                        print(
                            f"  → Failed to get valid batch response after {self.retry_attempts + 1} attempts"
                        )
                        break

            except Exception as e:
                if attempt < self.retry_attempts:
                    self.total_retries += 1
                    print(
                        f"  → Error during batch scoring: {e}, retrying... (attempt {attempt + 1})"
                    )
                    time.sleep(1)
                    continue
                else:
                    print(
                        f"  → Final batch failure after {self.retry_attempts + 1} attempts: {e}"
                    )
                    break

        # If we get here, all attempts failed - return failed papers
        self.total_failures += len(papers)
        failed_papers = []
        for paper in papers:
            enhanced_paper = paper.copy()
            enhanced_paper["llm_score"] = None
            enhanced_paper["llm_explanation"] = "Batch scoring failed after retries"
            enhanced_paper["scored_by"] = self.model_alias
            enhanced_paper["scored_at"] = datetime.now().isoformat()
            failed_papers.append(enhanced_paper)

        return failed_papers

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

    def print_batch_progress(self, batch_start_index: int, scored_papers: List[Dict]):
        """
        Print progress information for a batch of papers.

        Args:
            batch_start_index: Starting index of the batch
            scored_papers: List of scored papers from the batch
        """
        if self.start_time is None:
            return

        # Calculate progress for the entire batch
        batch_end_index = batch_start_index + len(scored_papers) - 1
        elapsed = time.time() - self.start_time
        progress_pct = ((batch_end_index + 1) / self.total_papers) * 100

        # Estimate time remaining
        if batch_end_index > 0:
            avg_time_per_paper = elapsed / (batch_end_index + 1)
            remaining_papers = self.total_papers - (batch_end_index + 1)
            eta_seconds = remaining_papers * avg_time_per_paper
            eta_minutes = eta_seconds / 60
            eta_str = f"ETA: {eta_minutes:.1f}m"
        else:
            eta_str = "ETA: calculating..."

        # Print batch summary
        batch_scores = [
            p.get("llm_score") for p in scored_papers if p.get("llm_score") is not None
        ]
        if batch_scores:
            avg_score = sum(batch_scores) / len(batch_scores)
            score_range = f"{min(batch_scores):.1f}-{max(batch_scores):.1f}"
            score_str = f"Batch avg: {avg_score:.1f} (range: {score_range}) | "
        else:
            score_str = "Batch: FAILED | "

        print(
            f"[{batch_start_index + 1:3d}-{batch_end_index + 1:3d}/{self.total_papers}] ({progress_pct:5.1f}%) {eta_str} | {score_str}Scored {len(scored_papers)} papers"
        )

        # Optionally print individual paper titles in batch
        if len(scored_papers) <= 3:  # Only for small batches
            for i, paper in enumerate(scored_papers):
                title = paper.get("title", "Untitled")
                score = paper.get("llm_score")
                display_title = title[:50] + "..." if len(title) > 50 else title
                score_str = f"{score:.1f}" if score is not None else "FAIL"
                print(f"    └─ {score_str}: {display_title}")

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

    if not scoring_config.get("score_calculation_prompt", "").strip():
        warnings.append("Missing score_calculation_prompt - will use generic default")

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

    # Validate batch_size
    batch_size = scoring_config.get("batch_size")
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size < 1:
            errors.append("batch_size must be a positive integer or null")
        elif batch_size > 10:
            warnings.append(
                f"Large batch_size ({batch_size}) may cause API timeouts or token limit issues"
            )

    # Print warnings (non-blocking)
    if warnings:
        print("⚠️  Configuration warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print()

    return errors  # Only return blocking errors


def validate_rescoring_config(config: Dict) -> List[str]:
    """
    Validate the rescoring configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    rescoring_config = config.get("rescoring", {})
    scoring_config = config.get("scoring", {})

    # Check required parameters
    model_alias = rescoring_config.get("model_alias")
    if not model_alias:
        errors.append("model_alias is required in rescoring configuration")

    # Validate retry attempts
    retry_attempts = rescoring_config.get("retry_attempts", 2)
    if not isinstance(retry_attempts, int) or retry_attempts < 0:
        errors.append("retry_attempts must be a non-negative integer")

    # Validate include_metadata
    include_metadata = rescoring_config.get("include_metadata", [])
    if not include_metadata:
        errors.append("include_metadata cannot be empty")

    valid_metadata_fields = {
        "title",
        "abstract",
        "authors",
        "categories",
        "published",
        "updated",
        "llm_summary",
        "summary_key_contributions",
        "summary_confidence",
        "llm_score",
    }
    for field in include_metadata:
        if field not in valid_metadata_fields:
            errors.append(f"Invalid metadata field: {field}")

    # Validate batch_size
    batch_size = rescoring_config.get("batch_size")
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size < 1:
            errors.append("batch_size must be a positive integer or null")

    # Check that we can inherit prompts if they're null
    inheritable_prompts = [
        "research_context_prompt",
        "scoring_strategy_prompt",
        "score_calculation_prompt",
    ]

    for prompt_name in inheritable_prompts:
        rescoring_value = rescoring_config.get(prompt_name)
        scoring_value = scoring_config.get(prompt_name)

        # If rescoring prompt is null, check that scoring has a value
        if rescoring_value is None and not scoring_value:
            errors.append(
                f"{prompt_name} is null in rescoring config but no fallback found in scoring config"
            )

    return errors


def test_scoring_utilities():
    """
    Test function to validate scoring utilities without making API calls.
    Updated to include comprehensive batching tests.
    """
    print("=== TESTING SCORING UTILITIES ===")

    # Test 1: Default prompts
    print("\n1. Testing default prompts...")
    defaults = get_default_scoring_prompts()
    print(f"✅ Found {len(defaults)} default prompts:")
    for name in defaults.keys():
        print(f"   - {name}")

    # Test 2: JSON format functions
    print("\n2. Testing JSON format functions...")
    try:
        single_format = get_single_paper_json_format()
        batch_format = get_batch_json_format()

        print("✅ JSON format functions working:")
        print(f"   - Single format length: {len(single_format)} chars")
        print(f"   - Batch format length: {len(batch_format)} chars")
        print(f"   - Single contains 'score': {'score' in single_format}")
        print(f"   - Batch contains 'paper_id': {'paper_id' in batch_format}")

    except Exception as e:
        print(f"❌ JSON format functions failed: {e}")
        return

    # Test 3: Configuration validation (single and batch)
    print("\n3. Testing configuration validation...")

    test_configs = {
        "valid_single": {
            "scoring": {
                "model_alias": "test-model",
                "batch_size": None,  # Single mode
                "include_metadata": ["title", "abstract"],
            }
        },
        "valid_batch": {
            "scoring": {
                "model_alias": "test-model",
                "batch_size": 3,  # Batch mode
                "include_metadata": ["title", "abstract"],
            }
        },
        "invalid_batch_size": {
            "scoring": {
                "model_alias": "test-model",
                "batch_size": 0,  # Invalid
                "include_metadata": ["title", "abstract"],
            }
        },
        "large_batch_warning": {
            "scoring": {
                "model_alias": "test-model",
                "batch_size": 15,  # Should trigger warning
                "include_metadata": ["title", "abstract"],
            }
        },
    }

    for config_name, config in test_configs.items():
        validation_errors = validate_scoring_config(config)
        if config_name.startswith("invalid"):
            expected_result = "❌ (expected)"
            success = len(validation_errors) > 0
        else:
            expected_result = "✅"
            success = len(validation_errors) == 0

        if success:
            print(f"   {config_name}: {expected_result}")
        else:
            print(f"   {config_name}: ❌ Unexpected result - {validation_errors}")

    # Test 4: ScoringEngine initialization (single vs batch)
    print("\n4. Testing ScoringEngine initialization...")

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

    # Test single mode
    try:
        single_config = test_configs["valid_single"]
        engine_single = ScoringEngine(single_config, mock_llm_manager, "test-model")
        print("✅ Single mode ScoringEngine initialized")
        print(f"   Batch size: {engine_single.batch_size}")
        has_single_json = '"score":' in engine_single.system_prompt
        print(f"   System prompt contains single JSON: {has_single_json}")

    except Exception as e:
        print(f"❌ Single mode ScoringEngine failed: {e}")
        return

    # Test batch mode
    try:
        batch_config = test_configs["valid_batch"]
        engine_batch = ScoringEngine(batch_config, mock_llm_manager, "test-model")
        print("✅ Batch mode ScoringEngine initialized")
        print(f"   Batch size: {engine_batch.batch_size}")
        has_batch_json = '"paper_id"' in engine_batch.system_prompt
        print(f"   System prompt contains batch JSON: {has_batch_json}")

    except Exception as e:
        print(f"❌ Batch mode ScoringEngine failed: {e}")
        return

    # Test 5: Content extraction (single vs batch)
    print("\n5. Testing content extraction...")

    mock_papers = [
        {
            "id": "test123",
            "title": "Test Paper 1: A Novel Approach",
            "abstract": "This paper presents a novel approach to testing...",
            "authors": ["Author One", "Author Two"],
            "categories": ["cs.AI", "cs.LG"],
        },
        {
            "id": "test456",
            "title": "Test Paper 2: Advanced Methods",
            "abstract": "This work extends previous methods by introducing...",
            "authors": ["Author Three"],
            "categories": ["cs.CV"],
        },
        {
            "id": "test789",
            "title": "Test Paper 3: Comprehensive Analysis",
            "abstract": "We provide a comprehensive analysis of existing approaches...",
            "authors": ["Author Four", "Author Five"],
            "categories": ["stat.ML"],
        },
    ]

    try:
        # Single paper extraction
        single_content = engine_single._extract_paper_content(mock_papers[0])
        print("✅ Single paper content extraction successful")
        print(f"   Length: {len(single_content)} characters")

        # Batch extraction
        batch_content = engine_batch._extract_batch_content(mock_papers)
        print("✅ Batch content extraction successful")
        print(f"   Length: {len(batch_content)} characters")
        print(f"   Contains paper separators: {'=== PAPER_' in batch_content}")
        print(f"   Number of papers detected: {batch_content.count('=== PAPER_')}")

    except Exception as e:
        print(f"❌ Content extraction failed: {e}")

    # Test 6: Response validation (single vs batch)
    print("\n6. Testing response validation...")

    single_test_responses = [
        '{"score": 7.5, "explanation": "Good methodology"}',  # Valid
        '{"score": "8", "explanation": "Score as string"}',  # Should convert
        '{"score": 7.5}',  # Missing explanation
        '{"explanation": "Missing score"}',  # Missing score
        "Not JSON at all",  # Invalid
    ]

    batch_test_responses = [
        """[
            {"paper_id": "paper_1", "score": 7.5, "explanation": "Good paper 1"},
            {"paper_id": "paper_2", "score": 6.2, "explanation": "Decent paper 2"},
            {"paper_id": "paper_3", "score": 8.1, "explanation": "Excellent paper 3"}
        ]""",  # Valid 3-paper batch
        """[
            {"paper_id": "paper_1", "score": 7.5, "explanation": "Good paper"},
            {"paper_id": "paper_2", "score": 6.2}
        ]""",  # Missing explanation in second paper
        """[
            {"paper_id": "wrong_id", "score": 7.5, "explanation": "Wrong paper ID"},
            {"paper_id": "paper_2", "score": 6.2, "explanation": "Good paper"}
        ]""",  # Wrong paper_id format
        '{"score": 7.5, "explanation": "Single when batch expected"}',  # Single when batch expected
        "Not JSON array",  # Invalid
    ]

    print("   Single paper validation:")
    for i, response in enumerate(single_test_responses, 1):
        try:
            is_valid, parsed = engine_single._validate_score_response(response, 1)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"    Test {i}: {status}")
        except Exception as e:
            print(f"    Test {i}: ❌ Exception - {e}")

    print("   Batch validation:")
    for i, response in enumerate(batch_test_responses, 1):
        try:
            is_valid, parsed = engine_batch._validate_score_response(response, 3)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"    Test {i}: {status}")
            if is_valid and isinstance(parsed, list):
                print(f"      → Parsed {len(parsed)} papers")
        except Exception as e:
            print(f"    Test {i}: ❌ Exception - {e}")

    # Test 7: Mock batch scoring workflow
    print("\n7. Testing mock batch scoring workflow...")

    try:
        # Mock the API call to return valid batch response
        def mock_api_call(content):
            if engine_batch.batch_size > 1:
                return """[
                    {"paper_id": "paper_1", "score": 7.5, "explanation": "Novel approach with good validation"},
                    {"paper_id": "paper_2", "score": 6.2, "explanation": "Incremental improvement, limited scope"},
                    {"paper_id": "paper_3", "score": 8.1, "explanation": "Excellent methodology and comprehensive evaluation"}
                ]"""
            else:
                return '{"score": 7.5, "explanation": "Good paper overall"}'

        # Temporarily replace the API call method
        original_method = engine_batch._call_llm_api
        engine_batch._call_llm_api = mock_api_call

        # Test batch scoring
        batch_results = engine_batch._score_paper_batch(mock_papers)

        print("✅ Mock batch scoring successful")
        print(f"   Processed {len(batch_results)} papers")
        print(
            f"   All have scores: {all(p.get('llm_score') is not None for p in batch_results)}"
        )
        print(
            f"   Score range: {min(p['llm_score'] for p in batch_results):.1f} - {max(p['llm_score'] for p in batch_results):.1f}"
        )

        # Restore original method
        engine_batch._call_llm_api = original_method

    except Exception as e:
        print(f"❌ Mock batch scoring failed: {e}")

    print("\n=== BATCHING TESTS COMPLETE ===")
    print("\nTo test with real API calls:")
    print("1. Set up your API keys")
    print("2. Configure a real model in config/llm.yaml")
    print("3. Run: python src/score_utils.py --test-real --batch")


def test_real_configuration(test_api_call: bool = False, test_batching: bool = False):
    """
    Test function using real YAML configuration files.

    Args:
        test_api_call: If True, makes a real API call to test connectivity
        test_batching: If True, tests batch scoring (requires test_api_call=True)
    """
    print("=== TESTING WITH REAL CONFIGURATION ===")
    if test_batching:
        print("🔄 BATCHING TESTS ENABLED")

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

    # Test 4: Enhanced model configuration testing
    print("\n4. Testing LLM model configuration (with batch considerations)...")
    try:
        import yaml
        from llm_utils import LLMManager

        with open("config/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print("✅ Loaded config/config.yaml")

        llm_manager = LLMManager()
        print("✅ Loaded config/llm.yaml via LLMManager")
    except Exception as e:
        print(f"❌ Error loading configurations: {e}")
        return

    scoring_config = config.get("scoring", {})
    model_alias = scoring_config.get("model_alias")
    batch_size = scoring_config.get("batch_size", 1)

    if not model_alias:
        print("❌ No model_alias specified")
        return

    try:
        model_config = llm_manager.get_model_config(model_alias)
        print(f"✅ Model '{model_alias}' configuration loaded:")
        print(f"   Provider: {model_config.get('provider')}")
        print(f"   Model: {model_config.get('model')}")
        print(f"   Batch size configured: {batch_size}")

        # Check context window for batching
        context_window = model_config.get(
            "context_window", model_config.get("max_tokens", 4000)
        )
        if batch_size and batch_size > 1:
            print(f"   Context window: {context_window} tokens")
            estimated_system_tokens = len(get_batch_json_format()) // 4
            print(f"   Estimated system prompt tokens: ~{estimated_system_tokens}")

    except Exception as e:
        print(f"❌ Error getting model configuration: {e}")
        return

    # Test 5: Initialize ScoringEngine with real config
    print("\n5. Testing ScoringEngine with real configuration...")
    try:
        scoring_engine = ScoringEngine(config, llm_manager, model_alias)
        print("✅ ScoringEngine initialized successfully")
        print(f"   Batch size: {scoring_engine.batch_size}")
        print(
            f"   System prompt length: {len(scoring_engine.system_prompt)} characters"
        )

        if scoring_engine.batch_size > 1:
            print("   🔄 Batch mode active")
            print(
                f"   System prompt contains batch format: {'paper_id' in scoring_engine.system_prompt}"
            )
        else:
            print("   📄 Single paper mode")

    except Exception as e:
        print(f"❌ ScoringEngine initialization failed: {e}")
        return

    # Test 6: Enhanced paper data testing
    print("\n6. Testing with sample papers...")

    # Create test papers for batching
    test_papers = [
        {
            "id": "2501.12345v1",
            "title": "Novel Deep Learning Architecture for Scientific Discovery",
            "abstract": "We present a novel neural architecture that combines attention mechanisms with graph neural networks for scientific discovery tasks. Our approach demonstrates significant improvements over existing baselines across multiple domains.",
            "authors": ["Test Author One", "Test Author Two"],
            "categories": ["cs.LG", "cs.AI"],
            "published": "2025-01-15T10:00:00Z",
        },
        {
            "id": "2501.12346v1",
            "title": "Efficient Bayesian Inference for Large-Scale Data Analysis",
            "abstract": "This work introduces a scalable variational inference framework for Bayesian models with applications to astronomical surveys. We achieve orders of magnitude speedup while maintaining statistical accuracy.",
            "authors": ["Researcher Alpha", "Researcher Beta"],
            "categories": ["stat.ML", "astro-ph.IM"],
            "published": "2025-01-15T11:00:00Z",
        },
        {
            "id": "2501.12347v1",
            "title": "Interpretable Machine Learning for Scientific Applications",
            "abstract": "We develop a framework for interpretable ML that provides uncertainty quantification and causal insights. Applications to climate science and materials discovery demonstrate practical utility.",
            "authors": ["Dr. Gamma"],
            "categories": ["cs.LG", "physics.data-an"],
            "published": "2025-01-15T12:00:00Z",
        },
    ]

    try:
        if scoring_engine.batch_size > 1:
            # Test batch content extraction
            batch_content = scoring_engine._extract_batch_content(test_papers)
            print("✅ Batch content extraction successful")
            print(f"   Content length: {len(batch_content)} characters")
            print(f"   Number of papers: {batch_content.count('=== PAPER_')}")
        else:
            # Test single content extraction
            single_content = scoring_engine._extract_paper_content(test_papers[0])
            print("✅ Single paper content extraction successful")
            print(f"   Content length: {len(single_content)} characters")

    except Exception as e:
        print(f"❌ Content extraction failed: {e}")
        return

    # Test 7: API connectivity (enhanced for batching)
    if test_api_call:
        print("\n7. Testing API connectivity...")

        # Check API key/server availability
        provider = model_config.get("provider")
        if provider in ["ollama", "lmstudio", "local", "custom"]:
            base_url = model_config.get("base_url", "http://localhost:11434")
            print(f"   Checking local server: {base_url}")
            try:
                import requests

                _ = requests.get(base_url, timeout=3)
                print("✅ Local server reachable")
            except:
                print("❌ Local server not reachable")
                return
        elif not model_config.get("api_key"):
            print("❌ No API key available")
            return

        print(f"   Testing API call ({provider})...")

        try:
            if scoring_engine.batch_size > 1 and test_batching:
                # Test batch API call
                print("   🔄 Testing batch API call...")
                batch_content = scoring_engine._extract_batch_content(test_papers)
                response_text = scoring_engine._call_llm_api(batch_content)

                if response_text:
                    print("✅ Batch API call successful")
                    print(f"   Response length: {len(response_text)} characters")

                    # Test batch validation
                    is_valid, parsed_data = scoring_engine._validate_score_response(
                        response_text, len(test_papers)
                    )
                    if is_valid:
                        print("✅ Batch response validation successful")
                        print(f"   Parsed {len(parsed_data)} paper scores")
                        scores = [item.get("score") for item in parsed_data]
                        print(f"   Score range: {min(scores):.1f} - {max(scores):.1f}")

                        # Test full batch scoring workflow
                        print("   🔄 Testing full batch workflow...")
                        scored_papers = scoring_engine._score_paper_batch(test_papers)
                        success_count = sum(
                            1 for p in scored_papers if p.get("llm_score") is not None
                        )
                        print(
                            f"✅ Batch workflow complete: {success_count}/{len(test_papers)} papers scored"
                        )

                    else:
                        print("❌ Batch response validation failed")
                        print(f"   Raw response preview: {response_text[:200]}...")
                else:
                    print("❌ Batch API call failed")

            else:
                # Test single API call
                single_content = scoring_engine._extract_paper_content(test_papers[0])
                response_text = scoring_engine._call_llm_api(single_content)

                if response_text:
                    print("✅ Single API call successful")
                    is_valid, parsed_data = scoring_engine._validate_score_response(
                        response_text, 1
                    )
                    if is_valid:
                        print("✅ Single response validation successful")
                        print(f"   Score: {parsed_data.get('score')}")
                    else:
                        print("❌ Single response validation failed")
                else:
                    print("❌ Single API call failed")

        except Exception as e:
            print(f"❌ API testing failed: {e}")

    else:
        print("\n7. Skipping API connectivity test")

    print("\n=== REAL CONFIGURATION TEST COMPLETE ===")

    if test_batching and test_api_call:
        print("\n✅ Full batch-enabled end-to-end test completed!")
    elif test_api_call:
        print("\n✅ Single-mode API test completed!")
    else:
        print(
            "\nℹ️  Configuration test completed. Use --test-api --batch for full testing."
        )

    print("\nNext steps:")
    print("1. Run: python src/fetch_papers.py")
    if batch_size and batch_size > 1:
        print(f"2. Run: python src/score_papers.py (will use batch_size={batch_size})")
    else:
        print("2. Run: python src/score_papers.py")
        print("   To enable batching, set batch_size: 3 in config.yaml")


def create_test_configurations() -> Dict[str, Dict]:
    """
    Create various test configurations for validation.

    Returns:
        Dictionary of test configurations
    """
    return {
        "minimal_single": {
            "scoring": {
                "model_alias": "test-model",
                "include_metadata": ["title", "abstract"],
                # batch_size omitted = defaults to single mode
            }
        },
        "explicit_single": {
            "scoring": {
                "model_alias": "test-model",
                "batch_size": None,  # Explicit single mode
                "include_metadata": ["title", "abstract"],
            }
        },
        "small_batch": {
            "scoring": {
                "model_alias": "test-model",
                "batch_size": 3,
                "include_metadata": ["title", "abstract"],
                "retry_attempts": 2,
            }
        },
        "large_batch": {
            "scoring": {
                "model_alias": "test-model",
                "batch_size": 8,
                "include_metadata": ["title", "abstract", "categories"],
                "retry_attempts": 1,  # Reduce retries for large batches
            }
        },
        "full_metadata_batch": {
            "scoring": {
                "model_alias": "test-model",
                "batch_size": 5,
                "include_metadata": [
                    "title",
                    "abstract",
                    "authors",
                    "categories",
                    "published",
                ],
                "retry_attempts": 3,
                "research_context_prompt": "Custom context for batch testing...",
                "scoring_strategy_prompt": "Custom strategy for batch testing...",
                "score_format_prompt": "Custom format for batch testing...",
            }
        },
    }


def test_rescoring_configuration():
    """Test rescoring configuration validation with various scenarios."""
    print("=== TESTING RESCORING CONFIGURATION ===")

    # Test configuration scenarios
    test_configs = {
        "valid_with_inheritance": {
            "scoring": {
                "model_alias": "test-model",
                "research_context_prompt": "Original context",
                "scoring_strategy_prompt": "Original strategy",
                "score_calculation_prompt": "Original calculation",
            },
            "rescoring": {
                "model_alias": "test-model",
                "include_metadata": ["title", "abstract", "llm_summary"],
                "research_context_prompt": None,  # Inherit
                "scoring_strategy_prompt": "New strategy",  # Override
                "score_calculation_prompt": None,  # Inherit
            },
        },
        "valid_full_override": {
            "scoring": {
                "model_alias": "test-model",
                "research_context_prompt": "Original context",
            },
            "rescoring": {
                "model_alias": "test-model",
                "include_metadata": [
                    "title",
                    "llm_summary",
                    "summary_key_contributions",
                ],
                "research_context_prompt": "New context",
                "scoring_strategy_prompt": "New strategy",
                "score_calculation_prompt": "New calculation",
            },
        },
        "invalid_missing_model": {
            "rescoring": {"include_metadata": ["title", "abstract"]}
        },
        "invalid_bad_inheritance": {
            "scoring": {},  # No prompts to inherit from
            "rescoring": {
                "model_alias": "test-model",
                "include_metadata": ["title"],
                "research_context_prompt": None,  # Can't inherit - nothing there
            },
        },
        "invalid_metadata": {
            "rescoring": {
                "model_alias": "test-model",
                "include_metadata": ["invalid_field"],
            }
        },
    }

    for config_name, config in test_configs.items():
        print(f"\nTesting {config_name}:")
        validation_errors = validate_rescoring_config(config)

        if config_name.startswith("invalid"):
            if validation_errors:
                print(f"   ✅ Correctly identified errors: {len(validation_errors)}")
                for error in validation_errors[:2]:  # Show first 2 errors
                    print(f"      - {error}")
            else:
                print(f"   ❌ Should have found validation errors")
        else:
            if not validation_errors:
                print(f"   ✅ Valid configuration passed")
            else:
                print(f"   ❌ Unexpected validation errors:")
                for error in validation_errors:
                    print(f"      - {error}")

    print(f"\n=== RESCORING CONFIGURATION TEST COMPLETE ===")


def test_rescoring_workflow():
    """Test the rescoring workflow with mock data."""
    print("=== TESTING RESCORING WORKFLOW ===")

    # Create mock summarized papers
    mock_papers = [
        {
            "id": "2501.12345v1",
            "title": "Novel Deep Learning Architecture for Scientific Discovery",
            "abstract": "We present a novel neural architecture...",
            "llm_score": 7.2,
            "llm_explanation": "Strong methodological contribution",
            "llm_summary": """
## CORE CONTRIBUTION
This paper introduces a transformer-based architecture specifically designed for scientific discovery tasks, incorporating domain-specific attention mechanisms and uncertainty quantification.

## METHODOLOGY  
The approach combines multi-head attention with graph neural networks, using a novel attention mechanism that incorporates scientific priors. The model is trained on diverse scientific datasets with a custom loss function.

## RESULTS
Evaluation on 5 scientific domains shows 15-30% improvement over baselines. Statistical significance confirmed through extensive ablation studies and cross-validation.

## IMPLICATIONS
This work opens new directions for AI-assisted scientific discovery and provides a general framework applicable across scientific domains.
            """,
            "summary_key_contributions": [
                "Novel transformer architecture for scientific discovery",
                "Domain-specific attention mechanisms",
                "15-30% improvement over baselines",
            ],
            "summary_confidence": 0.9,
        },
        {
            "id": "2501.12346v1",
            "title": "Statistical Methods for Large-Scale Survey Analysis",
            "abstract": "This work presents statistical methods...",
            "llm_score": 6.5,
            "llm_explanation": "Solid statistical contribution",
            "llm_summary": """
## CORE CONTRIBUTION
Development of robust statistical methods for handling selection effects and systematic uncertainties in large astronomical surveys.

## METHODOLOGY
Uses hierarchical Bayesian modeling with MCMC sampling. Introduces novel priors for handling missing data and selection biases.

## RESULTS  
Applied to SDSS data, shows improved parameter estimation accuracy. Reduces systematic biases by 40% compared to standard methods.

## IMPLICATIONS
Critical for next-generation surveys like LSST. Methods are generalizable to other large-scale scientific surveys.
            """,
            "summary_key_contributions": [
                "Hierarchical Bayesian methods for survey data",
                "Novel priors for selection effects",
                "40% reduction in systematic biases",
            ],
            "summary_confidence": 0.8,
        },
    ]

    print(f"Created {len(mock_papers)} mock papers with summaries")

    # Test configuration for rescoring
    test_config = {
        "scoring": {
            "research_context_prompt": "Original research context for testing",
            "scoring_strategy_prompt": "Original scoring strategy",
            "score_calculation_prompt": "Original score calculation",
        },
        "rescoring": {
            "model_alias": "test-model",
            "include_metadata": [
                "title",
                "abstract",
                "llm_summary",
                "summary_key_contributions",
            ],
            "batch_size": 2,
            "retry_attempts": 1,
            "research_context_prompt": None,  # Should inherit
            "scoring_strategy_prompt": "Enhanced scoring strategy for full content",
            "score_calculation_prompt": None,  # Should inherit
        },
    }

    # Validate configuration
    validation_errors = validate_rescoring_config(test_config)
    if validation_errors:
        print(f"❌ Configuration validation failed:")
        for error in validation_errors:
            print(f"   - {error}")
        return

    print(f"✅ Configuration validation passed")

    # Test prompt inheritance logic (simplified version)
    rescoring_config = test_config["rescoring"].copy()
    scoring_config = test_config["scoring"]

    inheritable_prompts = [
        "research_context_prompt",
        "scoring_strategy_prompt",
        "score_calculation_prompt",
    ]
    inherited_count = 0

    for prompt_name in inheritable_prompts:
        if rescoring_config.get(prompt_name) is None:
            inherited_value = scoring_config.get(prompt_name)
            if inherited_value:
                rescoring_config[prompt_name] = inherited_value
                inherited_count += 1
                print(f"✅ Inherited {prompt_name}")

    print(f"✅ Prompt inheritance working: {inherited_count} prompts inherited")

    # Test metadata extraction would happen here
    # (This would use the existing ScoringEngine._extract_paper_content method)

    print(f"✅ Mock rescoring workflow test completed")
    print(f"Next step: Run real rescoring with actual summarized papers")


# Example usage and testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] in ["--test", "-t"]:
            test_scoring_utilities()
        elif sys.argv[1] in ["--test-real", "--real", "-r"]:
            test_real_configuration(test_api_call=False, test_batching=False)
        elif sys.argv[1] in ["--test-api", "--api", "-a"]:
            test_real_configuration(test_api_call=True, test_batching=False)
        elif sys.argv[1] in ["--test-batch", "--batch", "-b"]:
            test_real_configuration(test_api_call=True, test_batching=True)
        elif sys.argv[1] in ["--test-rescore", "--rescore", "-rs"]:
            test_rescoring_configuration()
            test_rescoring_workflow()
        elif sys.argv[1] in ["--defaults", "-d"]:
            display_default_prompts()
        elif sys.argv[1] in ["--configs", "-c"]:
            print("=== TEST CONFIGURATIONS ===")
            configs = create_test_configurations()
            for name, config in configs.items():
                batch_size = config["scoring"].get("batch_size", "null")
                print(f"\n{name} (batch_size: {batch_size}):")
                print(yaml.dump(config, default_flow_style=False, indent=2))
        elif sys.argv[1] in ["--help", "-h"]:
            print("Scoring Utilities Test Module")
            print("Usage:")
            print(
                "  python score_utils.py --test          # Run mock tests (includes batching)"
            )
            print("  python score_utils.py --test-real     # Test with real configs")
            print("  python score_utils.py --test-api      # Test real configs + API")
            print(
                "  python score_utils.py --test-batch    # Test real configs + API + batching"
            )
            print(
                "  python score_utils.py --test-rescore  # Test rescoring configuration"
            )
            print("  python score_utils.py --defaults      # Show default prompts")
            print("  python score_utils.py --configs       # Show test configurations")
            print("  python score_utils.py --help          # Show this help")
        else:
            print("Unknown option. Use --help for usage information.")
    else:
        print("Scoring Utilities Module")
        print("Run with --help for usage options")
        print("Add --rescore to test rescoring functionality")

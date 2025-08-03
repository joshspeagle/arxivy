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


#!/usr/bin/env python3
"""
Scoring Utilities

Helper functions for LLM-based paper scoring including prompt construction,
response validation, retry logic, and progress tracking.

Dependencies:
    pip install openai anthropic google-generativeai

Usage:
    from scoring_utils import ScoringEngine
    
    engine = ScoringEngine(config, llm_manager)
    score_data = engine.score_paper(paper_data)
"""

import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import traceback


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
Rate each component from 0-10:
- Relevance score (0-10): ___
- Contribution score (0-10): ___  
- Impact score (0-10): ___

Calculate: Final Score = (Relevance × 0.4) + (Contribution × 0.3) + (Impact × 0.3)
Round to one decimal place.

Provide result as JSON:
{
  "score": <final_calculated_score>,
  "explanation": "Brief assessment covering all three components."
}

Do not include any text outside of this JSON structure.
        """.strip()
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
            self.model_config = llm_manager.get_model_config(model_alias)
            self.client = llm_manager.get_client(model_alias)
        except Exception as e:
            raise ValueError(f"Failed to initialize model '{model_alias}': {e}")
        
        # Scoring configuration
        self.scoring_config = config.get("scoring", {})
        self.retry_attempts = self.scoring_config.get("retry_attempts", 2)
        self.include_metadata = self.scoring_config.get("include_metadata", ["title", "abstract"])
        
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
        research_context = self.scoring_config.get("research_context_prompt", "").strip()
        if not research_context:
            research_context = defaults["research_context_prompt"]
            used_defaults.append("research_context_prompt")
        
        scoring_strategy = self.scoring_config.get("scoring_strategy_prompt", "").strip()
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
            print(f"   For better results, customize these prompts in your config file.")
            print()
        
        # Combine the three parts with clear separators
        prompt_parts = [
            f"RESEARCH CONTEXT:\n{research_context}",
            f"SCORING CRITERIA:\n{scoring_strategy}",
            f"OUTPUT FORMAT:\n{score_format}"
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
    
    def _validate_score_response(self, response_text: str) -> Tuple[bool, Optional[Dict]]:
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
                json_match = re.search(r'\{[^}]*"score"[^}]*\}', response_text, re.DOTALL)
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
                        {"role": "user", "content": f"Please score this paper:\n\n{paper_content}"}
                    ],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 1000)
                )
                return response.choices[0].message.content
            
            elif provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_config["model"],
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": f"Please score this paper:\n\n{paper_content}"}
                    ],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 1000)
                )
                return response.content[0].text
            
            elif provider == "google":
                model = self.client.GenerativeModel(self.model_config["model"])
                prompt = f"{self.system_prompt}\n\nPlease score this paper:\n\n{paper_content}"
                response = model.generate_content(
                    prompt,
                    generation_config=self.client.types.GenerationConfig(
                        temperature=self.model_config.get("temperature", 0.1),
                        max_output_tokens=self.model_config.get("max_tokens", 1000)
                    )
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
        full_prompt = f"{self.system_prompt}\n\nPlease score this paper:\n\n{paper_content}"
        
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.model_config.get("temperature", 0.1),
                "num_predict": self.model_config.get("max_tokens", 1000)
            }
        }
        
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=120  # Ollama can be slow
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
                api_key="not-needed"  # LM Studio doesn't require API key
            )
            
            response = local_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Please score this paper:\n\n{paper_content}"}
                ],
                temperature=self.model_config.get("temperature", 0.1),
                max_tokens=self.model_config.get("max_tokens", 1000)
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
            api_format = self.model_config.get("api_format", "openai")  # openai, ollama, or custom
            
            if api_format == "openai":
                # OpenAI-compatible API (like LM Studio, vLLM, etc.)
                import openai
                local_client = openai.OpenAI(
                    base_url=base_url,
                    api_key=self.model_config.get("api_key", "not-needed")
                )
                
                response = local_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Please score this paper:\n\n{paper_content}"}
                    ],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 1000)
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
                        "num_predict": self.model_config.get("max_tokens", 1000)
                    }
                }
                
                response = requests.post(
                    f"{base_url}/api/generate",
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "")
            
            else:
                # Custom API format - user needs to implement
                raise ValueError(f"Unsupported api_format: {api_format}. Use 'openai' or 'ollama'.")
                
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
                        print(f"  → Invalid response format, retrying... (attempt {attempt + 1})")
                        time.sleep(1)
                        continue
                    else:
                        print(f"  → Failed to get valid response after {self.retry_attempts + 1} attempts")
                        break
                        
            except Exception as e:
                if attempt < self.retry_attempts:
                    self.total_retries += 1
                    print(f"  → Error during scoring: {e}, retrying... (attempt {attempt + 1})")
                    time.sleep(1)
                    continue
                else:
                    print(f"  → Final failure after {self.retry_attempts + 1} attempts: {e}")
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
    
    def print_progress(self, current_index: int, paper_title: str):
        """
        Print progress information.
        
        Args:
            current_index: Current paper index (0-based)
            paper_title: Title of current paper
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
        display_title = paper_title[:60] + "..." if len(paper_title) > 60 else paper_title
        
        print(f"[{current_index + 1:3d}/{self.total_papers}] ({progress_pct:5.1f}%) {eta_str} | {display_title}")
    
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
    
    valid_metadata_fields = {"title", "abstract", "authors", "categories", "published", "updated"}
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
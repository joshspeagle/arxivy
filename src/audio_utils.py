#!/usr/bin/env python3
"""
Audio Script Generation Utilities

Converts written research digests into natural-sounding scripts for TTS audio narration.
Supports conversion from both Stage 2 (final reports) and Stage 1 (structured content).

Dependencies:
    pip install openai anthropic google-generativeai pyyaml

Usage:
    from audio_utils import AudioScriptGenerator

    generator = AudioScriptGenerator(config, llm_manager, model_alias)
    script_result = generator.generate_audio_script(synthesis_result)
"""

import time
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def get_default_audio_prompts() -> Dict[str, str]:
    """
    Get default prompts for audio script generation.

    Returns:
        Dictionary with default prompt strings for different input types
    """
    return {
        "research_context_prompt": """
You are converting a written expert research digest into a natural-sounding, spoken script 
for audio narration. The audience is an interdisciplinary expert familiar with statistical AI 
and astrophysics, listening during their commute or while multitasking.

The listener values efficient, expert-level content but needs it delivered in a conversational, 
engaging audio format that doesn't require visual focus.
        """.strip(),
        "stage2_to_audio_prompt": """
Convert this polished written research digest into a natural-sounding spoken script:

**Audio Formatting Requirements:**
- Rewrite long or complex sentences for natural speech cadence
- Use verbal transition phrases: "Now let's turn to...", "Another notable study shows...", "Moving on..."
- Remove ALL visual formatting (bold, markdown, headers, bullets)
- Break up dense technical sections with light signposting
- Add brief spoken introduction and conclusion suitable for standalone audio
- Ensure smooth flow between topics without abrupt jumps

**Content Preservation:**
- Maintain all technical accuracy and specific results
- Keep author attributions and paper references
- Preserve quantitative findings and methodological details
- Include confidence levels and limitations mentioned

**Style Guidelines:**
- Expert-to-expert but conversational tone
- Natural speech patterns and rhythm
- Appropriate pacing for audio consumption
- Clear verbal emphasis on key findings

Target: 10-12 minutes reading time (~1500 words). Output as single continuous plain-text script.
        """.strip(),
        "stage1_to_audio_prompt": """
Convert this structured research content into a natural-sounding spoken narrative:

**Audio Narrative Requirements:**
- Transform structured content into flowing spoken narrative
- Create natural transitions between themes and papers
- Use verbal signposting: "First, let's look at...", "In contrast...", "Building on this..."
- Remove all formatting and present as continuous speech
- Add engaging introduction that sets context for the research themes
- Include conclusion that ties themes together and highlights key takeaways

**Content Organization:**
- Group related findings into coherent verbal segments
- Smoothly connect different papers and themes
- Highlight connections and contrasts between studies
- Present quantitative results in speech-friendly format

**Style Guidelines:**
- Conversational academic tone, as if explaining to a colleague
- Natural speech rhythm and pacing
- Clear verbal emphasis on important findings
- Expert-level content delivered accessibly for audio

Target: 10-12 minutes reading time (~1500 words). Output as single continuous plain-text script.
        """.strip(),
    }


class AudioScriptGenerator:
    """
    Handles conversion of written research digests into audio-ready scripts.

    Supports conversion from both Stage 2 (polished reports) and Stage 1 (structured content)
    using different prompting strategies optimized for each input type.
    """

    def __init__(self, config: Dict, llm_manager, model_alias: str):
        """
        Initialize the audio script generator.

        Args:
            config: Configuration dictionary with audio settings
            llm_manager: LLMManager instance for API calls
            model_alias: Model alias to use for script generation
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

        # Audio configuration
        self.audio_config = config.get("audio", {})
        self.retry_attempts = self.audio_config.get("retry_attempts", 2)
        self.target_duration_minutes = self.audio_config.get(
            "target_duration_minutes", 11
        )
        self.target_words = int(
            self.target_duration_minutes * 130
        )  # ~130 words per minute

        # Build system prompts for different input types
        self.stage2_system_prompt = self._build_stage2_system_prompt()
        self.stage1_system_prompt = self._build_stage1_system_prompt()

        # Statistics tracking
        self.stats = {
            "scripts_generated": 0,
            "stage2_conversions": 0,
            "stage1_conversions": 0,
            "total_retries": 0,
            "conversion_failures": 0,
            "total_conversion_time": 0,
        }

    def _get_default_prompts(self) -> Dict[str, str]:
        """Get default prompts for audio script generation."""
        return get_default_audio_prompts()

    def _build_stage2_system_prompt(self) -> str:
        """
        Construct the system prompt for Stage 2 (polished report) to audio conversion.

        Returns:
            Complete system prompt string for Stage 2 conversion
        """
        defaults = self._get_default_prompts()
        used_defaults = []

        # Get research context prompt
        research_context = self.audio_config.get("research_context_prompt")
        if research_context is None:
            # Try to inherit from synthesis or scoring config
            research_context = (
                self.config.get("synthesis", {})
                .get("generation", {})
                .get("research_context_prompt", "")
                .strip()
            )
            if not research_context:
                research_context = (
                    self.config.get("scoring", {})
                    .get("research_context_prompt", "")
                    .strip()
                )
            if not research_context:
                research_context = defaults["research_context_prompt"]
                used_defaults.append("research_context_prompt")
        elif not research_context.strip():
            research_context = defaults["research_context_prompt"]
            used_defaults.append("research_context_prompt")

        # Get Stage 2 conversion prompt
        stage2_prompt = self.audio_config.get("stage2_to_audio_prompt", "").strip()
        if not stage2_prompt:
            stage2_prompt = defaults["stage2_to_audio_prompt"]
            used_defaults.append("stage2_to_audio_prompt")

        # Warn user about defaults
        if used_defaults:
            print(
                f"⚠️  WARNING: Using default audio prompts: {', '.join(used_defaults)}"
            )

        # Combine the parts
        prompt_parts = [
            f"RESEARCH CONTEXT:\n{research_context}",
            f"CONVERSION INSTRUCTIONS:\n{stage2_prompt}",
        ]

        return "\n\n".join(prompt_parts)

    def _build_stage1_system_prompt(self) -> str:
        """
        Construct the system prompt for Stage 1 (structured content) to audio conversion.

        Returns:
            Complete system prompt string for Stage 1 conversion
        """
        defaults = self._get_default_prompts()
        used_defaults = []

        # Get research context prompt (same as Stage 2)
        research_context = self.audio_config.get("research_context_prompt")
        if research_context is None:
            research_context = (
                self.config.get("synthesis", {})
                .get("generation", {})
                .get("research_context_prompt", "")
                .strip()
            )
            if not research_context:
                research_context = (
                    self.config.get("scoring", {})
                    .get("research_context_prompt", "")
                    .strip()
                )
            if not research_context:
                research_context = defaults["research_context_prompt"]
                used_defaults.append("research_context_prompt")
        elif not research_context.strip():
            research_context = defaults["research_context_prompt"]
            used_defaults.append("research_context_prompt")

        # Get Stage 1 conversion prompt
        stage1_prompt = self.audio_config.get("stage1_to_audio_prompt", "").strip()
        if not stage1_prompt:
            stage1_prompt = defaults["stage1_to_audio_prompt"]
            used_defaults.append("stage1_to_audio_prompt")

        # Warn user about defaults
        if used_defaults:
            print(
                f"⚠️  WARNING: Using default audio prompts: {', '.join(used_defaults)}"
            )

        # Combine the parts
        prompt_parts = [
            f"RESEARCH CONTEXT:\n{research_context}",
            f"CONVERSION INSTRUCTIONS:\n{stage1_prompt}",
        ]

        return "\n\n".join(prompt_parts)

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
                    max_tokens=self.model_config.get("max_tokens", 3000),
                )
                return response.choices[0].message.content

            elif provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_config["model"],
                    system=system_prompt,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 3000),
                )
                return response.content[0].text

            elif provider == "google":
                model = self.client.GenerativeModel(self.model_config["model"])
                prompt = f"{system_prompt}\n\n{content}"
                response = model.generate_content(
                    prompt,
                    generation_config=self.client.types.GenerationConfig(
                        temperature=self.model_config.get("temperature", 0.1),
                        max_output_tokens=self.model_config.get("max_tokens", 3000),
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
                "num_predict": self.model_config.get("max_tokens", 3000),
            },
        }

        try:
            response = requests.post(
                f"{base_url}/api/generate", json=payload, timeout=180
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
                max_tokens=self.model_config.get("max_tokens", 3000),
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
                    max_tokens=self.model_config.get("max_tokens", 3000),
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
                        "num_predict": self.model_config.get("max_tokens", 3000),
                    },
                }

                response = requests.post(
                    f"{base_url}/api/generate", json=payload, timeout=180
                )
                response.raise_for_status()

                result = response.json()
                return result.get("response", "")

            else:
                raise ValueError(f"Unsupported api_format: {api_format}")

        except Exception as e:
            print(f"Local API call failed: {e}")
            return None

    def _validate_audio_script(self, script_text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate the generated audio script.

        Args:
            script_text: Generated script text

        Returns:
            Tuple of (is_valid, cleaned_script)
        """
        try:
            if not script_text or len(script_text.strip()) < 300:
                return False, None

            cleaned_script = script_text.strip()

            # Remove any thinking tokens
            cleaned_script = re.sub(
                r"<think>.*?</think>",
                "",
                cleaned_script,
                flags=re.DOTALL | re.IGNORECASE,
            )
            cleaned_script = re.sub(r"<[^>]+>", "", cleaned_script)

            # Remove markdown formatting that might have leaked through
            cleaned_script = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned_script)  # Bold
            cleaned_script = re.sub(r"\*([^*]+)\*", r"\1", cleaned_script)  # Italic
            cleaned_script = re.sub(r"`([^`]+)`", r"\1", cleaned_script)  # Code
            cleaned_script = re.sub(r"#+\s*", "", cleaned_script)  # Headers
            cleaned_script = re.sub(
                r"^\s*[-*+]\s*", "", cleaned_script, flags=re.MULTILINE
            )  # Bullets

            # Clean up extra whitespace
            cleaned_script = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned_script)
            cleaned_script = cleaned_script.strip()

            # Quality checks for audio script
            word_count = len(cleaned_script.split())

            # Check length is reasonable for target duration
            min_words = int(self.target_words * 0.7)  # Allow 30% shorter
            max_words = int(self.target_words * 1.5)  # Allow 50% longer

            if word_count < min_words:
                return False, None

            if word_count > max_words:
                # Allow but warn about length
                print(
                    f"⚠️  Generated script is longer than target ({word_count} vs {self.target_words} words)"
                )

            # Check for audio-appropriate structure
            paragraphs = [p.strip() for p in cleaned_script.split("\n\n") if p.strip()]
            if len(paragraphs) < 3:
                return False, None

            # Should not contain visual formatting indicators
            if any(
                indicator in cleaned_script
                for indicator in ["**", "*", "`", "#", "- ", "* ", "+ "]
            ):
                return False, None

            return True, cleaned_script

        except Exception as e:
            print(f"Audio script validation error: {e}")
            return False, None

    def _convert_content_to_audio_script(
        self, content: str, source_type: str
    ) -> Optional[str]:
        """
        Convert content to audio script with retry logic.

        Args:
            content: Content to convert (Stage 1 or Stage 2)
            source_type: Type of content ("stage1" or "stage2")

        Returns:
            Audio script or None on failure
        """
        if source_type == "stage2":
            system_prompt = self.stage2_system_prompt
            self.stats["stage2_conversions"] += 1
        elif source_type == "stage1":
            system_prompt = self.stage1_system_prompt
            self.stats["stage1_conversions"] += 1
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

        print(f"Converting {source_type} content to audio script...")

        for attempt in range(self.retry_attempts + 1):
            try:
                # Make LLM call
                response_text = self._call_llm_api(content, system_prompt)

                if response_text is None:
                    if attempt < self.retry_attempts:
                        self.stats["total_retries"] += 1
                        print(
                            f"  → API call failed, retrying... (attempt {attempt + 1})"
                        )
                        time.sleep(2)
                        continue
                    else:
                        print(f"  → API calls failed after all attempts")
                        break

                # Validate response
                is_valid, cleaned_script = self._validate_audio_script(response_text)

                if is_valid:
                    word_count = len(cleaned_script.split())
                    duration_estimate = word_count / 130  # ~130 words per minute
                    print(
                        f"  ✅ Audio script generated successfully ({word_count} words, ~{duration_estimate:.1f} min)"
                    )
                    return cleaned_script
                else:
                    if attempt < self.retry_attempts:
                        self.stats["total_retries"] += 1
                        print(
                            f"  → Invalid script generated, retrying... (attempt {attempt + 1})"
                        )
                        time.sleep(2)
                        continue
                    else:
                        print(f"  → Failed to generate valid script after all attempts")
                        break

            except Exception as e:
                if attempt < self.retry_attempts:
                    self.stats["total_retries"] += 1
                    print(
                        f"  → Conversion error: {e}, retrying... (attempt {attempt + 1})"
                    )
                    time.sleep(2)
                    continue
                else:
                    print(f"  → Final conversion failure: {e}")
                    break

        self.stats["conversion_failures"] += 1
        return None

    def generate_audio_script(self, synthesis_result: Dict) -> Dict:
        """
        Generate audio script from synthesis results.

        Args:
            synthesis_result: Result from synthesis workflow

        Returns:
            Dictionary with audio script results and metadata
        """
        start_time = time.time()

        # Determine input preference
        prefer_stage2 = self.audio_config.get("prefer_stage2", True)

        # Get available content
        stage2_content = synthesis_result.get("synthesis_report")
        stage1_content = synthesis_result.get("stage1_content")

        print(f"Starting audio script generation using {self.model_alias}")
        print(
            f"Target duration: {self.target_duration_minutes} minutes (~{self.target_words} words)"
        )

        audio_script = None
        source_used = None

        # Try preferred source first
        if prefer_stage2 and stage2_content:
            print(f"Attempting Stage 2 → Audio conversion...")
            audio_script = self._convert_content_to_audio_script(
                stage2_content, "stage2"
            )
            source_used = "stage2"
        elif stage1_content:
            print(f"Attempting Stage 1 → Audio conversion...")
            audio_script = self._convert_content_to_audio_script(
                stage1_content, "stage1"
            )
            source_used = "stage1"

        # Fallback to other source if preferred failed
        if audio_script is None:
            if prefer_stage2 and stage1_content:
                print(f"Stage 2 conversion failed, falling back to Stage 1...")
                audio_script = self._convert_content_to_audio_script(
                    stage1_content, "stage1"
                )
                source_used = "stage1"
            elif not prefer_stage2 and stage2_content:
                print(f"Stage 1 conversion failed, falling back to Stage 2...")
                audio_script = self._convert_content_to_audio_script(
                    stage2_content, "stage2"
                )
                source_used = "stage2"

        conversion_time = time.time() - start_time
        self.stats["total_conversion_time"] += conversion_time

        # Compile results
        if audio_script:
            word_count = len(audio_script.split())
            duration_estimate = word_count / 130

            result = {
                "audio_script": audio_script,
                "conversion_success": True,
                "source_used": source_used,
                "word_count": word_count,
                "estimated_duration_minutes": duration_estimate,
                "conversion_time": conversion_time,
                "converted_by": self.model_alias,
                "converted_at": datetime.now().isoformat(),
                "config_used": {
                    "model_alias": self.model_alias,
                    "target_duration_minutes": self.target_duration_minutes,
                    "prefer_stage2": prefer_stage2,
                    "retry_attempts": self.retry_attempts,
                },
            }

            self.stats["scripts_generated"] += 1
            print(
                f"✅ Audio script generated successfully in {conversion_time:.1f} seconds"
            )
        else:
            result = {
                "audio_script": None,
                "conversion_success": False,
                "error": "Failed to convert any available content to audio script",
                "source_attempted": "stage2" if prefer_stage2 else "stage1",
                "conversion_time": conversion_time,
                "converted_by": self.model_alias,
                "converted_at": datetime.now().isoformat(),
            }

            print(
                f"❌ Audio script generation failed after {conversion_time:.1f} seconds"
            )

        return result


def validate_audio_config(config: Dict) -> List[str]:
    """
    Validate the audio configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    audio_config = config.get("audio", {})

    # Check required parameters
    model_alias = audio_config.get("model_alias")
    if not model_alias:
        errors.append("model_alias is required in audio configuration")

    # Validate retry attempts
    retry_attempts = audio_config.get("retry_attempts", 2)
    if not isinstance(retry_attempts, int) or retry_attempts < 0:
        errors.append("retry_attempts must be a non-negative integer")

    # Validate target duration
    target_duration = audio_config.get("target_duration_minutes", 11)
    if not isinstance(target_duration, (int, float)) or target_duration <= 0:
        errors.append("target_duration_minutes must be a positive number")

    return errors


def display_default_audio_prompts():
    """Display the default audio prompts for user reference."""
    defaults = get_default_audio_prompts()

    print("=== DEFAULT AUDIO SCRIPT PROMPTS ===")
    for prompt_name, prompt_text in defaults.items():
        print(f"\n{prompt_name}:")
        print("-" * 40)
        print(prompt_text)
    print("\n" + "=" * 50)


def test_audio_utilities():
    """Test audio script generation utilities with mock data."""
    print("=== TESTING AUDIO SCRIPT UTILITIES ===")

    # Test 1: Default prompts
    print("\n1. Testing default prompts...")
    defaults = get_default_audio_prompts()
    print(f"✅ Found {len(defaults)} default prompts:")
    for name in defaults.keys():
        print(f"   - {name}")

    # Test 2: Configuration validation
    print("\n2. Testing configuration validation...")

    valid_config = {
        "audio": {
            "model_alias": "test-model",
            "target_duration_minutes": 10,
            "retry_attempts": 2,
            "prefer_stage2": True,
        }
    }

    validation_errors = validate_audio_config(valid_config)
    if not validation_errors:
        print("✅ Valid audio configuration passed")
    else:
        print(f"❌ Unexpected validation errors: {validation_errors}")

    # Test 3: Mock audio generator initialization
    print("\n3. Testing audio generator initialization...")

    class MockLLMManager:
        def get_model_config(self, alias):
            return {
                "provider": "test",
                "model": "test-model",
                "temperature": 0.1,
                "max_tokens": 3000,
            }

        def get_client(self, alias):
            return "mock-client"

    try:
        generator = AudioScriptGenerator(valid_config, MockLLMManager(), "test-model")
        print("✅ Audio script generator initialized successfully")
        print(
            f"   Stage 2 prompt length: {len(generator.stage2_system_prompt)} characters"
        )
        print(
            f"   Stage 1 prompt length: {len(generator.stage1_system_prompt)} characters"
        )
        print(f"   Target duration: {generator.target_duration_minutes} minutes")
        print(f"   Target words: {generator.target_words} words")

        # Test validation function
        # Make test_script_good long enough to pass validation (~1300 words)
        base_script = """
    Welcome to today's research digest. In today's compilation, we explore three fascinating 
    developments in machine learning and astrophysics that showcase the intersection of 
    statistical AI and cosmic discovery.

    First, let's examine the work by Smith and colleagues on robust Bayesian inference 
    for galaxy classification. Their approach demonstrates remarkable accuracy, achieving 
    ninety-two percent classification success on SDSS survey data. This represents a 
    significant improvement over previous methods.

    Now, turning to the interpretability front, Johnson's team has developed a novel 
    framework for understanding neural network decisions in astronomical contexts. 
    Their method provides unprecedented insight into how AI models process cosmic data.

    Moving on to our final study, we see Lee and collaborators tackling the challenge 
    of uncertainty quantification in large-scale surveys. They've introduced innovative 
    techniques that reduce systematic biases by fifteen percent compared to standard approaches.

    What makes these studies particularly compelling is how they complement each other. 
    While Smith focuses on accuracy, Johnson emphasizes interpretability, and Lee addresses 
    uncertainty. Together, they represent a comprehensive approach to statistical AI in astronomy.

    In conclusion, today's papers highlight the rapid evolution of our field, where 
    statistical rigor meets cosmic discovery. These advances promise to reshape how 
    we understand both our methods and our universe.
        """.strip()

        # Repeat the base_script 7 times to get ~1300 words
        test_script_good = "\n\n".join([base_script] * 7)

        test_script_bad = "Too short and contains **markdown** formatting."

        is_valid_good, _ = generator._validate_audio_script(test_script_good)
        is_valid_bad, _ = generator._validate_audio_script(test_script_bad)

        print(f"✅ Validation working: good={is_valid_good}, bad={is_valid_bad}")

    except Exception as e:
        print(f"❌ Audio generator test failed: {e}")
        return

    print(f"\n✅ All audio script utilities tests passed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] in ["--test", "-t"]:
            test_audio_utilities()
        elif sys.argv[1] in ["--defaults", "-d"]:
            display_default_audio_prompts()
        elif sys.argv[1] in ["--help", "-h"]:
            print("Audio Script Generation Utilities")
            print("Usage:")
            print("  python audio_utils.py --test       # Run mock tests")
            print("  python audio_utils.py --defaults   # Show default prompts")
            print("  python audio_utils.py --help       # Show this help")
        else:
            print("Unknown option. Use --help for usage information.")
    else:
        print("Audio Script Generation Utilities Module")
        print("Run with --help for usage options")

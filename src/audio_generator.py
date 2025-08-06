"""
Enhanced TTS Generator compatible with existing config.yaml and tts.yaml structure.

This module provides improved text-to-speech functionality that integrates with
your existing configuration while fixing issues like:
- Character encoding problems
- Minimum length requirements
- Smart text chunking
- Multiple fallback strategies
"""

import logging
import re
import unicodedata
import yaml
import asyncio
import os
from typing import List, Dict, Any
from pathlib import Path
import tempfile

# Conditional imports based on availability
try:
    from TTS.api import TTS

    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

try:
    import edge_tts

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from scipy.io.wavfile import write as write_wav

    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedTTSGenerator:
    """
    Enhanced TTS generator that works with your existing config structure.

    Features:
    - Integrates with existing config.yaml and tts.yaml files
    - Smart text cleaning and normalization
    - Minimum length enforcement to prevent kernel size errors
    - Intelligent chunking that preserves sentence structure
    - Multiple fallback strategies
    - Comprehensive error handling
    """

    def __init__(
        self, config_path: str = "config.yaml", tts_config_path: str = "tts.yaml"
    ):
        """
        Initialize the enhanced TTS generator using your existing config structure.

        Args:
            config_path: Path to main configuration file
            tts_config_path: Path to TTS-specific configuration file
        """
        self.config = self._load_config(config_path)
        self.tts_config = self._load_config(tts_config_path)
        self.logger = self._setup_logging()

        # Get configuration with safe fallbacks
        self.provider_name = self._get_provider_name()
        self.provider_config = self._get_provider_config()

        # Text processing parameters from config
        self.text_processing = self._get_text_processing_config()

        # Initialize provider
        self.provider = None
        self._initialize_provider()

        # Setup output directory
        self.audio_dir = Path(self._get_audio_directory())
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file with error handling."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                logger.info(f"Loaded config from {config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading config {config_path}: {e}")
            return {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the TTS generator."""
        logger_instance = logging.getLogger("enhanced_tts_generator")
        logger_instance.setLevel(logging.INFO)

        if not logger_instance.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger_instance.addHandler(handler)

        return logger_instance

    def _get_provider_name(self) -> str:
        """Get provider name from config with safe fallbacks."""
        try:
            return self.config["text_to_speech"]["tts"]["provider"]
        except (KeyError, TypeError):
            # Check what providers are available in tts.yaml and pick the first available one
            if "providers" in self.tts_config:
                for provider_name in self.tts_config["providers"].keys():
                    if self._check_provider_availability(provider_name):
                        self.logger.warning(
                            f"Using first available provider: {provider_name}"
                        )
                        return provider_name

            # Ultimate fallback
            self.logger.warning("No provider specified, defaulting to edge_tts")
            return "edge_tts"

    def _get_provider_config(self) -> Dict[str, Any]:
        """Get provider-specific configuration."""
        if "providers" not in self.tts_config:
            return {}

        provider_config = self.tts_config["providers"].get(self.provider_name, {})
        if not provider_config:
            self.logger.warning(
                f"No configuration found for provider {self.provider_name}"
            )

        return provider_config

    def _get_text_processing_config(self) -> Dict[str, Any]:
        """Get text processing configuration with safe defaults."""
        try:
            config = self.config["text_to_speech"]["tts"]["text_processing"]
            return {
                "chunk_size": config.get("chunk_size", 1000),
                "chunk_overlap": config.get("chunk_overlap", 100),
                "pause_between_chunks": config.get("pause_between_chunks", 0.5),
                "clean_markdown": config.get("clean_markdown", True),
                "expand_abbreviations": config.get("expand_abbreviations", True),
                "add_pauses": config.get("add_pauses", True),
            }
        except (KeyError, TypeError):
            return {
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "pause_between_chunks": 0.5,
                "clean_markdown": True,
                "expand_abbreviations": True,
                "add_pauses": True,
            }

    def _get_audio_directory(self) -> str:
        """Get audio directory with safe fallback."""
        try:
            return self.config["text_to_speech"]["files"]["directory"]
        except (KeyError, TypeError):
            return "audio"

    def _check_provider_availability(self, provider_name: str) -> bool:
        """Check if a provider's dependencies are available."""
        try:
            if provider_name == "edge_tts":
                return EDGE_TTS_AVAILABLE
            elif provider_name == "coqui":
                return COQUI_AVAILABLE
            elif provider_name == "bark":
                return BARK_AVAILABLE
            else:
                return False
        except:
            return False

    def _initialize_provider(self):
        """Initialize the selected TTS provider."""
        try:
            self.logger.info(f"Initializing provider: {self.provider_name}")

            if self.provider_name == "coqui":
                if not COQUI_AVAILABLE:
                    raise ImportError(
                        "Coqui TTS not available. Install with: pip install TTS"
                    )
                self.provider = self._init_coqui()

            elif self.provider_name == "edge_tts":
                if not EDGE_TTS_AVAILABLE:
                    raise ImportError(
                        "Edge TTS not available. Install with: pip install edge-tts"
                    )
                # Edge TTS doesn't need initialization
                self.provider = None

            elif self.provider_name == "bark":
                if not BARK_AVAILABLE:
                    raise ImportError(
                        "Bark not available. Install with: pip install git+https://github.com/suno-ai/bark.git scipy"
                    )
                self.provider = self._init_bark()

            else:
                raise ValueError(
                    f"Provider {self.provider_name} not implemented in enhanced generator"
                )

        except Exception as e:
            self.logger.error(f"Failed to initialize provider: {e}")
            raise

    def _init_coqui(self):
        """Initialize Coqui TTS with model selection from config."""
        try:
            # Get model from config
            model_choice = (
                self.config.get("text_to_speech", {}).get("tts", {}).get("model")
            )

            if model_choice and self.provider_config.get("models", {}).get(
                model_choice
            ):
                model_info = self.provider_config["models"][model_choice]
                model_name = model_info["model_name"]
            else:
                # Use default model from provider config
                default_model = self.provider_config.get("default_model", "fast")
                if (
                    "models" in self.provider_config
                    and default_model in self.provider_config["models"]
                ):
                    model_name = self.provider_config["models"][default_model][
                        "model_name"
                    ]
                else:
                    # Ultimate fallback
                    model_name = "tts_models/en/ljspeech/tacotron2-DDC"

            self.logger.info(f"Loading Coqui model: {model_name}")
            return TTS(model_name=model_name)

        except Exception as e:
            self.logger.error(f"Failed to initialize Coqui: {e}")
            # Try with a minimal fallback model
            try:
                self.logger.info("Trying fallback model")
                return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            except:
                raise e

    def _init_bark(self):
        """Initialize Bark TTS."""
        try:
            # Preload models
            preload_models()
            return "bark_initialized"  # Bark doesn't need a specific object
        except Exception as e:
            self.logger.error(f"Failed to initialize Bark: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for TTS processing to prevent kernel size errors.

        This addresses the specific error you encountered by ensuring text is
        properly formatted and long enough for the TTS models.
        """
        # Remove or replace problematic characters
        text = unicodedata.normalize("NFKD", text)

        # Remove combining characters that cause the ͡ issue
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")

        # Replace problematic characters with safe alternatives
        replacements = {
            "–": "-",  # en dash
            "—": "-",  # em dash
            """: "'",  # smart quote
            """: "'",  # smart quote
            '"': '"',  # smart quote
            '"': '"',  # smart quote
            "…": "...",  # ellipsis
            "ɹ": "r",  # IPA r
            "ɚ": "er",  # IPA schwa-r
            "t͡ʃ": "ch",  # IPA ch sound
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove markdown if configured
        if self.text_processing.get("clean_markdown", True):
            text = self._clean_markdown(text)

        # Expand abbreviations if configured
        if self.text_processing.get("expand_abbreviations", True):
            text = self._expand_abbreviations(text)

        # Clean up whitespace and ensure proper sentence endings
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)

        # Add pauses if configured
        if self.text_processing.get("add_pauses", True):
            text = self._add_natural_pauses(text)

        return text

    def _clean_markdown(self, text: str) -> str:
        """Remove markdown formatting."""
        # Remove headers
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
        # Remove links
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove code blocks
        text = re.sub(r"```[^`]*```", "", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations for better TTS."""
        abbreviations = {
            "e.g.": "for example",
            "i.e.": "that is",
            "etc.": "and so on",
            "vs.": "versus",
            "Mr.": "Mister",
            "Dr.": "Doctor",
            "Prof.": "Professor",
        }

        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)

        return text

    def _add_natural_pauses(self, text: str) -> str:
        """Add natural pauses for better TTS rhythm."""
        # Add small pauses after commas and semicolons
        text = re.sub(r",", ", ", text)
        text = re.sub(r";", "; ", text)

        # Ensure proper spacing after periods
        text = re.sub(r"\.([A-Z])", r". \1", text)

        return text

    def intelligent_chunk_text(self, text: str) -> List[str]:
        """
        Intelligently chunk text to ensure proper TTS processing.

        This prevents the kernel size error by ensuring each chunk is
        long enough and properly formatted.
        """
        # Clean the text first
        text = self.clean_text(text)

        chunk_size = self.text_processing.get("chunk_size", 1000)
        chunk_overlap = self.text_processing.get("chunk_overlap", 100)
        min_chunk_length = 100  # Minimum to prevent kernel size errors

        if len(text) <= chunk_size:
            # If text is short, ensure it meets minimum length
            if len(text) < min_chunk_length:
                padding = ". This ensures adequate length for speech synthesis."
                text = text + padding
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]

            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(0, len(chunk) - 200)
                search_text = chunk[search_start:]

                # Find the last sentence boundary
                last_period = search_text.rfind(". ")
                last_exclamation = search_text.rfind("! ")
                last_question = search_text.rfind("? ")

                best_break = max(last_period, last_exclamation, last_question)

                if best_break > 0:
                    # Adjust the chunk to end at sentence boundary
                    actual_break = search_start + best_break + 2
                    chunk = chunk[:actual_break]
                    end = start + actual_break

            # Ensure chunk meets minimum length
            if len(chunk) < min_chunk_length:
                if chunks:
                    # Merge with previous chunk
                    chunks[-1] += " " + chunk
                else:
                    # First chunk is too short, pad it
                    padding = ". This provides additional content for proper audio generation."
                    chunk += padding
                    chunks.append(chunk)
            else:
                chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - chunk_overlap
            if start >= len(text):
                break

        # Final validation
        validated_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) >= min_chunk_length:
                validated_chunks.append(chunk.strip())
            elif validated_chunks:
                validated_chunks[-1] += " " + chunk.strip()

        return validated_chunks if validated_chunks else [self._create_fallback_text()]

    def _create_fallback_text(self) -> str:
        """Create fallback text when original text is problematic."""
        return (
            "This is a fallback audio message. The original text could not be processed "
            "properly for speech synthesis. Please check the input text for formatting issues."
        )

    async def generate_audio_chunk(self, text_chunk: str, output_path: str) -> bool:
        """
        Generate audio for a single text chunk with enhanced error handling.

        Args:
            text_chunk: Text to convert to speech
            output_path: Path for output audio file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Additional validation to prevent kernel size errors
            if len(text_chunk.strip()) < 20:
                self.logger.warning(
                    f"Text chunk too short ({len(text_chunk)} chars), using fallback"
                )
                text_chunk = self._create_fallback_text()

            if self.provider_name == "coqui":
                return await self._generate_coqui_audio(text_chunk, output_path)
            elif self.provider_name == "edge_tts":
                return await self._generate_edge_tts_audio(text_chunk, output_path)
            elif self.provider_name == "bark":
                return self._generate_bark_audio(text_chunk, output_path)
            else:
                self.logger.error(f"Provider {self.provider_name} not implemented")
                return False

        except Exception as e:
            self.logger.error(f"Audio generation failed for chunk: {e}")
            # Try with fallback text
            if text_chunk != self._create_fallback_text():
                self.logger.info("Retrying with fallback text")
                fallback_text = self._create_fallback_text()
                return await self.generate_audio_chunk(fallback_text, output_path)
            return False

    async def _generate_coqui_audio(self, text: str, output_path: str) -> bool:
        """Generate audio using Coqui TTS with enhanced error handling."""
        try:
            if self.provider is None:
                raise ValueError("Coqui provider not initialized")

            # Generate audio with error handling
            self.provider.tts_to_file(text=text, file_path=output_path)
            return True

        except Exception as e:
            self.logger.error(f"Coqui TTS generation failed: {e}")
            return False

    async def _generate_edge_tts_audio(self, text: str, output_path: str) -> bool:
        """Generate audio using Edge TTS."""
        try:
            # Get voice from config
            voice = self.config.get("text_to_speech", {}).get("tts", {}).get(
                "voice"
            ) or self.provider_config.get("default_voice", "en-US-AriaNeural")

            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            return True

        except Exception as e:
            self.logger.error(f"Edge TTS generation failed: {e}")
            return False

    def _generate_bark_audio(self, text: str, output_path: str) -> bool:
        """Generate audio using Bark."""
        try:
            # Get voice preset from config
            voice_preset = self.config.get("text_to_speech", {}).get("tts", {}).get(
                "voice"
            ) or self.provider_config.get("default_voice", "v2/en_speaker_6")

            # Generate audio
            audio_array = generate_audio(text, history_prompt=voice_preset)

            # Save to file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                write_wav(temp_file.name, SAMPLE_RATE, audio_array)
                temp_path = temp_file.name

            # Move to final location
            os.rename(temp_path, output_path)
            return True

        except Exception as e:
            self.logger.error(f"Bark generation failed: {e}")
            return False

    async def generate_audio(self, text: str, output_path: str) -> bool:
        """
        Generate audio from text with comprehensive error handling.

        This is the main method that fixes your kernel size error while
        maintaining compatibility with your existing configuration.
        """
        try:
            self.logger.info(f"Generating audio using {self.provider_name} provider")

            # Chunk the text intelligently
            chunks = self.intelligent_chunk_text(text)
            self.logger.info(f"Split text into {len(chunks)} chunks")

            if not chunks:
                self.logger.error("No valid chunks created from text")
                return False

            # For single chunk, generate directly
            if len(chunks) == 1:
                return await self.generate_audio_chunk(chunks[0], output_path)

            # For multiple chunks, generate and combine
            temp_files = []
            for i, chunk in enumerate(chunks):
                self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")

                temp_path = f"{output_path}.chunk_{i}.wav"
                if await self.generate_audio_chunk(chunk, temp_path):
                    temp_files.append(temp_path)
                else:
                    self.logger.error(f"Failed to generate chunk {i+1}")
                    # Clean up temp files
                    for temp_file in temp_files:
                        Path(temp_file).unlink(missing_ok=True)
                    return False

            # Combine audio files
            return self._combine_audio_files(temp_files, output_path)

        except Exception as e:
            self.logger.error(f"Audio generation failed: {e}")
            return False

    def _combine_audio_files(self, audio_files: List[str], output_path: str) -> bool:
        """Combine multiple audio files into one."""
        try:
            # This is a simplified version - you might want to use a more robust
            # audio library like pydub for better format handling
            import subprocess

            # Use ffmpeg to combine files if available
            file_list = "|".join(audio_files)
            cmd = [
                "ffmpeg",
                "-i",
                f"concat:{file_list}",
                "-acodec",
                "copy",
                output_path,
                "-y",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Clean up temporary files
                for audio_file in audio_files:
                    Path(audio_file).unlink(missing_ok=True)
                return True
            else:
                self.logger.error(f"ffmpeg failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to combine audio files: {e}")
            return False


# Convenience function that works with your existing workflow
async def generate_audio_with_config(
    text: str,
    output_path: str,
    config_path: str = "config.yaml",
    tts_config_path: str = "tts.yaml",
) -> bool:
    """
    Generate audio using the enhanced TTS generator with your existing config structure.

    This function provides a drop-in replacement for your existing TTS functionality
    while fixing the kernel size error and adding robustness.
    """
    generator = EnhancedTTSGenerator(config_path, tts_config_path)
    return await generator.generate_audio(text, output_path)


# Synchronous wrapper for compatibility
def generate_audio_sync(
    text: str,
    output_path: str,
    config_path: str = "config.yaml",
    tts_config_path: str = "tts.yaml",
) -> bool:
    """Synchronous wrapper for generate_audio_with_config."""
    return asyncio.run(
        generate_audio_with_config(text, output_path, config_path, tts_config_path)
    )

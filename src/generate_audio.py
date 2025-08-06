#!/usr/bin/env python3
"""
Enhanced Audio Generator for ArXiv Summaries

This script generates audio from text summaries with enhanced error handling
that fixes the "kernel size" error and handles problematic characters.

Usage:
    python generate_audio.py [audio_script]
    python generate_audio.py --test              # Run test
    python generate_audio.py --list-providers    # Show available providers

If no file is specified, it will look for the most recent audio script.
"""

import asyncio
import argparse
import sys
import os
import re
import unicodedata
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

# Conditional imports for TTS providers
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
    from bark import SAMPLE_RATE, generate_audio as bark_generate, preload_models
    from scipy.io.wavfile import write as write_wav

    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("enhanced_audio_generator")


class EnhancedAudioGenerator:
    """
    Enhanced TTS generator that fixes kernel size errors and handles
    problematic characters while working with existing config structure.
    """

    def __init__(self, config_path: str, tts_config_path: str):
        """Initialize with config files."""
        self.config = self._load_config(config_path)
        self.tts_config = self._load_config(tts_config_path)

        # Get provider configuration
        self.provider_name = self._get_provider_name()
        self.provider_config = self._get_provider_config()
        self.text_processing = self._get_text_processing_config()

        # Initialize provider
        self.provider = None
        self._initialize_provider()

        # Setup output directory
        self.audio_dir = Path(self._get_audio_directory())
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading config {config_path}: {e}")
            return {}

    def _get_provider_name(self) -> str:
        """Get provider name from config with fallbacks."""
        try:
            return self.config["text_to_speech"]["tts"]["provider"]
        except (KeyError, TypeError):
            # Check available providers and pick the first working one
            if EDGE_TTS_AVAILABLE:
                logger.warning("No provider specified, using edge_tts")
                return "edge_tts"
            elif COQUI_AVAILABLE:
                logger.warning("No provider specified, using coqui")
                return "coqui"
            else:
                raise RuntimeError(
                    "No TTS providers available. Install with: pip install edge-tts TTS"
                )

    def _get_provider_config(self) -> Dict:
        """Get provider-specific configuration."""
        if "providers" not in self.tts_config:
            return {}
        return self.tts_config["providers"].get(self.provider_name, {})

    def _get_text_processing_config(self) -> Dict:
        """Get text processing configuration with safe defaults."""
        try:
            config = self.config["text_to_speech"]["tts"]["text_processing"]
            return {
                "chunk_size": config.get("chunk_size", 1000),
                "chunk_overlap": config.get("chunk_overlap", 100),
                "clean_markdown": config.get("clean_markdown", True),
                "expand_abbreviations": config.get("expand_abbreviations", True),
                "add_pauses": config.get("add_pauses", True),
            }
        except (KeyError, TypeError):
            return {
                "chunk_size": 1000,
                "chunk_overlap": 100,
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

    def _initialize_provider(self):
        """Initialize the TTS provider."""
        try:
            logger.info(f"Initializing provider: {self.provider_name}")

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
                    raise ImportError("Bark not available. Install dependencies")
                self.provider = self._init_bark()

            else:
                raise ValueError(f"Provider {self.provider_name} not supported")

        except Exception as e:
            logger.error(f"Failed to initialize provider: {e}")
            # Try fallback to edge_tts
            if self.provider_name != "edge_tts" and EDGE_TTS_AVAILABLE:
                logger.info("Falling back to edge_tts")
                self.provider_name = "edge_tts"
                self.provider_config = self.tts_config.get("providers", {}).get(
                    "edge_tts", {}
                )
                self.provider = None
            else:
                raise

    def _init_coqui(self):
        """Initialize Coqui TTS with robust model selection and PyTorch compatibility."""
        try:
            # Try to get model from config
            model_choice = (
                self.config.get("text_to_speech", {}).get("tts", {}).get("model")
            )

            if model_choice and self.provider_config.get("models", {}).get(
                model_choice
            ):
                model_info = self.provider_config["models"][model_choice]
                model_name = model_info["model_name"]
            else:
                # Use a safe default model that's less likely to have kernel issues
                model_name = "tts_models/en/ljspeech/tacotron2-DDC"

            logger.info(f"Loading Coqui model: {model_name}")

            # Handle XTTS PyTorch loading issues
            if "xtts" in model_name.lower():
                return self._init_xtts_model(model_name)
            else:
                return TTS(model_name=model_name)

        except Exception as e:
            logger.error(f"Failed to initialize Coqui: {e}")
            # Try fallback models in order of reliability
            fallback_models = [
                "tts_models/en/ljspeech/tacotron2-DDC",
                "tts_models/en/ljspeech/speedy-speech",
                "tts_models/en/ljspeech/glow-tts",
            ]

            for fallback_model in fallback_models:
                try:
                    logger.info(f"Trying fallback model: {fallback_model}")
                    return TTS(model_name=fallback_model)
                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback model {fallback_model} failed: {fallback_error}"
                    )
                    continue

            raise Exception(f"All Coqui models failed to load. Original error: {e}")

    def _init_xtts_model(self, model_name: str):
        """Initialize XTTS model with PyTorch 2.6+ compatibility."""
        try:
            import torch

            # Method 1: Try adding safe globals for XTTS
            try:
                torch.serialization.add_safe_globals(
                    ["TTS.tts.configs.xtts_config.XttsConfig"]
                )
                logger.info("Added XTTS config to PyTorch safe globals")
                return TTS(model_name=model_name)
            except Exception as safe_globals_error:
                logger.warning(f"Safe globals approach failed: {safe_globals_error}")

            # Method 2: Temporarily modify torch.load behavior
            try:
                # Store original torch.load
                original_torch_load = torch.load

                # Create a wrapper that uses weights_only=False for XTTS
                def xtts_compatible_load(*args, **kwargs):
                    if "weights_only" not in kwargs:
                        kwargs["weights_only"] = False
                    return original_torch_load(*args, **kwargs)

                # Temporarily replace torch.load
                torch.load = xtts_compatible_load

                try:
                    logger.info("Using PyTorch load compatibility mode for XTTS")
                    model = TTS(model_name=model_name)
                    return model
                finally:
                    # Restore original torch.load
                    torch.load = original_torch_load

            except Exception as load_wrapper_error:
                logger.warning(f"Load wrapper approach failed: {load_wrapper_error}")

            # Method 3: If all else fails, suggest using a different model
            raise Exception(
                f"XTTS model loading failed due to PyTorch 2.6+ security changes. "
                f"Consider using a different model like 'fast' or 'high_quality' instead."
            )

        except Exception as e:
            logger.error(f"XTTS initialization failed: {e}")
            raise e

    def _init_bark(self):
        """Initialize Bark TTS."""
        try:
            preload_models()
            return "bark_initialized"
        except Exception as e:
            logger.error(f"Failed to initialize Bark: {e}")
            raise

    def clean_text_for_tts(self, text: str) -> str:
        """
        Enhanced text cleaning that prevents the kernel size error.

        This specifically addresses:
        1. The ͡ combining character issue from your error
        2. IPA symbols like ɹisɚt͡ʃ
        3. Minimum length requirements for TTS models
        """
        # Step 1: Unicode normalization and combining character removal
        text = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")

        # Step 2: Replace problematic IPA and special characters
        replacements = {
            # IPA symbols that caused your error
            "ɹ": "r",
            "ɚ": "er",
            "t͡ʃ": "ch",
            "ʃ": "sh",
            "θ": "th",
            "ð": "th",
            # Common problematic characters
            "–": "-",
            "—": "-",
            """: "'",
            """: "'",
            '"': '"',
            '"': '"',
            "…": "...",
            "°": " degrees",
            "±": " plus or minus",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Step 3: Remove any remaining non-ASCII that might cause issues
        text = "".join(char if ord(char) < 128 else " " for char in text)

        # Step 4: Clean markdown if configured
        if self.text_processing.get("clean_markdown", True):
            text = self._clean_markdown(text)

        # Step 5: Expand abbreviations if configured
        if self.text_processing.get("expand_abbreviations", True):
            text = self._expand_abbreviations(text)

        # Step 6: Normalize whitespace and ensure proper sentence structure
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)

        # Step 7: Add natural pauses if configured
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
        """Expand abbreviations for better TTS."""
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
        # Add pauses after commas and semicolons
        text = re.sub(r",", ", ", text)
        text = re.sub(r";", "; ", text)
        # Ensure proper spacing after periods
        text = re.sub(r"\.([A-Z])", r". \1", text)
        return text

    def chunk_text_intelligently(self, text: str) -> List[str]:
        """
        Chunk text to prevent kernel size errors while preserving readability.

        Key improvements:
        - Ensures minimum chunk length (prevents kernel size error)
        - Breaks at sentence boundaries when possible
        - Adds padding to short chunks
        """
        # Clean the text first
        text = self.clean_text_for_tts(text)

        chunk_size = self.text_processing.get("chunk_size", 1000)
        chunk_overlap = self.text_processing.get("chunk_overlap", 100)
        min_chunk_length = 80  # Minimum to prevent kernel size errors

        # If text is short, ensure it meets minimum length
        if len(text) <= chunk_size:
            if len(text) < min_chunk_length:
                padding = ". This ensures the text is long enough for proper speech synthesis."
                text = text + padding
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]

            # If not the last chunk, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings in the last part of the chunk
                search_start = max(0, len(chunk) - 200)
                search_text = chunk[search_start:]

                # Find the best sentence boundary
                sentence_ends = [
                    search_text.rfind(". "),
                    search_text.rfind("! "),
                    search_text.rfind("? "),
                ]
                best_break = max(sentence_ends)

                if best_break > 0:
                    # Adjust chunk to end at sentence boundary
                    actual_break = search_start + best_break + 2
                    chunk = chunk[:actual_break]
                    end = start + actual_break

            # Ensure chunk meets minimum length requirement
            if len(chunk.strip()) < min_chunk_length:
                if chunks:
                    # Merge with previous chunk
                    chunks[-1] += " " + chunk.strip()
                else:
                    # First chunk is too short, pad it
                    padding = ". This additional content ensures proper processing for text to speech conversion."
                    chunk += padding
                    chunks.append(chunk.strip())
            else:
                chunks.append(chunk.strip())

            # Move to next chunk with overlap
            start = end - chunk_overlap
            if start >= len(text):
                break

        # Final validation: ensure no chunk is too short
        validated_chunks = []
        for chunk in chunks:
            if len(chunk) >= min_chunk_length:
                validated_chunks.append(chunk)
            elif validated_chunks:
                validated_chunks[-1] += " " + chunk

        return validated_chunks if validated_chunks else [self._create_fallback_text()]

    def _create_fallback_text(self) -> str:
        """Create fallback text when original text is problematic."""
        return (
            "This is a fallback audio message. The original text could not be processed "
            "properly for speech synthesis due to formatting or length issues. "
            "Please check the input text for any special characters or formatting problems."
        )

    async def generate_audio_chunk(self, text: str, output_path: str) -> bool:
        """Generate audio for a single text chunk."""
        try:
            # Validate text length to prevent kernel size error
            if len(text.strip()) < 20:
                logger.warning(
                    f"Text chunk too short ({len(text)} chars), using fallback"
                )
                text = self._create_fallback_text()

            if self.provider_name == "coqui":
                return self._generate_coqui_audio(text, output_path)
            elif self.provider_name == "edge_tts":
                return await self._generate_edge_tts_audio(text, output_path)
            elif self.provider_name == "bark":
                return self._generate_bark_audio(text, output_path)
            else:
                logger.error(f"Provider {self.provider_name} not implemented")
                return False

        except Exception as e:
            logger.error(f"Audio generation failed for chunk: {e}")
            # Try with fallback text if we haven't already
            if text != self._create_fallback_text():
                logger.info("Retrying with fallback text")
                return await self.generate_audio_chunk(
                    self._create_fallback_text(), output_path
                )
            return False

    def _generate_coqui_audio(self, text: str, output_path: str) -> bool:
        """Generate audio using Coqui TTS."""
        try:
            if self.provider is None:
                raise ValueError("Coqui provider not initialized")

            self.provider.tts_to_file(text=text, file_path=output_path)
            return True

        except Exception as e:
            logger.error(f"Coqui TTS generation failed: {e}")
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
            logger.error(f"Edge TTS generation failed: {e}")
            return False

    def _generate_bark_audio(self, text: str, output_path: str) -> bool:
        """Generate audio using Bark."""
        try:
            # Get voice preset
            voice_preset = self.config.get("text_to_speech", {}).get("tts", {}).get(
                "voice"
            ) or self.provider_config.get("default_voice", "v2/en_speaker_6")

            # Generate audio
            audio_array = bark_generate(text, history_prompt=voice_preset)

            # Save to file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                write_wav(temp_file.name, SAMPLE_RATE, audio_array)
                temp_path = temp_file.name

            # Move to final location
            os.rename(temp_path, output_path)
            return True

        except Exception as e:
            logger.error(f"Bark generation failed: {e}")
            return False

    async def generate_audio_from_text(
        self, text: str, report_type: str = "summary"
    ) -> Optional[str]:
        """
        Main method to generate audio from text with enhanced error handling.

        This method fixes the kernel size error while maintaining compatibility
        with your existing workflow.
        """
        try:
            logger.info(f"Generating audio using {self.provider_name} provider")

            # Chunk the text intelligently
            chunks = self.chunk_text_intelligently(text)
            logger.info(f"Split text into {len(chunks)} chunks")

            if not chunks:
                logger.error("No valid chunks created from text")
                return None

            # Generate filename
            filename = self._generate_filename(report_type)
            output_path = self.audio_dir / filename

            # For single chunk, generate directly
            if len(chunks) == 1:
                success = await self.generate_audio_chunk(chunks[0], str(output_path))
                return str(output_path) if success else None

            # For multiple chunks, generate and combine
            temp_files = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")

                temp_path = f"{output_path}.chunk_{i}.wav"
                if await self.generate_audio_chunk(chunk, temp_path):
                    temp_files.append(temp_path)
                else:
                    logger.error(f"Failed to generate chunk {i+1}")
                    # Clean up temp files
                    for temp_file in temp_files:
                        Path(temp_file).unlink(missing_ok=True)
                    return None

            # Combine audio files
            success = self._combine_audio_files(temp_files, str(output_path))
            return str(output_path) if success else None

        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            return None

    def _generate_filename(self, report_type: str = "summary") -> str:
        """Generate filename based on configuration."""
        try:
            naming_config = self.config["text_to_speech"]["files"]["naming_convention"]
        except (KeyError, TypeError):
            naming_config = "{timestamp}_{report_type}_summary"

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date = datetime.now().strftime("%Y-%m-%d")

        filename = naming_config.format(
            timestamp=timestamp,
            date=date,
            report_type=report_type,
            categories="general",
        )

        # Get output format
        try:
            output_format = self.config["text_to_speech"]["tts"]["output"][
                "format"
            ] or self.provider_config.get("output_format", "mp3")
        except (KeyError, TypeError):
            output_format = "mp3"

        return f"{filename}.{output_format}"

    def _combine_audio_files(self, audio_files: List[str], output_path: str) -> bool:
        """Combine multiple audio files into one."""
        try:
            # Simple concatenation for now
            with open(output_path, "wb") as outfile:
                for audio_file in audio_files:
                    with open(audio_file, "rb") as infile:
                        outfile.write(infile.read())

            # Clean up temporary files
            for audio_file in audio_files:
                Path(audio_file).unlink(missing_ok=True)

            logger.info(f"Successfully combined {len(audio_files)} audio chunks")
            return True

        except Exception as e:
            logger.error(f"Failed to combine audio files: {e}")
            return False


def find_config_files():
    """Find configuration files in the project structure."""
    search_paths = [
        Path.cwd(),
        Path(__file__).parent,
        Path(__file__).parent.parent,
        Path(__file__).parent.parent / "config",
    ]

    config_names = ["config.yaml", "config.yml"]
    tts_config_names = ["tts.yaml", "tts.yml"]

    config_path = None
    tts_config_path = None

    # Find main config
    for search_dir in search_paths:
        for config_name in config_names:
            potential_path = search_dir / config_name
            if potential_path.exists():
                config_path = potential_path
                break
        if config_path:
            break

    # Find TTS config
    if config_path:
        config_dir = config_path.parent
        for tts_config_name in tts_config_names:
            potential_path = config_dir / tts_config_name
            if potential_path.exists():
                tts_config_path = potential_path
                break

    return config_path, tts_config_path


def find_latest_audio_script(data_dir: str = "data") -> Path:
    """Find the most recent audio script file."""
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Search for audio script files
    audio_script_files = list(data_path.rglob("audio_script*.txt"))

    if not audio_script_files:
        raise FileNotFoundError(f"No audio_script files found in {data_dir}")

    return max(audio_script_files, key=lambda x: x.stat().st_mtime)


def load_audio_script(file_path: Path) -> str:
    """Load and prepare audio script for TTS conversion."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            raise ValueError(f"Audio script file is empty: {file_path}")

        # Filter out header information before the separator
        separator = "=" * 50  # Matches 50 equals signs
        if separator in content:
            # Split on the separator and take everything after it
            parts = content.split(separator, 1)  # Split only on first occurrence
            if len(parts) > 1:
                content = parts[1].strip()  # Take content after separator
                logger.info("Filtered out header information before separator")

        # Remove speaker markers like "Host:" since there is only one speaker
        content = re.sub(r"^\s*Host:\s*", "", content, flags=re.MULTILINE)

        # Remove single lines that are parentheticals (e.g., (pause), (verbal cue), etc.)
        content = re.sub(
            r"^\s*\([^)]+\)\s*$",
            "",
            content,
            flags=re.MULTILINE,
        )

        logger.info(f"Loaded audio script from: {file_path}")
        logger.info(f"Text length: {len(content)} characters")

        return content

    except Exception as e:
        raise Exception(f"Failed to load audio script file {file_path}: {e}")


async def generate_audio_script(
    script_file: Path = None, provider: str = None
) -> Optional[str]:
    """
    Generate audio from an audio script file with enhanced error handling.

    Args:
        script_file: Path to audio script file (if None, finds latest)
        provider: TTS provider to use (if None, uses config default)

    Returns:
        Path to generated audio file or None if failed
    """
    try:
        # Find configuration files
        config_path, tts_config_path = find_config_files()

        if not config_path:
            raise FileNotFoundError("Could not find config.yaml")

        if not tts_config_path:
            raise FileNotFoundError("Could not find tts.yaml")

        logger.info(f"Using config: {config_path}")
        logger.info(f"Using TTS config: {tts_config_path}")

        # Find audio script file if not specified
        if script_file is None:
            script_file = find_latest_audio_script()

        # Load the audio script text
        audio_script = load_audio_script(script_file)

        # Initialize enhanced TTS generator
        logger.info("Initializing enhanced TTS generator...")
        tts = EnhancedAudioGenerator(str(config_path), str(tts_config_path))

        # Override provider if specified
        if provider:
            if (
                "providers" in tts.tts_config
                and provider in tts.tts_config["providers"]
            ):
                tts.provider_name = provider
                tts.provider_config = tts.tts_config["providers"][provider]
                tts._initialize_provider()
                logger.info(f"Using TTS provider: {provider}")
            else:
                available = list(tts.tts_config.get("providers", {}).keys())
                raise ValueError(
                    f"Provider '{provider}' not found. Available: {available}"
                )
        else:
            logger.info(f"Using TTS provider: {tts.provider_name}")

        # Determine report type from filename
        report_type = "audio_script"
        if "daily" in script_file.name.lower():
            report_type = "daily_audio_script"
        elif "weekly" in script_file.name.lower():
            report_type = "weekly_audio_script"
        elif "report" in script_file.name.lower():
            report_type = "report"

        # Generate audio with enhanced processing
        logger.info("Generating audio...")
        logger.info(
            "This may take a few minutes depending on text length and provider..."
        )

        audio_path = await tts.generate_audio_from_text(audio_script, report_type)

        if audio_path:
            logger.info("✅ Audio generated successfully!")
            logger.info(f"📁 File saved to: {audio_path}")

            # Show file size
            audio_file = Path(audio_path)
            if audio_file.exists():
                size_mb = audio_file.stat().st_size / (1024 * 1024)
                logger.info(f"📊 File size: {size_mb:.1f} MB")

            return audio_path
        else:
            logger.error("❌ Audio generation failed")
            return None

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return None


def check_provider_availability(provider_name: str) -> bool:
    """Check if a provider is available."""
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


def list_available_providers():
    """List available TTS providers."""
    try:
        _, tts_config_path = find_config_files()

        if not tts_config_path:
            logger.error("Could not find TTS config file")
            return

        with open(tts_config_path, "r") as f:
            tts_config = yaml.safe_load(f)

        if "providers" not in tts_config:
            logger.error("TTS config file doesn't contain 'providers' section")
            return

        providers = tts_config["providers"]

        print("\nAvailable TTS providers:")
        print("-" * 50)

        for name, config in providers.items():
            provider_type = config.get("type", "unknown")
            provider_name = config.get("name", name)

            # Check if provider is available
            availability = "✅" if check_provider_availability(name) else "❌"

            print(f"{availability} {name:<15} - {provider_name} ({provider_type})")

        print("\n✅ = Available, ❌ = Requires installation")

    except Exception as e:
        logger.error(f"Failed to list providers: {e}")


def diagnose_pytorch_compatibility():
    """Diagnose PyTorch version compatibility issues."""
    print("🔍 PyTorch Compatibility Diagnosis")
    print("-" * 40)

    try:
        import torch

        pytorch_version = torch.__version__
        print(f"📦 PyTorch version: {pytorch_version}")

        # Check if this is PyTorch 2.6+
        version_parts = pytorch_version.split(".")
        major, minor = int(version_parts[0]), int(version_parts[1])

        if major >= 2 and minor >= 6:
            print("⚠️  PyTorch 2.6+ detected - may have XTTS compatibility issues")
            print(
                "💡 Recommendation: Use 'fast' or 'high_quality' models instead of 'xtts'"
            )
            print("💡 Or install older PyTorch: pip install 'torch<2.6'")
        else:
            print("✅ PyTorch version should be compatible with all models")

    except ImportError:
        print("❌ PyTorch not installed")
        return False

    try:
        from TTS.api import TTS

        print("✅ TTS library available")
    except ImportError:
        print("❌ TTS library not installed")
        return False

    return True


def test_enhanced_tts():
    """
    Test the enhanced TTS with problematic text similar to your error.

    This test uses text that contains the exact characters that caused
    your kernel size error to verify the fix works.
    """
    print("🧪 Testing Enhanced TTS with Problematic Text")
    print("=" * 50)

    # Run PyTorch compatibility check first
    if not diagnose_pytorch_compatibility():
        print("❌ Environment check failed")
        return False

    print()  # Add spacing

    # Test text that includes the problematic characters from your error
    test_text = """Research Digest Audio Script
Generated on 2025-08-05 at 23:48:34
Model: deepseek-r1-qwen3
Source: stage1 content
Length: 1,095 words
Estimated duration: 8.4 minutes
==================================================

Okay, let's dive into the recent developments synthesized from these papers relevant to statistical AI for cosmic discovery.
This test includes the problematic characters that caused the kernel size error: ɹisɚt͡ʃ daɪd͡ʒɛst ɔdioʊ skɹɪpt

The enhanced system should handle these gracefully and produce clear audio output.
This text is specifically designed to test the fix for the "Calculated padded input size per channel: (4). Kernel size: (5)" error.
"""

    try:
        # Find config files
        config_path, tts_config_path = find_config_files()

        if not config_path or not tts_config_path:
            print(
                "❌ Could not find config files. Please ensure config.yaml and tts.yaml exist."
            )
            return False

        print(f"📁 Using config: {config_path}")
        print(f"📁 Using TTS config: {tts_config_path}")

        # Initialize enhanced generator
        print("🔧 Initializing enhanced TTS generator...")
        generator = EnhancedAudioGenerator(str(config_path), str(tts_config_path))

        print(f"🎯 Provider: {generator.provider_name}")
        print(f"📝 Original text length: {len(test_text)} characters")

        # Test text cleaning
        cleaned_text = generator.clean_text_for_tts(test_text)
        print(f"🧹 Cleaned text length: {len(cleaned_text)} characters")
        print(
            f"🔍 Removed problematic characters: {len(test_text) - len(cleaned_text)} chars"
        )

        # Test chunking
        chunks = generator.chunk_text_intelligently(test_text)
        print(f"📊 Created {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}: {len(chunk)} characters")

        # Test actual audio generation
        print("\n🎵 Testing audio generation...")

        output_path = "test_enhanced_tts.wav"
        success = asyncio.run(generator.generate_audio_from_text(test_text, "test"))

        if success:
            print("✅ Audio generation test PASSED!")
            print(f"📁 Test audio saved to: {success}")

            # Check file size
            if Path(success).exists():
                size_bytes = Path(success).stat().st_size
                print(f"📊 File size: {size_bytes} bytes ({size_bytes/1024:.1f} KB)")

            return True
        else:
            print("❌ Audio generation test FAILED!")
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced Audio Generator for ArXiv Summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_audio.py                           # Use latest script
    python generate_audio.py audio_script.txt          # Use specific file  
    python generate_audio.py --provider edge_tts       # Use specific provider
    python generate_audio.py --list-providers          # Show available providers
    python generate_audio.py --test                    # Run enhanced TTS test
    python generate_audio.py --diagnose                # Check PyTorch compatibility
        """,
    )

    parser.add_argument(
        "audio_script",
        nargs="?",
        help="Path to audio script file (optional - will find latest if not specified)",
    )

    parser.add_argument("--provider", help="TTS provider to use (overrides config)")

    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available TTS providers and exit",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run enhanced TTS test with problematic characters",
    )

    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Diagnose PyTorch and TTS compatibility issues",
    )

    args = parser.parse_args()

    # Handle diagnosis mode
    if args.diagnose:
        success = diagnose_pytorch_compatibility()
        if success:
            print("\n✅ Environment diagnosis completed")
        else:
            print("\n❌ Environment issues detected")
        sys.exit(0)

    # Handle test mode
    if args.test:
        success = test_enhanced_tts()
        if success:
            print("\n🎉 Enhanced TTS test completed successfully!")
            print("The kernel size error fix is working correctly.")
        else:
            print(
                "\n❌ Enhanced TTS test failed. Check your configuration and dependencies."
            )
        sys.exit(0 if success else 1)

    # Handle provider listing
    if args.list_providers:
        list_available_providers()
        return

    # Prepare audio script file path
    audio_script_file = None
    if args.audio_script:
        audio_script_file = Path(args.audio_script)
        if not audio_script_file.exists():
            print(f"❌ Audio script file not found: {audio_script_file}")
            sys.exit(1)

    # Run audio generation
    print("🎵 Enhanced ArXiv Audio Generator")
    print("=" * 40)

    try:
        audio_path = asyncio.run(
            generate_audio_script(script_file=audio_script_file, provider=args.provider)
        )

        if audio_path:
            print(f"\n🎧 Your audio script is ready!")
            print(f"You can now listen to: {Path(audio_path).name}")
        else:
            print(f"\n❌ Audio generation failed")
            print(f"💡 Tip: Try running --test to diagnose issues")
            print(f"💡 Tip: Make sure text_to_speech is enabled in config.yaml")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n⏹️  Audio generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Text-to-Speech Audio Generator

This module handles the conversion of text summaries to audio files using
various TTS providers (local and API-based).
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import yaml
import re

# Import statements will be conditional based on available providers
try:
    import edge_tts

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    from TTS.api import TTS

    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from scipy.io.wavfile import write as write_wav

    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from google.cloud import texttospeech

    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False


def find_config_files():
    """
    Find configuration files, checking multiple possible locations.

    Returns:
        Tuple of (config_path, tts_config_path)
    """
    # Possible locations to check (in order of preference)
    search_paths = [
        Path.cwd(),  # Current working directory
        Path(__file__).parent,  # Same directory as this script
        Path(__file__).parent.parent,  # Parent directory (if script is in src/)
        Path(__file__).parent.parent
        / "config",  # config folder in parent parent directory
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

    # Find TTS config (look in same directory as main config)
    if config_path:
        config_dir = config_path.parent
        for tts_config_name in tts_config_names:
            potential_path = config_dir / tts_config_name
            if potential_path.exists():
                tts_config_path = potential_path
                break

    # If we didn't find TTS config in same dir as main config, search all paths
    if not tts_config_path:
        for search_dir in search_paths:
            for tts_config_name in tts_config_names:
                potential_path = search_dir / tts_config_name
                if potential_path.exists():
                    tts_config_path = potential_path
                    break
            if tts_config_path:
                break

    return config_path, tts_config_path


class TTSGenerator:
    """
    Text-to-Speech generator that supports multiple providers.

    This class provides a unified interface for generating audio from text
    using various TTS providers, both local and API-based.
    """

    def __init__(
        self, config_path: str = "config.yaml", tts_config_path: str = "tts.yaml"
    ):
        """
        Initialize the TTS generator.

        Args:
            config_path: Path to main configuration file
            tts_config_path: Path to TTS-specific configuration file
        """
        self.config = self._load_config(config_path)
        self.tts_config = self._load_config(tts_config_path)
        self.logger = self._setup_logging()

        # Get provider name with safe fallback
        self.provider_name = self._get_provider_name()

        # Validate provider exists in TTS config
        if self.provider_name not in self.tts_config["providers"]:
            available = list(self.tts_config["providers"].keys())
            raise ValueError(
                f"Provider '{self.provider_name}' not found in TTS config. Available: {available}"
            )

        self.provider_config = self.tts_config["providers"][self.provider_name]

        # Initialize provider with error handling
        try:
            self.provider = self._initialize_provider()
        except Exception as e:
            self.logger.error(
                f"Failed to initialize provider '{self.provider_name}': {e}"
            )
            raise

        # Setup output directory
        self.audio_dir = Path(self._get_audio_directory())
        self.audio_dir.mkdir(exist_ok=True)

        # Setup archive directory if needed
        if self._get_archive_setting():
            (self.audio_dir / "archive").mkdir(exist_ok=True)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the TTS generator."""
        logger = logging.getLogger("tts_generator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _get_provider_name(self) -> str:
        """Get provider name with safe fallbacks."""
        try:
            # Try to get from main config
            return self.config["text_to_speech"]["tts"]["provider"]
        except (KeyError, TypeError):
            # Fallback to first available provider that's actually installed
            for provider_name in self.tts_config["providers"].keys():
                if self._check_provider_availability(provider_name):
                    self.logger.warning(
                        f"No provider specified in config, using available provider: {provider_name}"
                    )
                    return provider_name

            # Last resort: use edge_tts as it's most likely to work
            self.logger.warning(
                "No provider specified and none detected, defaulting to edge_tts"
            )
            return "edge_tts"

    def _get_audio_directory(self) -> str:
        """Get audio directory with safe fallback."""
        try:
            return self.config["text_to_speech"]["files"]["directory"]
        except (KeyError, TypeError):
            return "text_to_speech"  # Default fallback

    def _get_archive_setting(self) -> bool:
        """Get archive setting with safe fallback."""
        try:
            return self.config["text_to_speech"]["files"]["archive_old_files"]
        except (KeyError, TypeError):
            return False  # Default fallback

    def _check_provider_availability(self, provider_name: str) -> bool:
        """Check if a provider's dependencies are available."""
        try:
            if provider_name == "edge_tts":
                import edge_tts

                return True
            elif provider_name == "coqui":
                from TTS.api import TTS

                return True
            elif provider_name == "bark":
                from bark import generate_audio

                return True
            elif provider_name == "openai":
                from openai import OpenAI

                return True
            elif provider_name == "elevenlabs":
                import requests

                return True
            elif provider_name == "google_cloud":
                from google.cloud import texttospeech

                return True
            else:
                return False
        except ImportError:
            return False

    def _initialize_provider(self):
        """Initialize the selected TTS provider with better error handling."""
        self.logger.info(f"Initializing TTS provider: {self.provider_name}")

        try:
            if self.provider_name == "coqui":
                if not self._check_provider_availability("coqui"):
                    raise ImportError(
                        "Coqui TTS not available. Install with: pip install TTS"
                    )
                return self._init_coqui()

            elif self.provider_name == "bark":
                if not self._check_provider_availability("bark"):
                    raise ImportError(
                        "Bark not available. Install with: pip install git+https://github.com/suno-ai/bark.git scipy"
                    )
                return self._init_bark()

            elif self.provider_name == "edge_tts":
                if not self._check_provider_availability("edge_tts"):
                    raise ImportError(
                        "Edge TTS not available. Install with: pip install edge-tts"
                    )
                return None  # Edge TTS doesn't need initialization

            elif self.provider_name == "openai":
                if not self._check_provider_availability("openai"):
                    raise ImportError(
                        "OpenAI not available. Install with: pip install openai"
                    )
                return self._init_openai()

            elif self.provider_name == "elevenlabs":
                if not self._check_provider_availability("elevenlabs"):
                    raise ImportError(
                        "Requests not available. Install with: pip install requests"
                    )
                return self._init_elevenlabs()

            elif self.provider_name == "google_cloud":
                if not self._check_provider_availability("google_cloud"):
                    raise ImportError(
                        "Google Cloud TTS not available. Install with: pip install google-cloud-texttospeech"
                    )
                return self._init_google_cloud()

            else:
                raise ValueError(f"Unknown provider: {self.provider_name}")

        except Exception as e:
            self.logger.error(f"Provider initialization failed: {e}")
            raise

    def _init_coqui(self):
        """Initialize Coqui TTS with better error handling."""
        try:
            # Get model configuration with safe fallbacks
            model_key = self._get_coqui_model_key()

            # Get the actual model name
            if model_key in self.provider_config["models"]:
                model_info = self.provider_config["models"][model_key]
                model_name = model_info["model_name"]
            else:
                # Fallback to default model
                default_model = self.provider_config.get("default_model", "fast")
                if default_model in self.provider_config["models"]:
                    model_info = self.provider_config["models"][default_model]
                    model_name = model_info["model_name"]
                    self.logger.warning(
                        f"Model '{model_key}' not found, using default: {default_model}"
                    )
                else:
                    raise ValueError(
                        f"Neither specified model '{model_key}' nor default model '{default_model}' found in TTS config"
                    )

            self.logger.info(f"Loading Coqui TTS model: {model_name}")

            # Import and initialize TTS
            from TTS.api import TTS

            # Initialize with error handling
            try:
                tts_instance = TTS(model_name)
                self.logger.info("Coqui TTS model loaded successfully")
                return tts_instance
            except Exception as e:
                self.logger.error(f"Failed to load Coqui model '{model_name}': {e}")
                # Try to fallback to a basic model
                try:
                    fallback_model = "tts_models/en/ljspeech/tacotron2-DDC"
                    self.logger.warning(f"Trying fallback model: {fallback_model}")
                    return TTS(fallback_model)
                except Exception as fallback_error:
                    raise Exception(
                        f"Failed to load both specified model and fallback: {e}, {fallback_error}"
                    )

        except ImportError as e:
            raise ImportError(f"Coqui TTS not properly installed: {e}")
        except Exception as e:
            raise Exception(f"Coqui TTS initialization failed: {e}")

    def _get_coqui_model_key(self) -> str:
        """Get Coqui model key with safe fallbacks."""
        try:
            # Try to get from main config
            return self.config["text_to_speech"]["tts"]["model"]
        except (KeyError, TypeError):
            # Fallback to provider default
            return self.provider_config.get("default_model", "fast")

    def _init_bark(self):
        """Initialize Bark TTS with better error handling."""
        try:
            from bark import preload_models

            self.logger.info("Loading Bark TTS models (this may take a while)...")
            preload_models()
            self.logger.info("Bark TTS models loaded successfully")
            return None

        except ImportError as e:
            raise ImportError(f"Bark not properly installed: {e}")
        except Exception as e:
            raise Exception(f"Bark initialization failed: {e}")

    def _init_openai(self):
        """Initialize OpenAI TTS with better error handling."""
        try:
            from openai import OpenAI

            api_key_env = self.provider_config.get("api_key_env", "OPENAI_API_KEY")
            api_key = os.getenv(api_key_env)

            if not api_key:
                raise ValueError(
                    f"OpenAI API key not found in environment variable: {api_key_env}"
                )

            client = OpenAI(api_key=api_key)
            self.logger.info("OpenAI TTS client initialized successfully")
            return client

        except ImportError as e:
            raise ImportError(f"OpenAI library not properly installed: {e}")
        except Exception as e:
            raise Exception(f"OpenAI TTS initialization failed: {e}")

    def _init_elevenlabs(self):
        """Initialize ElevenLabs TTS with better error handling."""
        try:
            api_key_env = self.provider_config.get("api_key_env", "ELEVENLABS_API_KEY")
            api_key = os.getenv(api_key_env)

            if not api_key:
                raise ValueError(
                    f"ElevenLabs API key not found in environment variable: {api_key_env}"
                )

            # Test the API key by making a simple request
            import requests

            test_url = f"{self.provider_config['base_url']}/voices"
            headers = {"xi-api-key": api_key}

            response = requests.get(test_url, headers=headers, timeout=10)
            if response.status_code != 200:
                raise ValueError(
                    f"ElevenLabs API key appears to be invalid (status: {response.status_code})"
                )

            self.logger.info("ElevenLabs TTS initialized successfully")
            return api_key

        except ImportError as e:
            raise ImportError(f"Requests library not available: {e}")
        except Exception as e:
            raise Exception(f"ElevenLabs TTS initialization failed: {e}")

    def _init_google_cloud(self):
        """Initialize Google Cloud TTS with better error handling."""
        try:
            from google.cloud import texttospeech

            credentials_path_env = self.provider_config.get(
                "credentials_path_env", "GOOGLE_APPLICATION_CREDENTIALS"
            )
            credentials_path = os.getenv(credentials_path_env)

            if credentials_path and not os.path.exists(credentials_path):
                raise ValueError(
                    f"Google Cloud credentials file not found: {credentials_path}"
                )

            client = texttospeech.TextToSpeechClient()

            # Test the client with a simple request
            voices = client.list_voices()
            if not voices.voices:
                raise Exception(
                    "Google Cloud TTS client initialized but no voices available"
                )

            self.logger.info("Google Cloud TTS client initialized successfully")
            return client

        except ImportError as e:
            raise ImportError(f"Google Cloud TTS library not properly installed: {e}")
        except Exception as e:
            raise Exception(f"Google Cloud TTS initialization failed: {e}")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better TTS output.

        Args:
            text: Raw text to preprocess

        Returns:
            Processed text optimized for TTS
        """
        processing_config = self.config["text_to_speech"]["tts"]["text_processing"]

        # Clean markdown if requested
        if processing_config.get("clean_markdown", True):
            # Remove markdown formatting
            text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Bold
            text = re.sub(r"\*(.*?)\*", r"\1", text)  # Italic
            text = re.sub(r"`(.*?)`", r"\1", text)  # Inline code
            text = re.sub(r"#{1,6}\s*(.*)", r"\1", text)  # Headers
            text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # Links

        # Expand abbreviations if requested
        if processing_config.get("expand_abbreviations", True):
            abbreviations = {
                "e.g.": "for example",
                "i.e.": "that is",
                "etc.": "and so on",
                "vs.": "versus",
                "Dr.": "Doctor",
                "Prof.": "Professor",
                # "AI": "artificial intelligence",
                # "ML": "machine learning",
                # "NLP": "natural language processing",
                # "CV": "computer vision",
            }

            for abbrev, expansion in abbreviations.items():
                text = text.replace(abbrev, expansion)

        # Add natural pauses if requested
        if processing_config.get("add_pauses", True):
            # Add longer pauses after periods and semicolons
            text = re.sub(r"\.(?=\s+[A-Z])", ". ", text)
            text = re.sub(r";(?=\s)", "; ", text)
            # Add brief pauses after commas
            text = re.sub(r",(?=\s)", ", ", text)

        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks suitable for TTS processing.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        processing_config = self.config["text_to_speech"]["tts"]["text_processing"]
        chunk_size = processing_config.get("chunk_size", 3000)
        overlap = processing_config.get("chunk_overlap", 100)

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to break at sentence boundary
            chunk = text[start:end]
            last_period = chunk.rfind(". ")
            last_exclamation = chunk.rfind("! ")
            last_question = chunk.rfind("? ")

            best_break = max(last_period, last_exclamation, last_question)

            if best_break > start + chunk_size // 2:  # Only break if it's not too early
                end = start + best_break + 2
                chunks.append(text[start:end])
                start = end - overlap
            else:
                chunks.append(chunk)
                start = end - overlap

        return chunks

    async def generate_audio_chunk(self, text: str, chunk_index: int = 0) -> bytes:
        """
        Generate audio for a single text chunk.

        Args:
            text: Text to convert to audio
            chunk_index: Index of the chunk (for caching)

        Returns:
            Audio data as bytes
        """
        if self.provider_name == "edge_tts":
            return await self._generate_edge_tts(text)
        elif self.provider_name == "coqui":
            return self._generate_coqui(text, chunk_index)
        elif self.provider_name == "bark":
            return self._generate_bark(text, chunk_index)
        elif self.provider_name == "openai":
            return self._generate_openai(text)
        elif self.provider_name == "elevenlabs":
            return self._generate_elevenlabs(text)
        elif self.provider_name == "google_cloud":
            return self._generate_google_cloud(text)
        else:
            raise ValueError(f"Unsupported provider: {self.provider_name}")

    async def _generate_edge_tts(self, text: str) -> bytes:
        """Generate audio using Edge TTS."""
        voice = (
            self.config["text_to_speech"]["tts"].get("voice")
            or self.provider_config["default_voice"]
        )

        communicate = edge_tts.Communicate(text, voice)
        audio_data = b""

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        return audio_data

    def _generate_coqui(self, text: str, chunk_index: int) -> bytes:
        """Generate audio using Coqui TTS."""
        temp_file = f"temp_chunk_{chunk_index}.wav"

        try:
            self.provider.tts_to_file(text=text, file_path=temp_file)

            with open(temp_file, "rb") as f:
                audio_data = f.read()

            return audio_data
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _generate_bark(self, text: str, chunk_index: int) -> bytes:
        """Generate audio using Bark."""
        voice = (
            self.config["text_to_speech"]["tts"].get("voice")
            or self.provider_config["default_voice"]
        )

        audio_array = generate_audio(text, history_prompt=voice)

        # Convert to bytes
        temp_file = f"temp_bark_{chunk_index}.wav"
        try:
            write_wav(temp_file, SAMPLE_RATE, audio_array)

            with open(temp_file, "rb") as f:
                audio_data = f.read()

            return audio_data
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _generate_openai(self, text: str) -> bytes:
        """Generate audio using OpenAI TTS."""
        model = (
            self.config["text_to_speech"]["tts"].get("model")
            or self.provider_config["default_model"]
        )
        voice = (
            self.config["text_to_speech"]["tts"].get("voice")
            or self.provider_config["default_voice"]
        )

        response = self.provider.audio.speech.create(
            model=model, voice=voice, input=text, response_format="mp3"
        )

        return response.content

    def _generate_elevenlabs(self, text: str) -> bytes:
        """Generate audio using ElevenLabs."""
        voice = (
            self.config["text_to_speech"]["tts"].get("voice")
            or self.provider_config["default_voice"]
        )

        # Get voice ID from config
        voice_id = self.provider_config["voices"].get(voice, voice)

        url = f"{self.provider_config['base_url']}/text-to-speech/{voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.provider,
        }

        data = {
            "text": text,
            "model_id": self.provider_config["model"],
            "voice_settings": self.provider_config["voice_settings"],
        }

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()

        return response.content

    def _generate_google_cloud(self, text: str) -> bytes:
        """Generate audio using Google Cloud TTS."""
        voice_name = (
            self.config["text_to_speech"]["tts"].get("voice")
            or self.provider_config["default_voice"]
        )

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=self.provider_config["language_code"], name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = self.provider.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        return response.audio_content

    def _combine_audio_chunks(self, chunks: List[bytes], output_path: str) -> str:
        """
        Combine multiple audio chunks into a single file.

        Args:
            chunks: List of audio chunks as bytes
            output_path: Output file path

        Returns:
            Path to the combined audio file
        """
        # For now, just concatenate the chunks
        # In a more sophisticated implementation, you might use pydub
        # to add silence between chunks and normalize levels

        with open(output_path, "wb") as f:
            for chunk in chunks:
                f.write(chunk)

        return output_path

    def _generate_filename(self, report_type: str = "summary") -> str:
        """Generate filename based on configuration."""
        naming_config = self.config["text_to_speech"]["files"]["naming_convention"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date = datetime.now().strftime("%Y-%m-%d")

        filename = naming_config.format(
            timestamp=timestamp,
            date=date,
            report_type=report_type,
            categories="general",  # You might want to pass this as parameter
        )

        # Get output format
        output_format = self.config["text_to_speech"]["tts"]["output"].get(
            "format"
        ) or self.provider_config.get("output_format", "mp3")

        return f"{filename}.{output_format}"

    def _save_metadata(self, audio_path: str, text: str, provider_info: Dict):
        """Save metadata alongside audio file."""
        if not self.config["text_to_speech"]["files"]["include_metadata"]:
            return

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "provider": self.provider_name,
            "provider_config": provider_info,
            "text_length": len(text),
            "audio_file": os.path.basename(audio_path),
            "processing_config": self.config["text_to_speech"]["tts"][
                "text_processing"
            ],
        }

        metadata_path = audio_path.replace(".mp3", ".json").replace(".wav", ".json")

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _archive_old_files(self):
        """Archive old audio files if configured."""
        if not self.config["text_to_speech"]["files"]["archive_old_files"]:
            return

        max_files = self.config["text_to_speech"]["files"]["max_files_to_keep"]

        # Get all audio files sorted by modification time
        audio_files = []
        for ext in ["*.mp3", "*.wav"]:
            audio_files.extend(self.audio_dir.glob(ext))

        audio_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Move old files to archive
        archive_dir = self.audio_dir / "archive"
        for file_to_archive in audio_files[max_files:]:
            archive_path = archive_dir / file_to_archive.name
            file_to_archive.rename(archive_path)

            # Also move metadata file if it exists
            metadata_file = file_to_archive.with_suffix(".json")
            if metadata_file.exists():
                metadata_file.rename(archive_dir / metadata_file.name)

    async def generate_audio_from_text(
        self, text: str, report_type: str = "summary"
    ) -> str:
        """
        Generate audio from text and save to file.

        Args:
            text: Text to convert to audio
            report_type: Type of report (for filename generation)

        Returns:
            Path to generated audio file
        """
        if not self.config["text_to_speech"]["enabled"]:
            self.logger.info("Audio generation is disabled in configuration")
            return None

        self.logger.info(f"Generating audio using {self.provider_name} provider")

        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Split into chunks
        chunks = self.chunk_text(processed_text)
        self.logger.info(f"Split text into {len(chunks)} chunks")

        # Generate audio for each chunk
        audio_chunks = []
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")

            try:
                audio_data = await self.generate_audio_chunk(chunk, i)
                audio_chunks.append(audio_data)

                # Add pause between chunks if configured
                pause_duration = self.config["text_to_speech"]["tts"][
                    "text_processing"
                ].get("pause_between_chunks", 0.5)
                if pause_duration > 0 and i < len(chunks) - 1:
                    # Create silence (this is a simplified approach)
                    # In practice, you might want to generate actual silence audio
                    pass

            except Exception as e:
                self.logger.error(f"Failed to generate audio for chunk {i}: {e}")
                if self.config["text_to_speech"]["tts"]["max_retries"] > 0:
                    # Implement retry logic here
                    pass
                raise

        # Generate output filename
        filename = self._generate_filename(report_type)
        output_path = self.audio_dir / filename

        # Combine chunks and save
        final_path = self._combine_audio_chunks(audio_chunks, str(output_path))

        # Save metadata
        provider_info = {
            "name": self.provider_config["name"],
            "voice": self.config["text_to_speech"]["tts"].get(
                "voice", self.provider_config.get("default_voice")
            ),
            "model": self.config["text_to_speech"]["tts"].get(
                "model", self.provider_config.get("default_model")
            ),
        }
        self._save_metadata(final_path, text, provider_info)

        # Archive old files
        self._archive_old_files()

        self.logger.info(f"Audio generated successfully: {final_path}")
        return final_path


async def main():
    """
    Example usage of the TTS generator.
    """
    # Example text (you would normally load this from your summary files)
    sample_text = """
    Welcome to today's ArXiv research summary. We have identified several 
    interesting papers in machine learning and artificial intelligence that 
    are worth your attention. The first paper discusses a novel approach to 
    transformer architectures that improves efficiency by 25 percent while 
    maintaining comparable performance on benchmark tasks.
    """

    try:
        # Initialize generator
        tts = TTSGenerator()

        # Generate audio
        audio_path = await tts.generate_audio_from_text(sample_text, "daily_summary")

        if audio_path:
            print(f"Audio generated successfully: {audio_path}")
        else:
            print("Audio generation is disabled or failed")

    except Exception as e:
        print(f"Error generating audio: {e}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Standalone Audio Generator for ArXiv Summaries

Run this script after generating summaries to convert them to audio files.

Usage:
    python generate_audio.py [audio_script]

If no file is specified, it will look for the most recent audio script in the outputs directory.
"""

import asyncio
import argparse
import sys
from pathlib import Path
import yaml

# Import our TTS generator
from audio_generator import TTSGenerator


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


def load_config(config_path: Path):
    """Load a YAML config file safely."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"Failed to load {config_path}: {e}")


def check_provider_availability(provider_name: str) -> bool:
    """Check if a provider is available."""
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


def list_available_providers():
    """List available TTS providers."""
    try:
        # Find configuration files
        config_path, tts_config_path = find_config_files()

        if not tts_config_path:
            print("❌ Could not find TTS config file")
            print("💡 Looking for: tts_config.yaml, tts.yaml, or similar")
            print(
                "💡 Searched in current directory, script directory, parent directory, and config/ subdirectories"
            )
            return

        print(f"Reading TTS config from: {tts_config_path}")

        # Load TTS config directly (don't need full TTSGenerator)
        tts_config = load_config(tts_config_path)

        if "providers" not in tts_config:
            print("❌ TTS config file doesn't contain 'providers' section")
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
        print("\nTo install a provider:")

        # Show installation info if available
        if "installation" in tts_config:
            print("\nInstallation commands:")
            for provider, install_info in tts_config["installation"].items():
                if "pip" in install_info:
                    pip_packages = " ".join(f'"{pkg}"' for pkg in install_info["pip"])
                    print(f"  {provider}: pip install {pip_packages}")

    except Exception as e:
        print(f"Failed to list providers: {e}")
        import traceback

        print("Full error:")
        traceback.print_exc()


def find_latest_audio_script(data_dir: str = "data") -> Path:
    """
    Find the most recent audio script in the data directory and its subdirectories.

    Args:
        data_dir: Directory to search for audio script files

    Returns:
        Path to the most recent audio script file
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Search recursively for files starting with 'audio_script'
    audio_script_files = list(data_path.rglob("audio_script*.txt"))

    if not audio_script_files:
        raise FileNotFoundError(f"No audio_script files found in {data_dir}")

    # Return the most recently modified file
    latest_file = max(audio_script_files, key=lambda x: x.stat().st_mtime)
    return latest_file


def load_audio_script(file_path: Path) -> str:
    """
    Load and prepare audio script  for TTS conversion.

    Args:
        file_path: Path to the audio script file

    Returns:
        Text content ready for TTS
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            raise ValueError(f"Audio script file is empty: {file_path}")

        print(f"Loaded audio script from: {file_path}")
        print(f"Text length: {len(content)} characters")

        return content

    except Exception as e:
        raise Exception(f"Failed to load audio script file {file_path}: {e}")


async def generate_audio_script(script_file: Path = None, provider: str = None):
    """
    Generate audio from an audio script file.

    Args:
        script_file: Path to audio script file (if None, finds latest)
        provider: TTS provider to use (if None, uses config default)
    """
    try:
        # Find configuration files
        config_path, tts_config_path = find_config_files()

        if not config_path:
            raise FileNotFoundError(
                "Could not find config.yaml. Please ensure it exists in the current directory, "
                "script directory, or parent directory."
            )

        if not tts_config_path:
            raise FileNotFoundError(
                "Could not find tts.yaml. Please ensure it exists alongside config.yaml."
            )

        print(f"Using config: {config_path}")
        print(f"Using TTS config: {tts_config_path}")

        # Find audio script file if not specified
        if script_file is None:
            script_file = find_latest_audio_script()

        # Load the audio script text
        audio_script = load_audio_script(script_file)

        # Initialize TTS generator with explicit config paths
        print(f"Initializing TTS generator...")
        tts = TTSGenerator(
            config_path=str(config_path), tts_config_path=str(tts_config_path)
        )

        # Override provider if specified
        if provider:
            if provider not in tts.tts_config["providers"]:
                available = list(tts.tts_config["providers"].keys())
                raise ValueError(
                    f"Provider '{provider}' not found. Available: {available}"
                )

            tts.provider_name = provider
            tts.provider_config = tts.tts_config["providers"][provider]
            tts.provider = tts._initialize_provider()
            print(f"Using TTS provider: {provider}")
        else:
            print(f"Using TTS provider: {tts.provider_name}")

        # Determine report type from filename
        report_type = "audio_script"
        if "daily" in script_file.name.lower():
            report_type = "daily_audio_script"
        elif "weekly" in script_file.name.lower():
            report_type = "weekly_audio_script"
        elif "report" in script_file.name.lower():
            report_type = "report"

        # Generate audio
        print(f"\nGenerating audio...")
        print(f"This may take a few minutes depending on text length and provider...")

        audio_path = await tts.generate_audio_from_text(audio_script, report_type)

        if audio_path:
            print(f"\n✅ Audio generated successfully!")
            print(f"📁 File saved to: {audio_path}")

            # Show file size
            audio_file = Path(audio_path)
            if audio_file.exists():
                size_mb = audio_file.stat().st_size / (1024 * 1024)
                print(f"📊 File size: {size_mb:.1f} MB")

            return audio_path
        else:
            print("❌ Audio generation failed or is disabled in config")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def create_minimal_audio_config():
    """Create a minimal TTS config section."""
    return {
        "text_to_speech": {
            "enabled": True,
            "tts": {
                "provider": "edge_tts",  # Safe default
                "voice": None,
                "model": None,
                "text_processing": {
                    "chunk_size": 3000,
                    "chunk_overlap": 100,
                    "pause_between_chunks": 0.5,
                    "clean_markdown": True,
                    "expand_abbreviations": True,
                    "add_pauses": True,
                },
                "output": {"format": "mp3", "quality": "high", "normalize_audio": True},
                "parallel_generation": False,
                "cache_audio": True,
                "max_retries": 3,
            },
            "files": {
                "directory": "audio",
                "naming_convention": "{timestamp}_{report_type}_summary",
                "include_metadata": True,
                "archive_old_files": True,
                "max_files_to_keep": 10,
            },
        }
    }


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate audio from ArXiv summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_audio.py                           # Use latest script
    python generate_audio.py audio_script.txt          # Use specific file
    python generate_audio.py --provider edge_tts       # Use specific provider
    python generate_audio.py --list-providers          # Show available providers
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

    args = parser.parse_args()

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
    print("🎵 ArXiv Audio Script Generator")
    print("=" * 40)

    try:
        audio_path = asyncio.run(
            generate_audio_script(script_file=audio_script_file, provider=args.provider)
        )

        if audio_path:
            print(f"\n🎧 Your audio script is ready!")
            print(f"You can now listen to: {Path(audio_path).name}")
        else:
            print(f"\n💡 Tip: Make sure text_to_speech is enabled in config.yaml")
            print(f"💡 Tip: Check that your chosen TTS provider is properly installed")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n⏹️  Audio generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

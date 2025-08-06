#!/usr/bin/env python3
"""
ArXiv Audio Script Creation with LLMs

Converts synthesis reports into natural-sounding scripts for TTS audio narration.
Supports conversion from both Stage 2 (final reports) and Stage 1 (structured content).

Dependencies:
    pip install openai anthropic google-generativeai pyyaml

Usage:
    python create_audio_script.py

Configuration:
    Edit the 'audio' section in config/config.yaml
"""

import json
import yaml
import os
import sys
import glob
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_utils import LLMManager
from audio_utils import AudioScriptGenerator, validate_audio_config


class AudioScriptWorkflow:
    """
    Main class for the audio script creation workflow.

    Orchestrates:
    1. Loading synthesis results with both Stage 1 and Stage 2 content
    2. Converting content to audio-ready scripts using LLM
    3. Saving audio scripts and metadata
    """

    def __init__(self, config: Dict):
        """
        Initialize the audio script workflow.

        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.audio_config = config.get("audio", {})

        # Initialize LLM manager
        self.llm_manager = LLMManager()

        # Get model configuration
        self.model_alias = self.audio_config.get("model_alias")
        if not self.model_alias:
            raise ValueError("model_alias must be specified in audio configuration")

        # Initialize audio script generator
        self.generator = AudioScriptGenerator(
            config, self.llm_manager, self.model_alias
        )

        # File paths and data
        self.input_files = {}
        self.synthesis_result = {}
        self.audio_result = {}

    def find_most_recent_synthesis_file(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the most recent synthesis results file.

        Returns:
            Tuple of (filepath, format_type) or (None, None) if not found
        """
        data_dir = self.config.get("synthesis", {}).get(
            "synthesis_storage_dir", "./data"
        )

        # Look for synthesis results files
        synthesis_pattern = os.path.join(data_dir, "synthesis_results_*.json")
        synthesis_files = glob.glob(synthesis_pattern)

        if not synthesis_files:
            # Fallback to default data directory
            data_dir = self.config.get("output", {}).get("base_dir", "./data")
            synthesis_pattern = os.path.join(data_dir, "synthesis_results_*.json")
            synthesis_files = glob.glob(synthesis_pattern)

        if not synthesis_files:
            return None, None

        # Get most recent by modification time
        most_recent = max(synthesis_files, key=os.path.getmtime)
        return most_recent, "json"

    def auto_detect_input_files(self) -> Dict[str, str]:
        """
        Auto-detect the most recent synthesis results files.

        Returns:
            Dictionary mapping format -> filepath
        """
        # Check if input file is explicitly specified
        input_file = self.audio_config.get("input_file")

        if input_file:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Specified input file not found: {input_file}")

            # Determine format from extension
            if input_file.endswith(".json"):
                return {"json": input_file}
            else:
                raise ValueError(f"Unsupported file format: {input_file}")

        # Auto-detect most recent file
        filepath, format_type = self.find_most_recent_synthesis_file()

        if not filepath:
            return {}

        return {format_type: filepath}

    def load_synthesis_results(self, filepath: str) -> Dict:
        """
        Load synthesis results from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Synthesis results dictionary
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                results = json.load(f)

            if not isinstance(results, dict):
                raise ValueError(
                    "JSON file must contain a synthesis results dictionary"
                )

            return results

        except Exception as e:
            raise RuntimeError(f"Failed to load synthesis results from {filepath}: {e}")

    def setup_input_files(self):
        """
        Setup input file paths based on configuration.
        """
        # Auto-detect or use specified input files
        self.input_files = self.auto_detect_input_files()

        if not self.input_files:
            raise FileNotFoundError(
                "No synthesis results found. Run synthesize_papers.py first, "
                "or specify input_file in config."
            )

        print(f"Input file detected:")
        for format_type, path in self.input_files.items():
            print(f"  {format_type.upper()}: {path}")

    def load_synthesis_data(self) -> Dict:
        """
        Load synthesis results from input files.

        Returns:
            Synthesis results dictionary
        """
        # Load from the detected file
        format_type, filepath = list(self.input_files.items())[0]
        synthesis_result = self.load_synthesis_results(filepath)

        # Check what content is available
        has_stage2 = synthesis_result.get("synthesis_report") is not None
        has_stage1 = synthesis_result.get("stage1_content") is not None
        synthesis_success = synthesis_result.get("synthesis_success", False)

        print(f"Loaded synthesis results:")
        print(f"  Synthesis successful: {synthesis_success}")
        print(f"  Stage 2 content available: {has_stage2}")
        print(f"  Stage 1 content available: {has_stage1}")

        if not has_stage2 and not has_stage1:
            raise RuntimeError(
                "No Stage 1 or Stage 2 content found in synthesis results"
            )

        if has_stage2:
            stage2_length = len(synthesis_result["synthesis_report"])
            print(f"  Stage 2 length: {stage2_length:,} characters")

        if has_stage1:
            stage1_length = len(synthesis_result["stage1_content"])
            print(f"  Stage 1 length: {stage1_length:,} characters")

        return synthesis_result

    def create_audio_script(self, synthesis_result: Dict) -> Dict:
        """
        Generate audio script from synthesis results.

        Args:
            synthesis_result: Synthesis results dictionary

        Returns:
            Audio script result dictionary
        """
        print(f"\nSTARTING AUDIO SCRIPT GENERATION...")

        # Generate audio script
        audio_result = self.generator.generate_audio_script(synthesis_result)

        # Store results
        self.audio_result = audio_result

        # Show results
        if audio_result.get("conversion_success"):
            word_count = audio_result.get("word_count", 0)
            duration = audio_result.get("estimated_duration_minutes", 0)
            source = audio_result.get("source_used", "unknown")
            conversion_time = audio_result.get("conversion_time", 0)

            print(f"✅ Audio script generated successfully")
            print(f"   Source: {source} content")
            print(f"   Length: {word_count:,} words (~{duration:.1f} minutes)")
            print(f"   Generation time: {conversion_time:.1f} seconds")
        else:
            error = audio_result.get("error", "Unknown error")
            print(f"❌ Audio script generation failed: {error}")

        return audio_result

    def save_audio_script(self) -> Tuple[str, Optional[str]]:
        """
        Save audio script to files.

        Returns:
            Tuple of (results_path, script_path) where script_path may be None if conversion failed
        """
        output_dir = self.audio_config.get("audio_storage_dir", "./data")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save comprehensive results
        results_path = os.path.join(output_dir, f"audio_results_{timestamp}.json")

        try:
            os.makedirs(os.path.dirname(results_path), exist_ok=True)

            # Create comprehensive results
            comprehensive_results = {
                "audio_generation_timestamp": datetime.now().isoformat(),
                **self.audio_result,
                "input_file": (
                    list(self.input_files.values())[0] if self.input_files else None
                ),
            }

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
            print(f"Saved audio results: {results_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save audio results: {e}")

        # Save standalone script if generation was successful
        script_path = None
        if self.audio_result.get("conversion_success") and self.audio_result.get(
            "audio_script"
        ):
            script_path = os.path.join(output_dir, f"audio_script_{timestamp}.txt")
            try:
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(f"# Research Digest Audio Script\n\n")
                    f.write(
                        f"Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}\n"
                    )
                    f.write(
                        f"Model: {self.audio_result.get('converted_by', 'unknown')}\n"
                    )
                    f.write(
                        f"Source: {self.audio_result.get('source_used', 'unknown')} content\n"
                    )
                    f.write(
                        f"Length: {self.audio_result.get('word_count', 0):,} words\n"
                    )
                    f.write(
                        f"Estimated duration: {self.audio_result.get('estimated_duration_minutes', 0):.1f} minutes\n"
                    )
                    f.write(f"\n" + "=" * 50 + "\n\n")
                    f.write(self.audio_result["audio_script"])
                print(f"Saved audio script: {script_path}")
            except Exception as e:
                print(f"Warning: Could not save audio script: {e}")

        return results_path, script_path

    def print_final_summary(self):
        """Print a final summary of the audio script generation."""
        print(f"\n" + "=" * 50)
        print("AUDIO SCRIPT GENERATION COMPLETE")
        print("=" * 50)

        if self.audio_result.get("conversion_success"):
            print(f"✅ Audio script generated successfully")

            # Show key metrics
            word_count = self.audio_result.get("word_count", 0)
            duration = self.audio_result.get("estimated_duration_minutes", 0)
            source = self.audio_result.get("source_used", "unknown")

            print(f"   Content source: {source}")
            print(f"   Script length: {word_count:,} words")
            print(f"   Estimated duration: {duration:.1f} minutes")

            # Show target comparison
            target_duration = self.generator.target_duration_minutes
            if abs(duration - target_duration) / target_duration > 0.3:
                if duration > target_duration:
                    print(f"   ⚠️  Script is longer than target ({target_duration} min)")
                else:
                    print(
                        f"   ⚠️  Script is shorter than target ({target_duration} min)"
                    )

            print(f"\nNext steps:")
            print(f"1. Review the generated script for accuracy and flow")
            print(f"2. Use TTS software to convert script to audio")
            print(f"3. Consider editing for optimal spoken delivery")

        else:
            print(f"❌ Audio script generation failed")
            error = self.audio_result.get("error", "Unknown error")
            print(f"   Error: {error}")

            print(f"\nTroubleshooting:")
            print(f"1. Check that synthesis results contain Stage 1 or Stage 2 content")
            print(f"2. Verify audio configuration and model settings")
            print(f"3. Review synthesis quality and retry if needed")

    def run(self):
        """
        Run the complete audio script creation workflow.
        """
        try:
            print("Audio Script Creation Workflow")
            print("==============================")

            # Setup and load synthesis data
            self.setup_input_files()
            synthesis_result = self.load_synthesis_data()

            # Generate audio script
            audio_result = self.create_audio_script(synthesis_result)

            # Save results
            results_path, script_path = self.save_audio_script()

            # Print final summary
            self.print_final_summary()

            # Show file locations
            print(f"\nFiles saved:")
            print(f"  Results: {results_path}")
            if script_path:
                print(f"  Script: {script_path}")

        except Exception as e:
            print(f"Error during audio script workflow: {e}")
            import traceback

            traceback.print_exc()
            raise


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")


def validate_audio_workflow_config(config: Dict) -> List[str]:
    """
    Validate the complete audio workflow configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Validate audio configuration
    audio_errors = validate_audio_config(config)
    errors.extend([f"Audio: {err}" for err in audio_errors])

    return errors


def test_audio_setup():
    """Test the audio script setup and configuration."""
    print("=== TESTING AUDIO SCRIPT WORKFLOW SETUP ===")

    # Test 1: Load configuration
    print("\n1. Testing configuration loading...")
    try:
        config = load_config()
        print("✅ Configuration loaded successfully")

        # Check for audio section
        if "audio" not in config:
            print("❌ No 'audio' section found in config.yaml")
            print("   Add the audio section to your configuration")
            return 1
        else:
            print("✅ Audio section found in configuration")

    except FileNotFoundError:
        print("❌ config/config.yaml not found")
        return 1
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1

    # Test 2: Validate audio configuration
    print("\n2. Testing audio configuration validation...")
    try:
        validation_errors = validate_audio_workflow_config(config)
        if validation_errors:
            print("❌ Audio configuration validation errors:")
            for error in validation_errors:
                print(f"   - {error}")
            return 1
        else:
            print("✅ Audio configuration validation passed")
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return 1

    # Test 3: Check LLM model configuration
    print("\n3. Testing LLM model configuration...")
    try:
        audio_config = config.get("audio", {})
        model_alias = audio_config.get("model_alias")

        if not model_alias:
            print("❌ No model_alias specified in audio configuration")
            return 1

        llm_manager = LLMManager()
        model_config = llm_manager.get_model_config(model_alias)
        print(f"✅ Model '{model_alias}' configuration loaded")
        print(f"   Provider: {model_config.get('provider')}")
        print(f"   Model: {model_config.get('model')}")

    except Exception as e:
        print(f"❌ Error with LLM model configuration: {e}")
        return 1

    # Test 4: Check for synthesis input files
    print("\n4. Checking for synthesis results input files...")
    try:
        # Check synthesis storage directory first
        synthesis_dir = config.get("synthesis", {}).get(
            "synthesis_storage_dir", "./data"
        )
        synthesis_pattern = os.path.join(synthesis_dir, "synthesis_results_*.json")
        synthesis_files = glob.glob(synthesis_pattern)

        # Fallback to main data directory
        if not synthesis_files:
            data_dir = config.get("output", {}).get("base_dir", "./data")
            synthesis_pattern = os.path.join(data_dir, "synthesis_results_*.json")
            synthesis_files = glob.glob(synthesis_pattern)

        if synthesis_files:
            latest_file = max(synthesis_files, key=os.path.getmtime)
            print(f"✅ Found synthesis results files for audio conversion")
            print(f"   Latest: {latest_file}")

            # Quick check of file content
            try:
                with open(latest_file, "r", encoding="utf-8") as f:
                    results = json.load(f)

                has_stage2 = results.get("synthesis_report") is not None
                has_stage1 = results.get("stage1_content") is not None
                synthesis_success = results.get("synthesis_success", False)

                print(f"   Synthesis successful: {synthesis_success}")
                print(f"   Stage 2 content: {'✓' if has_stage2 else '✗'}")
                print(f"   Stage 1 content: {'✓' if has_stage1 else '✗'}")

                if not has_stage2 and not has_stage1:
                    print("⚠️  No usable content found - run synthesize_papers.py first")
                elif not synthesis_success:
                    print("⚠️  Synthesis was not successful - may have limited content")

            except Exception as e:
                print(f"⚠️  Could not analyze file content: {e}")

        else:
            print("⚠️  No synthesis_results_*.json files found")
            print("   Complete the workflow through synthesize_papers.py first")

    except Exception as e:
        print(f"❌ Error checking input files: {e}")
        return 1

    print(f"\n=== AUDIO SCRIPT SETUP TEST COMPLETE ===")

    # Check if we're ready for audio script generation
    if synthesis_files:
        print(f"\n✅ Ready to create audio scripts!")
        print(f"   Execute: python src/create_audio_script.py")
    else:
        print(f"\n❌ Complete the synthesis workflow first, then create audio scripts")

    return 0


def main():
    """Main function to run the audio script creation workflow."""
    print("ArXiv Audio Script Creation with LLMs")
    print("=====================================")

    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--test", "-t"]:
            return test_audio_setup()
        elif sys.argv[1] in ["--help", "-h"]:
            print("Usage:")
            print("  python create_audio_script.py          # Create audio script")
            print("  python create_audio_script.py --test   # Test audio setup")
            print("  python create_audio_script.py --help   # Show this help")
            return 0

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return 1

    # Validate audio configuration
    validation_errors = validate_audio_workflow_config(config)
    if validation_errors:
        print("Configuration validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        return 1

    # Display streamlined configuration
    audio_config = config.get("audio", {})

    print(
        f"Configuration: {audio_config.get('model_alias', 'not specified')} | "
        f"Target: {audio_config.get('target_duration_minutes', 11)} min | "
        f"Prefer: {'Stage 2' if audio_config.get('prefer_stage2', True) else 'Stage 1'}"
    )

    # Create and run audio script workflow
    try:
        workflow = AudioScriptWorkflow(config)
        workflow.run()
        return 0

    except Exception as e:
        print(f"Audio script workflow failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

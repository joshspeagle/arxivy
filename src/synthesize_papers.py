#!/usr/bin/env python3
"""
ArXiv Paper Synthesis with LLMs

Synthesizes selected papers into comprehensive reports using configurable LLM models.
Integrates paper selection with theme-based synthesis for final research summaries.

Dependencies:
    pip install openai anthropic google-generativeai pyyaml numpy

Usage:
    python synthesize_papers.py

Configuration:
    Edit the 'synthesis' section in config/config.yaml
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
from selection_utils import SynthesisSelector, validate_selection_config
from synthesis_utils import PaperSynthesizer, validate_synthesis_config


class OutputFormatter:
    """Helper class to format output consistently and avoid repetition."""

    def __init__(self, config: Dict):
        self.score_field = (
            config.get("synthesis", {})
            .get("selection", {})
            .get("score_field", "rescored_llm_score")
        )

    def format_paper_summary(self, paper: Dict, index: Optional[int] = None) -> str:
        """Format a single paper for display."""
        title = paper.get("title", "Untitled")
        score = paper.get(self.score_field, "N/A")

        # Truncate title if too long
        display_title = title[:70] + "..." if len(title) > 70 else title

        prefix = f"{index}. " if index is not None else ""
        return f"{prefix}[{score}] {display_title}"

    def format_paper_list(self, papers: List[Dict], max_display: int = 5) -> List[str]:
        """Format a list of papers for display."""
        lines = []
        for i, paper in enumerate(papers[:max_display], 1):
            lines.append("  " + self.format_paper_summary(paper, i))

        if len(papers) > max_display:
            lines.append(f"  ... and {len(papers) - max_display} more")

        return lines

    def format_score_stats(self, papers: List[Dict]) -> str:
        """Format score statistics for a list of papers."""
        if not papers:
            return "No papers with valid scores"

        scores = [
            p[self.score_field] for p in papers if p.get(self.score_field) is not None
        ]
        if not scores:
            return f"No papers with valid {self.score_field} scores"

        return f"Score range: {min(scores):.1f} - {max(scores):.1f}, avg: {sum(scores)/len(scores):.1f}"


class SynthesisWorkflow:
    """
    Main class for the paper synthesis workflow.

    Orchestrates:
    1. Loading rescored papers with full summaries
    2. Paper selection using sophisticated algorithms
    3. LLM-based synthesis into comprehensive reports
    4. Results compilation and storage
    """

    def __init__(self, config: Dict):
        """
        Initialize the synthesis workflow.

        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.synthesis_config = config.get("synthesis", {})

        # Initialize LLM manager
        self.llm_manager = LLMManager()

        # Get model configuration
        self.model_alias = self.synthesis_config.get("model_alias")
        if not self.model_alias:
            raise ValueError("model_alias must be specified in synthesis configuration")

        # Initialize components
        self.selector = SynthesisSelector(config)
        self.synthesizer = PaperSynthesizer(config, self.llm_manager, self.model_alias)
        self.formatter = OutputFormatter(config)

        # File paths and data
        self.input_files = {}
        self.papers = []
        self.selected_papers = []
        self.synthesis_result = {}
        self.selection_stats = {}

    def find_most_recent_rescored_file(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the most recent rescored papers file.

        Returns:
            Tuple of (filepath, format_type) or (None, None) if not found
        """
        data_dir = self.config.get("output", {}).get("base_dir", "./data")

        # Look for rescored papers files
        rescored_pattern = os.path.join(data_dir, "rescored_papers_*.json")
        rescored_files = glob.glob(rescored_pattern)

        if not rescored_files:
            return None, None

        # Get most recent by modification time
        most_recent = max(rescored_files, key=os.path.getmtime)
        return most_recent, "json"

    def auto_detect_input_files(self) -> Dict[str, str]:
        """
        Auto-detect the most recent rescored papers files.

        Returns:
            Dictionary mapping format -> filepath
        """
        # Check if input file is explicitly specified
        input_file = self.synthesis_config.get("input_file")

        if input_file:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Specified input file not found: {input_file}")

            # Determine format from extension
            if input_file.endswith(".json"):
                return {"json": input_file}
            else:
                raise ValueError(f"Unsupported file format: {input_file}")

        # Auto-detect most recent file
        filepath, format_type = self.find_most_recent_rescored_file()

        if not filepath:
            return {}

        return {format_type: filepath}

    def load_papers_from_json(self, filepath: str) -> List[Dict]:
        """
        Load papers from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            List of paper dictionaries
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                papers = json.load(f)

            if not isinstance(papers, list):
                raise ValueError("JSON file must contain a list of papers")

            return papers

        except Exception as e:
            raise RuntimeError(f"Failed to load papers from {filepath}: {e}")

    def setup_input_files(self):
        """
        Setup input file paths based on configuration.
        """
        # Auto-detect or use specified input files
        self.input_files = self.auto_detect_input_files()

        if not self.input_files:
            raise FileNotFoundError(
                "No input files found. Run rescore_papers.py first, "
                "or specify input_file in config."
            )

        print(f"Input file detected:")
        for format_type, path in self.input_files.items():
            print(f"  {format_type.upper()}: {path}")

    def load_papers(self) -> List[Dict]:
        """
        Load papers from input files.

        Returns:
            List of paper dictionaries with valid scores
        """
        # Load from the detected file
        format_type, filepath = list(self.input_files.items())[0]
        papers = self.load_papers_from_json(filepath)

        # Filter to papers with valid scores
        score_field = self.formatter.score_field
        valid_papers = [p for p in papers if p.get(score_field) is not None]

        print(
            f"Loaded {len(papers)} papers, {len(valid_papers)} with valid {score_field} scores"
        )

        if not valid_papers:
            raise RuntimeError(f"No papers with valid {score_field} scores found")

        # Show score distribution only once
        print(f"  {self.formatter.format_score_stats(valid_papers)}")

        return valid_papers

    def select_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Select papers for synthesis using the configured selection algorithm.

        Args:
            papers: List of papers to select from

        Returns:
            List of selected papers
        """
        print(f"\nSELECTING PAPERS FOR SYNTHESIS...")

        # Run selection algorithm
        selection_result = self.selector.select_papers(papers)
        selected_papers = selection_result["selected_papers"]

        # Store selection statistics
        self.selection_stats = selection_result["statistics"]

        print(f"Selected {len(selected_papers)} papers for synthesis")
        if selected_papers:
            print(f"  {self.formatter.format_score_stats(selected_papers)}")

        return selected_papers

    def synthesize_papers(self, papers: List[Dict]) -> Dict:
        """
        Synthesize papers into a comprehensive report.

        Args:
            papers: List of selected papers

        Returns:
            Synthesis result dictionary
        """
        if not papers:
            raise RuntimeError("No papers provided for synthesis")

        print(f"\nSYNTHESIZING {len(papers)} PAPERS...")

        # Synthesize papers
        synthesis_result = self.synthesizer.synthesize_papers(papers)

        # Store synthesis results
        self.synthesis_result = synthesis_result

        # Brief synthesis status
        if synthesis_result.get("synthesis_success"):
            synthesis_time = synthesis_result.get("synthesis_time", 0)
            report_length = len(synthesis_result.get("synthesis_report", ""))
            print(
                f"✅ Synthesis completed in {synthesis_time:.1f}s ({report_length:,} characters)"
            )
        else:
            error = synthesis_result.get("error", "Unknown error")
            print(f"❌ Synthesis failed: {error}")

        return synthesis_result

    def create_synthesis_metadata(self) -> Dict:
        """Create minimal metadata for synthesis results."""
        return {
            "synthesis_timestamp": datetime.now().isoformat(),
            "model_used": self.model_alias,
            "total_papers_loaded": len(self.papers),
            "papers_selected": len(self.selected_papers),
            "synthesis_success": self.synthesis_result.get("synthesis_success", False),
            "synthesis_time": self.synthesis_result.get("synthesis_time", 0),
        }

    def save_synthesis_results(self) -> Tuple[str, Optional[str]]:
        """
        Save synthesis results to files.

        Returns:
            Tuple of (results_path, report_path) where report_path may be None if synthesis failed
        """
        output_dir = self.config.get("output", {}).get("base_dir", "./data")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create streamlined results (avoid duplication)
        synthesis_metadata = self.create_synthesis_metadata()

        # Only include essential data to avoid bloat
        streamlined_results = {
            **synthesis_metadata,
            "synthesis_report": self.synthesis_result.get("synthesis_report"),
            "synthesis_confidence": self.synthesis_result.get("synthesis_confidence"),
            "papers_metadata": [
                {
                    "id": p.get("id"),
                    "title": p.get("title"),
                    "score": p.get(self.formatter.score_field),
                    # Include only essential fields, not full paper content
                }
                for p in self.selected_papers
            ],
            "selection_summary": {
                "method": self.selection_stats.get("config_used", {}).get("percentile"),
                "score_threshold": self.selection_stats.get("config_used", {}).get(
                    "score_threshold"
                ),
                "final_count": self.selection_stats.get("final_count"),
            },
            "config_used": {
                "model_alias": self.model_alias,
                "score_field": self.formatter.score_field,
            },
        }

        # Save streamlined results
        results_path = os.path.join(output_dir, f"synthesis_results_{timestamp}.json")

        try:
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(streamlined_results, f, indent=2, ensure_ascii=False)
            print(f"Saved synthesis results: {results_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save synthesis results: {e}")

        # Save standalone report only if synthesis was successful
        report_path = None
        if self.synthesis_result.get("synthesis_success") and self.synthesis_result.get(
            "synthesis_report"
        ):
            report_path = os.path.join(output_dir, f"synthesis_report_{timestamp}.md")
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    # Create a clean report without redundant metadata
                    f.write(f"# Research Synthesis Report\n\n")
                    f.write(
                        f"*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n\n"
                    )
                    f.write(f"**Papers Analyzed:** {len(self.selected_papers)} | ")
                    f.write(f"**Model:** {self.model_alias}\n\n")
                    f.write("---\n\n")
                    f.write(self.synthesis_result["synthesis_report"])
                print(f"Saved synthesis report: {report_path}")
            except Exception as e:
                print(f"Warning: Could not save standalone report: {e}")

        return results_path, report_path

    def print_final_summary(self):
        """Print a concise final summary without repetition."""
        print(f"\n" + "=" * 50)
        print("SYNTHESIS WORKFLOW COMPLETE")
        print("=" * 50)

        # Single summary with all key information
        print(
            f"Papers processed: {len(self.papers)} → {len(self.selected_papers)} selected"
        )

        if self.synthesis_result.get("synthesis_success"):
            print(f"✅ Synthesis successful")

            # Show only top 3 papers to avoid repetition
            if self.selected_papers:
                print(f"\nTop papers synthesized:")
                lines = self.formatter.format_paper_list(
                    self.selected_papers, max_display=3
                )
                for line in lines:
                    print(line)
        else:
            print(
                f"❌ Synthesis failed: {self.synthesis_result.get('error', 'Unknown error')}"
            )

    def run(self):
        """
        Run the complete synthesis workflow with streamlined output.
        """
        try:
            print("Paper Synthesis Workflow")
            print("========================")

            # Setup and load papers
            self.setup_input_files()
            self.papers = self.load_papers()

            # Select papers for synthesis
            self.selected_papers = self.select_papers(self.papers)

            if not self.selected_papers:
                print("No papers selected for synthesis.")
                return

            # Synthesize papers
            synthesis_result = self.synthesize_papers(self.selected_papers)

            # Save results
            results_path, report_path = self.save_synthesis_results()

            # Print concise final summary
            self.print_final_summary()

            # Show next steps only if relevant
            if self.synthesis_result.get("synthesis_success"):
                print(f"\nFiles saved:")
                print(f"  Results: {results_path}")
                if report_path:
                    print(f"  Report:  {report_path}")
            else:
                print(f"\nTroubleshooting:")
                print(f"  1. Check synthesis configuration")
                print(f"  2. Verify selected papers have required metadata")

        except Exception as e:
            print(f"Error during synthesis workflow: {e}")
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


def validate_synthesis_workflow_config(config: Dict) -> List[str]:
    """
    Validate the complete synthesis workflow configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Validate synthesis configuration
    synthesis_errors = validate_synthesis_config(config)
    errors.extend([f"Synthesis: {err}" for err in synthesis_errors])

    # Validate selection configuration
    selection_errors = validate_selection_config(config, "synthesis")
    errors.extend([f"Selection: {err}" for err in selection_errors])

    return errors


def test_synthesis_setup():
    """Test the synthesis setup and configuration."""
    print("=== TESTING SYNTHESIS WORKFLOW SETUP ===")

    # Test 1: Load configuration
    print("\n1. Testing configuration loading...")
    try:
        config = load_config()
        print("✅ Configuration loaded successfully")

        # Check for synthesis section
        if "synthesis" not in config:
            print("❌ No 'synthesis' section found in config.yaml")
            print("   Add the synthesis section to your configuration")
            return 1
        else:
            print("✅ Synthesis section found in configuration")

    except FileNotFoundError:
        print("❌ config/config.yaml not found")
        return 1
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1

    # Test 2: Validate synthesis configuration
    print("\n2. Testing synthesis configuration validation...")
    try:
        validation_errors = validate_synthesis_workflow_config(config)
        if validation_errors:
            print("❌ Synthesis configuration validation errors:")
            for error in validation_errors:
                print(f"   - {error}")
            return 1
        else:
            print("✅ Synthesis configuration validation passed")
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return 1

    # Test 3: Check LLM model configuration
    print("\n3. Testing LLM model configuration...")
    try:
        synthesis_config = config.get("synthesis", {})
        model_alias = synthesis_config.get("model_alias")

        if not model_alias:
            print("❌ No model_alias specified in synthesis configuration")
            return 1

        llm_manager = LLMManager()
        model_config = llm_manager.get_model_config(model_alias)
        print(f"✅ Model '{model_alias}' configuration loaded")
        print(f"   Provider: {model_config.get('provider')}")
        print(f"   Model: {model_config.get('model')}")

    except Exception as e:
        print(f"❌ Error with LLM model configuration: {e}")
        return 1

    # Test 4: Check for input files
    print("\n4. Checking for rescored papers input files...")
    try:
        data_dir = config.get("output", {}).get("base_dir", "./data")
        rescored_pattern = os.path.join(data_dir, "rescored_papers_*.json")
        rescored_files = glob.glob(rescored_pattern)

        if rescored_files:
            latest_file = max(rescored_files, key=os.path.getmtime)
            print(f"✅ Found rescored papers files for synthesis")
            print(f"   Latest: {latest_file}")

            # Quick check of file content
            try:
                with open(latest_file, "r", encoding="utf-8") as f:
                    papers = json.load(f)

                score_field = synthesis_config.get("selection", {}).get(
                    "score_field", "rescored_llm_score"
                )
                valid_papers = [p for p in papers if p.get(score_field) is not None]
                print(
                    f"   Papers with {score_field}: {len(valid_papers)}/{len(papers)}"
                )

                if len(valid_papers) == 0:
                    print(
                        "⚠️  No papers have rescored scores - run rescore_papers.py first"
                    )

            except Exception as e:
                print(f"⚠️  Could not analyze file content: {e}")

        else:
            print("⚠️  No rescored_papers_*.json files found")
            print("   Complete the workflow first through rescore_papers.py")

    except Exception as e:
        print(f"❌ Error checking input files: {e}")
        return 1

    print(f"\n=== SYNTHESIS SETUP TEST COMPLETE ===")

    # Check if we're ready for synthesis
    if rescored_files:
        print(f"\n✅ Ready to run synthesis!")
        print(f"   Execute: python src/synthesize_papers.py")
    else:
        print(f"\n❌ Complete the workflow first, then run synthesis")

    return 0


def main():
    """Main function to run the synthesis workflow."""
    print("ArXiv Paper Synthesis with LLMs")
    print("===============================")

    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--test", "-t"]:
            return test_synthesis_setup()
        elif sys.argv[1] in ["--help", "-h"]:
            print("Usage:")
            print("  python synthesize_papers.py          # Run synthesis workflow")
            print("  python synthesize_papers.py --test   # Test synthesis setup")
            print("  python synthesize_papers.py --help   # Show this help")
            return 0

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return 1

    # Validate synthesis configuration
    validation_errors = validate_synthesis_workflow_config(config)
    if validation_errors:
        print("Configuration validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        return 1

    # Display streamlined configuration
    synthesis_config = config.get("synthesis", {})
    selection_config = synthesis_config.get("selection", {})

    print(
        f"Configuration: {synthesis_config.get('model_alias', 'not specified')} | "
        f"Score field: {selection_config.get('score_field', 'rescored_llm_score')} | "
        f"Selection: top {selection_config.get('percentile', 'not set')}% with score ≥ {selection_config.get('score_threshold', 'not set')}"
    )

    # Create and run synthesis workflow
    try:
        workflow = SynthesisWorkflow(config)
        workflow.run()
        return 0

    except Exception as e:
        print(f"Synthesis workflow failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

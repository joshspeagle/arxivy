#!/usr/bin/env python3
"""
ArXiv Paper Scoring with LLMs

Scores papers from the arxiv fetcher using configurable LLM models.

Dependencies:
    pip install openai anthropic google-generativeai pyyaml pandas

Usage:
    python score_papers.py

Configuration:
    Edit the 'scoring' section in config/config.yaml
"""

import json
import yaml
import os
import sys
from pathlib import Path
from typing import List, Dict
import glob

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_utils import LLMManager
from score_utils import ScoringEngine, validate_scoring_config


class PaperScorer:
    """
    Main class for scoring papers using LLMs with file format preservation.
    """

    def __init__(self, config: Dict):
        """
        Initialize the paper scorer.

        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.scoring_config = config.get("scoring", {})

        # Initialize LLM manager
        self.llm_manager = LLMManager()

        # Get model configuration
        self.model_alias = self.scoring_config.get("model_alias")
        if not self.model_alias:
            raise ValueError("model_alias must be specified in scoring configuration")

        # Initialize scoring engine
        self.scoring_engine = ScoringEngine(config, self.llm_manager, self.model_alias)

        # File paths
        self.input_files = {}
        self.output_files = {}
        self.papers = []

    def auto_detect_input_files(self) -> Dict[str, str]:
        """
        Auto-detect the most recent arxiv papers files from the data directory.

        Returns:
            Dictionary mapping format -> filepath
        """
        output_dir = self.config.get("output", {}).get("base_dir", "./data")

        # Look for arxiv_papers_*.json files
        json_pattern = os.path.join(output_dir, "arxiv_papers_*.json")

        json_files = glob.glob(json_pattern)

        # Filter out already scored files
        json_files = [f for f in json_files if "_scored" not in f]

        # Get the most recent files
        detected_files = {}

        if json_files:
            # Sort by modification time, get most recent
            latest_json = max(json_files, key=os.path.getmtime)
            detected_files["json"] = latest_json

        return detected_files

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

    def save_papers_to_json(self, papers: List[Dict], filepath: str):
        """
        Save papers to JSON file.

        Args:
            papers: List of paper dictionaries
            filepath: Output file path
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(papers)} scored papers to {filepath}")

        except Exception as e:
            raise RuntimeError(f"Failed to save papers to {filepath}: {e}")

    def generate_output_filepath(self, input_filepath: str) -> str:
        """
        Generate output filepath by adding "_scored" before the extension.

        Args:
            input_filepath: Original input file path

        Returns:
            Output file path
        """
        path = Path(input_filepath)
        stem = path.stem
        suffix = path.suffix

        # Add _scored to the filename
        new_filename = f"{stem}_scored{suffix}"
        return str(path.parent / new_filename)

    def setup_input_output_files(self):
        """
        Setup input and output file paths based on configuration.
        """
        # Check if input file is explicitly specified
        input_file = self.scoring_config.get("input_file")

        if input_file:
            # Use specified input file
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Specified input file not found: {input_file}")

            # Determine format from extension
            if input_file.endswith(".json"):
                self.input_files["json"] = input_file
            else:
                raise ValueError(f"Unsupported file format: {input_file}")

        else:
            # Auto-detect input files
            self.input_files = self.auto_detect_input_files()

            if not self.input_files:
                raise FileNotFoundError(
                    "No input files found. Run fetch_papers.py first or specify input_file in config."
                )

        # Generate output file paths
        for format_type, input_path in self.input_files.items():
            self.output_files[format_type] = self.generate_output_filepath(
                input_path, format_type
            )

        print(f"Input files detected:")
        for format_type, path in self.input_files.items():
            print(f"  {format_type.upper()}: {path}")

        print(f"Output files will be:")
        for format_type, path in self.output_files.items():
            print(f"  {format_type.upper()}: {path}")
        print()

    def load_papers(self) -> List[Dict]:
        """
        Load papers from input files (prefer JSON if available).

        Returns:
            List of paper dictionaries
        """
        # Prefer JSON format if available
        if "json" in self.input_files:
            papers = self.load_papers_from_json(self.input_files["json"])
            print(f"Loaded {len(papers)} papers from JSON file")
        else:
            raise RuntimeError("No valid input files found")

        return papers

    def apply_paper_limit(self, papers: List[Dict]) -> List[Dict]:
        """
        Apply max_papers limit to the list of papers.

        Args:
            papers: Full list of papers

        Returns:
            Limited list of papers
        """
        max_papers = self.scoring_config.get("max_papers")

        if max_papers is None or max_papers <= 0:
            return papers

        if len(papers) > max_papers:
            limited_papers = papers[:max_papers]
            print(f"Limited to first {max_papers} papers (out of {len(papers)} total)")
            print(
                f"Note: Paper order follows arXiv category order from fetch configuration"
            )
            return limited_papers

        return papers

    def score_all_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Score all papers using the LLM with batching support.

        Args:
            papers: List of papers to score

        Returns:
            List of scored papers
        """
        # Use the new batch-aware scoring method
        scored_papers = self.scoring_engine.score_papers_batch(papers)
        return scored_papers

    def print_scoring_summary(self, papers: List[Dict]):
        """
        Print a summary of the scoring results.
        Updated to show batch information.
        """
        print(f"\n=== SCORING SUMMARY ===")

        # Count successful scores
        successful_scores = [p for p in papers if p.get("llm_score") is not None]
        failed_scores = [p for p in papers if p.get("llm_score") is None]

        print(f"Total papers processed: {len(papers)}")
        print(f"Successfully scored: {len(successful_scores)}")
        print(f"Failed to score: {len(failed_scores)}")

        # Show batch configuration
        batch_size = self.scoring_engine.batch_size
        if batch_size > 1:
            num_batches = (
                len(papers) + batch_size - 1
            ) // batch_size  # Ceiling division
            print(
                f"Batch configuration: {batch_size} papers per batch ({num_batches} batches total)"
            )
        else:
            print(f"Batch configuration: Single paper mode")

        if successful_scores:
            scores = [p["llm_score"] for p in successful_scores]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)

            print(f"\nScore statistics:")
            print(f"  Average: {avg_score:.2f}")
            print(f"  Range: {min_score} - {max_score}")

            # Show top 3 scoring papers
            sorted_papers = sorted(
                successful_scores, key=lambda x: x["llm_score"], reverse=True
            )
            print(f"\nTop 3 scoring papers:")
            for i, paper in enumerate(sorted_papers[:3], 1):
                title = paper.get("title", "Untitled")
                score = paper["llm_score"]
                print(
                    f"  {i}. [{score}] {title[:80]}{'...' if len(title) > 80 else ''}"
                )

        if failed_scores:
            print(f"\nFailed papers:")
            for paper in failed_scores[:3]:  # Show first 3 failures
                title = paper.get("title", "Untitled")
                print(f"  - {title[:80]}{'...' if len(title) > 80 else ''}")
            if len(failed_scores) > 3:
                print(f"  ... and {len(failed_scores) - 3} more")

    def save_scored_papers(self, papers: List[Dict]):
        """
        Save scored papers to all configured output formats.

        Args:
            papers: List of scored papers
        """
        print(f"\nSaving scored papers...")

        for format_type, output_path in self.output_files.items():
            if format_type == "json":
                self.save_papers_to_json(papers, output_path)

    def run(self):
        """
        Run the complete paper scoring workflow.
        """
        try:
            # Setup input/output files
            self.setup_input_output_files()

            # Load papers
            papers = self.load_papers()

            # Apply paper limit
            papers = self.apply_paper_limit(papers)

            if not papers:
                print("No papers to score after applying limits.")
                return

            # Score papers
            scored_papers = self.score_all_papers(papers)

            # Save results
            self.save_scored_papers(scored_papers)

            # Print summary
            self.print_scoring_summary(scored_papers)

        except Exception as e:
            print(f"Error during scoring workflow: {e}")
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


def main():
    """Main function to run the paper scoring workflow."""
    print("ArXiv Paper Scoring with LLMs")
    print("=============================")

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return 1

    # Validate scoring configuration
    validation_errors = validate_scoring_config(config)
    if validation_errors:
        print("Configuration validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        return 1

    # Display configuration
    scoring_config = config.get("scoring", {})
    print(f"Configuration:")
    print(f"  Model: {scoring_config.get('model_alias', 'not specified')}")
    print(f"  Max papers: {scoring_config.get('max_papers', 'unlimited')}")
    print(f"  Retry attempts: {scoring_config.get('retry_attempts', 2)}")
    print(f"  Metadata fields: {', '.join(scoring_config.get('include_metadata', []))}")
    print()

    # Create and run scorer
    try:
        scorer = PaperScorer(config)
        scorer.run()

        print("\nScoring workflow completed successfully!")
        return 0

    except Exception as e:
        print(f"Scoring workflow failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

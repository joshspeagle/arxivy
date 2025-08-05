#!/usr/bin/env python3
"""
ArXiv Paper Re-scoring with LLMs

Re-scores papers using full summaries and content from the summarization workflow.
Uses configurable LLM models with enhanced metadata including summaries.

Dependencies:
    pip install openai anthropic google-generativeai pyyaml pandas

Usage:
    python rescore_papers.py

Configuration:
    Edit the 'rescoring' section in config/config.yaml
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
from score_utils import ScoringEngine, validate_rescoring_config


class PaperRescorer:
    """
    Main class for re-scoring papers using LLMs with enhanced content from summaries.
    """

    def __init__(self, config: Dict):
        """
        Initialize the paper rescorer.

        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.rescoring_config = config.get("rescoring", {})
        self.scoring_config = config.get("scoring", {})  # For inheritance

        # Initialize LLM manager
        self.llm_manager = LLMManager()

        # Get model configuration
        self.model_alias = self.rescoring_config.get("model_alias")
        if not self.model_alias:
            raise ValueError("model_alias must be specified in rescoring configuration")

        # Resolve inherited prompts
        self._resolve_inherited_prompts()

        # Initialize scoring engine with resolved rescoring config
        self.scoring_engine = ScoringEngine(
            {"scoring": self.resolved_rescoring_config},
            self.llm_manager,
            self.model_alias,
        )

        # File paths
        self.input_files = {}
        self.output_files = {}
        self.papers = []

    def _resolve_inherited_prompts(self):
        """
        Resolve prompts that should be inherited from scoring config when null.
        """
        self.resolved_rescoring_config = self.rescoring_config.copy()

        # List of prompts that can be inherited
        inheritable_prompts = [
            "research_context_prompt",
            "scoring_strategy_prompt",
            "score_calculation_prompt",
        ]

        for prompt_name in inheritable_prompts:
            if self.resolved_rescoring_config.get(prompt_name) is None:
                # Inherit from scoring config
                inherited_value = self.scoring_config.get(prompt_name)
                if inherited_value:
                    self.resolved_rescoring_config[prompt_name] = inherited_value
                    print(f"ℹ️  Inheriting {prompt_name} from scoring configuration")
                else:
                    print(
                        f"⚠️  Warning: {prompt_name} is null and no value found in scoring config"
                    )

    def auto_detect_input_files(self) -> Dict[str, str]:
        """
        Auto-detect the most recent summarization results files from the data directory.

        Returns:
            Dictionary mapping format -> filepath
        """
        output_dir = self.config.get("output", {}).get("base_dir", "./data")

        # Look for summarization_results_*.json files
        json_pattern = os.path.join(output_dir, "summarization_results_*.json")
        json_files = glob.glob(json_pattern)

        # Filter out already rescored files (though unlikely)
        json_files = [f for f in json_files if "_rescored" not in f]

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

            print(f"Saved {len(papers)} rescored papers to {filepath}")

        except Exception as e:
            raise RuntimeError(f"Failed to save papers to {filepath}: {e}")

    def generate_output_filepath(self, input_filepath: str) -> str:
        """
        Generate output filepath by adding "_rescored" before the extension.

        Args:
            input_filepath: Original input file path

        Returns:
            Output file path
        """
        path = Path(input_filepath)
        stem = path.stem
        suffix = path.suffix

        # Replace summarization_results with rescored_papers
        if "summarization_results" in stem:
            new_stem = stem.replace("summarization_results", "rescored_papers")
        else:
            new_stem = f"{stem}_rescored"

        new_filename = f"{new_stem}{suffix}"
        return str(path.parent / new_filename)

    def setup_input_output_files(self):
        """
        Setup input and output file paths based on configuration.
        """
        # Check if input file is explicitly specified
        input_file = self.rescoring_config.get("input_file")

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
                    "No input files found. Run summarize_papers.py first or specify input_file in config."
                )

        # Generate output file paths
        for format_type, input_path in self.input_files.items():
            self.output_files[format_type] = self.generate_output_filepath(input_path)

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

    def filter_summarized_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Filter papers to only include those with successful summaries.

        Args:
            papers: Full list of papers

        Returns:
            List of papers with summaries
        """
        summarized_papers = [
            p
            for p in papers
            if p.get("llm_summary") is not None
            and len(p.get("llm_summary", "").strip()) > 0
        ]

        print(
            f"Papers with summaries available: {len(summarized_papers)} out of {len(papers)}"
        )

        if len(summarized_papers) < len(papers):
            skipped = len(papers) - len(summarized_papers)
            print(f"Skipping {skipped} papers without summaries")

        return summarized_papers

    def apply_paper_limit(self, papers: List[Dict]) -> List[Dict]:
        """
        Apply max_papers limit to the list of papers.

        Args:
            papers: Full list of papers

        Returns:
            Limited list of papers
        """
        max_papers = self.rescoring_config.get("max_papers")

        if max_papers is None or max_papers <= 0:
            return papers

        if len(papers) > max_papers:
            # Sort by original score if available, otherwise use first N
            if all(p.get("llm_score") is not None for p in papers):
                limited_papers = sorted(
                    papers, key=lambda x: x["llm_score"], reverse=True
                )[:max_papers]
                print(
                    f"Limited to top {max_papers} papers by original score (out of {len(papers)} total)"
                )
            else:
                limited_papers = papers[:max_papers]
                print(
                    f"Limited to first {max_papers} papers (out of {len(papers)} total)"
                )

            return limited_papers

        return papers

    def rescore_all_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Re-score all papers using the LLM with enhanced metadata including summaries.

        Args:
            papers: List of papers to rescore

        Returns:
            List of rescored papers
        """
        # Rename llm_score to original_llm_score and remove llm_explanation to avoid conflicts
        enhanced_papers = []
        for paper in papers:
            enhanced_paper = paper.copy()

            # Preserve original scoring information
            if "llm_score" in enhanced_paper:
                enhanced_paper["original_llm_score"] = enhanced_paper["llm_score"]
                del enhanced_paper["llm_score"]
            if "llm_explanation" in enhanced_paper:
                enhanced_paper["original_llm_explanation"] = enhanced_paper[
                    "llm_explanation"
                ]
                del enhanced_paper["llm_explanation"]
            if "scored_by" in enhanced_paper:
                enhanced_paper["originally_scored_by"] = enhanced_paper["scored_by"]
                del enhanced_paper["scored_by"]
            if "scored_at" in enhanced_paper:
                enhanced_paper["originally_scored_at"] = enhanced_paper["scored_at"]
                del enhanced_paper["scored_at"]

            enhanced_papers.append(enhanced_paper)

        # Use the existing scoring engine with enhanced metadata
        rescored_papers = self.scoring_engine.score_papers_batch(enhanced_papers)

        # Rename the new scores to be clear about rescoring
        final_papers = []
        for paper in rescored_papers:
            final_paper = paper.copy()

            # Rename new scoring information to be explicit about rescoring
            if "llm_score" in final_paper:
                final_paper["rescored_llm_score"] = final_paper["llm_score"]
                del final_paper["llm_score"]
            if "llm_explanation" in final_paper:
                final_paper["rescored_llm_explanation"] = final_paper["llm_explanation"]
                del final_paper["llm_explanation"]
            if "scored_by" in final_paper:
                final_paper["rescored_by"] = final_paper["scored_by"]
                del final_paper["scored_by"]
            if "scored_at" in final_paper:
                final_paper["rescored_at"] = final_paper["scored_at"]
                del final_paper["scored_at"]

            final_papers.append(final_paper)

        return final_papers

    def print_rescoring_summary(self, papers: List[Dict]):
        """
        Print a summary of the rescoring results with comparison to original scores.
        """
        print(f"\n=== RESCORING SUMMARY ===")

        # Count successful rescores
        successful_rescores = [
            p for p in papers if p.get("rescored_llm_score") is not None
        ]
        failed_rescores = [p for p in papers if p.get("rescored_llm_score") is None]

        print(f"Total papers processed: {len(papers)}")
        print(f"Successfully rescored: {len(successful_rescores)}")
        print(f"Failed to rescore: {len(failed_rescores)}")

        # Show batch configuration
        batch_size = self.scoring_engine.batch_size
        if batch_size > 1:
            num_batches = (len(papers) + batch_size - 1) // batch_size
            print(
                f"Batch configuration: {batch_size} papers per batch ({num_batches} batches total)"
            )
        else:
            print(f"Batch configuration: Single paper mode")

        if successful_rescores:
            # Score comparison analysis
            original_scores = [
                p.get("original_llm_score")
                for p in successful_rescores
                if p.get("original_llm_score") is not None
            ]
            new_scores = [p["rescored_llm_score"] for p in successful_rescores]

            if original_scores and len(original_scores) == len(new_scores):
                avg_original = sum(original_scores) / len(original_scores)
                avg_new = sum(new_scores) / len(new_scores)

                print(f"\nScore comparison:")
                print(f"  Original average: {avg_original:.2f}")
                print(f"  Rescored average: {avg_new:.2f}")
                print(f"  Average change: {avg_new - avg_original:+.2f}")

                # Count papers that improved, stayed same, or decreased
                improved = sum(
                    1
                    for i, (orig, new) in enumerate(zip(original_scores, new_scores))
                    if new > orig + 0.1
                )
                decreased = sum(
                    1
                    for i, (orig, new) in enumerate(zip(original_scores, new_scores))
                    if new < orig - 0.1
                )
                unchanged = len(original_scores) - improved - decreased

                print(f"  Papers improved: {improved}")
                print(f"  Papers unchanged: {unchanged}")
                print(f"  Papers decreased: {decreased}")

            # Show top rescored papers
            sorted_papers = sorted(
                successful_rescores, key=lambda x: x["rescored_llm_score"], reverse=True
            )
            print(f"\nTop 3 rescored papers:")
            for i, paper in enumerate(sorted_papers[:3], 1):
                title = paper.get("title", "Untitled")
                new_score = paper["rescored_llm_score"]
                orig_score = paper.get("original_llm_score", "N/A")
                change = (
                    f"({new_score - orig_score:+.1f})" if orig_score != "N/A" else ""
                )
                print(
                    f"  {i}. [{new_score}] {change} {title[:70]}{'...' if len(title) > 70 else ''}"
                )

        if failed_rescores:
            print(f"\nFailed papers:")
            for paper in failed_rescores[:3]:  # Show first 3 failures
                title = paper.get("title", "Untitled")
                print(f"  - {title[:80]}{'...' if len(title) > 80 else ''}")
            if len(failed_rescores) > 3:
                print(f"  ... and {len(failed_rescores) - 3} more")

    def save_rescored_papers(self, papers: List[Dict]):
        """
        Save rescored papers to all configured output formats.

        Args:
            papers: List of rescored papers
        """
        print(f"\nSaving rescored papers...")

        for format_type, output_path in self.output_files.items():
            if format_type == "json":
                self.save_papers_to_json(papers, output_path)

    def run(self):
        """
        Run the complete paper rescoring workflow.
        """
        try:
            # Setup input/output files
            self.setup_input_output_files()

            # Load papers
            papers = self.load_papers()

            # Filter to papers with summaries
            papers = self.filter_summarized_papers(papers)

            if not papers:
                print("No papers with summaries found for rescoring.")
                return

            # Apply paper limit
            papers = self.apply_paper_limit(papers)

            if not papers:
                print("No papers to rescore after applying limits.")
                return

            # Rescore papers
            rescored_papers = self.rescore_all_papers(papers)

            # Save results
            self.save_rescored_papers(rescored_papers)

            # Print summary
            self.print_rescoring_summary(rescored_papers)

        except Exception as e:
            print(f"Error during rescoring workflow: {e}")
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


def main():
    """Main function to run the paper rescoring workflow."""
    print("ArXiv Paper Re-scoring with LLMs")
    print("=================================")

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return 1

    # Validate rescoring configuration
    validation_errors = validate_rescoring_config(config)
    if validation_errors:
        print("Configuration validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        return 1

    # Display configuration
    rescoring_config = config.get("rescoring", {})
    print(f"Configuration:")
    print(f"  Model: {rescoring_config.get('model_alias', 'not specified')}")
    print(f"  Max papers: {rescoring_config.get('max_papers', 'unlimited')}")
    print(f"  Retry attempts: {rescoring_config.get('retry_attempts', 2)}")
    print(f"  Batch size: {rescoring_config.get('batch_size', 'single paper mode')}")
    print(
        f"  Metadata fields: {', '.join(rescoring_config.get('include_metadata', []))}"
    )

    # Show prompt inheritance info
    inheritable_prompts = [
        "research_context_prompt",
        "scoring_strategy_prompt",
        "score_calculation_prompt",
    ]
    inherited = [p for p in inheritable_prompts if rescoring_config.get(p) is None]
    if inherited:
        print(f"  Inheriting from scoring config: {', '.join(inherited)}")
    print()

    # Create and run rescorer
    try:
        rescorer = PaperRescorer(config)
        rescorer.run()

        print("\nRescoring workflow completed successfully!")
        return 0

    except Exception as e:
        print(f"Rescoring workflow failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
ArXiv Paper Summarization with LLMs

Summarizes papers from the PDF processing workflow using configurable LLM models.
Supports both extracted text and direct PDF processing with chunking for large documents.

Dependencies:
    pip install openai anthropic google-generativeai pyyaml PyPDF2 pymupdf pdfplumber

Usage:
    python summarize_papers.py

Configuration:
    Edit the 'summarization' section in config/config.yaml
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
from summarize_utils import PaperSummarizer, validate_summarization_config


class PaperSummarizationWorkflow:
    """
    Main class for the paper summarization workflow.
    """

    def __init__(self, config: Dict):
        """
        Initialize the summarization workflow.

        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.summ_config = config.get("summarization", {})

        # Initialize LLM manager
        self.llm_manager = LLMManager()

        # Get model configuration
        self.model_alias = self.summ_config.get("model_alias")
        if not self.model_alias:
            raise ValueError(
                "model_alias must be specified in summarization configuration"
            )

        # Initialize summarization engine
        self.summarizer = PaperSummarizer(config, self.llm_manager, self.model_alias)

        # File paths
        self.input_files = {}
        self.output_files = {}
        self.papers = []

    def find_most_recent_papers_file(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the most recent papers file from extraction or selection results.

        Returns:
            Tuple of (filepath, format_type) or (None, None) if not found
        """
        data_dir = self.config.get("output", {}).get("base_dir", "./data")

        # Look for extraction results first (preferred)
        extraction_pattern = os.path.join(data_dir, "extraction_results_*.json")
        extraction_files = glob.glob(extraction_pattern)

        # Look for selected papers as fallback
        selected_pattern = os.path.join(data_dir, "selected_papers_*.json")
        selected_files = glob.glob(selected_pattern)

        all_files = [(f, "extraction") for f in extraction_files] + [
            (f, "selected") for f in selected_files
        ]

        if not all_files:
            return None, None

        # Get most recent by modification time
        most_recent = max(all_files, key=lambda x: os.path.getmtime(x[0]))
        return most_recent

    def auto_detect_input_files(self) -> Dict[str, str]:
        """
        Auto-detect the most recent papers files.

        Returns:
            Dictionary mapping format -> filepath
        """
        # Check if input file is explicitly specified
        input_file = self.summ_config.get("input_file")

        if input_file:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Specified input file not found: {input_file}")

            # Determine format from content/filename
            if "extraction_results" in input_file:
                return {"extraction": input_file}
            elif "selected_papers" in input_file:
                return {"selected": input_file}
            else:
                return {"json": input_file}

        # Auto-detect most recent file
        filepath, format_type = self.find_most_recent_papers_file()

        if not filepath:
            return {}

        return {format_type: filepath}

    def load_papers_from_json(self, filepath: str, format_type: str) -> List[Dict]:
        """
        Load papers from JSON file.

        Args:
            filepath: Path to JSON file
            format_type: Type of file (extraction, selected, etc.)

        Returns:
            List of paper dictionaries
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle different file formats
            if format_type == "extraction" and isinstance(data, list):
                # extraction_results_*.json contains a list of papers
                papers = data
            elif format_type == "selected" and isinstance(data, list):
                # selected_papers_*.json contains a list of papers
                papers = data
            elif isinstance(data, list):
                # Generic list of papers
                papers = data
            else:
                raise ValueError("JSON file must contain a list of papers")

            return papers

        except Exception as e:
            raise RuntimeError(f"Failed to load papers from {filepath}: {e}")

    def generate_output_filepath(self, input_filepath: str, format_type: str) -> str:
        """
        Generate output filepath for summarization results.

        Args:
            input_filepath: Original input file path
            format_type: Type of input file

        Returns:
            Output file path
        """
        output_dir = self.config.get("output", {}).get("base_dir", "./data")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create descriptive filename
        filename = f"summarization_results_{timestamp}.json"
        return os.path.join(output_dir, filename)

    def setup_input_output_files(self):
        """
        Setup input and output file paths based on configuration.
        """
        # Auto-detect or use specified input files
        self.input_files = self.auto_detect_input_files()

        if not self.input_files:
            raise FileNotFoundError(
                "No input files found. Run extract_papers.py or download_papers.py first, "
                "or specify input_file in config."
            )

        # Generate output file path
        input_path = list(self.input_files.values())[0]
        format_type = list(self.input_files.keys())[0]
        self.output_path = self.generate_output_filepath(input_path, format_type)

        print(f"Input file detected:")
        for format_type, path in self.input_files.items():
            print(f"  {format_type.upper()}: {path}")

        print(f"Output file will be:")
        print(f"  JSON: {self.output_path}")
        print()

    def load_papers(self) -> List[Dict]:
        """
        Load papers from input files.

        Returns:
            List of paper dictionaries
        """
        # Load from the detected file
        format_type, filepath = list(self.input_files.items())[0]
        papers = self.load_papers_from_json(filepath, format_type)

        print(f"Loaded {len(papers)} papers from {format_type} file")

        # Analyze available content types
        extracted_text_count = 0
        pdf_only_count = 0
        no_content_count = 0

        for paper in papers:
            has_extracted_text = (
                paper.get("extraction_success")
                and paper.get("extraction_output_path")
                and os.path.exists(paper.get("extraction_output_path", ""))
            )
            has_pdf = (
                paper.get("pdf_downloaded")
                and paper.get("pdf_file_path")
                and os.path.exists(paper.get("pdf_file_path", ""))
            )

            if has_extracted_text:
                extracted_text_count += 1
            elif has_pdf:
                pdf_only_count += 1
            else:
                no_content_count += 1

        print(f"Content analysis:")
        print(f"  Papers with extracted text: {extracted_text_count}")
        print(f"  Papers with PDF only: {pdf_only_count}")
        print(f"  Papers with no content: {no_content_count}")

        # Filter summarizable papers
        summarizable_papers = []
        for paper in papers:
            has_extracted_text = (
                paper.get("extraction_success")
                and paper.get("extraction_output_path")
                and os.path.exists(paper.get("extraction_output_path", ""))
            )
            has_pdf = (
                paper.get("pdf_downloaded")
                and paper.get("pdf_file_path")
                and os.path.exists(paper.get("pdf_file_path", ""))
            )

            if has_extracted_text or has_pdf:
                summarizable_papers.append(paper)

        print(f"Total papers available for summarization: {len(summarizable_papers)}")

        if not summarizable_papers:
            raise RuntimeError("No papers with available text or PDF content found")

        return summarizable_papers

    def apply_paper_limit(self, papers: List[Dict]) -> List[Dict]:
        """
        Apply max_papers limit to the list of papers.

        Args:
            papers: Full list of papers

        Returns:
            Limited list of papers
        """
        max_papers = self.summ_config.get("max_papers")

        if max_papers is None or max_papers <= 0:
            return papers

        if len(papers) > max_papers:
            # Sort by score if available, otherwise use first N
            if all(p.get("llm_score") is not None for p in papers):
                limited_papers = sorted(
                    papers, key=lambda x: x["llm_score"], reverse=True
                )[:max_papers]
                print(
                    f"Limited to top {max_papers} papers by score (out of {len(papers)} total)"
                )
            else:
                limited_papers = papers[:max_papers]
                print(
                    f"Limited to first {max_papers} papers (out of {len(papers)} total)"
                )

            return limited_papers

        return papers

    def summarize_all_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Summarize all papers using the LLM.

        Args:
            papers: List of papers to summarize

        Returns:
            List of summarized papers
        """
        print(f"Starting to summarize {len(papers)} papers using {self.model_alias}")
        print(
            f"Processing preference: {'extracted text > PDF direct' if self.summarizer.prefer_extracted_text else 'PDF direct > extracted text'}"
        )
        print()

        summarized_papers = []

        # Setup progress tracking
        self.summarizer.start_time = datetime.now()

        for i, paper in enumerate(papers):
            # Print progress
            title = paper.get("title", "Untitled")
            print(
                f"[{i+1:3d}/{len(papers)}] Summarizing: {title[:70]}{'...' if len(title) > 70 else ''}"
            )

            # Summarize paper
            summarized_paper = self.summarizer.summarize_paper(paper)
            summarized_papers.append(summarized_paper)

            # Print result
            if summarized_paper.get("llm_summary"):
                confidence = summarized_paper.get("summary_confidence", 0)
                source = summarized_paper.get("summary_source", "unknown")
                chunked = (
                    "chunked"
                    if summarized_paper.get("summary_chunked", False)
                    else "single"
                )

                # Show processing details
                if source == "extracted":
                    mode_info = f"text-{chunked}"
                elif source == "pdf":
                    mode_info = "pdf-direct"
                else:
                    mode_info = source

                print(f"    ✅ Success ({mode_info}, confidence: {confidence:.2f})")
            else:
                error = summarized_paper.get("summary_error", "Unknown error")
                print(f"    ❌ Failed: {error}")

        return summarized_papers

    def save_summarized_papers(self, papers: List[Dict]):
        """
        Save summarized papers to JSON file.

        Args:
            papers: List of summarized papers
        """
        print(f"\nSaving summarized papers...")

        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(papers)} summarized papers to {self.output_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to save papers to {self.output_path}: {e}")

    def print_summarization_summary(self, papers: List[Dict]):
        """
        Print a summary of the summarization results.
        """
        print(f"\n=== SUMMARIZATION SUMMARY ===")

        # Count successful summaries
        successful_summaries = [p for p in papers if p.get("llm_summary") is not None]
        failed_summaries = [p for p in papers if p.get("llm_summary") is None]

        print(f"Total papers processed: {len(papers)}")
        print(f"Successfully summarized: {len(successful_summaries)}")
        print(f"Failed to summarize: {len(failed_summaries)}")

        # Statistics from summarizer
        print(f"Total retries: {self.summarizer.total_retries}")
        print(f"Papers requiring chunking: {self.summarizer.total_chunked}")

        if successful_summaries:
            # Analyze confidence scores
            confidences = [p.get("summary_confidence", 0) for p in successful_summaries]
            avg_confidence = sum(confidences) / len(confidences)
            print(f"Average confidence: {avg_confidence:.2f}")

            # Source breakdown
            sources = {}
            for paper in successful_summaries:
                source = paper.get("summary_source", "unknown")
                sources[source] = sources.get(source, 0) + 1

            print(f"Sources used:")
            for source, count in sources.items():
                print(f"  {source}: {count} papers")

            # Show sample summaries
            print(f"\nSample summaries:")
            for i, paper in enumerate(successful_summaries[:2], 1):
                title = paper.get("title", "Untitled")
                summary = paper.get("llm_summary", "")
                summary_preview = (
                    summary[:200].replace("\n", " ") + "..."
                    if len(summary) > 200
                    else summary
                )
                print(f"\n{i}. {title[:60]}{'...' if len(title) > 60 else ''}")
                print(f"   {summary_preview}")

        if failed_summaries:
            print(f"\nFailed papers:")
            for paper in failed_summaries[:3]:  # Show first 3 failures
                title = paper.get("title", "Untitled")
                error = paper.get("summary_error", "Unknown error")
                print(f"  - {title[:60]}{'...' if len(title) > 60 else ''}")
                print(f"    Error: {error}")
            if len(failed_summaries) > 3:
                print(f"  ... and {len(failed_summaries) - 3} more")

    def run(self):
        """
        Run the complete paper summarization workflow.
        """
        try:
            # Setup input/output files
            self.setup_input_output_files()

            # Load papers
            papers = self.load_papers()

            # Apply paper limit
            papers = self.apply_paper_limit(papers)

            if not papers:
                print("No papers to summarize after applying limits.")
                return

            # Summarize papers
            summarized_papers = self.summarize_all_papers(papers)

            # Save results
            self.save_summarized_papers(summarized_papers)

            # Print summary
            self.print_summarization_summary(summarized_papers)

        except Exception as e:
            print(f"Error during summarization workflow: {e}")
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
    """Main function to run the paper summarization workflow."""
    print("ArXiv Paper Summarization with LLMs")
    print("===================================")

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return 1

    # Validate summarization configuration
    validation_errors = validate_summarization_config(config)
    if validation_errors:
        print("Configuration validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        return 1

    # Display configuration
    summ_config = config.get("summarization", {})
    print(f"Configuration:")
    print(f"  Model: {summ_config.get('model_alias', 'not specified')}")
    print(f"  Max papers: {summ_config.get('max_papers', 'unlimited')}")
    print(f"  Retry attempts: {summ_config.get('retry_attempts', 2)}")
    print(f"  Prefer extracted text: {summ_config.get('prefer_extracted_text', True)}")
    print(
        f"  Text chunking - Max single chunk: {summ_config.get('max_single_chunk_tokens', 15000)} tokens"
    )
    print(
        f"  Text chunking - Max chunk size: {summ_config.get('max_chunk_tokens', 8000)} tokens"
    )
    print(f"  Note: PDF direct processing does not use chunking")
    print()

    # Create and run summarizer
    try:
        workflow = PaperSummarizationWorkflow(config)
        workflow.run()

        print("\nSummarization workflow completed successfully!")
        return 0

    except Exception as e:
        print(f"Summarization workflow failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
PDF Processing Workflow

Main script that orchestrates paper selection and PDF downloading.
Integrates selection algorithms with PDF download management.

Dependencies:
    pip install requests tqdm python-magic pyyaml pandas numpy

Usage:
    python process_pdfs.py

Configuration:
    Edit the 'pdf_processing' section in config/config.yaml
"""

import json
import yaml
import pandas as pd
import os
import sys
from typing import List, Dict
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from selection_utils import (
        PaperSelector,
        validate_selection_config,
        find_most_recent_scored_file,
        load_scored_papers,
    )
    from pdf_utils import PDFDownloader, validate_download_config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure selection_utils.py and pdf_utils.py are in the src/ directory.")
    sys.exit(1)


class PDFProcessor:
    """
    Main class for the complete PDF processing workflow.

    Orchestrates:
    1. Loading scored papers
    2. Paper selection using sophisticated algorithms
    3. PDF downloading with rate limiting and verification
    4. Results compilation and storage
    """

    def __init__(self, config: Dict):
        """
        Initialize the PDF processor.

        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.pdf_config = config.get("pdf_processing", {})

        # Initialize components
        self.selector = PaperSelector(config)
        self.downloader = PDFDownloader(config)

        # File paths
        self.input_files = {}
        self.output_files = {}
        self.papers = []
        self.selected_papers = []
        self.download_results = {}

    def auto_detect_input_files(self) -> Dict[str, str]:
        """
        Auto-detect the most recent scored papers files.

        Returns:
            Dictionary mapping format -> filepath
        """
        # Check if input file is explicitly specified
        input_file = self.pdf_config.get("input_file")

        if input_file:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Specified input file not found: {input_file}")

            # Determine format from extension
            if input_file.endswith(".json"):
                return {"json": input_file}
            else:
                raise ValueError(f"Unsupported file format: {input_file}")

        # Auto-detect using the same logic as selection_utils
        filepath, format_type = find_most_recent_scored_file()

        if not filepath:
            return {}

        return {format_type: filepath}

    def load_papers(self) -> List[Dict]:
        """
        Load scored papers from input files.

        Returns:
            List of paper dictionaries
        """
        self.input_files = self.auto_detect_input_files()

        if not self.input_files:
            raise FileNotFoundError(
                "No scored papers found. Run score_papers.py first or specify input_file in config."
            )

        # Load papers (prefer JSON format)
        if "json" in self.input_files:
            papers = load_scored_papers(self.input_files["json"], "json")
            print(
                f"Loaded {len(papers)} papers from JSON file: {self.input_files['json']}"
            )
        else:
            raise RuntimeError("No valid input files found")

        # Filter papers with valid scores
        valid_papers = [p for p in papers if p.get("llm_score") is not None]
        print(f"Papers with valid scores: {len(valid_papers)}")

        if not valid_papers:
            raise RuntimeError("No papers with valid scores found")

        self.papers = valid_papers
        return valid_papers

    def select_papers(self) -> List[Dict]:
        """
        Run paper selection algorithm.

        Returns:
            List of selected papers
        """
        print(f"\n" + "=" * 50)
        print("PAPER SELECTION")
        print("=" * 50)

        # Run selection algorithm
        selection_result = self.selector.select_papers(self.papers)
        selected_papers = selection_result["selected_papers"]

        # Store selection results
        self.selected_papers = selected_papers
        self.selection_stats = selection_result["statistics"]

        return selected_papers

    def download_pdfs(self) -> Dict:
        """
        Download PDFs for selected papers.

        Returns:
            Download results dictionary
        """
        if not self.selected_papers:
            raise RuntimeError("No papers selected for download")

        print(f"\n" + "=" * 50)
        print("PDF DOWNLOAD")
        print("=" * 50)

        # Download PDFs
        download_results = self.downloader.download_papers(self.selected_papers)

        # Store download results
        self.download_results = download_results

        return download_results

    def save_processing_results(self) -> Dict[str, str]:
        """
        Save complete processing results to files.

        Returns:
            Dictionary of saved file paths
        """
        output_dir = self.config.get("output", {}).get("base_dir", "./data")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_files = {}

        # Save selected papers with download results
        enhanced_papers = []
        download_results_by_id = {
            r["paper_id"]: r for r in self.download_results.get("results", [])
        }

        for paper in self.selected_papers:
            enhanced_paper = paper.copy()
            paper_id = paper.get("id", "unknown")

            # Add download information
            if paper_id in download_results_by_id:
                download_info = download_results_by_id[paper_id]
                enhanced_paper.update(
                    {
                        "pdf_downloaded": download_info["success"],
                        "pdf_file_path": (
                            download_info["storage_path"]
                            if download_info["success"]
                            else None
                        ),
                        "pdf_file_size": download_info["file_size"],
                        "pdf_download_error": download_info["error"],
                        "pdf_download_skipped": download_info["skipped"],
                    }
                )

            # Add selection metadata
            enhanced_paper["selected_at"] = self.selection_stats["selection_timestamp"]
            enhanced_paper["selection_method"] = "pdf_processing_workflow"

            enhanced_papers.append(enhanced_paper)

        # Determine input format and save enhanced selected papers in the same format
        input_format = None
        if self.input_files:
            input_file = list(self.input_files.values())[0]
            if input_file.endswith(".json"):
                input_format = "json"

        if input_format == "json":
            selected_json_path = os.path.join(
                output_dir, f"selected_papers_{timestamp}.json"
            )
            with open(selected_json_path, "w", encoding="utf-8") as f:
                json.dump(enhanced_papers, f, indent=2, ensure_ascii=False)
            saved_files["selected_papers_json"] = selected_json_path
            print(f"Saved selected papers: {selected_json_path}")
        else:
            # Default to JSON if format cannot be determined
            selected_json_path = os.path.join(
                output_dir, f"selected_papers_{timestamp}.json"
            )
            with open(selected_json_path, "w", encoding="utf-8") as f:
                json.dump(enhanced_papers, f, indent=2, ensure_ascii=False)
            saved_files["selected_papers_json"] = selected_json_path
            print(f"Saved selected papers: {selected_json_path}")

        # Save complete processing report
        report = {
            "processing_timestamp": datetime.now().isoformat(),
            "input_file": (
                list(self.input_files.values())[0] if self.input_files else None
            ),
            "total_papers_loaded": len(self.papers),
            "papers_selected": len(self.selected_papers),
            "selection_statistics": self.selection_stats,
            "download_statistics": {
                "total_attempted": self.download_results.get("total_papers", 0),
                "successful_downloads": self.download_results.get(
                    "successful_downloads", 0
                ),
                "failed_downloads": self.download_results.get("failed_downloads", 0),
                "skipped_downloads": self.download_results.get("skipped_downloads", 0),
                "total_time": self.download_results.get("total_time", 0),
                "total_bytes": self.download_results.get("total_bytes", 0),
                "config_used": self.download_results.get("config_used", {}),
            },
            "config_used": {
                "selection": self.pdf_config.get("selection", {}),
                "download": self.pdf_config.get("download", {}),
            },
        }

        report_path = os.path.join(
            output_dir, f"pdf_processing_report_{timestamp}.json"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        saved_files["processing_report"] = report_path
        print(f"Saved processing report: {report_path}")

        return saved_files

    def print_final_summary(self):
        """Print a comprehensive final summary."""
        print(f"\n" + "=" * 50)
        print("FINAL SUMMARY")
        print("=" * 50)

        print(f"Papers processed: {len(self.papers)}")
        print(f"Papers selected: {len(self.selected_papers)}")

        if self.download_results:
            print(
                f"PDFs downloaded: {self.download_results.get('successful_downloads', 0)}"
            )
            print(
                f"Downloads failed: {self.download_results.get('failed_downloads', 0)}"
            )
            print(
                f"Downloads skipped: {self.download_results.get('skipped_downloads', 0)}"
            )

            if self.download_results.get("total_bytes", 0) > 0:
                size_mb = self.download_results["total_bytes"] / (1024 * 1024)
                print(f"Total downloaded: {size_mb:.1f} MB")

        # Show successful papers
        successful_papers = [
            p
            for p in self.selected_papers
            if any(
                r["paper_id"] == p.get("id") and r["success"]
                for r in self.download_results.get("results", [])
            )
        ]

        if successful_papers:
            print(f"\nSuccessfully processed papers:")
            for paper in successful_papers[:5]:  # Show first 5
                title = paper.get("title", "Untitled")
                score = paper.get("llm_score", "N/A")
                print(f"  [{score}] {title[:60]}{'...' if len(title) > 60 else ''}")

            if len(successful_papers) > 5:
                print(f"  ... and {len(successful_papers) - 5} more")

        # Show next steps
        print(f"\nNext steps:")
        if successful_papers:
            print(f"1. Review downloaded PDFs in the storage directory")
            print(f"2. Implement text extraction from PDFs (if desired)")
            print(f"3. Run detailed summarization workflow")
        else:
            print(f"1. Check download errors and retry if needed")
            print(f"2. Adjust selection or download configuration")

    def run(self):
        """
        Run the complete PDF processing workflow.
        """
        try:
            print("PDF Processing Workflow")
            print("======================")

            # Load papers
            papers = self.load_papers()

            # Select papers
            selected_papers = self.select_papers()

            if not selected_papers:
                print("No papers selected for processing.")
                return

            # Download PDFs
            download_results = self.download_pdfs()

            # Save results
            saved_files = self.save_processing_results()

            # Print final summary
            self.print_final_summary()

            print(f"\nFiles saved:")
            for file_type, filepath in saved_files.items():
                print(f"  {file_type}: {filepath}")

            print(f"\nPDF processing workflow completed successfully!")

        except Exception as e:
            print(f"Error during PDF processing workflow: {e}")
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


def validate_pdf_processing_config(config: Dict) -> List[str]:
    """
    Validate the complete PDF processing configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Validate selection configuration
    selection_errors = validate_selection_config(config)
    errors.extend([f"Selection: {err}" for err in selection_errors])

    # Validate download configuration
    download_errors = validate_download_config(config)
    errors.extend([f"Download: {err}" for err in download_errors])

    # Check that pdf_processing section exists
    if "pdf_processing" not in config:
        errors.append("pdf_processing section missing from configuration")

    return errors


def main():
    """Main function to run the PDF processing workflow."""
    print("ArXiv PDF Processing Workflow")
    print("============================")

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return 1

    # Validate configuration
    validation_errors = validate_pdf_processing_config(config)
    if validation_errors:
        print("Configuration validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        return 1

    # Display configuration summary
    pdf_config = config.get("pdf_processing", {})
    selection_config = pdf_config.get("selection", {})
    download_config = pdf_config.get("download", {})

    print(f"Configuration summary:")
    print(
        f"  Selection - Percentile: {selection_config.get('percentile', 'not set')}%, "
        f"Score threshold: {selection_config.get('score_threshold', 'not set')}"
    )
    print(
        f"  Selection - Min papers: {selection_config.get('min_papers', 'not set')}, "
        f"Max papers: {selection_config.get('max_papers', 'not set')}"
    )
    print(
        f"  Selection - Random papers: {selection_config.get('n_random', 'not set')}, "
        f"Temperature: {selection_config.get('temperature', 'not set')}"
    )
    print(
        f"  Download - Rate limit: {download_config.get('rate_limit_delay', 'not set')}s, "
        f"Max retries: {download_config.get('max_retries', 'not set')}"
    )
    print(f"  Download - Storage: {download_config.get('storage_dir', 'not set')}")
    print()

    # Create and run processor
    try:
        processor = PDFProcessor(config)
        processor.run()
        return 0

    except Exception as e:
        print(f"PDF processing workflow failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

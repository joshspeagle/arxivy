#!/usr/bin/env python3
"""
PDF Text Extraction Workflow

Main script that orchestrates text extraction from downloaded PDFs.
Handles file discovery, extraction processing, quality validation, and cleanup.

Dependencies:
    pip install PyPDF2 pdfplumber pymupdf pyyaml pandas

Usage:
    python extract_papers.py

Configuration:
    Edit the 'pdf_processing.extraction' section in config/config.yaml
"""

import json
import yaml
import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import glob
from datetime import datetime

# Progress bar
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from extract_utils import TextExtractor, validate_extraction_config
except ImportError as e:
    print(f"Error importing extract_utils: {e}")
    print("Make sure extract_utils.py is in the src/ directory.")
    sys.exit(1)


class PaperExtractor:
    """
    Main class for the PDF text extraction workflow.

    Orchestrates:
    1. Loading papers with PDF download information
    2. Text extraction from PDFs using multiple methods
    3. Quality validation and error handling
    4. Results compilation and storage
    5. Optional PDF cleanup after extraction
    """

    def __init__(self, config: Dict):
        """
        Initialize the paper extractor.

        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.pdf_config = config.get("pdf_processing", {})
        self.extraction_config = self.pdf_config.get("extraction", {})

        # Initialize text extractor
        self.extractor = TextExtractor(config)

        # File paths and data
        self.input_files = {}
        self.papers_with_pdfs = []
        self.extraction_results = []

        # PDF cleanup configuration
        self.cleanup_pdfs_after_extraction = self.extraction_config.get(
            "cleanup_pdfs_after_extraction", False
        )

        # Statistics tracking
        self.stats = {
            "papers_loaded": 0,
            "papers_with_pdfs": 0,
            "extractions_attempted": 0,
            "extractions_successful": 0,
            "extractions_failed": 0,
            "quality_failures": 0,
            "total_extraction_time": 0,
            "total_text_chars": 0,
            "pdfs_cleaned_up": 0,
        }

    def find_most_recent_selected_papers_file(
        self,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the most recent selected papers file with PDF information.

        Returns:
            Tuple of (filepath, format_type) or (None, None) if not found
        """
        data_dir = self.config.get("output", {}).get("base_dir", "./data")

        # Look for selected papers files
        json_pattern = os.path.join(data_dir, "selected_papers_*.json")

        json_files = glob.glob(json_pattern)

        all_files = [(f, "json") for f in json_files]

        if not all_files:
            return None, None

        # Get most recent by modification time
        most_recent = max(all_files, key=lambda x: os.path.getmtime(x[0]))
        return most_recent

    def load_papers_with_pdfs(self, filepath: str) -> List[Dict]:
        """
        Load papers with PDF information from file.

        Args:
            filepath: Path to the selected papers file

        Returns:
            List of paper dictionaries with PDF information
        """
        if filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                papers = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        return papers

    def auto_detect_input_files(self) -> Dict[str, str]:
        """
        Auto-detect the most recent selected papers files.

        Returns:
            Dictionary mapping format -> filepath
        """
        # Check if input file is explicitly specified
        input_file = self.extraction_config.get("input_file")

        if input_file:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Specified input file not found: {input_file}")

            # Determine format from extension
            if input_file.endswith(".json"):
                return {"json": input_file}
            else:
                raise ValueError(f"Unsupported file format: {input_file}")

        # Auto-detect using selected papers files
        filepath, format_type = self.find_most_recent_selected_papers_file()

        if not filepath:
            return {}

        return {format_type: filepath}

    def load_papers(self) -> List[Dict]:
        """
        Load papers with PDF information from input files.

        Returns:
            List of papers with successful PDF downloads
        """
        self.input_files = self.auto_detect_input_files()

        if not self.input_files:
            raise FileNotFoundError(
                "No selected papers found. Run download_papers.py first or specify input_file in config."
            )

        # Load papers (prefer JSON format)
        if "json" in self.input_files:
            papers = self.load_papers_with_pdfs(self.input_files["json"])
            print(
                f"Loaded {len(papers)} papers from JSON file: {self.input_files['json']}"
            )
        else:
            raise RuntimeError("No valid input files found")

        # Filter papers with successful PDF downloads
        papers_with_pdfs = []
        for paper in papers:
            if (
                paper.get("pdf_downloaded")
                and paper.get("pdf_file_path")
                and Path(paper["pdf_file_path"]).exists()
            ):
                papers_with_pdfs.append(paper)

        print(f"Papers with successfully downloaded PDFs: {len(papers_with_pdfs)}")

        if not papers_with_pdfs:
            raise RuntimeError("No papers with valid PDF files found")

        self.papers_with_pdfs = papers_with_pdfs
        self.stats["papers_loaded"] = len(papers)
        self.stats["papers_with_pdfs"] = len(papers_with_pdfs)

        return papers_with_pdfs

    def extract_text_from_papers(self) -> List[Dict]:
        """
        Extract text from all papers with PDFs.

        Returns:
            List of extraction results
        """
        if not self.papers_with_pdfs:
            raise RuntimeError("No papers with PDFs loaded")

        print(f"\n" + "=" * 50)
        print("TEXT EXTRACTION")
        print("=" * 50)
        print(f"Extracting text from {len(self.papers_with_pdfs)} PDFs...")
        print(f"Extraction method: {self.extractor.method}")
        print(f"Text format: {self.extractor.text_format}")
        print(
            f"Quality validation: {'enabled' if self.extractor.require_abstract_keywords else 'disabled'}"
        )
        print()

        results = []

        # Setup progress bar if available
        if HAS_TQDM:
            papers_iter = tqdm(
                self.papers_with_pdfs, desc="Extracting text", unit="paper"
            )
        else:
            papers_iter = self.papers_with_pdfs

        for i, paper in enumerate(papers_iter):
            self.stats["extractions_attempted"] += 1

            # Get PDF path
            pdf_path = Path(paper["pdf_file_path"])

            # Extract text
            extraction_result = self.extractor.extract_from_pdf(pdf_path, paper)

            # Enhance result with paper information
            enhanced_result = paper.copy()
            enhanced_result.update(
                {
                    "extraction_success": extraction_result["success"],
                    "extraction_method": extraction_result["method_used"],
                    "extracted_text_length": extraction_result["text_length"],
                    "extraction_output_path": extraction_result["output_path"],
                    "extraction_quality_valid": extraction_result["quality_valid"],
                    "extraction_quality_message": extraction_result["quality_message"],
                    "extraction_time": extraction_result["extraction_time"],
                    "extraction_errors": extraction_result["errors"],
                    "extraction_metadata": extraction_result["metadata"],
                }
            )

            # Include extracted text if not saving to file
            if not self.extractor.save_extracted_text:
                enhanced_result["extracted_text"] = extraction_result["extracted_text"]

            results.append(enhanced_result)

            # Update statistics
            if extraction_result["success"]:
                self.stats["extractions_successful"] += 1
                self.stats["total_text_chars"] += extraction_result["text_length"]
                if not extraction_result["quality_valid"]:
                    self.stats["quality_failures"] += 1
            else:
                self.stats["extractions_failed"] += 1

            self.stats["total_extraction_time"] += extraction_result["extraction_time"]

            # Progress update for non-tqdm display
            if not HAS_TQDM:
                status = "OK" if extraction_result["success"] else "FAIL"
                if (
                    extraction_result["success"]
                    and not extraction_result["quality_valid"]
                ):
                    status = "WARN"

                title = paper.get("title", "Untitled")[:50]
                method = extraction_result.get("method_used", "none")
                chars = extraction_result.get("text_length", 0)

                print(
                    f"[{i+1:3d}/{len(self.papers_with_pdfs)}] {status} | {method:>10} | {chars:>6} chars | {title}{'...' if len(paper.get('title', '')) > 50 else ''}"
                )

        self.extraction_results = results
        return results

    def cleanup_pdfs(self):
        """
        Clean up PDF files and directories after successful extraction.
        """
        if not self.cleanup_pdfs_after_extraction:
            print("PDF cleanup disabled")
            return

        print(f"\n" + "=" * 50)
        print("PDF CLEANUP")
        print("=" * 50)

        # Get PDF storage directory
        pdf_storage_dir = Path(
            self.config.get("pdf_processing", {})
            .get("download", {})
            .get("storage_dir", "./data/pdfs")
        )

        if not pdf_storage_dir.exists():
            print(f"PDF storage directory does not exist: {pdf_storage_dir}")
            return

        # Count successful extractions
        successful_extractions = [
            r for r in self.extraction_results if r.get("extraction_success", False)
        ]

        if not successful_extractions:
            print("No successful extractions, skipping PDF cleanup")
            return

        print(
            f"Cleaning up PDF directory after {len(successful_extractions)} successful extractions..."
        )

        try:
            # Remove entire PDF directory structure
            shutil.rmtree(pdf_storage_dir)
            self.stats["pdfs_cleaned_up"] = (
                len(list(pdf_storage_dir.rglob("*.pdf")))
                if pdf_storage_dir.exists()
                else 0
            )
            print(f"✅ Removed PDF directory: {pdf_storage_dir}")

        except Exception as e:
            print(f"❌ Error during PDF cleanup: {e}")

    def save_extraction_results(self) -> Dict[str, str]:
        """
        Save extraction results to files.

        Returns:
            Dictionary of saved file paths
        """
        output_dir = self.config.get("output", {}).get("base_dir", "./data")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_files = {}

        # Save extraction results as JSON
        results_json_path = os.path.join(
            output_dir, f"extraction_results_{timestamp}.json"
        )
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(self.extraction_results, f, indent=2, ensure_ascii=False)
        saved_files["extraction_results_json"] = results_json_path
        print(f"Saved extraction results: {results_json_path}")

        # Save comprehensive extraction report
        report = {
            "extraction_timestamp": datetime.now().isoformat(),
            "input_file": (
                list(self.input_files.values())[0] if self.input_files else None
            ),
            "extraction_statistics": self.stats,
            "extractor_statistics": self.extractor.stats,
            "successful_extractions": len(
                [
                    r
                    for r in self.extraction_results
                    if r.get("extraction_success", False)
                ]
            ),
            "failed_extractions": len(
                [
                    r
                    for r in self.extraction_results
                    if not r.get("extraction_success", False)
                ]
            ),
            "quality_issues": len(
                [
                    r
                    for r in self.extraction_results
                    if r.get("extraction_success", False)
                    and not r.get("extraction_quality_valid", True)
                ]
            ),
            "config_used": {
                "extraction": self.extraction_config,
                "pdf_cleanup_enabled": self.cleanup_pdfs_after_extraction,
            },
            "text_storage_directory": (
                self.extractor.text_storage_dir
                if self.extractor.save_extracted_text
                else None
            ),
        }

        report_path = os.path.join(output_dir, f"extraction_report_{timestamp}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        saved_files["extraction_report"] = report_path
        print(f"Saved extraction report: {report_path}")

        return saved_files

    def print_final_summary(self):
        """Print a comprehensive final summary."""
        print(f"\n" + "=" * 50)
        print("FINAL SUMMARY")
        print("=" * 50)

        print(f"Papers loaded: {self.stats['papers_loaded']}")
        print(f"Papers with PDFs: {self.stats['papers_with_pdfs']}")
        print(f"Extractions attempted: {self.stats['extractions_attempted']}")
        print(f"Extractions successful: {self.stats['extractions_successful']}")
        print(f"Extractions failed: {self.stats['extractions_failed']}")

        if self.stats["quality_failures"] > 0:
            print(f"Quality validation issues: {self.stats['quality_failures']}")

        if self.stats["total_extraction_time"] > 0:
            print(
                f"Total extraction time: {self.stats['total_extraction_time']:.1f} seconds"
            )
            avg_time = (
                self.stats["total_extraction_time"]
                / self.stats["extractions_attempted"]
            )
            print(f"Average time per paper: {avg_time:.1f} seconds")

        if self.stats["total_text_chars"] > 0:
            print(
                f"Total text extracted: {self.stats['total_text_chars']:,} characters"
            )
            avg_chars = (
                self.stats["total_text_chars"] / self.stats["extractions_successful"]
            )
            print(f"Average text per paper: {avg_chars:,.0f} characters")

        if self.cleanup_pdfs_after_extraction:
            if self.stats["pdfs_cleaned_up"] > 0:
                print(f"PDFs cleaned up: Yes (freed disk space)")
            else:
                print(f"PDFs cleaned up: Attempted")

        # Show method usage statistics
        if self.extractor.stats.get("method_usage"):
            print(f"\nExtraction methods used:")
            for method, count in self.extractor.stats["method_usage"].items():
                print(f"  {method}: {count} papers")

        # Show successful extractions
        successful_results = [
            r for r in self.extraction_results if r.get("extraction_success", False)
        ]

        if successful_results:
            print(f"\nSuccessfully extracted papers:")
            for result in successful_results[:5]:  # Show first 5
                title = result.get("title", "Untitled")
                chars = result.get("extracted_text_length", 0)
                method = result.get("extraction_method", "unknown")
                quality = "✓" if result.get("extraction_quality_valid", True) else "⚠"
                print(
                    f"  {quality} [{chars:>6} chars, {method}] {title[:60]}{'...' if len(title) > 60 else ''}"
                )

            if len(successful_results) > 5:
                print(f"  ... and {len(successful_results) - 5} more")

        # Show next steps
        print(f"\nNext steps:")
        if successful_results:
            if self.extractor.save_extracted_text:
                print(
                    f"1. Review extracted text files in: {self.extractor.text_storage_dir}"
                )
            print(f"2. Apply detailed summarization workflow")
            print(f"3. Run final scoring based on full paper content")
        else:
            print(f"1. Check extraction errors and configuration")
            print(f"2. Verify PDF files are valid and accessible")
            print(f"3. Consider adjusting extraction parameters")

    def run(self):
        """
        Run the complete text extraction workflow.
        """
        try:
            print("PDF Text Extraction Workflow")
            print("============================")

            # Load papers
            papers = self.load_papers()

            # Extract text from PDFs
            results = self.extract_text_from_papers()

            # Clean up PDFs if configured
            if self.cleanup_pdfs_after_extraction:
                self.cleanup_pdfs()

            # Save results
            saved_files = self.save_extraction_results()

            # Print final summary
            self.print_final_summary()

            print(f"\nFiles saved:")
            for file_type, filepath in saved_files.items():
                print(f"  {file_type}: {filepath}")

            print(f"\nText extraction workflow completed successfully!")

        except Exception as e:
            print(f"Error during text extraction workflow: {e}")
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


def validate_extraction_workflow_config(config: Dict) -> List[str]:
    """
    Validate the complete extraction workflow configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Validate extraction configuration
    extraction_errors = validate_extraction_config(config)
    errors.extend(extraction_errors)

    # Check that pdf_processing section exists
    if "pdf_processing" not in config:
        errors.append("pdf_processing section missing from configuration")

    return errors


def main():
    """Main function to run the text extraction workflow."""
    print("ArXiv PDF Text Extraction Workflow")
    print("==================================")

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return 1

    # Validate configuration
    validation_errors = validate_extraction_workflow_config(config)
    if validation_errors:
        print("Configuration validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        return 1

    # Display configuration summary
    pdf_config = config.get("pdf_processing", {})
    extraction_config = pdf_config.get("extraction", {})

    print(f"Configuration summary:")
    print(f"  Method: {extraction_config.get('method', 'not set')}")
    print(f"  Fallback methods: {extraction_config.get('fallback_methods', 'not set')}")
    print(f"  Text format: {extraction_config.get('text_format', 'not set')}")
    print(
        f"  Min text length: {extraction_config.get('min_text_length', 'not set')} chars"
    )
    print(
        f"  Quality validation: {extraction_config.get('require_abstract_keywords', 'not set')}"
    )
    print(
        f"  Save extracted text: {extraction_config.get('save_extracted_text', 'not set')}"
    )
    print(
        f"  Storage directory: {extraction_config.get('text_storage_dir', 'not set')}"
    )
    print(
        f"  PDF cleanup: {extraction_config.get('cleanup_pdfs_after_extraction', 'not set')}"
    )
    print()

    # Create and run extractor
    try:
        extractor = PaperExtractor(config)
        extractor.run()
        return 0

    except Exception as e:
        print(f"Text extraction workflow failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

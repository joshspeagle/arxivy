#!/usr/bin/env python3
"""
PDF Download and Management Utilities

Handles downloading, verification, and organization of arXiv PDFs.
Implements rate limiting, retry logic, and robust error handling.

Dependencies:
    pip install requests tqdm python-magic

Usage:
    from pdf_utils import PDFDownloader

    downloader = PDFDownloader(config)
    results = downloader.download_papers(selected_papers)
"""

import time
import requests
import yaml
import re
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import logging

# Optional dependencies for file verification
try:
    import magic

    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

# Progress bar
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class PDFDownloader:
    """
    Handles PDF downloading with rate limiting, retries, and verification.

    Features:
    - Rate limiting for arXiv compliance
    - Retry logic for failed downloads
    - File verification and integrity checks
    - Progress tracking and statistics
    - Organized storage with configurable naming
    - Skip already downloaded files
    """

    def __init__(self, config: Dict):
        """
        Initialize the PDF downloader.

        Args:
            config: Configuration dictionary containing pdf_processing settings
        """
        self.config = config
        self.download_config = config.get("pdf_processing", {}).get("download", {})

        # Download parameters
        self.rate_limit_delay = self.download_config.get("rate_limit_delay", 3.0)
        self.max_retries = self.download_config.get("max_retries", 3)
        self.timeout = self.download_config.get("timeout", 30)

        # Storage parameters
        self.storage_dir = self.download_config.get("storage_dir", "./data/pdfs")
        self.organize_by_date = self.download_config.get("organize_by_date", True)
        self.filename_format = self.download_config.get(
            "filename_format", "{arxiv_id}_{title_slug}.pdf"
        )
        self.max_filename_length = self.download_config.get("max_filename_length", 100)

        # Verification parameters
        self.verify_downloads = self.download_config.get("verify_downloads", True)
        self.min_file_size = self.download_config.get("min_file_size", 10000)  # 10KB

        # Setup session for connection reuse
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "arxivy/1.0 (https://github.com/joshspeagle/arxivy) Python-requests"
            }
        )

        # Statistics tracking
        self.stats = {
            "attempted": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "total_bytes": 0,
            "total_time": 0,
            "retries": 0,
        }

        # Setup logging
        self._setup_logging()

        # Validate configuration
        self._validate_config()

    def _setup_logging(self):
        """Setup logging for download operations."""
        self.logger = logging.getLogger("pdf_downloader")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _validate_config(self):
        """Validate download configuration parameters."""
        if self.rate_limit_delay < 0:
            raise ValueError("rate_limit_delay must be non-negative")

        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

        if self.min_file_size < 0:
            raise ValueError("min_file_size must be non-negative")

        if self.max_filename_length < 10:
            raise ValueError("max_filename_length must be at least 10")

    def _create_title_slug(self, title: str) -> str:
        """
        Create a filesystem-safe slug from paper title.

        Args:
            title: Paper title

        Returns:
            Filesystem-safe slug
        """
        if not title:
            return "untitled"

        # Remove/replace problematic characters
        slug = re.sub(r"[^\w\s-]", "", title.lower())
        slug = re.sub(r"[\s_-]+", "_", slug)
        slug = slug.strip("_-")

        # Limit to up to 5 words, 100 characters, or max_filename_length - 20 (padding)
        words = slug.split("_")
        limited_words = words[:5]
        limited_slug = "_".join(limited_words)
        max_slug_length = min(self.max_filename_length - 20, 100, len(limited_slug))
        if len(limited_slug) > max_slug_length:
            limited_slug = limited_slug[:max_slug_length].rstrip("_-")

        return limited_slug or "untitled"

    def _generate_filename(self, paper: Dict) -> str:
        """
        Generate filename for a paper based on configuration.

        Args:
            paper: Paper dictionary

        Returns:
            Generated filename
        """
        arxiv_id = paper.get("id", "unknown")
        title = paper.get("title", "Untitled")
        title_slug = self._create_title_slug(title)

        # Format filename
        filename = self.filename_format.format(arxiv_id=arxiv_id, title_slug=title_slug)

        # Ensure .pdf extension
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        # Final length check
        if len(filename) > self.max_filename_length:
            # Truncate title_slug to fit
            available_length = self.max_filename_length - len(arxiv_id) - 10  # margin
            title_slug = title_slug[:available_length].rstrip("_-")
            filename = f"{arxiv_id}_{title_slug}.pdf"

        return filename

    def _get_storage_path(self, paper: Dict) -> Path:
        """
        Get the full storage path for a paper.

        Args:
            paper: Paper dictionary

        Returns:
            Full path where PDF should be stored
        """
        base_path = Path(self.storage_dir)

        if self.organize_by_date:
            # Use paper's published date or current date
            date_str = paper.get("published", "")
            if date_str:
                try:
                    # Parse ISO date format (2023-01-15T10:00:00Z)
                    date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    date_folder = date_obj.strftime("%Y-%m-%d")
                except:
                    # Fallback to current date
                    date_folder = datetime.now().strftime("%Y-%m-%d")
            else:
                date_folder = datetime.now().strftime("%Y-%m-%d")

            storage_path = base_path / date_folder
        else:
            storage_path = base_path

        # Create directory if it doesn't exist
        storage_path.mkdir(parents=True, exist_ok=True)

        filename = self._generate_filename(paper)
        return storage_path / filename

    def _get_pdf_url(self, paper: Dict) -> str:
        """
        Get the PDF URL for a paper.

        Args:
            paper: Paper dictionary

        Returns:
            PDF URL
        """
        # Check if paper already has a PDF URL
        if "pdf_url" in paper and paper["pdf_url"]:
            return paper["pdf_url"]

        # Construct PDF URL from arXiv ID
        arxiv_id = paper.get("id", "")
        if not arxiv_id:
            raise ValueError("Paper missing arXiv ID")

        # Handle different arXiv ID formats
        if "/" in arxiv_id:
            # Old format: subject-class/YYMMnnn
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        else:
            # New format: YYMM.nnnn
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        return pdf_url

    def _verify_pdf_file(self, filepath: Path) -> Tuple[bool, str]:
        """
        Verify that downloaded file is a valid PDF.

        Args:
            filepath: Path to the downloaded file

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not filepath.exists():
            return False, "File does not exist"

        # Check file size
        file_size = filepath.stat().st_size
        if file_size < self.min_file_size:
            return False, f"File too small ({file_size} bytes)"

        # Check file type using magic if available
        if HAS_MAGIC:
            try:
                file_type = magic.from_file(str(filepath), mime=True)
                if file_type != "application/pdf":
                    return False, f"Wrong file type: {file_type}"
            except Exception as e:
                self.logger.warning(f"Magic file type detection failed: {e}")

        # Basic PDF header check
        try:
            with open(filepath, "rb") as f:
                header = f.read(8)
                if not header.startswith(b"%PDF-"):
                    return False, "Invalid PDF header"
        except Exception as e:
            return False, f"Cannot read file: {e}"

        return True, "Valid PDF"

    def _download_single_pdf(self, paper: Dict) -> Dict:
        """
        Download a single PDF with retry logic.

        Args:
            paper: Paper dictionary

        Returns:
            Download result dictionary
        """
        arxiv_id = paper.get("id", "unknown")
        pdf_url = self._get_pdf_url(paper)
        storage_path = self._get_storage_path(paper)

        result = {
            "paper_id": arxiv_id,
            "pdf_url": pdf_url,
            "storage_path": str(storage_path),
            "success": False,
            "error": None,
            "file_size": 0,
            "download_time": 0,
            "retries_used": 0,
            "skipped": False,
        }

        # Check if file already exists
        if storage_path.exists():
            if self.verify_downloads:
                is_valid, error_msg = self._verify_pdf_file(storage_path)
                if is_valid:
                    result["success"] = True
                    result["skipped"] = True
                    result["file_size"] = storage_path.stat().st_size
                    self.stats["skipped"] += 1
                    return result
                else:
                    self.logger.info(
                        f"Existing file invalid ({error_msg}), re-downloading: {arxiv_id}"
                    )
                    storage_path.unlink()  # Delete invalid file
            else:
                result["success"] = True
                result["skipped"] = True
                result["file_size"] = storage_path.stat().st_size
                self.stats["skipped"] += 1
                return result

        # Attempt download with retries
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()

                # Make request
                response = self.session.get(pdf_url, timeout=self.timeout, stream=True)
                response.raise_for_status()

                # Write file
                temp_path = storage_path.with_suffix(".pdf.tmp")
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # Move temp file to final location
                temp_path.rename(storage_path)

                download_time = time.time() - start_time
                file_size = storage_path.stat().st_size

                # Verify download if configured
                if self.verify_downloads:
                    is_valid, error_msg = self._verify_pdf_file(storage_path)
                    if not is_valid:
                        storage_path.unlink()  # Delete invalid file
                        raise ValueError(f"Downloaded file invalid: {error_msg}")

                # Success!
                result.update(
                    {
                        "success": True,
                        "file_size": file_size,
                        "download_time": download_time,
                        "retries_used": attempt,
                    }
                )

                self.stats["successful"] += 1
                self.stats["total_bytes"] += file_size
                self.stats["total_time"] += download_time
                self.stats["retries"] += attempt

                return result

            except Exception as e:
                error_msg = str(e)
                self.logger.warning(
                    f"Download attempt {attempt + 1} failed for {arxiv_id}: {error_msg}"
                )

                # Clean up any partial files
                temp_path = storage_path.with_suffix(".pdf.tmp")
                if temp_path.exists():
                    temp_path.unlink()
                if storage_path.exists():
                    storage_path.unlink()

                if attempt < self.max_retries:
                    # Wait before retry (exponential backoff)
                    wait_time = self.rate_limit_delay * (2**attempt)
                    time.sleep(wait_time)
                else:
                    # Final failure
                    result.update({"error": error_msg, "retries_used": attempt})
                    self.stats["failed"] += 1
                    return result

        return result

    def download_papers(self, papers: List[Dict]) -> Dict:
        """
        Download PDFs for a list of papers.

        Args:
            papers: List of paper dictionaries

        Returns:
            Download results dictionary
        """
        print(f"Starting PDF downloads for {len(papers)} papers...")
        print(f"Storage directory: {self.storage_dir}")
        print(f"Rate limit: {self.rate_limit_delay}s between downloads")
        print(f"Max retries: {self.max_retries}")
        print()

        # Reset statistics
        self.stats = {k: 0 for k in self.stats}
        self.stats["total_time"] = 0

        start_time = time.time()
        results = []

        # Setup progress bar if available
        if HAS_TQDM:
            papers_iter = tqdm(papers, desc="Downloading PDFs", unit="paper")
        else:
            papers_iter = papers

        for i, paper in enumerate(papers_iter):
            self.stats["attempted"] += 1

            # Download paper
            result = self._download_single_pdf(paper)
            results.append(result)

            # Progress update for non-tqdm display
            if not HAS_TQDM:
                status = (
                    "SKIP"
                    if result["skipped"]
                    else ("OK" if result["success"] else "FAIL")
                )
                title = paper.get("title", "Untitled")[:50]
                print(
                    f"[{i+1:3d}/{len(papers)}] {status} | {title}{'...' if len(paper.get('title', '')) > 50 else ''}"
                )

            # Rate limiting (skip for last item and skipped downloads)
            if i < len(papers) - 1 and not result["skipped"]:
                time.sleep(self.rate_limit_delay)

        total_time = time.time() - start_time

        # Compile final results
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        skipped_results = [r for r in results if r["skipped"]]

        summary = {
            "total_papers": len(papers),
            "successful_downloads": len(successful_results),
            "failed_downloads": len(failed_results),
            "skipped_downloads": len(skipped_results),
            "total_time": total_time,
            "total_bytes": self.stats["total_bytes"],
            "average_file_size": (
                self.stats["total_bytes"] / len(successful_results)
                if successful_results
                else 0
            ),
            "download_rate": (
                len(successful_results) / total_time if total_time > 0 else 0
            ),
            "results": results,
            "config_used": {
                "rate_limit_delay": self.rate_limit_delay,
                "max_retries": self.max_retries,
                "storage_dir": self.storage_dir,
                "verify_downloads": self.verify_downloads,
            },
        }

        self._print_download_summary(summary)
        return summary

    def _print_download_summary(self, summary: Dict):
        """Print a detailed download summary."""
        print(f"\n=== DOWNLOAD COMPLETE ===")
        print(f"Total papers: {summary['total_papers']}")
        print(f"Successful downloads: {summary['successful_downloads']}")
        print(f"Failed downloads: {summary['failed_downloads']}")
        print(f"Skipped (already exist): {summary['skipped_downloads']}")
        print(f"Total time: {summary['total_time']:.1f} seconds")

        if summary["total_bytes"] > 0:
            size_mb = summary["total_bytes"] / (1024 * 1024)
            print(f"Total downloaded: {size_mb:.1f} MB")
            print(f"Average file size: {summary['average_file_size']/1024:.1f} KB")

        if summary["download_rate"] > 0:
            print(f"Download rate: {summary['download_rate']:.2f} files/second")

        # Show failed downloads
        failed_results = [r for r in summary["results"] if not r["success"]]
        if failed_results:
            print(f"\nFailed downloads:")
            for result in failed_results[:5]:  # Show first 5
                print(f"  - {result['paper_id']}: {result['error']}")
            if len(failed_results) > 5:
                print(f"  ... and {len(failed_results) - 5} more")


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
        print(f"Config file not found at {config_path}")
        return {}
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def validate_download_config(config: Dict) -> List[str]:
    """
    Validate the PDF download configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    pdf_config = config.get("pdf_processing", {})
    download_config = pdf_config.get("download", {})

    # Check parameters
    rate_limit = download_config.get("rate_limit_delay", 3.0)
    if not isinstance(rate_limit, (int, float)) or rate_limit < 0:
        errors.append("rate_limit_delay must be a non-negative number")

    max_retries = download_config.get("max_retries", 3)
    if not isinstance(max_retries, int) or max_retries < 0:
        errors.append("max_retries must be a non-negative integer")

    timeout = download_config.get("timeout", 30)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        errors.append("timeout must be a positive number")

    min_file_size = download_config.get("min_file_size", 10000)
    if not isinstance(min_file_size, int) or min_file_size < 0:
        errors.append("min_file_size must be a non-negative integer")

    storage_dir = download_config.get("storage_dir", "./data/pdfs")
    if not isinstance(storage_dir, str) or not storage_dir.strip():
        errors.append("storage_dir must be a non-empty string")

    return errors


def test_pdf_download():
    """Test PDF download functionality with mock data."""
    print("=== TESTING PDF DOWNLOAD FUNCTIONALITY ===")

    # Create test configuration
    test_config = {
        "pdf_processing": {
            "download": {
                "rate_limit_delay": 1.0,  # Faster for testing
                "max_retries": 2,
                "timeout": 15,
                "storage_dir": "./data/test_pdfs",
                "organize_by_date": False,  # Simpler for testing
                "verify_downloads": True,
                "min_file_size": 1000,  # Smaller for testing
            }
        }
    }

    # Create test papers (these should be real arXiv papers for testing)
    test_papers = [
        {
            "id": "1706.03762",  # "Attention Is All You Need" (Transformer paper)
            "title": "Attention Is All You Need",
            "published": "2017-06-12T17:58:34Z",
        },
        {
            "id": "2312.00752",  # Mamba paper
            "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
            "published": "2023-12-01T18:01:34Z",
        },
    ]

    print(f"Testing with {len(test_papers)} papers")
    print("Note: This will attempt real downloads for testing")

    try:
        # Test downloader initialization
        downloader = PDFDownloader(test_config)
        print("✅ PDFDownloader initialized successfully")

        # Test filename generation
        for paper in test_papers:
            filename = downloader._generate_filename(paper)
            storage_path = downloader._get_storage_path(paper)
            print(f"✅ Generated filename for {paper['id']}: {filename}")
            print(f"   Storage path: {storage_path}")

        # Test URL generation
        for paper in test_papers:
            pdf_url = downloader._get_pdf_url(paper)
            print(f"✅ Generated URL for {paper['id']}: {pdf_url}")

        print("\n" + "=" * 50)
        print("Ready to test actual downloads")
        print("This will download real PDFs - continue? (y/n): ", end="")

        response = input().lower().strip()
        if response == "y":
            results = downloader.download_papers(test_papers)
            print(f"✅ Download test completed")

            # Show detailed results
            for result in results["results"]:
                status = "SUCCESS" if result["success"] else "FAILED"
                if result["skipped"]:
                    status = "SKIPPED"
                print(f"  {result['paper_id']}: {status}")
                if result["error"]:
                    print(f"    Error: {result['error']}")
        else:
            print("Skipping actual download test")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n=== PDF DOWNLOAD TEST COMPLETE ===")


def test_pdf_download_config():
    """Test PDF download configuration."""
    print("=== TESTING PDF DOWNLOAD CONFIGURATION ===")

    # This would integrate with selection results
    # For now, show how it would work
    print("Testing with configuration validation...")

    try:
        config = load_config()

        if not config:
            print("❌ No configuration loaded")
            return

        validation_errors = validate_download_config(config)
        if validation_errors:
            print("❌ Configuration validation errors:")
            for error in validation_errors:
                print(f"   - {error}")
        else:
            print("✅ Configuration validation passed")

            # Show configuration
            download_config = config.get("pdf_processing", {}).get("download", {})
            print(f"Download configuration:")
            print(
                f"  Rate limit: {download_config.get('rate_limit_delay', 'not set')}s"
            )
            print(f"  Max retries: {download_config.get('max_retries', 'not set')}")
            print(f"  Storage dir: {download_config.get('storage_dir', 'not set')}")
            print(
                f"  Organize by date: {download_config.get('organize_by_date', 'not set')}"
            )
            print(
                f"  Verify downloads: {download_config.get('verify_downloads', 'not set')}"
            )

    except Exception as e:
        print(f"❌ Test failed: {e}")


# Example usage and testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] in ["--test", "-t"]:
            test_pdf_download()
        elif sys.argv[1] in ["--config", "-c"]:
            test_pdf_download_config()
        elif sys.argv[1] in ["--help", "-h"]:
            print("PDF Download Utilities")
            print("Usage:")
            print("  python pdf_utils.py --test       # Test download functionality")
            print("  python pdf_utils.py --config     # Test real configuration")
            print("  python pdf_utils.py --help       # Show this help")
        else:
            print("Unknown option. Use --help for usage information.")
    else:
        print("PDF Download Utilities Module")
        print("Run with --help for usage options")

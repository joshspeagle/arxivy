#!/usr/bin/env python3
"""
PDF Text Extraction Utilities

Handles text extraction from PDFs using multiple backends with fallback support.
Includes quality validation, structure preservation, and metadata generation.

Dependencies:
    pip install PyPDF2 pdfplumber pymupdf

Usage:
    from extraction_utils import TextExtractor

    extractor = TextExtractor(config)
    result = extractor.extract_from_pdf(pdf_path, paper_metadata)
"""

import re
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import logging
import os

# PDF extraction libraries (optional imports)
PDF_LIBRARIES = {}

try:
    import PyPDF2

    PDF_LIBRARIES["pypdf2"] = PyPDF2
except ImportError:
    PDF_LIBRARIES["pypdf2"] = None

try:
    import pdfplumber

    PDF_LIBRARIES["pdfplumber"] = pdfplumber
except ImportError:
    PDF_LIBRARIES["pdfplumber"] = None

try:
    import fitz  # PyMuPDF

    PDF_LIBRARIES["pymupdf"] = fitz
except ImportError:
    PDF_LIBRARIES["pymupdf"] = None


class TextExtractor:
    """
    Handles PDF text extraction with multiple backends and quality validation.

    Features:
    - Multiple extraction methods with automatic fallback
    - Quality validation against known abstracts
    - Structure preservation in multiple output formats
    - Metadata generation and error tracking
    - Configurable text processing and cleanup
    """

    def __init__(self, config: Dict):
        """
        Initialize the text extractor.

        Args:
            config: Configuration dictionary containing pdf_processing.extraction settings
        """
        self.config = config
        self.extraction_config = config.get("pdf_processing", {}).get("extraction", {})

        # Extraction parameters
        self.method = self.extraction_config.get("method", "auto")
        self.fallback_methods = self.extraction_config.get(
            "fallback_methods", ["pdfplumber", "pymupdf", "pypdf2"]
        )
        self.extract_text = self.extraction_config.get("extract_text", True)

        # Text processing parameters
        self.preserve_formatting = self.extraction_config.get(
            "preserve_formatting", False
        )
        self.remove_headers_footers = self.extraction_config.get(
            "remove_headers_footers", True
        )
        self.min_text_length = self.extraction_config.get("min_text_length", 1000)

        # Quality parameters
        self.max_extraction_errors = self.extraction_config.get(
            "max_extraction_errors", 5
        )
        self.require_abstract_keywords = self.extraction_config.get(
            "require_abstract_keywords", True
        )

        # Output parameters
        self.text_format = self.extraction_config.get("text_format", "markdown")
        self.include_metadata = self.extraction_config.get("include_metadata", True)
        self.save_extracted_text = self.extraction_config.get(
            "save_extracted_text", True
        )
        self.text_storage_dir = self.extraction_config.get(
            "text_storage_dir", "./data/extracted_text"
        )

        # Setup logging
        self._setup_logging()

        # Validate configuration and check available libraries
        self._validate_config()
        self._check_available_libraries()

        # Statistics tracking
        self.stats = {
            "attempted": 0,
            "successful": 0,
            "failed": 0,
            "quality_failed": 0,
            "total_chars": 0,
            "total_time": 0,
            "method_usage": {},
            "error_types": {},
        }

    def _setup_logging(self):
        """Setup logging for extraction operations."""
        self.logger = logging.getLogger("text_extractor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _validate_config(self):
        """Validate extraction configuration parameters."""
        if self.method not in ["auto", "pdfplumber", "pymupdf", "pypdf2", "none"]:
            raise ValueError(f"Invalid extraction method: {self.method}")

        if self.min_text_length < 0:
            raise ValueError("min_text_length must be non-negative")

        if self.max_extraction_errors < 0:
            raise ValueError("max_extraction_errors must be non-negative")

        if self.text_format not in ["markdown", "plain", "json"]:
            raise ValueError(f"Invalid text_format: {self.text_format}")

    def _check_available_libraries(self):
        """Check which PDF libraries are available and log warnings."""
        available = [lib for lib, module in PDF_LIBRARIES.items() if module is not None]
        unavailable = [lib for lib, module in PDF_LIBRARIES.items() if module is None]

        if unavailable:
            self.logger.warning(
                f"PDF libraries not available: {', '.join(unavailable)}"
            )
            self.logger.info(
                "Install missing libraries: pip install " + " ".join(unavailable)
            )

        if not available:
            raise RuntimeError(
                "No PDF extraction libraries available. Install at least one: pip install PyPDF2 pdfplumber pymupdf"
            )

        self.available_methods = available
        self.logger.info(f"Available extraction methods: {', '.join(available)}")

    def _get_extraction_methods(self) -> List[str]:
        """
        Get ordered list of extraction methods to try.

        Returns:
            List of method names in order of preference
        """
        if self.method == "none":
            return []
        elif self.method == "auto":
            # Use all available methods in fallback order
            return [
                method
                for method in self.fallback_methods
                if method in self.available_methods
            ]
        elif self.method in self.available_methods:
            # Use specified method with fallbacks
            methods = [self.method]
            for fallback in self.fallback_methods:
                if fallback != self.method and fallback in self.available_methods:
                    methods.append(fallback)
            return methods
        else:
            raise ValueError(f"Requested method '{self.method}' not available")

    def _extract_with_pypdf2(self, pdf_path: Path) -> Tuple[str, Dict]:
        """Extract text using PyPDF2."""
        metadata = {"method": "pypdf2", "pages_processed": 0, "errors": []}
        text_parts = []

        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                metadata["total_pages"] = len(reader.pages)

                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(page_text)
                        metadata["pages_processed"] += 1
                    except Exception as e:
                        error_msg = f"Page {page_num + 1}: {str(e)}"
                        metadata["errors"].append(error_msg)
                        if len(metadata["errors"]) > self.max_extraction_errors:
                            break

                extracted_text = "\n\n".join(text_parts)
                return extracted_text, metadata

        except Exception as e:
            metadata["errors"].append(f"Reader error: {str(e)}")
            return "", metadata

    def _extract_with_pdfplumber(self, pdf_path: Path) -> Tuple[str, Dict]:
        """Extract text using pdfplumber."""
        metadata = {"method": "pdfplumber", "pages_processed": 0, "errors": []}
        text_parts = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata["total_pages"] = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(page_text)
                        metadata["pages_processed"] += 1
                    except Exception as e:
                        error_msg = f"Page {page_num + 1}: {str(e)}"
                        metadata["errors"].append(error_msg)
                        if len(metadata["errors"]) > self.max_extraction_errors:
                            break

                extracted_text = "\n\n".join(text_parts)
                return extracted_text, metadata

        except Exception as e:
            metadata["errors"].append(f"Reader error: {str(e)}")
            return "", metadata

    def _extract_with_pymupdf(self, pdf_path: Path) -> Tuple[str, Dict]:
        """Extract text using PyMuPDF (fitz)."""
        metadata = {"method": "pymupdf", "pages_processed": 0, "errors": []}
        text_parts = []

        try:
            doc = fitz.open(pdf_path)
            metadata["total_pages"] = len(doc)

            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                    metadata["pages_processed"] += 1
                except Exception as e:
                    error_msg = f"Page {page_num + 1}: {str(e)}"
                    metadata["errors"].append(error_msg)
                    if len(metadata["errors"]) > self.max_extraction_errors:
                        break

            doc.close()
            extracted_text = "\n\n".join(text_parts)
            return extracted_text, metadata

        except Exception as e:
            metadata["errors"].append(f"Reader error: {str(e)}")
            return "", metadata

    def _process_extracted_text(self, text: str) -> str:
        """
        Process and clean extracted text based on configuration.

        Args:
            text: Raw extracted text

        Returns:
            Processed text
        """
        if not text:
            return text

        processed_text = text

        # Remove undefined or non-printable characters (e.g., nulls)
        processed_text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", processed_text)

        # Fix missing spaces between words
        processed_text = self._fix_missing_spaces(processed_text)

        # Handle column layout issues
        processed_text = self._handle_column_layout(processed_text)

        # Remove headers/footers if configured
        if self.remove_headers_footers:
            processed_text = self._remove_headers_footers(processed_text)

        # Handle formatting based on preserve_formatting setting
        if not self.preserve_formatting:
            # Clean up whitespace and line breaks
            processed_text = re.sub(
                r"\n+", "\n", processed_text
            )  # Multiple newlines -> single
            processed_text = re.sub(
                r"[ \t]+", " ", processed_text
            )  # Multiple spaces -> single
            processed_text = processed_text.strip()

        return processed_text

    def _remove_headers_footers(self, text: str) -> str:
        """
        Remove common header/footer patterns from text.

        Args:
            text: Input text

        Returns:
            Text with headers/footers removed
        """
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # Skip common header/footer patterns (basic heuristics)
            if (
                len(line) < 5  # Very short lines
                or re.match(r"^\d+$", line)  # Page numbers only
                or re.match(r"^Page \d+", line, re.IGNORECASE)  # "Page N"
                or line.count("...") > 3  # Dotted lines
                or len(set(line.replace(" ", ""))) <= 2
            ):  # Repeated characters
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _format_text_output(self, text: str, metadata: Dict) -> str:
        """
        Format extracted text according to configured output format.

        Args:
            text: Processed text
            metadata: Extraction metadata

        Returns:
            Formatted text
        """
        if self.text_format == "plain":
            return text

        elif self.text_format == "json":
            output = {
                "extracted_text": text,
                "metadata": metadata if self.include_metadata else {},
            }
            return json.dumps(output, indent=2, ensure_ascii=False)

        elif self.text_format == "markdown":
            # Attempt to add structure to markdown
            formatted_text = self._add_markdown_structure(text)

            if self.include_metadata:
                # Add metadata as YAML frontmatter
                frontmatter = "---\n"
                frontmatter += (
                    f"extraction_method: {metadata.get('method', 'unknown')}\n"
                )
                frontmatter += (
                    f"pages_processed: {metadata.get('pages_processed', 0)}\n"
                )
                frontmatter += f"total_pages: {metadata.get('total_pages', 0)}\n"
                frontmatter += f"extraction_date: {datetime.now().isoformat()}\n"
                if metadata.get("errors"):
                    frontmatter += f"extraction_errors: {len(metadata['errors'])}\n"
                frontmatter += "---\n\n"

                return frontmatter + formatted_text
            else:
                return formatted_text

        return text

    def _add_markdown_structure(self, text: str) -> str:
        """
        Add basic markdown structure to text.

        Args:
            text: Input text

        Returns:
            Text with markdown formatting
        """
        lines = text.split("\n")
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append("")
                continue

            # Detect potential headings (basic heuristics)
            if len(line) < 100 and (  # Not too long
                line.isupper()  # ALL CAPS
                or re.match(
                    r"^\d+\.?\s+[A-Z]", line
                )  # "1. Introduction" or "1 Introduction"
                or re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$", line)
            ):  # Title Case

                # Determine heading level based on context/format
                if line.isupper() or any(
                    keyword in line.lower()
                    for keyword in [
                        "abstract",
                        "introduction",
                        "conclusion",
                        "references",
                    ]
                ):
                    formatted_lines.append(f"## {line}")
                else:
                    formatted_lines.append(f"### {line}")
            else:
                formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def _fix_missing_spaces(self, text: str) -> str:
        """Fix missing spaces between words using heuristics."""
        # Add space before capital letters following lowercase
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

        # Add space after periods followed by capital letters
        text = re.sub(r"\.([A-Z])", r". \1", text)

        # Add space before numbers following letters
        text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)

        return text

    def _handle_column_layout(self, text: str) -> str:
        """Attempt to fix column layout issues."""
        lines = text.split("\n")
        processed_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # If line is very long, it might be concatenated columns
            if len(line) > 150:
                # Try to split on sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line)
                processed_lines.extend(sentences)
            else:
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def _validate_extraction_quality(
        self, extracted_text: str, paper_metadata: Dict
    ) -> Tuple[bool, str]:
        """
        Validate extraction quality by comparing with known abstract.

        Args:
            extracted_text: Extracted text from PDF
            paper_metadata: Paper metadata including original abstract

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.require_abstract_keywords:
            return True, "Quality validation disabled"

        original_abstract = paper_metadata.get("abstract", "")
        if not original_abstract or len(original_abstract) < 50:
            return True, "No original abstract available for comparison"

        # Clean both abstracts for comparison
        def clean_for_comparison(text):
            # Remove LaTeX commands and formatting
            text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)  # \command{content}
            text = re.sub(r"\\[a-zA-Z]+", "", text)  # \command
            text = re.sub(r"\{[^}]*\}", "", text)  # {content}
            text = re.sub(r"\$[^$]*\$", "", text)  # $math$

            # Normalize whitespace and punctuation
            text = re.sub(r"[^\w\s]", " ", text.lower())
            text = re.sub(r"\s+", " ", text.strip())
            return text

        clean_original = clean_for_comparison(original_abstract)
        clean_extracted = clean_for_comparison(extracted_text)

        # Extract key words from original abstract (ignore common words)
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "we",
            "us",
            "our",
            "ours",
            "you",
            "your",
            "yours",
            "they",
            "them",
            "their",
            "theirs",
        }

        original_words = [
            word
            for word in clean_original.split()
            if len(word) > 3 and word not in stop_words
        ]

        if not original_words:
            return True, "No significant words in original abstract"

        # Check how many key words appear in extracted text
        found_words = [word for word in original_words if word in clean_extracted]
        match_ratio = len(found_words) / len(original_words)

        # Require at least 30% of key words to be found
        if match_ratio >= 0.3:
            return (
                True,
                f"Quality validation passed ({match_ratio:.1%} key words found)",
            )
        else:
            return (
                False,
                f"Quality validation failed ({match_ratio:.1%} key words found, minimum 30% required)",
            )

    def _assess_format_quality(self, text: str) -> Tuple[float, str]:
        """Assess the formatting quality of extracted text."""
        total_chars = len(text)
        if total_chars == 0:
            return 0.0, "No text extracted"

        # Check for missing spaces (consecutive letters without spaces)
        missing_space_ratio = len(re.findall(r"[a-z][A-Z]", text)) / max(total_chars, 1)

        # Check for reasonable word lengths
        words = text.split()
        if not words:
            return 0.0, "No words found"

        avg_word_length = sum(len(word) for word in words) / len(words)
        very_long_words = sum(1 for word in words if len(word) > 20)
        long_word_ratio = very_long_words / len(words)

        # Score based on quality indicators (0-1 scale)
        quality_score = 1.0
        quality_score -= min(missing_space_ratio * 10, 0.5)  # Penalize missing spaces
        quality_score -= min(long_word_ratio * 2, 0.3)  # Penalize very long words

        if avg_word_length > 15:  # Abnormally long average word length
            quality_score -= 0.2

        issues = []
        if missing_space_ratio > 0.05:
            issues.append("missing spaces between words")
        if long_word_ratio > 0.1:
            issues.append("abnormally long words")
        if avg_word_length > 15:
            issues.append("high average word length")

        message = f"Quality score: {quality_score:.2f}"
        if issues:
            message += f" (issues: {', '.join(issues)})"

        return quality_score, message

    def _get_output_path(self, pdf_path: Path, paper_metadata: Dict) -> Path:
        """
        Generate output path for extracted text, mirroring PDF structure.

        Args:
            pdf_path: Path to source PDF
            paper_metadata: Paper metadata

        Returns:
            Path where extracted text should be saved
        """
        # Get relative path from PDF storage directory
        pdf_storage_dir = Path(
            self.config.get("pdf_processing", {})
            .get("download", {})
            .get("storage_dir", "./data/pdfs")
        )

        try:
            # Get relative path from PDF storage to this PDF
            relative_path = pdf_path.relative_to(pdf_storage_dir)
        except ValueError:
            # PDF is not under expected storage directory, use filename only
            relative_path = Path(pdf_path.name)

        # Replace .pdf extension with appropriate text extension
        text_extensions = {"markdown": ".md", "plain": ".txt", "json": ".json"}
        text_extension = text_extensions.get(self.text_format, ".txt")
        text_filename = relative_path.with_suffix(text_extension)

        # Create full output path
        output_path = Path(self.text_storage_dir) / text_filename

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return output_path

    def extract_from_pdf(self, pdf_path: Path, paper_metadata: Dict) -> Dict:
        """
        Extract text from a single PDF with fallback methods.

        Args:
            pdf_path: Path to PDF file
            paper_metadata: Paper metadata for quality validation

        Returns:
            Extraction result dictionary
        """
        result = {
            "pdf_path": str(pdf_path),
            "paper_id": paper_metadata.get("id", "unknown"),
            "success": False,
            "method_used": None,
            "extracted_text": "",
            "text_length": 0,
            "output_path": None,
            "quality_valid": False,
            "quality_message": "",
            "extraction_time": 0,
            "metadata": {},
            "errors": [],
        }

        if not self.extract_text or self.method == "none":
            result["success"] = True
            result["method_used"] = "none"
            return result

        if not pdf_path.exists():
            result["errors"].append(f"PDF file not found: {pdf_path}")
            return result

        start_time = datetime.now()
        methods_to_try = self._get_extraction_methods()

        if not methods_to_try:
            result["errors"].append("No extraction methods available")
            return result

        # Try each extraction method
        for method in methods_to_try:
            try:
                self.logger.info(
                    f"Trying extraction method: {method} for {pdf_path.name}"
                )

                if method == "pypdf2":
                    extracted_text, metadata = self._extract_with_pypdf2(pdf_path)
                elif method == "pdfplumber":
                    extracted_text, metadata = self._extract_with_pdfplumber(pdf_path)
                elif method == "pymupdf":
                    extracted_text, metadata = self._extract_with_pymupdf(pdf_path)
                else:
                    continue

                # Update method usage stats
                self.stats["method_usage"][method] = (
                    self.stats["method_usage"].get(method, 0) + 1
                )

                # Check if extraction produced sufficient text
                if len(extracted_text.strip()) < self.min_text_length:
                    error_msg = f"Extracted text too short ({len(extracted_text)} chars) with {method}"
                    result["errors"].append(error_msg)
                    continue

                # Process extracted text
                processed_text = self._process_extracted_text(extracted_text)

                # Validate quality
                quality_valid, quality_message = self._validate_extraction_quality(
                    processed_text, paper_metadata
                )

                # Assess format quality
                format_quality_score, format_quality_message = (
                    self._assess_format_quality(processed_text)
                )

                if format_quality_score < 0.3:  # Adjustable threshold
                    error_msg = f"Format quality too poor ({format_quality_score:.2f}): {format_quality_message}"
                    result["errors"].append(error_msg)
                    continue

                if not quality_valid:
                    result["errors"].append(
                        f"Quality validation failed with {method}: {quality_message}"
                    )
                    continue

                # Success! Format and save the text
                formatted_text = self._format_text_output(processed_text, metadata)

                # Save to file if configured
                output_path = None
                if self.save_extracted_text:
                    output_path = self._get_output_path(pdf_path, paper_metadata)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(formatted_text)

                # Update result
                result.update(
                    {
                        "success": True,
                        "method_used": method,
                        "extracted_text": formatted_text,
                        "text_length": len(processed_text),
                        "output_path": str(output_path) if output_path else None,
                        "quality_valid": quality_valid,
                        "quality_message": quality_message,
                        "format_quality_score": format_quality_score,
                        "format_quality_message": format_quality_message,
                        "metadata": metadata,
                    }
                )

                break  # Success, stop trying other methods

            except Exception as e:
                error_msg = f"Method {method} failed: {str(e)}"
                result["errors"].append(error_msg)
                self.logger.warning(error_msg)
                continue

        # Calculate extraction time
        end_time = datetime.now()
        result["extraction_time"] = (end_time - start_time).total_seconds()

        # Update statistics
        if result["success"]:
            self.stats["successful"] += 1
            self.stats["total_chars"] += result["text_length"]
            if not result["quality_valid"]:
                self.stats["quality_failed"] += 1
        else:
            self.stats["failed"] += 1
            # Track error types
            for error in result["errors"]:
                error_type = error.split(":")[0] if ":" in error else "unknown"
                self.stats["error_types"][error_type] = (
                    self.stats["error_types"].get(error_type, 0) + 1
                )

        self.stats["total_time"] += result["extraction_time"]

        return result


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


def validate_extraction_config(config: Dict) -> List[str]:
    """
    Validate the PDF extraction configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    pdf_config = config.get("pdf_processing", {})
    extraction_config = pdf_config.get("extraction", {})

    # Check method
    method = extraction_config.get("method", "auto")
    if method not in ["auto", "pdfplumber", "pymupdf", "pypdf2", "none"]:
        errors.append(f"Invalid extraction method: {method}")

    # Check fallback methods
    fallback_methods = extraction_config.get("fallback_methods", [])
    valid_methods = ["pdfplumber", "pymupdf", "pypdf2"]
    for method in fallback_methods:
        if method not in valid_methods:
            errors.append(f"Invalid fallback method: {method}")

    # Check numeric parameters
    min_text_length = extraction_config.get("min_text_length", 1000)
    if not isinstance(min_text_length, int) or min_text_length < 0:
        errors.append("min_text_length must be a non-negative integer")

    max_errors = extraction_config.get("max_extraction_errors", 5)
    if not isinstance(max_errors, int) or max_errors < 0:
        errors.append("max_extraction_errors must be a non-negative integer")

    # Check text format
    text_format = extraction_config.get("text_format", "markdown")
    if text_format not in ["markdown", "plain", "json"]:
        errors.append(f"Invalid text_format: {text_format}")

    # Check storage directory
    text_storage_dir = extraction_config.get(
        "text_storage_dir", "./data/extracted_text"
    )
    if not isinstance(text_storage_dir, str) or not text_storage_dir.strip():
        errors.append("text_storage_dir must be a non-empty string")

    return errors


def test_extraction_utilities():
    """Test extraction utilities with mock setup."""
    print("=== TESTING EXTRACTION UTILITIES ===")

    # Check available PDF libraries
    available = [lib for lib, module in PDF_LIBRARIES.items() if module is not None]
    unavailable = [lib for lib, module in PDF_LIBRARIES.items() if module is None]

    print(f"Available PDF libraries: {', '.join(available) if available else 'None'}")
    if unavailable:
        print(f"Missing PDF libraries: {', '.join(unavailable)}")
        print("Install with: pip install " + " ".join(unavailable))

    if not available:
        print("❌ No PDF extraction libraries available")
        return

    # Test configuration
    test_config = {
        "pdf_processing": {
            "extraction": {
                "method": "auto",
                "fallback_methods": ["pdfplumber", "pymupdf", "pypdf2"],
                "extract_text": True,
                "preserve_formatting": False,
                "remove_headers_footers": True,
                "min_text_length": 500,  # Lower for testing
                "max_extraction_errors": 3,
                "require_abstract_keywords": False,  # Disable for testing
                "text_format": "markdown",
                "include_metadata": True,
                "save_extracted_text": False,  # Don't save during testing
                "text_storage_dir": "./data/test_extracted_text",
            }
        }
    }

    try:
        # Test extractor initialization
        extractor = TextExtractor(test_config)
        print("✅ TextExtractor initialized successfully")
        print(f"   Available methods: {', '.join(extractor.available_methods)}")
        print(f"   Primary method: {extractor.method}")

        # Test method selection
        methods = extractor._get_extraction_methods()
        print(f"✅ Method selection working: {methods}")

        # Test text processing
        test_text = "This is a test.\n\n\nMultiple   spaces    and\n\n\nnewlines."
        processed = extractor._process_extracted_text(test_text)
        print(f"✅ Text processing working")
        print(f"   Input: {repr(test_text[:50])}")
        print(f"   Output: {repr(processed[:50])}")

        # Test quality validation
        paper_metadata = {
            "abstract": "This paper presents a novel approach to machine learning using deep neural networks."
        }
        extracted_text = "In this work, we present a novel approach using machine learning and neural networks for classification."

        is_valid, message = extractor._validate_extraction_quality(
            extracted_text, paper_metadata
        )
        print(f"✅ Quality validation working: {is_valid} - {message}")

        # Test output formatting
        test_metadata = {
            "method": "test",
            "pages_processed": 5,
            "total_pages": 5,
            "errors": [],
        }
        formatted = extractor._format_text_output(
            "# Test Document\n\nThis is a test.", test_metadata
        )
        print(f"✅ Output formatting working (markdown with metadata)")

        print(f"\n✅ All extraction utilities tests passed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


def test_real_extraction_config():
    """Test extraction configuration validation with real config file."""
    print("=== TESTING EXTRACTION CONFIGURATION ===")

    # Load real configuration
    try:
        config = load_config()
        if not config:
            print("❌ No configuration loaded")
            return

        print("✅ Configuration loaded successfully")

        # Validate configuration
        validation_errors = validate_extraction_config(config)
        if validation_errors:
            print("❌ Configuration validation errors:")
            for error in validation_errors:
                print(f"   - {error}")
        else:
            print("✅ Configuration validation passed")

        # Show extraction configuration
        extraction_config = config.get("pdf_processing", {}).get("extraction", {})
        if extraction_config:
            print(f"Extraction configuration:")
            print(f"  Method: {extraction_config.get('method', 'not set')}")
            print(
                f"  Fallback methods: {extraction_config.get('fallback_methods', 'not set')}"
            )
            print(f"  Text format: {extraction_config.get('text_format', 'not set')}")
            print(
                f"  Min text length: {extraction_config.get('min_text_length', 'not set')}"
            )
            print(
                f"  Storage directory: {extraction_config.get('text_storage_dir', 'not set')}"
            )
            print(
                f"  Quality validation: {extraction_config.get('require_abstract_keywords', 'not set')}"
            )
        else:
            print("⚠️  No extraction configuration found")
            print("   Add pdf_processing.extraction section to config.yaml")

    except Exception as e:
        print(f"❌ Test failed: {e}")


def test_extraction_methods_comparison():
    """Test all available extraction methods on a real PDF and compare results."""
    print("=== COMPARING EXTRACTION METHODS ON REAL PDF ===")

    # Find a PDF file to test with
    import glob

    pdf_storage_dir = "./data/test_pdfs"
    pdf_files = glob.glob(os.path.join(pdf_storage_dir, "**/*.pdf"), recursive=True)

    if not pdf_files:
        print("❌ No PDF files found for testing")
        print(f"   Expected PDFs in: {pdf_storage_dir}")
        print("   Run download_papers.py first to get test PDFs")
        return

    test_pdf = Path(pdf_files[0])  # Use first PDF found
    print(f"Testing extraction methods on: {test_pdf.name}")
    print()

    # Test configuration for each method
    methods_to_test = ["pypdf2", "pdfplumber", "pymupdf"]
    available_methods = [
        lib for lib, module in PDF_LIBRARIES.items() if module is not None
    ]

    results = {}

    for method in methods_to_test:
        if method not in available_methods:
            print(f"⏭️  Skipping {method} (not installed)")
            continue

        print(f"🔄 Testing {method}...")

        # Create test config for this method
        test_config = {
            "pdf_processing": {
                "extraction": {
                    "method": method,
                    "extract_text": True,
                    "preserve_formatting": False,
                    "remove_headers_footers": True,
                    "min_text_length": 100,
                    "max_extraction_errors": 10,
                    "require_abstract_keywords": False,  # Skip quality validation
                    "text_format": "plain",
                    "include_metadata": True,
                    "save_extracted_text": False,
                }
            }
        }

        try:
            # Create extractor and extract text
            extractor = TextExtractor(test_config)
            paper_metadata = {"abstract": "test"}  # Dummy metadata
            result = extractor.extract_from_pdf(test_pdf, paper_metadata)

            if result["success"]:
                extracted_text = result["extracted_text"]

                # Assess format quality
                quality_score, quality_message = extractor._assess_format_quality(
                    extracted_text
                )

                results[method] = {
                    "success": True,
                    "text": extracted_text,
                    "length": len(extracted_text),
                    "quality_score": quality_score,
                    "quality_message": quality_message,
                    "extraction_time": result["extraction_time"],
                    "pages_processed": result["metadata"].get("pages_processed", 0),
                }

                print(
                    f"✅ {method}: {len(extracted_text):,} chars, {quality_score:.2f} quality, {result['extraction_time']:.1f}s"
                )

            else:
                results[method] = {"success": False, "errors": result["errors"]}
                print(f"❌ {method}: Failed - {'; '.join(result['errors'][:2])}")

        except Exception as e:
            results[method] = {"success": False, "errors": [str(e)]}
            print(f"❌ {method}: Exception - {e}")

    # Compare results
    print(f"\n=== COMPARISON RESULTS ===")
    successful_results = {k: v for k, v in results.items() if v.get("success")}

    if not successful_results:
        print("❌ No methods succeeded")
        return

    # Show quality ranking
    quality_ranking = sorted(
        successful_results.items(),
        key=lambda x: x[1].get("quality_score", 0),
        reverse=True,
    )

    print(f"Quality ranking:")
    for i, (method, result) in enumerate(quality_ranking, 1):
        score = result.get("quality_score", 0)
        chars = result.get("length", 0)
        time_taken = result.get("extraction_time", 0)
        print(
            f"  {i}. {method}: {score:.2f} quality, {chars:,} chars, {time_taken:.1f}s"
        )

    # Show text previews
    print(f"\n=== TEXT PREVIEWS (first 300 chars) ===")
    for method, result in quality_ranking:
        if result.get("success") and result.get("text"):
            preview = result["text"][:300].replace("\n", " ").strip()
            print(f"\n{method.upper()}:")
            print(f"  {preview}...")

            # Check for obvious formatting issues
            issues = []
            if len(re.findall(r"[a-z][A-Z]", preview)) > 5:
                issues.append("missing spaces")
            if any(len(word) > 25 for word in preview.split()):
                issues.append("very long words")

            if issues:
                print(f"  ⚠️  Potential issues: {', '.join(issues)}")

    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    if quality_ranking:
        best_method = quality_ranking[0][0]
        best_score = quality_ranking[0][1].get("quality_score", 0)

        if best_score > 0.8:
            print(f"✅ Recommended: Use '{best_method}' method (high quality)")
        elif best_score > 0.6:
            print(f"⚠️  Recommended: Use '{best_method}' method (acceptable quality)")
            print(f"   Consider text post-processing improvements")
        else:
            print(f"❌ All methods show quality issues")
            print(f"   Best available: '{best_method}'")
            print(f"   Consider manual text processing or different PDFs")

        print(f"\nTo use the best method, set in config.yaml:")
        print(f"  extraction:")
        print(f'    method: "{best_method}"')


# Example usage and testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] in ["--test", "-t"]:
            test_extraction_utilities()
        elif sys.argv[1] in ["--test-real", "--real", "-r"]:
            test_real_extraction_config()
        elif sys.argv[1] in ["--compare", "--comp", "-c"]:
            test_extraction_methods_comparison()  # NEW
        elif sys.argv[1] in ["--help", "-h"]:
            print("Text Extraction Utilities")
            print("Usage:")
            print(
                "  python extraction_utils.py --test       # Test extraction functionality"
            )
            print(
                "  python extraction_utils.py --test-real  # Test with real configuration"
            )
            print(
                "  python extraction_utils.py --compare    # Compare extraction methods"
            )
            print("  python extraction_utils.py --help       # Show this help")
        else:
            print("Unknown option. Use --help for usage information.")
    else:
        print("Text Extraction Utilities Module")
        print("Run with --help for usage options")

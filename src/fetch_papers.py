#!/usr/bin/env python3
"""
ArXiv Paper Fetcher with Legacy Category Filtering

A modular script to fetch recent submissions from arXiv and save them to structured files.
This version filters out legacy categories and only keeps papers with modern arXiv subject-class categories.

Dependencies:
    pip install feedparser requests pyyaml

Usage:
    python arxiv_fetcher.py
    
Configuration:
    Edit the CONFIG dictionary below to customize categories, output format, etc.
"""

import feedparser
import json
import csv
import yaml
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import re
import time
import os
from pathlib import Path
from urllib.parse import urlencode


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
        print("Creating default config file...")
        
        # Create default config if file doesn't exist
        default_config = {
            "arxiv": {
                "categories": ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"],
                "days_back": 7,
                "max_papers_per_category": 100,
                "request_delay": 1.0,
                "arxiv_days_only": True,
                "filter_legacy_categories": True,
                "require_modern_categories": True
            },
            "output": {
                "format": "both",
                "base_dir": "./data"
            }
        }
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        return default_config


def is_modern_arxiv_category(category: str) -> bool:
    """
    Check if a category follows the modern arXiv subject-class format.
    
    Modern categories follow the pattern: subject-class.subclass
    Examples: cs.AI, math.CO, physics.data-an, q-bio.NC
    
    Args:
        category: Category string to check
        
    Returns:
        True if category is in modern format, False otherwise
    """
    # Pattern for modern arXiv categories: subject-class.subclass
    # subject-class can include letters, numbers, and hyphens
    # subclass can include letters, numbers, and hyphens
    pattern = r'^[a-z-]+\.[A-Za-z-]+$'
    
    return bool(re.match(pattern, category.strip()))


def filter_legacy_categories(categories: List[str]) -> List[str]:
    """
    Filter out legacy categories, keeping only modern arXiv subject-class categories.
    
    Args:
        categories: List of category strings
        
    Returns:
        List of modern categories only
    """
    return [cat for cat in categories if is_modern_arxiv_category(cat)]


def has_modern_categories(categories: List[str]) -> bool:
    """
    Check if a paper has at least one modern arXiv category.
    
    Args:
        categories: List of category strings
        
    Returns:
        True if at least one category is modern format
    """
    return any(is_modern_arxiv_category(cat) for cat in categories)


def get_known_arxiv_subjects() -> Set[str]:
    """
    Return a set of known arXiv subject classes for validation.
    
    Returns:
        Set of valid subject-class prefixes
    """
    return {
        'astro-ph',  # Astrophysics
        'cond-mat',  # Condensed Matter
        'cs',        # Computer Science
        'econ',      # Economics
        'eess',      # Electrical Engineering and Systems Science
        'gr-qc',     # General Relativity and Quantum Cosmology
        'hep-ex',    # High Energy Physics - Experiment
        'hep-lat',   # High Energy Physics - Lattice
        'hep-ph',    # High Energy Physics - Phenomenology
        'hep-th',    # High Energy Physics - Theory
        'math',      # Mathematics
        'math-ph',   # Mathematical Physics
        'nlin',      # Nonlinear Sciences
        'nucl-ex',   # Nuclear Experiment
        'nucl-th',   # Nuclear Theory
        'physics',   # Physics
        'q-bio',     # Quantitative Biology
        'q-fin',     # Quantitative Finance
        'quant-ph',  # Quantum Physics
        'stat'       # Statistics
    }


def is_valid_arxiv_category(category: str) -> bool:
    """
    Check if a category is both modern format and from a known arXiv subject.
    
    Args:
        category: Category string to validate
        
    Returns:
        True if category is valid modern arXiv category
    """
    if not is_modern_arxiv_category(category):
        return False
    
    subject = category.split('.')[0]
    return subject in get_known_arxiv_subjects()


class ArxivFetcher:
    """
    A class to fetch and process arXiv papers using the arXiv API.
    
    The arXiv API uses the ATOM format and allows querying by category,
    date range, and other parameters.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the ArxivFetcher.
        
        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.base_url = "http://export.arxiv.org/api/query"
        self.papers = []
        self.filtered_papers_count = 0  # Track how many papers were filtered out
        
        # Create output directory if it doesn't exist
        output_dir = self.config.get("output", {}).get("base_dir", "./data")
        os.makedirs(output_dir, exist_ok=True)
    
    def find_arxiv_date_range_with_papers(self, days_back: int, category: str = None) -> tuple:
        """
        Find a date range that actually contains papers by shifting backwards if needed.
        
        Args:
            days_back: Desired number of days to look back
            category: ArXiv category to test (None for all categories)
            
        Returns:
            Tuple of (start_date, end_date, days_shifted) where days_shifted 
            indicates how many extra days we had to go back
        """
        max_attempts = 10  # Prevent infinite loops
        days_shifted = 0
        
        for attempt in range(max_attempts):
            # Calculate date range: shift start back by days_shifted, keep same window size
            end_date = datetime.now() - timedelta(days=days_shifted)
            start_date = end_date - timedelta(days=days_back)
            
            # Test if this date range has papers
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            
            if category:
                test_query = f"(cat:{category}) AND submittedDate:[{start_str}* TO {end_str}*]"
            else:
                test_query = f"submittedDate:[{start_str}* TO {end_str}*]"
            
            # Quick test with max_results=1 to see if any papers exist
            params = {
                "search_query": test_query,
                "start": 0,
                "max_results": 1,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            test_url = f"{self.base_url}?{urlencode(params)}"
            test_feed = feedparser.parse(test_url)
            
            if len(test_feed.entries) > 0:
                # Found papers! Return this date range
                if days_shifted > 0:
                    print(f"  → Shifted search window back {days_shifted} days to find papers")
                    print(f"  → Using date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                return start_date, end_date, days_shifted
            
            # No papers found, shift back one more day
            days_shifted += 1
            if attempt < max_attempts - 1:
                print(f"  → No papers found in current range, shifting back 1 day...")
        
        # If we get here, couldn't find papers even after max_attempts
        print(f"  → Warning: No papers found even after shifting back {days_shifted} days")
        end_date = datetime.now() - timedelta(days=days_shifted)
        start_date = end_date - timedelta(days=days_back)
        return start_date, end_date, days_shifted

    def calculate_arxiv_date_range(self, days_back: int, arxiv_days_only: bool = True, category: str = None) -> tuple:
        """
        Calculate the date range for arXiv queries, with smart fallback when arxiv_days_only is True.
        
        Args:
            days_back: Number of days to look back
            arxiv_days_only: If True, use smart fallback to find actual papers
            category: ArXiv category for testing (when using smart fallback)
            
        Returns:
            Tuple of (start_date, end_date) as datetime objects
        """
        if not arxiv_days_only:
            # Use calendar days (original behavior)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            return start_date, end_date
        
        # Use smart fallback to find date range with actual papers
        start_date, end_date, days_shifted = self.find_arxiv_date_range_with_papers(days_back, category)
        return start_date, end_date
    
    def build_query(self, category: str = None) -> str:
        """
        Build the search query for the arXiv API.
        
        Args:
            category: ArXiv category (e.g., 'cs.AI'). If None, searches all categories.
            
        Returns:
            Formatted query string for the arXiv API
        """
        if category:
            query = f"cat:{category}"
        else:
            query = "all"
            
        # Add date filtering if specified
        days_back = self.config.get("arxiv", {}).get("days_back", 0)
        if days_back > 0:
            arxiv_days_only = self.config.get("arxiv", {}).get("arxiv_days_only", False)
            start_date, end_date = self.calculate_arxiv_date_range(days_back, arxiv_days_only, category)
            
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            
            date_filter = f"submittedDate:[{start_str}* TO {end_str}*]"
            
            if category:
                query = f"({query}) AND {date_filter}"
            else:
                query = date_filter
                
        return query
    
    def process_paper_categories(self, paper: Dict) -> Dict:
        """
        Process and filter categories for a paper based on configuration.
        
        Args:
            paper: Paper dictionary with categories
            
        Returns:
            Updated paper dictionary with filtered categories
        """
        original_categories = paper.get('categories', [])
        
        # Configuration options
        filter_legacy = self.config.get("arxiv", {}).get("filter_legacy_categories", True)
        require_modern = self.config.get("arxiv", {}).get("require_modern_categories", True)
        
        if filter_legacy:
            # Filter out legacy categories
            modern_categories = filter_legacy_categories(original_categories)
            paper['categories'] = modern_categories
            
            # Track filtering statistics
            if len(modern_categories) < len(original_categories):
                paper['filtered_categories'] = list(set(original_categories) - set(modern_categories))
        
        return paper
    
    def should_include_paper(self, paper: Dict) -> bool:
        """
        Determine if a paper should be included based on category filtering rules.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            True if paper should be included, False otherwise
        """
        require_modern = self.config.get("arxiv", {}).get("require_modern_categories", True)
        
        if require_modern:
            # Only include papers that have at least one modern category
            return len(paper.get('categories', [])) > 0
        
        return True
    
    def fetch_papers(self, category: str = None) -> List[Dict]:
        """
        Fetch papers from arXiv for a given category.
        
        Args:
            category: ArXiv category to search (None for all categories)
            
        Returns:
            List of dictionaries containing paper information
        """
        query = self.build_query(category)
        max_results = self.config.get("arxiv", {}).get("max_papers_per_category", 100)
        
        # Properly encode URL parameters
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        # Construct the API URL with proper encoding
        url = f"{self.base_url}?{urlencode(params)}"
        
        print(f"Fetching papers for category: {category or 'all'}")
        print(f"Query: {query}")
        
        # Show date range for debugging
        days_back = self.config.get("arxiv", {}).get("days_back", 0)
        if days_back > 0:
            arxiv_days_only = self.config.get("arxiv", {}).get("arxiv_days_only", False)
            start_date, end_date = self.calculate_arxiv_date_range(days_back, arxiv_days_only, category)
            search_type = "smart search (with fallback)" if arxiv_days_only else "calendar days"
            print(f"Date range ({days_back} days, {search_type}): {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Parse the ATOM feed
        try:
            feed = feedparser.parse(url)
            
            if feed.bozo:
                print(f"Warning: Feed parsing had issues for {category}")
                
            papers = []
            filtered_count = 0
            
            for entry in feed.entries:
                # Extract arXiv ID from the entry ID
                arxiv_id = entry.id.split('/')[-1]
                
                # Parse authors
                authors = []
                if hasattr(entry, 'authors'):
                    authors = [author.name for author in entry.authors]
                elif hasattr(entry, 'author'):
                    authors = [entry.author]
                
                # Parse categories
                categories = []
                if hasattr(entry, 'tags'):
                    categories = [tag.term for tag in entry.tags]
                
                # Find PDF link
                pdf_url = None
                abs_url = entry.link
                
                if hasattr(entry, 'links'):
                    for link in entry.links:
                        if link.type == 'application/pdf':
                            pdf_url = link.href
                        elif link.type == 'text/html':
                            abs_url = link.href
                
                # Construct PDF URL if not found
                if not pdf_url:
                    pdf_url = f"http://arxiv.org/pdf/{arxiv_id}.pdf"
                
                paper = {
                    "id": arxiv_id,
                    "title": entry.title.replace('\n', ' ').strip(),
                    "authors": authors,
                    "abstract": entry.summary.replace('\n', ' ').strip(),
                    "categories": categories,
                    "published": entry.published if hasattr(entry, 'published') else None,
                    "updated": entry.updated if hasattr(entry, 'updated') else None,
                    "pdf_url": pdf_url,
                    "abs_url": abs_url,
                    "fetched_category": category
                }
                
                # Process categories (filter legacy if configured)
                paper = self.process_paper_categories(paper)
                
                # Check if paper should be included
                if self.should_include_paper(paper):
                    papers.append(paper)
                else:
                    filtered_count += 1
            
            print(f"Found {len(papers)} papers for category {category or 'all'}")
            if filtered_count > 0:
                print(f"  → Filtered out {filtered_count} papers with only legacy categories")
            
            self.filtered_papers_count += filtered_count
            return papers
            
        except Exception as e:
            print(f"Error fetching papers for category {category}: {e}")
            return []
    
    def fetch_all_categories(self) -> List[Dict]:
        """
        Fetch papers for all specified categories.
        
        Returns:
            Combined list of all papers from all categories
        """
        all_papers = []
        categories = self.config.get("arxiv", {}).get("categories", [])
        request_delay = self.config.get("arxiv", {}).get("request_delay", 1.0)
        
        if not categories:
            papers = self.fetch_papers(None)
            all_papers.extend(papers)
        else:
            for i, category in enumerate(categories):
                papers = self.fetch_papers(category)
                all_papers.extend(papers)
                
                if i < len(categories) - 1:
                    time.sleep(request_delay)
        
        # Remove duplicates
        seen_ids = set()
        unique_papers = []
        
        for paper in all_papers:
            if paper["id"] not in seen_ids:
                seen_ids.add(paper["id"])
                unique_papers.append(paper)
        
        self.papers = unique_papers
        print(f"Total unique papers found: {len(unique_papers)}")
        if self.filtered_papers_count > 0:
            print(f"Total papers filtered out: {self.filtered_papers_count}")
        return unique_papers
    
    def save_to_json(self, filename: str = None) -> str:
        """Save papers to a JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arxiv_papers_{timestamp}.json"
        
        output_dir = self.config.get("output", {}).get("base_dir", "./data")
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.papers, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.papers)} papers to {filepath}")
        return filepath
    
    def save_to_csv(self, filename: str = None) -> str:
        """Save papers to a CSV file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arxiv_papers_{timestamp}.csv"
        
        output_dir = self.config.get("output", {}).get("base_dir", "./data")
        filepath = os.path.join(output_dir, filename)
        
        if not self.papers:
            print("No papers to save")
            return filepath
        
        fieldnames = [
            'id', 'title', 'authors', 'abstract', 'categories', 
            'published', 'updated', 'pdf_url', 'abs_url', 'fetched_category'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for paper in self.papers:
                row = paper.copy()
                row['authors'] = '; '.join(paper['authors'])
                row['categories'] = '; '.join(paper['categories'])
                # Don't include filtered_categories in CSV output
                if 'filtered_categories' in row:
                    del row['filtered_categories']
                writer.writerow(row)
        
        print(f"Saved {len(self.papers)} papers to {filepath}")
        return filepath
    
    def save_papers(self) -> Dict[str, str]:
        """Save papers according to the configured output format."""
        filepaths = {}
        output_format = self.config.get("output", {}).get("format", "both")
        
        if output_format in ["json", "both"]:
            filepaths["json"] = self.save_to_json()
        
        if output_format in ["csv", "both"]:
            filepaths["csv"] = self.save_to_csv()
        
        return filepaths
    
    def print_summary(self):
        """Print a summary of the fetched papers."""
        if not self.papers:
            print("No papers fetched.")
            return
        
        print(f"\n=== SUMMARY ===")
        print(f"Total papers: {len(self.papers)}")
        
        if self.filtered_papers_count > 0:
            print(f"Papers filtered out (legacy categories only): {self.filtered_papers_count}")
        
        # Count by category
        category_counts = {}
        for paper in self.papers:
            for cat in paper['categories']:
                category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print(f"Categories represented ({len(category_counts)} unique):")
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat}: {count} papers")
        
        # Show validation of categories
        print(f"\nCategory validation:")
        legacy_categories = set()
        invalid_categories = set()
        
        for paper in self.papers:
            for cat in paper['categories']:
                if not is_modern_arxiv_category(cat):
                    legacy_categories.add(cat)
                elif not is_valid_arxiv_category(cat):
                    invalid_categories.add(cat)
        
        if legacy_categories:
            print(f"  Legacy categories found: {len(legacy_categories)}")
            print(f"    {', '.join(sorted(legacy_categories))}")
        
        if invalid_categories:
            print(f"  Invalid/unknown categories found: {len(invalid_categories)}")
            print(f"    {', '.join(sorted(invalid_categories))}")
        
        if not legacy_categories and not invalid_categories:
            print(f"  ✓ All categories are valid modern arXiv categories")
        
        # Show a few example titles
        print(f"\nExample titles:")
        for i, paper in enumerate(self.papers[:3]):
            print(f"  {i+1}. {paper['title']}")


def main():
    """Main function to run the arXiv fetcher."""
    print("ArXiv Paper Fetcher (Enhanced with Legacy Category Filtering)")
    print("============================================================")
    
    # Load configuration
    config = load_config()
    
    # Show configuration
    filter_legacy = config.get("arxiv", {}).get("filter_legacy_categories", True)
    require_modern = config.get("arxiv", {}).get("require_modern_categories", True)
    
    print(f"Configuration:")
    print(f"  Filter legacy categories: {filter_legacy}")
    print(f"  Require modern categories: {require_modern}")
    print()
    
    # Create fetcher instance
    fetcher = ArxivFetcher(config)
    
    # Fetch papers
    papers = fetcher.fetch_all_categories()
    
    # Save papers
    if papers:
        filepaths = fetcher.save_papers()
        fetcher.print_summary()
        
        print(f"\nFiles saved:")
        for format_type, filepath in filepaths.items():
            print(f"  {format_type.upper()}: {filepath}")
    else:
        print("No papers found with the current configuration.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
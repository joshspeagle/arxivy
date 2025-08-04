#!/usr/bin/env python3
"""
Paper Selection Utilities

Implements sophisticated paper selection algorithms for PDF processing.
Supports percentile-based selection with score thresholds, min/max constraints,
and temperature-based random sampling for exploration.

Usage:
    from selection_utils import PaperSelector

    selector = PaperSelector(config)
    selected_papers = selector.select_papers(scored_papers)
"""

import numpy as np
import yaml
import os
from typing import List, Dict, Tuple
from datetime import datetime
import json


class PaperSelector:
    """
    Sophisticated paper selection engine with multiple criteria and constraints.

    Implements a multi-stage selection process:
    1. Base selection using percentile and score thresholds
    2. Constraint adjustment to meet min/max paper requirements
    3. Optional random paper addition with temperature-based sampling
    """

    def __init__(self, config: Dict):
        """
        Initialize the paper selector.

        Args:
            config: Configuration dictionary containing pdf_processing settings
        """
        self.config = config
        self.selection_config = config.get("pdf_processing", {}).get("selection", {})

        # Selection parameters
        self.percentile = self.selection_config.get("percentile", 20)
        self.score_threshold = self.selection_config.get("score_threshold", 7.0)
        self.min_papers = self.selection_config.get("min_papers", 5)
        self.max_papers = self.selection_config.get("max_papers", 25)

        # Random sampling parameters
        self.n_random = self.selection_config.get("n_random", 5)
        self.temperature = self.selection_config.get("temperature", 0.3)
        self.random_seed = self.selection_config.get("random_seed", None)

        # Validation
        self._validate_config()

        # Statistics tracking
        self.selection_stats = {}

    def _validate_config(self):
        """Validate selection configuration parameters."""
        if not 0 <= self.percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")

        if self.min_papers < 0:
            raise ValueError("min_papers must be non-negative")

        if self.max_papers < self.min_papers:
            raise ValueError("max_papers must be >= min_papers")

        if self.n_random < 0:
            raise ValueError("n_random must be non-negative")

        if self.temperature is not None and self.temperature < 0:
            raise ValueError(
                "temperature must be positive or None (for uniform sampling)"
            )

        if self.random_seed is not None:
            if not isinstance(self.random_seed, int):
                raise ValueError("random_seed must be an integer")
            np.random.seed(self.random_seed)

    def _calculate_percentile_threshold(self, scores: List[float]) -> float:
        """
        Calculate the score threshold for a given percentile.

        Args:
            scores: List of scores (sorted in descending order)

        Returns:
            Score threshold for the percentile
        """
        if not scores:
            return float("-inf")

        # Convert percentile to index (top X% means (100-X) percentile)
        percentile_rank = 100 - self.percentile
        threshold_score = np.percentile(scores, percentile_rank)

        return threshold_score

    def _apply_base_filters(self, papers: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Apply base selection filters (percentile AND score threshold).

        Args:
            papers: List of papers with scores

        Returns:
            Tuple of (selected_papers, filter_stats)
        """
        # Filter out papers without scores
        valid_papers = [p for p in papers if p.get("llm_score") is not None]

        if not valid_papers:
            return [], {"error": "No papers with valid scores found"}

        # Sort by score (descending)
        sorted_papers = sorted(valid_papers, key=lambda x: x["llm_score"], reverse=True)
        scores = [p["llm_score"] for p in sorted_papers]

        # Calculate percentile threshold
        percentile_threshold = self._calculate_percentile_threshold(scores)

        # Apply both filters (AND logic)
        selected_papers = []
        for paper in sorted_papers:
            score = paper["llm_score"]
            if score >= percentile_threshold and score >= self.score_threshold:
                selected_papers.append(paper)

        filter_stats = {
            "total_papers": len(papers),
            "valid_papers": len(valid_papers),
            "percentile_threshold": percentile_threshold,
            "score_threshold": self.score_threshold,
            "base_selected": len(selected_papers),
            "score_range": (
                f"{min(scores):.1f} - {max(scores):.1f}" if scores else "N/A"
            ),
        }

        return selected_papers, filter_stats

    def _adjust_for_constraints(
        self, papers: List[Dict], selected_papers: List[Dict]
    ) -> Tuple[List[Dict], Dict]:
        """
        Adjust selection to meet min/max paper constraints.

        Args:
            papers: All valid papers (sorted by score, descending)
            selected_papers: Papers selected by base filters

        Returns:
            Tuple of (adjusted_selected_papers, adjustment_stats)
        """
        n_selected = len(selected_papers)
        adjustment_stats = {"original_count": n_selected, "adjustments_made": []}

        # If we're within bounds, no adjustment needed
        if self.min_papers <= n_selected <= self.max_papers:
            adjustment_stats["final_count"] = n_selected
            adjustment_stats["adjustments_made"].append("No adjustment needed")
            return selected_papers, adjustment_stats

        # Sort all papers by score for constraint adjustment
        all_sorted = sorted(
            [p for p in papers if p.get("llm_score") is not None],
            key=lambda x: x["llm_score"],
            reverse=True,
        )

        if n_selected < self.min_papers:
            # Need more papers - relax constraints
            needed = self.min_papers - n_selected

            # Strategy: Take top papers that weren't selected by base filters
            selected_ids = {p["id"] for p in selected_papers}
            additional_papers = []

            for paper in all_sorted:
                if paper["id"] not in selected_ids:
                    additional_papers.append(paper)
                    if len(additional_papers) >= needed:
                        break

            selected_papers.extend(additional_papers)
            adjustment_stats["adjustments_made"].append(
                f"Added {len(additional_papers)} papers to reach min_papers"
            )

        elif n_selected > self.max_papers:
            # Too many papers - keep only the top ones
            selected_papers = selected_papers[: self.max_papers]
            removed = n_selected - self.max_papers
            adjustment_stats["adjustments_made"].append(
                f"Removed {removed} papers to meet max_papers"
            )

        adjustment_stats["final_count"] = len(selected_papers)
        return selected_papers, adjustment_stats

    def _temperature_sample(self, papers: List[Dict], n_samples: int) -> List[Dict]:
        """
        Sample papers using temperature-based probability weighting.

        Args:
            papers: List of papers to sample from (with scores)
            n_samples: Number of papers to sample

        Returns:
            List of sampled papers
        """
        if not papers or n_samples <= 0:
            return []

        n_samples = min(n_samples, len(papers))

        if self.temperature is None:
            # Uniform sampling (temperature = infinity)
            indices = np.random.choice(len(papers), size=n_samples, replace=False)
            return [papers[i] for i in indices]

        # Extract scores
        scores = np.array([p["llm_score"] for p in papers])

        if self.temperature == 0:
            # Deterministic top-N sampling (temperature = 0)
            sorted_papers = sorted(papers, key=lambda x: x["llm_score"], reverse=True)
            return sorted_papers[:n_samples]

        # Temperature-weighted sampling
        # Apply temperature scaling
        scaled_scores = scores / self.temperature

        # Convert to probabilities (softmax)
        # Subtract max for numerical stability
        scaled_scores = scaled_scores - np.max(scaled_scores)
        exp_scores = np.exp(scaled_scores)
        probabilities = exp_scores / np.sum(exp_scores)

        # Sample without replacement
        sampled_indices = np.random.choice(
            len(papers), size=n_samples, replace=False, p=probabilities
        )

        return [papers[i] for i in sampled_indices]

    def _add_random_papers(
        self, all_papers: List[Dict], selected_papers: List[Dict]
    ) -> Tuple[List[Dict], Dict]:
        """
        Add random papers using temperature-based sampling.

        Args:
            all_papers: All papers with scores
            selected_papers: Already selected papers

        Returns:
            Tuple of (final_selected_papers, random_stats)
        """
        if self.n_random <= 0:
            return selected_papers, {"random_papers_added": 0}

        # Find papers not already selected
        selected_ids = {p["id"] for p in selected_papers}
        remaining_papers = [
            p
            for p in all_papers
            if p["id"] not in selected_ids and p.get("llm_score") is not None
        ]

        if not remaining_papers:
            return selected_papers, {
                "random_papers_added": 0,
                "note": "No remaining papers available for random selection",
            }

        # Sample random papers
        n_to_sample = min(self.n_random, len(remaining_papers))
        random_papers = self._temperature_sample(remaining_papers, n_to_sample)

        # Add to selected papers
        final_selected = selected_papers + random_papers

        random_stats = {
            "random_papers_requested": self.n_random,
            "random_papers_available": len(remaining_papers),
            "random_papers_added": len(random_papers),
            "temperature": self.temperature,
            "random_score_range": (
                f"{min(p['llm_score'] for p in random_papers):.1f} - {max(p['llm_score'] for p in random_papers):.1f}"
                if random_papers
                else "N/A"
            ),
        }

        return final_selected, random_stats

    def select_papers(self, papers: List[Dict]) -> Dict:
        """
        Main paper selection method implementing the full algorithm.

        Args:
            papers: List of scored papers

        Returns:
            Dictionary containing selected papers and detailed statistics
        """
        print(f"Starting paper selection...")
        print(
            f"Selection criteria: top {self.percentile}% AND score >= {self.score_threshold}"
        )
        print(f"Constraints: {self.min_papers}-{self.max_papers} papers")
        if self.n_random > 0:
            temp_str = (
                "uniform" if self.temperature is None else f"T={self.temperature}"
            )
            print(f"Random addition: {self.n_random} papers ({temp_str} sampling)")
        print()

        # Stage 1: Apply base filters
        print("Stage 1: Applying base filters (percentile AND score threshold)...")
        base_selected, filter_stats = self._apply_base_filters(papers)
        print(f"  Base selection: {filter_stats['base_selected']} papers")
        print(f"  Percentile threshold: {filter_stats['percentile_threshold']:.2f}")
        print(f"  Score range in dataset: {filter_stats['score_range']}")

        # Stage 2: Adjust for constraints
        print(f"\nStage 2: Adjusting for min/max constraints...")
        valid_papers = [p for p in papers if p.get("llm_score") is not None]
        constrained_selected, adjustment_stats = self._adjust_for_constraints(
            valid_papers, base_selected
        )
        print(f"  After constraints: {adjustment_stats['final_count']} papers")
        for adjustment in adjustment_stats["adjustments_made"]:
            print(f"  - {adjustment}")

        # Stage 3: Add random papers
        print(f"\nStage 3: Adding random papers...")
        final_selected, random_stats = self._add_random_papers(
            valid_papers, constrained_selected
        )
        print(f"  Random papers added: {random_stats['random_papers_added']}")
        if random_stats["random_papers_added"] > 0:
            print(f"  Random paper score range: {random_stats['random_score_range']}")

        # Compile final statistics
        final_stats = {
            "selection_timestamp": datetime.now().isoformat(),
            "config_used": {
                "percentile": self.percentile,
                "score_threshold": self.score_threshold,
                "min_papers": self.min_papers,
                "max_papers": self.max_papers,
                "n_random": self.n_random,
                "temperature": self.temperature,
            },
            "filter_stats": filter_stats,
            "adjustment_stats": adjustment_stats,
            "random_stats": random_stats,
            "final_count": len(final_selected),
        }

        # Sort final selection by score for consistent output
        final_selected = sorted(
            final_selected, key=lambda x: x["llm_score"], reverse=True
        )

        print(f"\n=== SELECTION COMPLETE ===")
        print(f"Final selection: {len(final_selected)} papers")
        if final_selected:
            scores = [p["llm_score"] for p in final_selected]
            print(f"Selected score range: {min(scores):.1f} - {max(scores):.1f}")

        return {"selected_papers": final_selected, "statistics": final_stats}

    def print_selection_details(self, result: Dict):
        """
        Print detailed information about the selected papers.

        Args:
            result: Result dictionary from select_papers()
        """
        selected_papers = result["selected_papers"]
        stats = result["statistics"]

        print(f"\n=== SELECTED PAPERS DETAILS ===")

        # Show papers by category
        main_papers = (
            selected_papers[: -stats["random_stats"]["random_papers_added"]]
            if stats["random_stats"]["random_papers_added"] > 0
            else selected_papers
        )
        random_papers = (
            selected_papers[-stats["random_stats"]["random_papers_added"] :]
            if stats["random_stats"]["random_papers_added"] > 0
            else []
        )

        if main_papers:
            print(f"\nMain selection ({len(main_papers)} papers):")
            for i, paper in enumerate(main_papers, 1):
                title = paper.get("title", "Untitled")
                score = paper["llm_score"]
                categories = ", ".join(
                    paper.get("categories", [])[:3]
                )  # Show first 3 categories
                print(
                    f"  {i:2d}. [{score:4.1f}] {title[:60]}{'...' if len(title) > 60 else ''}"
                )
                print(f"      Categories: {categories}")

        if random_papers:
            print(f"\nRandom selection ({len(random_papers)} papers):")
            for i, paper in enumerate(random_papers, 1):
                title = paper.get("title", "Untitled")
                score = paper["llm_score"]
                categories = ", ".join(paper.get("categories", [])[:3])
                print(
                    f"  {i:2d}. [{score:4.1f}] {title[:60]}{'...' if len(title) > 60 else ''}"
                )
                print(f"      Categories: {categories}")

        # Category distribution
        print(f"\nCategory distribution:")
        category_counts = {}
        for paper in selected_papers:
            for cat in paper.get("categories", []):
                category_counts[cat] = category_counts.get(cat, 0) + 1

        for cat, count in sorted(
            category_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"  {cat}: {count} papers")


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
        print("Creating default config with PDF processing section...")

        # Create default config if file doesn't exist
        default_config = {
            "pdf_processing": {
                "selection": {
                    "percentile": 20,
                    "score_threshold": 7.0,
                    "min_papers": 5,
                    "max_papers": 25,
                    "n_random": 5,
                    "temperature": 1.0,
                    "random_seed": None,
                }
            }
        }

        # Ensure config directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)

        return default_config


def validate_selection_config(config: Dict) -> List[str]:
    """
    Validate the paper selection configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    pdf_config = config.get("pdf_processing", {})
    selection_config = pdf_config.get("selection", {})

    # Check required parameters
    percentile = selection_config.get("percentile")
    if percentile is None:
        errors.append("percentile is required in pdf_processing.selection")
    elif not 0 <= percentile <= 100:
        errors.append("percentile must be between 0 and 100")

    score_threshold = selection_config.get("score_threshold")
    if score_threshold is None:
        errors.append("score_threshold is required in pdf_processing.selection")

    min_papers = selection_config.get("min_papers", 0)
    max_papers = selection_config.get("max_papers", 100)

    if min_papers < 0:
        errors.append("min_papers must be non-negative")

    if max_papers < min_papers:
        errors.append("max_papers must be >= min_papers")

    n_random = selection_config.get("n_random", 0)
    if n_random < 0:
        errors.append("n_random must be non-negative")

    temperature = selection_config.get("temperature")
    if temperature is not None and temperature <= 0:
        errors.append("temperature must be positive or null (for uniform sampling)")

    random_seed = selection_config.get("random_seed")
    if random_seed is not None and not isinstance(random_seed, int):
        errors.append("random_seed must be an integer")

    return errors


def find_most_recent_scored_file():
    """
    Find the most recent scored papers file in the data directory.

    Returns:
        Tuple of (filepath, format_type) or (None, None) if not found
    """
    import glob

    data_dir = "./data"

    # Look for scored files
    json_pattern = os.path.join(data_dir, "*_scored.json")

    json_files = glob.glob(json_pattern)

    all_files = [(f, "json") for f in json_files]

    if not all_files:
        return None, None

    # Get most recent by modification time
    most_recent = max(all_files, key=lambda x: os.path.getmtime(x[0]))
    return most_recent


def load_scored_papers(filepath: str) -> List[Dict]:
    """
    Load scored papers from file.

    Args:
        filepath: Path to the scored papers file

    Returns:
        List of paper dictionaries
    """
    if filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Expected .json")


def test_selection_algorithm(random_seed: int = 42):
    """Test the selection algorithm with mock data."""
    print("=== TESTING PAPER SELECTION ALGORITHM ===")

    # Create mock papers with scores
    mock_papers = []
    np.random.seed(random_seed)  # For reproducible results

    for i in range(50):
        score = np.random.uniform(1, 10)  # Scores uniformly distributed between 1-10
        mock_papers.append(
            {
                "id": f"paper_{i:03d}",
                "title": f"Mock Paper {i}: Research Topic {i % 5}",
                "llm_score": round(score, 1),
                "categories": [f'cs.{np.random.choice(["AI", "LG", "CV", "CL"])}'],
            }
        )

    # Test configuration
    test_config = {
        "pdf_processing": {
            "selection": {
                "percentile": 25,  # Top 25%
                "score_threshold": 3.0,  # Score >= 3.0
                "min_papers": 8,  # At least 8 papers
                "max_papers": 15,  # At most 15 papers
                "n_random": 3,  # Add 3 random papers
                "temperature": 1.0,  # Default temperature
                "random_seed": random_seed,  # For reproducibility
            }
        }
    }

    print(f"Created {len(mock_papers)} mock papers")
    scores = [p["llm_score"] for p in mock_papers]
    print(
        f"Score distribution: {min(scores):.1f} - {max(scores):.1f}, avg: {np.mean(scores):.1f}"
    )

    # Test selection
    selector = PaperSelector(test_config)
    result = selector.select_papers(mock_papers)

    selector.print_selection_details(result)

    # Test different temperature values
    print(f"\n=== TESTING TEMPERATURE EFFECTS ===")
    temperatures = [0, 0.5, 1.0, 5.0, None]  # None = uniform

    for temp in temperatures:
        test_config["pdf_processing"]["selection"]["temperature"] = temp
        selector = PaperSelector(test_config)

        # Just test the random sampling part
        selected_ids = {f"paper_{i:03d}" for i in range(10)}  # Mock selected papers
        remaining = [p for p in mock_papers if p["id"] not in selected_ids]

        random_papers = selector._temperature_sample(remaining, 5)
        if random_papers:
            avg_score = np.mean([p["llm_score"] for p in random_papers])
            temp_str = "uniform" if temp is None else f"T={temp}"
            print(f"  {temp_str}: avg score = {avg_score:.2f}")

    print(f"\n=== SELECTION ALGORITHM TEST COMPLETE ===")


def test_real_selection():
    """Test the selection algorithm with real scored papers and configuration."""
    print("=== TESTING WITH REAL SCORED PAPERS ===")

    # Test 1: Find scored papers file
    print("\n1. Looking for scored papers file...")
    filepath, format_type = find_most_recent_scored_file()

    if not filepath:
        print("❌ No scored papers file found!")
        print("Expected files: *_scored.json in ./data/ directory")
        print("Run the scoring workflow first: python src/score_papers.py")
        return

    print(f"✅ Found: {filepath} ({format_type} format)")

    # Test 2: Load papers
    print("\n2. Loading scored papers...")
    try:
        papers = load_scored_papers(filepath, format_type)
        print(f"✅ Loaded {len(papers)} papers")

        # Filter papers with valid scores
        valid_papers = [p for p in papers if p.get("llm_score") is not None]
        print(f"✅ Papers with valid scores: {len(valid_papers)}")

        if not valid_papers:
            print("❌ No papers with valid scores found!")
            return

        # Show score distribution
        scores = [p["llm_score"] for p in valid_papers]
        print(
            f"Score distribution: {min(scores):.1f} - {max(scores):.1f}, avg: {sum(scores)/len(scores):.1f}"
        )

        # Show percentile thresholds for reference
        for pct in [10, 20, 25, 30]:
            threshold = np.percentile(scores, 100 - pct)
            count = sum(1 for s in scores if s >= threshold)
            print(f"  Top {pct}%: score >= {threshold:.1f} ({count} papers)")

    except Exception as e:
        print(f"❌ Error loading papers: {e}")
        return

    # Test 3: Load real configuration
    print("\n3. Loading real configuration...")
    try:
        config = load_config()
        print("✅ Loaded config/config.yaml")

        # Validate configuration
        validation_errors = validate_selection_config(config)
        if validation_errors:
            print("❌ Configuration validation errors:")
            for error in validation_errors:
                print(f"   - {error}")
            return
        else:
            print("✅ Configuration validation passed")

        # Show configuration being used
        selection_config = config.get("pdf_processing", {}).get("selection", {})
        print(f"Selection configuration:")
        print(f"  Percentile: {selection_config.get('percentile', 'not set')}")
        print(
            f"  Score threshold: {selection_config.get('score_threshold', 'not set')}"
        )
        print(f"  Min papers: {selection_config.get('min_papers', 'not set')}")
        print(f"  Max papers: {selection_config.get('max_papers', 'not set')}")
        print(f"  Random papers: {selection_config.get('n_random', 'not set')}")
        print(f"  Temperature: {selection_config.get('temperature', 'not set')}")
        print(f"  Random seed: {selection_config.get('random_seed', 'not set')}")

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return

    # Test 4: Run selection algorithm
    print("\n4. Running selection algorithm...")
    try:
        selector = PaperSelector(config)
        result = selector.select_papers(valid_papers)

        print("✅ Selection completed successfully")

        # Show detailed results
        selector.print_selection_details(result)

        # Show final statistics
        selected_papers = result["selected_papers"]
        stats = result["statistics"]

        print(f"\n=== FINAL SUMMARY ===")
        print(f"Total papers processed: {len(valid_papers)}")
        print(f"Papers selected: {len(selected_papers)}")

        if selected_papers:
            selected_scores = [p["llm_score"] for p in selected_papers]
            print(
                f"Selected score range: {min(selected_scores):.1f} - {max(selected_scores):.1f}"
            )
            print(
                f"Average selected score: {sum(selected_scores)/len(selected_scores):.1f}"
            )

            # Category distribution
            category_counts = {}
            for paper in selected_papers:
                for cat in paper.get("categories", []):
                    category_counts[cat] = category_counts.get(cat, 0) + 1

            if category_counts:
                print(f"Category representation:")
                for cat, count in sorted(
                    category_counts.items(), key=lambda x: x[1], reverse=True
                )[:5]:
                    print(f"  {cat}: {count} papers")

    except Exception as e:
        print(f"❌ Selection algorithm failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"\n=== REAL SELECTION TEST COMPLETE ===")
    print("Next steps:")
    print("1. Adjust selection parameters in config/config.yaml if needed")
    print("2. Download PDFs for selected papers from the arXiv API")
    print("3. Run: python src/process_pdfs.py")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] in ["--test", "-t"]:
            test_selection_algorithm()
        elif sys.argv[1] in ["--test-real", "-r"]:
            test_real_selection()
        elif sys.argv[1] in ["--help", "-h"]:
            print("Paper Selection Utilities")
            print("Usage:")
            print("  python selection_utils.py --test     # Run algorithm tests")
            print(
                "  python selection_utils.py --test-real  # Run real selection test with scored papers"
            )
            print("  python selection_utils.py --help     # Show this help")
        else:
            print("Unknown option. Use --help for usage information.")
    else:
        print("Paper Selection Utilities Module")
        print("Run with --help for usage options")

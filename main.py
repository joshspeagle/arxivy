#!/usr/bin/env python3
"""
ArXiv Paper Analysis - Main Workflow Orchestrator

This script runs the complete ArXiv paper analysis workflow from start to finish.
It orchestrates all the individual workflow components in the correct order.

Usage:
    python main.py                    # Run complete workflow
    python main.py --audio           # Include audio generation
    python main.py --test            # Test all components
    python main.py --help            # Show help

Dependencies:
    All dependencies from individual workflow components
    See requirements.txt for complete list

Configuration:
    Edit config/config.yaml and config/llm.yaml before running
"""

import argparse
import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def print_header():
    """Print the workflow header."""
    print("=" * 70)
    print("ArXiv Paper Analysis - Complete Workflow")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_step_header(step_num: int, step_name: str, description: str):
    """Print a step header."""
    print(f"\n{'='*50}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'='*50}")
    print(f"{description}")
    print()


def print_step_footer(step_num: int, step_name: str, success: bool, duration: float):
    """Print step completion status."""
    status = "✅ COMPLETED" if success else "❌ FAILED"
    print(f"\n{status} - Step {step_num}: {step_name}")
    print(f"Duration: {duration:.1f} seconds")
    print("-" * 50)


def run_step(
    step_func, step_num: int, step_name: str, description: str, required: bool = True
):
    """
    Run a workflow step with consistent error handling and timing.

    Args:
        step_func: Function to execute for this step
        step_num: Step number for display
        step_name: Human-readable step name
        description: Step description
        required: Whether failure should stop the workflow

    Returns:
        bool: True if step succeeded, False otherwise
    """
    print_step_header(step_num, step_name, description)
    start_time = time.time()

    try:
        result = step_func()
        duration = time.time() - start_time
        success = result == 0 if isinstance(result, int) else True

        print_step_footer(step_num, step_name, success, duration)

        if not success and required:
            print(f"\n❌ Required step failed: {step_name}")
            print("Workflow cannot continue.")
            return False

        return success

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n❌ Error in {step_name}: {e}")
        print_step_footer(step_num, step_name, False, duration)

        if required:
            print(f"\n❌ Required step failed: {step_name}")
            print("Workflow cannot continue.")
            return False

        return False


def step_1_fetch_papers():
    """Step 1: Fetch recent papers from arXiv."""
    from fetch_papers import main as fetch_main

    return fetch_main()


def step_2_score_papers():
    """Step 2: Score papers using LLM."""
    from score_papers import main as score_main

    return score_main()


def step_3_download_papers():
    """Step 3: Download PDFs of top-scoring papers."""
    from download_papers import main as download_main

    return download_main()


def step_4_extract_papers():
    """Step 4: Extract text from PDFs."""
    from extract_papers import main as extract_main

    return extract_main()


def step_5_summarize_papers():
    """Step 5: Summarize papers using LLM."""
    from summarize_papers import main as summarize_main

    return summarize_main()


def step_6_rescore_papers():
    """Step 6: Re-score papers based on summaries."""
    from rescore_papers import main as rescore_main

    return rescore_main()


def step_7_synthesize_papers():
    """Step 7: Synthesize final report."""
    from synthesize_papers import main as synthesize_main

    return synthesize_main()


def step_8_create_audio_script():
    """Step 8: Create audio script (optional)."""
    from create_audio_script import main as audio_script_main

    return audio_script_main()


def step_9_generate_audio():
    """Step 9: Generate audio file (optional)."""
    # Import and run generate_audio with default settings
    try:
        from generate_audio import main as generate_audio_main

        # Override sys.argv to avoid conflicts
        original_argv = sys.argv[:]
        sys.argv = ["generate_audio.py"]  # Default arguments
        result = generate_audio_main()
        sys.argv = original_argv
        return result
    except Exception as e:
        print(f"Audio generation failed: {e}")
        return 1


def test_all_components():
    """Test all workflow components."""
    print("=" * 70)
    print("TESTING ALL WORKFLOW COMPONENTS")
    print("=" * 70)

    test_results = []

    # Test each component
    components = [
        ("LLM Utils", "llm_utils"),
        ("Score Utils", "score_utils"),
        ("Selection Utils", "selection_utils"),
        ("PDF Utils", "pdf_utils"),
        ("Extract Utils", "extract_utils"),
        ("Summarize Utils", "summarize_utils"),
        ("Synthesis Utils", "synthesis_utils"),
        ("Audio Utils", "audio_utils"),
    ]

    for name, module in components:
        print(f"\nTesting {name}...")
        try:
            # Import and run test if available
            mod = __import__(module)
            if hasattr(mod, "main"):
                # Override sys.argv for test mode
                original_argv = sys.argv[:]
                sys.argv = [f"{module}.py", "--test"]
                result = mod.main()
                sys.argv = original_argv
                test_results.append((name, result == 0))
            else:
                print(f"  ✅ {name} imported successfully")
                test_results.append((name, True))
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            test_results.append((name, False))

    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print("=" * 50)
    for name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")

    failed_count = sum(1 for _, success in test_results if not success)
    if failed_count == 0:
        print(f"\n🎉 All {len(test_results)} components passed!")
        return 0
    else:
        print(f"\n❌ {failed_count}/{len(test_results)} components failed")
        return 1


def run_main_workflow(include_audio: bool = False):
    """
    Run the complete ArXiv workflow.

    Args:
        include_audio: Whether to include audio generation steps

    Returns:
        int: 0 for success, 1 for failure
    """
    print_header()

    workflow_start = time.time()
    failed_steps = []

    # Core workflow steps (all required)
    core_steps = [
        (
            step_1_fetch_papers,
            "Fetch Papers",
            "Retrieve recent papers from arXiv by category",
        ),
        (step_2_score_papers, "Score Papers", "Score papers for relevance using LLM"),
        (
            step_3_download_papers,
            "Download PDFs",
            "Download PDFs of top-scoring papers",
        ),
        (step_4_extract_papers, "Extract Text", "Extract text content from PDF files"),
        (step_5_summarize_papers, "Summarize Papers", "Generate summaries using LLM"),
        (
            step_6_rescore_papers,
            "Re-score Papers",
            "Re-score papers based on summaries",
        ),
        (
            step_7_synthesize_papers,
            "Synthesize Report",
            "Create comprehensive final report",
        ),
    ]

    # Run core workflow
    for i, (step_func, step_name, description) in enumerate(core_steps, 1):
        success = run_step(step_func, i, step_name, description, required=True)
        if not success:
            failed_steps.append(step_name)
            return 1  # Exit on any core step failure

    # Optional audio steps
    if include_audio:
        audio_steps = [
            (
                step_8_create_audio_script,
                "Create Audio Script",
                "Convert report to audio-ready script",
            ),
            (
                step_9_generate_audio,
                "Generate Audio",
                "Create audio file using text-to-speech",
            ),
        ]

        for i, (step_func, step_name, description) in enumerate(audio_steps, 8):
            success = run_step(step_func, i, step_name, description, required=False)
            if not success:
                failed_steps.append(step_name)

    # Final summary
    total_duration = time.time() - workflow_start
    print(f"\n{'='*70}")
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"Total duration: {total_duration/60:.1f} minutes")

    if failed_steps:
        print(f"❌ Failed steps: {', '.join(failed_steps)}")
        print("⚠️  Workflow completed with errors")
        return 1
    else:
        print("✅ All steps completed successfully!")
        print("\n🎉 Your ArXiv analysis workflow is complete!")
        print("\nNext steps:")
        print("  - Check the ./data/ directory for all generated files")
        print("  - Review the final synthesis report")
        if include_audio:
            print("  - Listen to the generated audio summary")
        print("  - Adjust configurations and re-run as needed")
        return 0


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="ArXiv Paper Analysis - Complete Workflow Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Run complete workflow
    python main.py --audio           # Include audio generation
    python main.py --test            # Test all components
    
Before running:
    1. Edit config/config.yaml to set your preferences
    2. Edit config/llm.yaml to configure your LLM API settings
    3. Ensure all dependencies are installed: pip install -r requirements.txt
        """,
    )

    parser.add_argument(
        "--audio",
        action="store_true",
        help="Include audio script and audio generation steps",
    )

    parser.add_argument(
        "--test", action="store_true", help="Test all workflow components"
    )

    args = parser.parse_args()

    # Handle test mode
    if args.test:
        return test_all_components()

    # Check that we're in the right directory
    if not Path("config/config.yaml").exists():
        print("❌ Error: config/config.yaml not found")
        print("Please run this script from the arxivy root directory")
        return 1

    # Run main workflow
    try:
        return run_main_workflow(include_audio=args.audio)
    except KeyboardInterrupt:
        print("\n\n❌ Workflow interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error in workflow: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

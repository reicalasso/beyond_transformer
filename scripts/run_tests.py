"""
Test runner configuration and utilities.
"""

import pytest
import sys
import os
from pathlib import Path


def run_tests(test_type="unit", markers=None, verbose=True):
    """
    Run tests with specified configuration.
    
    Args:
        test_type (str): Type of tests to run ("unit", "integration", "performance", "all")
        markers (list): List of markers to include/exclude
        verbose (bool): Whether to run in verbose mode
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Set up test arguments
    args = []
    
    # Add test paths based on type
    if test_type == "unit":
        args.extend(["tests/unit"])
    elif test_type == "integration":
        args.extend(["tests/unit", "-m", "integration"])
    elif test_type == "performance":
        args.extend(["tests/unit", "-m", "slow"])
    elif test_type == "all":
        args.extend(["tests"])
    
    # Add verbosity
    if verbose:
        args.append("-v")
    
    # Add markers
    if markers:
        marker_expr = " and ".join(markers)
        args.extend(["-m", marker_expr])
    
    # Add coverage
    args.extend(["--cov=src/nsm", "--cov-report=html", "--cov-report=term"])
    
    # Add junit xml report
    args.extend(["--junitxml=reports/junit/test-results.xml"])
    
    print(f"Running tests with arguments: {' '.join(args)}")
    
    # Run tests
    return pytest.main(args)


def run_unit_tests():
    """Run all unit tests."""
    return run_tests("unit")


def run_integration_tests():
    """Run integration tests."""
    return run_tests("integration")


def run_performance_tests():
    """Run performance tests."""
    return run_tests("performance")


def run_all_tests():
    """Run all tests."""
    return run_tests("all")


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NSM Test Runner")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "performance", "all"],
        default="unit",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--markers",
        nargs="*",
        help="Markers to include (e.g., 'gpu', 'slow')"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run tests quietly"
    )
    
    args = parser.parse_args()
    
    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "junit").mkdir(exist_ok=True)
    (reports_dir / "coverage").mkdir(exist_ok=True)
    
    # Run tests
    exit_code = run_tests(
        test_type=args.type,
        markers=args.markers,
        verbose=not args.quiet
    )
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
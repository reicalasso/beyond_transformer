#!/usr/bin/env python3
"""
Test Runner for PULSE Small-Scale Tests
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from pulse.tests.small_scale_tests import SmallScaleTester


def main():
    """Run all small-scale tests."""
    print("PULSE - Small-Scale Test Runner")
    print("=" * 50)
    
    # Create test directory
    test_dir = "test_logs"
    os.makedirs(test_dir, exist_ok=True)
    
    # Run tests
    tester = SmallScaleTester(log_dir=test_dir)
    results = tester.run_all_tests()
    
    # Summary
    print(f"\nTest logs saved to: {os.path.abspath(test_dir)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
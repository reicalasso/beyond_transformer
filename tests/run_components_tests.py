"""
Main test runner for PULSE components
"""

import os
import sys
import unittest

# Add tests to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "components"))

# Import all test modules
from test_layers import TestHybridAttention, TestPulseLayer
from test_router import TestTokenToStateRouter
from test_state_manager import TestStateManager


def create_test_suite():
    """Create a test suite with all component tests."""
    suite = unittest.TestSuite()

    # Add PulseLayer tests
    suite.addTest(unittest.makeSuite(TestPulseLayer))
    suite.addTest(unittest.makeSuite(TestHybridAttention))

    # Add TokenToStateRouter tests
    suite.addTest(unittest.makeSuite(TestTokenToStateRouter))

    # Add StateManager tests
    suite.addTest(unittest.makeSuite(TestStateManager))

    return suite


def run_all_tests():
    """Run all component tests."""
    print("Running PULSE Component Tests")
    print("=" * 40)

    # Create test suite
    suite = create_test_suite()

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

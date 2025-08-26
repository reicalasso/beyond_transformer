"""
Main test runner for NSM components
"""

import unittest
import sys
import os

# Add tests to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'components'))

# Import all test modules
from test_layers import TestNSMLayer, TestHybridAttention
from test_router import TestTokenToStateRouter
from test_state_manager import TestStateManager


def create_test_suite():
    """Create a test suite with all component tests."""
    suite = unittest.TestSuite()
    
    # Add NSMLayer tests
    suite.addTest(unittest.makeSuite(TestNSMLayer))
    suite.addTest(unittest.makeSuite(TestHybridAttention))
    
    # Add TokenToStateRouter tests
    suite.addTest(unittest.makeSuite(TestTokenToStateRouter))
    
    # Add StateManager tests
    suite.addTest(unittest.makeSuite(TestStateManager))
    
    return suite


def run_all_tests():
    """Run all component tests."""
    print("Running NSM Component Tests")
    print("=" * 40)
    
    # Create test suite
    suite = create_test_suite()
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
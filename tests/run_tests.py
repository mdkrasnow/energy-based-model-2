#!/usr/bin/env python3
"""
Test runner for Phase 1 Statistical Viability Testing

This script runs all unit tests for the phase1_statistical_viability.py module
and generates a comprehensive test report.
"""

import os
import sys
import unittest
import time
from pathlib import Path
import argparse

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def discover_and_run_tests(test_dir='tests/test_phase1', pattern='test_*.py', verbosity=2):
    """
    Discover and run all tests in the specified directory
    
    Args:
        test_dir: Directory containing tests
        pattern: Pattern to match test files
        verbosity: Test output verbosity level
    
    Returns:
        unittest.TestResult: Results of test execution
    """
    print(f"🔍 Discovering tests in: {test_dir}")
    print(f"📄 Test file pattern: {pattern}")
    print(f"📊 Verbosity level: {verbosity}")
    print("=" * 80)
    
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = str(project_root / test_dir)
    
    if not os.path.exists(start_dir):
        print(f"❌ Test directory not found: {start_dir}")
        return None
    
    test_suite = loader.discover(start_dir, pattern=pattern)
    
    # Count total tests
    test_count = test_suite.countTestCases()
    print(f"📝 Found {test_count} tests to run\n")
    
    if test_count == 0:
        print("⚠️  No tests found!")
        return None
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    
    print("🚀 Starting test execution...")
    print("=" * 80)
    start_time = time.time()
    
    result = runner.run(test_suite)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("=" * 80)
    print(f"⏱️  Test execution completed in {execution_time:.2f} seconds")
    
    return result


def print_test_summary(result):
    """
    Print a comprehensive test summary
    
    Args:
        result: unittest.TestResult object
    """
    if result is None:
        return
    
    print("\n" + "=" * 80)
    print("📊 TEST EXECUTION SUMMARY")
    print("=" * 80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"📈 Total tests run: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failures}")
    print(f"💥 Errors: {errors}")
    print(f"⏭️  Skipped: {skipped}")
    
    if total_tests > 0:
        success_rate = (passed / total_tests) * 100
        print(f"📊 Success rate: {success_rate:.1f}%")
    
    print("\n" + "=" * 80)
    
    if failures > 0:
        print("❌ FAILURES:")
        print("-" * 40)
        for test, traceback in result.failures:
            print(f"• {test}")
            print(f"  {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'See details above'}")
        print()
    
    if errors > 0:
        print("💥 ERRORS:")
        print("-" * 40)
        for test, traceback in result.errors:
            print(f"• {test}")
            print(f"  {traceback.split('Exception:')[-1].strip() if 'Exception:' in traceback else 'See details above'}")
        print()
    
    print("=" * 80)
    
    if passed == total_tests:
        print("🎉 ALL TESTS PASSED! 🎉")
    elif failures == 0 and errors == 0:
        print("✅ No failures or errors (some tests may have been skipped)")
    else:
        print("⚠️  Some tests failed - please review the output above")
    
    print("=" * 80)


def run_specific_test_class(test_class_name, verbosity=2):
    """
    Run a specific test class
    
    Args:
        test_class_name: Name of the test class to run
        verbosity: Test output verbosity level
    """
    print(f"🎯 Running specific test class: {test_class_name}")
    
    try:
        # Import the test module
        if test_class_name.startswith('TestExperimentRunner'):
            from tests.test_phase1.test_experiment_runner import TestExperimentRunner
            test_class = TestExperimentRunner
        elif test_class_name.startswith('TestPhase1ExperimentRunner'):
            from tests.test_phase1.test_phase1_runner import TestPhase1ExperimentRunner
            test_class = TestPhase1ExperimentRunner
        elif test_class_name.startswith('TestIntegration'):
            from tests.test_phase1.test_integration import TestIntegration
            test_class = TestIntegration
        elif test_class_name.startswith('TestUtility'):
            from tests.test_phase1.test_utilities import TestUtilityFunctions
            test_class = TestUtilityFunctions
        else:
            print(f"❌ Unknown test class: {test_class_name}")
            return None
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        return result
        
    except ImportError as e:
        print(f"❌ Failed to import test class: {e}")
        return None


def main():
    """Main function to run tests with command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run unit tests for phase1_statistical_viability.py'
    )
    parser.add_argument(
        '--test-dir', 
        default='tests/test_phase1',
        help='Directory containing tests'
    )
    parser.add_argument(
        '--pattern',
        default='test_*.py',
        help='Pattern to match test files'
    )
    parser.add_argument(
        '--verbosity',
        type=int,
        default=2,
        choices=[0, 1, 2],
        help='Test output verbosity (0=quiet, 1=normal, 2=verbose)'
    )
    parser.add_argument(
        '--class',
        dest='test_class',
        help='Run a specific test class only'
    )
    
    args = parser.parse_args()
    
    print("🧪 Phase 1 Statistical Viability - Test Runner")
    print("=" * 80)
    
    if args.test_class:
        result = run_specific_test_class(args.test_class, args.verbosity)
    else:
        result = discover_and_run_tests(args.test_dir, args.pattern, args.verbosity)
    
    print_test_summary(result)
    
    # Exit with appropriate code
    if result is None:
        sys.exit(1)  # No tests found or import error
    elif result.failures or result.errors:
        sys.exit(1)  # Tests failed
    else:
        sys.exit(0)  # All tests passed


if __name__ == '__main__':
    main()
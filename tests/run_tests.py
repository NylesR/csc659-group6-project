#!/usr/bin/env python3
"""
Test runner script for the Random Forest model tests.
Provides options for running tests with different verbosity levels and generating reports.
"""

import unittest
import sys
import os
import argparse
from io import StringIO
import time

def run_tests(verbosity=2, pattern=None, failfast=False):
    """
    Run the test suite with specified options.
    
    Args:
        verbosity (int): Verbosity level (0=quiet, 1=normal, 2=verbose)
        pattern (str): Pattern to match test names
        failfast (bool): Stop on first failure
    
    Returns:
        unittest.TestResult: Test results
    """
    # Discover and run tests
    loader = unittest.TestLoader()
    
    if pattern:
        loader.testNamePatterns = [pattern]
    
    # Discover tests in current directory (Test folder)
    start_dir = '.'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Create test runner
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        failfast=failfast,
        stream=sys.stdout
    )
    
    # Run tests and capture results
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        
        # Print failure details
        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"\n{test}:")
                print(traceback)
        
        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"\n{test}:")
                print(traceback)
    
    return result

def run_coverage_tests():
    """Run tests with coverage analysis."""
    try:
        import coverage
        
        # Start coverage measurement
        cov = coverage.Coverage()
        cov.start()
        
        # Run tests
        result = run_tests(verbosity=1)
        
        # Stop coverage measurement
        cov.stop()
        cov.save()
        
        # Generate coverage report
        print(f"\n{'='*60}")
        print("COVERAGE REPORT")
        print(f"{'='*60}")
        cov.report()
        
        return result
        
    except ImportError:
        print("Coverage package not installed. Install with: pip install coverage")
        return run_tests()

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description='Run Random Forest model tests')
    parser.add_argument(
        '-v', '--verbosity', 
        type=int, 
        default=2, 
        choices=[0, 1, 2],
        help='Verbosity level (0=quiet, 1=normal, 2=verbose)'
    )
    parser.add_argument(
        '-p', '--pattern', 
        type=str, 
        help='Pattern to match test names'
    )
    parser.add_argument(
        '-f', '--failfast', 
        action='store_true',
        help='Stop on first failure'
    )
    parser.add_argument(
        '-c', '--coverage', 
        action='store_true',
        help='Run with coverage analysis'
    )
    parser.add_argument(
        '--list', 
        action='store_true',
        help='List all available tests'
    )
    
    args = parser.parse_args()
    
    if args.list:
        # List all available tests
        loader = unittest.TestLoader()
        suite = loader.discover('.', pattern='test_*.py')
        
        print("Available tests:")
        print("="*60)
        
        def print_tests(suite, indent=0):
            for test in suite:
                if hasattr(test, '_tests'):
                    print_tests(test, indent + 2)
                else:
                    print(f"{' ' * indent}• {test}")
        
        print_tests(suite)
        return
    
    if args.coverage:
        result = run_coverage_tests()
    else:
        result = run_tests(
            verbosity=args.verbosity,
            pattern=args.pattern,
            failfast=args.failfast
        )
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    main() 
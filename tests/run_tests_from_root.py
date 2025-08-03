#!/usr/bin/env python3
"""
Script to run tests from the project root directory.
This script changes to the Test directory and runs the test suite.
"""

import os
import sys
import subprocess
import argparse

def run_tests_from_root(verbosity=2, pattern=None, failfast=False, coverage=False):
    """
    Run tests from the project root by changing to Test directory.
    
    Args:
        verbosity (int): Verbosity level (0=quiet, 1=normal, 2=verbose)
        pattern (str): Pattern to match test names
        failfast (bool): Stop on first failure
        coverage (bool): Run with coverage analysis
    """
    # Get the current directory (project root)
    project_root = os.getcwd()
    test_dir = os.path.join(project_root, 'tests')
    
    # Check if tests directory exists
    if not os.path.exists(test_dir):
        print("Error: tests directory not found!")
        print(f"Expected path: {test_dir}")
        return False
    
    # Change to Test directory
    os.chdir(test_dir)
    
    try:
        # Build command arguments
        cmd = [sys.executable, 'run_tests.py']
        
        if verbosity is not None:
            cmd.extend(['-v', str(verbosity)])
        
        if pattern:
            cmd.extend(['-p', pattern])
        
        if failfast:
            cmd.append('-f')
        
        if coverage:
            cmd.append('-c')
        
        # Run the tests
        print(f"Running tests from: {test_dir}")
        print(f"Command: {' '.join(cmd)}")
        print("="*60)
        
        result = subprocess.run(cmd, capture_output=False)
        
        return result.returncode == 0
        
    finally:
        # Change back to project root
        os.chdir(project_root)

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description='Run Random Forest model tests from project root')
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
        # List tests
        project_root = os.getcwd()
        test_dir = os.path.join(project_root, 'tests')
        os.chdir(test_dir)
        
        try:
            result = subprocess.run([sys.executable, 'run_tests.py', '--list'], 
                                  capture_output=False)
        finally:
            os.chdir(project_root)
        return
    
    # Run tests
    success = run_tests_from_root(
        verbosity=args.verbosity,
        pattern=args.pattern,
        failfast=args.failfast,
        coverage=args.coverage
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
"""
Final Test Runner for Phase 1 Statistical Viability Testing

This script runs the comprehensive test suite for phase1_statistical_viability.py
and provides a detailed summary of test coverage and results.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def main():
    """Run the comprehensive test suite"""
    
    print("🧪 PHASE 1 STATISTICAL VIABILITY - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("Testing: phase1_statistical_viability.py")
    print("Location: tests/test_phase1/test_comprehensive.py")
    print("=" * 80)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("📍 Working directory:", os.getcwd())
    print()
    
    # Run the comprehensive tests
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_phase1/test_comprehensive.py',
            '-v',
            '--tb=short'
        ], capture_output=True, text=True)
        
        execution_time = time.time() - start_time
        
        print("🚀 TEST EXECUTION OUTPUT:")
        print("-" * 40)
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  WARNINGS/ERRORS:")
            print("-" * 40)
            print(result.stderr)
        
        print("=" * 80)
        print("📊 FINAL TEST SUMMARY")
        print("=" * 80)
        print(f"⏱️  Execution time: {execution_time:.2f} seconds")
        print(f"📝 Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("🎉 ALL TESTS PASSED! 🎉")
            print()
            print("✅ Test Coverage Summary:")
            print("   • Constants and configuration validation")
            print("   • ExperimentRunner basic functionality")
            print("   • Phase1ExperimentRunner initialization and CSV logging")
            print("   • Model training and evaluation workflows (mocked)")
            print("   • Data structure handling")
            print("   • Error handling and edge cases")
            print()
            print("🔧 The phase1_statistical_viability.py module is ready for production use!")
        else:
            print("❌ SOME TESTS FAILED")
            print("   Please review the output above for details")
        
        print("=" * 80)
        return result.returncode
        
    except Exception as e:
        print(f"💥 ERROR: Failed to run tests: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
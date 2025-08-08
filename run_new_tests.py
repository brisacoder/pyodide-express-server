#!/usr/bin/env python3
"""
Quick Test Runner for New Tests

This script runs the newly added test modules:
- test_non_happy_paths
- test_sklearn
- test_matplotlib
- test_seaborn

These tests validate edge cases, scikit-learn functionality, and plotting capabilities.
"""

import subprocess
import sys
import requests


def check_server_running():
    """Check if server is running on port 3000."""
    try:
        response = requests.get("http://localhost:3000/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def run_test_module(module_name, test_name):
    """Run a specific test module."""
    print(f"🧪 Running {test_name}...")
    print("=" * 60)

    try:
        # Run the test module
        result = subprocess.run(
            [sys.executable, "-m", "unittest", f"tests.{module_name}", "-v"],
            capture_output=False,
            check=False,
            timeout=300,  # 5 minute timeout
        )

        success = result.returncode == 0
        if success:
            print(f"✅ {test_name} - All tests passed!")
        else:
            print(f"❌ {test_name} - Some tests failed!")

        return success

    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} - Tests timed out!")
        return False
    except Exception as e:
        print(f"💥 {test_name} - Error running tests: {e}")
        return False


def main():
    """Main entry point."""
    print("🚀 New Tests Runner for Pyodide Express Server")
    print("=" * 60)

    # Check if server is running
    if not check_server_running():
        print("❌ Server is not running on localhost:3000")
        print("   Please start the server first with: npm start")
        return 1

    print("✅ Server is running on localhost:3000")
    print()

    # Define new test categories
    test_categories = [
        ("test_non_happy_paths", "Non-Happy Paths Tests"),
        ("test_sklearn", "Scikit-Learn Tests"),
        ("test_matplotlib", "Matplotlib Plotting Tests"),
        ("test_seaborn", "Seaborn Plotting Tests"),
    ]

    total_tests = len(test_categories)
    passed_tests = 0

    print(f"Running {total_tests} test categories...")
    print()

    # Run each test category
    for module_name, test_name in test_categories:
        success = run_test_module(module_name, test_name)
        if success:
            passed_tests += 1
        print()  # Add spacing between test categories

    # Final summary
    print("=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"Test Categories Run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if passed_tests == total_tests:
        print("\n🎉 ALL NEW TESTS PASSED!")
        print("The new test modules are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test categories failed.")
        print("Please review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

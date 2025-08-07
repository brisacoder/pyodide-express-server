#!/usr/bin/env python3
"""
Simple Test Runner for Pyodide Express Server

This script runs tests against an existing server instance.
Make sure the server is running on localhost:3000 before running tests.
"""

import subprocess
import sys
import os
import requests


def check_server_running():
    """Check if server is running on port 3000."""
    try:
        response = requests.get("http://localhost:3000/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def run_basic_tests():
    """Run the basic API tests."""
    print("🧪 Running Basic API Tests...")
    print("=" * 50)

    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run just the basic API tests
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_api.APITestCase", "-v"],
        capture_output=False,
        check=False,
    )

    return result.returncode == 0


def main():
    """Main entry point."""
    print("🚀 Simple Test Runner for Pyodide Express Server")
    print("=" * 60)

    # Check if server is running
    if not check_server_running():
        print("❌ Server is not running on localhost:3000")
        print("   Please start the server first with: npm start")
        return 1

    print("✅ Server is running on localhost:3000")

    # Run basic tests
    success = run_basic_tests()

    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

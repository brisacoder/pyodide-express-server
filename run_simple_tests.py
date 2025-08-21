#!/usr/bin/env python3
"""
Simple Test Runner for Pyodide Express Server

This script runs tests against an existing server instance.
Make sure the server is running on localhost:3000 before running tests.
"""

import os
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


def run_dynamic_modules_tests():
    """Run the dynamic modules and execution robustness tests."""
    print("\n🔬 Running Dynamic Modules & Execution Robustness Tests...")
    print("=" * 60)

    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run the dynamic modules tests
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_dynamic_modules_and_execution_robustness", "-v"],
        capture_output=False,
        check=False,
    )

    return result.returncode == 0


def run_security_logging_tests():
    """Run the security logging tests."""
    print("\n🔐 Running Security Logging Tests...")
    print("=" * 40)

    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run the security logging tests
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_security_logging", "-v"],
        capture_output=False,
        check=False,
    )

    return result.returncode == 0


def run_function_return_patterns_tests():
    """Run the function return patterns tests."""
    print("\n🔧 Running Function Return Patterns Tests...")
    print("=" * 50)

    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run the function return patterns tests
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_function_return_patterns", "-v"],
        capture_output=False,
        check=False,
    )

    return result.returncode == 0


def run_container_filesystem_tests():
    """Run the container filesystem tests."""
    print("\n🐳 Running Container Filesystem Tests...")
    print("=" * 45)

    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run the container filesystem tests
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_container_filesystem", "-v"],
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
    basic_success = run_basic_tests()
    
    # Run dynamic modules tests
    dynamic_success = run_dynamic_modules_tests()
    
    # Run security logging tests
    security_success = run_security_logging_tests()
    
    # Run function return patterns tests
    patterns_success = run_function_return_patterns_tests()
    
    # Note: Container filesystem tests are skipped for native server testing
    # They are designed for containerized environments only
    print("\n🐳 Container Filesystem Tests: SKIPPED (native server mode)")
    print("   - These tests are designed for containerized environments")
    print("   - Use 'run_comprehensive_tests.py --categories container' for container testing")

    if basic_success and dynamic_success and security_success and patterns_success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        if not basic_success:
            print("   - Basic API tests failed")
        if not dynamic_success:
            print("   - Dynamic modules tests failed")
        if not security_success:
            print("   - Security logging tests failed")
        if not patterns_success:
            print("   - Function return patterns tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

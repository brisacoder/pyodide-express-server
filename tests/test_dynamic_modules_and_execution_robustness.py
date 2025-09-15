#!/usr/bin/env python3
"""
Comprehensive pytest test suite for dynamic module discovery and execution robustness.

This test suite covers:
1. Dynamic package installation and automatic availability
2. Complex Python code scenarios with string handling
3. Edge cases with various quote combinations and f-strings
4. Concurrent execution with newly installed packages
5. Cross-platform compatibility using pathlib
6. BDD-style test organization with comprehensive coverage

All tests use only the /api/execute-raw endpoint with proper API contract validation.
Tests are designed to work across Windows and Linux environments.
"""

import concurrent.futures
import time
from typing import Any, Dict
from uuid import uuid4

import pytest
import requests


# ========================================
# CONSTANTS & CONFIGURATION
# ========================================

class TestConfig:
    """Test configuration constants - no hardcoded values."""
    BASE_URL = "http://localhost:3000"
    DEFAULT_TIMEOUT = 30  # seconds
    LONG_TIMEOUT = 60     # seconds for complex operations
    REQUEST_TIMEOUT = 35  # requests library timeout (should be > DEFAULT_TIMEOUT)
    STRESS_TEST_SIZE = 200
    CONCURRENT_USERS = 5
    STRING_TEST_COUNT = 50

    # Expected API contract structure
    EXPECTED_SUCCESS_FIELDS = {"success", "data", "error", "meta"}
    EXPECTED_META_FIELDS = {"timestamp"}
    EXPECTED_DATA_FIELDS = {"result", "stdout", "stderr", "executionTime"}


# ========================================
# PYTEST FIXTURES
# ========================================

@pytest.fixture(scope="session")
def api_client():
    """
    Create a configured requests session for API testing.

    Returns:
        requests.Session: Configured session with appropriate timeouts

    Example:
        def test_example(api_client):
            response = api_client.post('/api/execute-raw', data='print("Hello")')
    """
    session = requests.Session()
    session.timeout = TestConfig.REQUEST_TIMEOUT
    return session


@pytest.fixture(scope="session")
def server_health_check(api_client):
    """
    Verify server is running and healthy before starting tests.

    Args:
        api_client: Configured requests session

    Raises:
        pytest.skip: If server is not accessible or healthy

    Example:
        This fixture runs automatically for session-scoped setup
    """
    try:
        response = api_client.get(f"{TestConfig.BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            pytest.skip("Server not running or not healthy")
    except requests.RequestException:
        pytest.skip("Server not accessible")


@pytest.fixture
def execution_timeout():
    """
    Provide default execution timeout for tests.
    
    Returns:
        int: Timeout value in seconds
        
    Example:
        def test_example(execution_timeout):
            result = execute_python_raw("print('test')", timeout=execution_timeout)
    """
    return TestConfig.DEFAULT_TIMEOUT


@pytest.fixture
def unique_test_id():
    """
    Generate unique ID for test isolation.
    
    Returns:
        str: Unique identifier for the test
        
    Example:
        def test_example(unique_test_id):
            code = f'test_id = "{unique_test_id}"'
    """
    return str(uuid4())[:8]


@pytest.fixture
def temp_file_cleanup():
    """
    Track temporary files for cleanup after test.
    
    Yields:
        List[str]: List to append temporary filenames for cleanup
        
    Example:
        def test_file_ops(temp_file_cleanup):
            filename = 'test.txt'
            temp_file_cleanup.append(filename)
            # File will be cleaned up automatically
    """
    temp_files = []
    yield temp_files
    
    # Cleanup any tracked temporary files
    for filename in temp_files:
        try:
            # Try to delete via API if needed
            requests.delete(f"{TestConfig.BASE_URL}/api/uploaded-files/{filename}", timeout=5)
        except requests.RequestException:
            pass  # File might not exist or be deletable


# ========================================
# HELPER FUNCTIONS
# ========================================

def execute_python_raw(
    api_client: requests.Session,
    code: str,
    timeout: int = TestConfig.DEFAULT_TIMEOUT
) -> Dict[str, Any]:
    """
    Execute Python code via /api/execute-raw endpoint with proper API contract validation.
    
    Args:
        api_client: Configured requests session
        code: Python code to execute as plain text
        timeout: Execution timeout in seconds
        
    Returns:
        dict: API response with validated structure
        
    Raises:
        AssertionError: If API contract is violated
        requests.RequestException: On network/HTTP errors
        
    Example:
        result = execute_python_raw(api_client, 'print("Hello World")')
        assert result["success"] is True
        assert "Hello World" in result["data"]["stdout"]
    """
    response = api_client.post(
        f"{TestConfig.BASE_URL}/api/execute-raw",
        data=code,
        headers={"Content-Type": "text/plain"},
        timeout=timeout + 5  # Add buffer for network
    )
    
    # Validate HTTP response
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    # Parse and validate API contract
    result = response.json()
    
    # Validate API contract structure
    assert isinstance(result, dict), f"Response must be dict, got {type(result)}"
    assert set(result.keys()) == TestConfig.EXPECTED_SUCCESS_FIELDS, \
        f"Response missing fields. Expected {TestConfig.EXPECTED_SUCCESS_FIELDS}, got {set(result.keys())}"
    
    # Validate success field
    assert isinstance(result["success"], bool), f"success must be boolean, got {type(result['success'])}"
    
    # Validate meta structure
    assert isinstance(result["meta"], dict), f"meta must be dict, got {type(result['meta'])}"
    assert TestConfig.EXPECTED_META_FIELDS.issubset(set(result["meta"].keys())), \
        f"meta missing required fields {TestConfig.EXPECTED_META_FIELDS}"
    
    # Validate conditional fields based on success
    if result["success"]:
        assert result["error"] is None, f"error must be None for success=True, got {result['error']}"
        assert result["data"] is not None, f"data must not be None for success=True"
        assert isinstance(result["data"], dict), f"data must be dict for success=True, got {type(result['data'])}"
        assert TestConfig.EXPECTED_DATA_FIELDS.issubset(set(result["data"].keys())), \
            f"data missing required fields {TestConfig.EXPECTED_DATA_FIELDS}"
    else:
        assert result["data"] is None, f"data must be None for success=False, got {result['data']}"
        assert result["error"] is not None, f"error must not be None for success=False"
        assert isinstance(result["error"], str), f"error must be string for success=False, got {type(result['error'])}"
    
    return result


def install_python_package(
    api_client: requests.Session,
    package_name: str,
    timeout: int = TestConfig.LONG_TIMEOUT
) -> Dict[str, Any]:
    """
    Install a Python package via API.
    
    Args:
        api_client: Configured requests session  
        package_name: Name of package to install
        timeout: Installation timeout in seconds
        
    Returns:
        dict: Installation result
        
    Raises:
        AssertionError: If installation fails
        
    Example:
        result = install_python_package(api_client, "jsonschema")
        assert result["success"] is True
    """
    response = api_client.post(
        f"{TestConfig.BASE_URL}/api/install-package",
        json={"package": package_name},
        timeout=timeout
    )
    assert response.status_code == 200, f"Package installation failed: {response.status_code}"
    return response.json()


# ========================================
# BDD-STYLE TEST CASES
# ========================================

class TestGivenServerIsRunning:
    """Test scenarios given that the server is running and healthy."""

    def test_given_server_running_when_health_check_then_returns_ok(
        self,
        api_client: requests.Session,
        server_health_check
    ):
        """
        Test basic server health verification.

        Given: Server is running
        When: Health check endpoint is called
        Then: Returns OK status with proper API contract

        Args:
            api_client: HTTP client session
            server_health_check: Fixture ensuring server health
        """
        response = api_client.get(f"{TestConfig.BASE_URL}/health")
        assert response.status_code == 200
        health_data = response.json()
        
        # Verify API contract
        assert health_data["success"] is True
        assert health_data["data"] is not None
        assert health_data["error"] is None
        assert "timestamp" in health_data["meta"]
        
        # Verify health status
        assert health_data["data"]["server"] == "running"
        assert health_data["data"]["status"] in ["ok", "degraded"]  # Allow degraded for pyodide init


class TestGivenBasicPythonExecution:
    """Test basic Python code execution scenarios."""

    def test_given_simple_code_when_executed_via_raw_api_then_returns_success(
        self,
        api_client: requests.Session,
        server_health_check,
        execution_timeout: int
    ):
        """
        Test basic Python code execution with API contract validation.
        
        Given: Simple Python print statement
        When: Executed via /api/execute-raw
        Then: Returns success with proper API contract structure
        
        Args:
            api_client: HTTP client session
            server_health_check: Server health verification
            execution_timeout: Test timeout configuration
            
        Example:
            This test validates the core API contract format:
            {
              "success": true,
              "data": {"result": "...", "stdout": "...", "stderr": "", "executionTime": 123},
              "error": null,
              "meta": {"timestamp": "..."}
            }
        """
        code = '''
import sys
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Hello from Pyodide!")
result = "Basic execution successful"
result
        '''
        
        result = execute_python_raw(api_client, code, execution_timeout)
        
        # Verify success
        assert result["success"] is True
        assert result["error"] is None
        assert result["data"] is not None
        
        # Verify execution data  
        data = result["data"]
        assert isinstance(data["executionTime"], (int, float))
        assert data["executionTime"] > 0
        assert "Python version:" in data["stdout"]
        assert "Hello from Pyodide!" in data["stdout"]
        assert "Basic execution successful" in data["result"]
        assert data["stderr"] == ""

    def test_given_syntax_error_when_executed_then_returns_proper_error_structure(
        self,
        api_client: requests.Session,
        server_health_check,
        execution_timeout: int
    ):
        """
        Test error handling with proper API contract.
        
        Given: Python code with syntax error
        When: Executed via /api/execute-raw  
        Then: Returns failure with proper error structure
        
        Args:
            api_client: HTTP client session
            server_health_check: Server health verification
            execution_timeout: Test timeout configuration
        """
        invalid_code = '''
print("Missing closing quote and parenthesis
invalid syntax here
        '''
        
        result = execute_python_raw(api_client, invalid_code, execution_timeout)
        
        # Verify error structure
        assert result["success"] is False
        assert result["data"] is None
        assert result["error"] is not None
        assert isinstance(result["error"], str)
        assert len(result["error"]) > 0  # Just verify we got some error message


class TestGivenCrossPlatformCompatibility:
    """Test cross-platform compatibility with pathlib and proper path handling."""

    def test_given_path_operations_when_using_pathlib_then_works_cross_platform(
        self,
        api_client: requests.Session,
        server_health_check,
        execution_timeout: int,
        unique_test_id: str
    ):
        """
        Test cross-platform path handling using pathlib.
        
        Given: Path operations needed for file handling
        When: Using pathlib for cross-platform compatibility
        Then: Paths work correctly on both Windows and Linux
        
        Args:
            api_client: HTTP client session
            server_health_check: Server health verification  
            execution_timeout: Test timeout configuration
            unique_test_id: Unique test identifier
        """
        code = f'''
from pathlib import Path
import os

# Test pathlib cross-platform compatibility
test_id = "{unique_test_id}"

# Create various paths using pathlib (cross-platform)
plots_dir = Path('/plots/matplotlib')
uploads_dir = Path('/uploads')
test_file = uploads_dir / f'test_{{test_id}}.txt'
nested_path = plots_dir / 'subfolder' / 'nested.png'

# Path operations
paths_info = {{
    "plots_dir_str": str(plots_dir),
    "uploads_dir_str": str(uploads_dir), 
    "test_file_str": str(test_file),
    "nested_path_str": str(nested_path),
    "plots_parent": str(plots_dir.parent),
    "test_file_name": test_file.name,
    "test_file_stem": test_file.stem,
    "test_file_suffix": test_file.suffix,
    "separator_used": os.sep,
    "pathlib_works": True
}}

# Verify path operations work
assert plots_dir.is_absolute(), "plots_dir should be absolute"
assert uploads_dir.is_absolute(), "uploads_dir should be absolute"
assert test_file.parent == uploads_dir, "Parent relationship should work"
assert nested_path.parents[1] == plots_dir, "Multi-level parent should work"

print(f"PATHS_INFO: {{paths_info}}")
paths_info
        '''
        
        result = execute_python_raw(api_client, code, execution_timeout)
        
        assert result["success"] is True
        data = result["data"]
        paths_info = data["result"]
        
        # Verify cross-platform paths
        assert paths_info["plots_dir_str"] == "/plots/matplotlib"
        assert paths_info["uploads_dir_str"] == "/uploads"
        assert f"test_{unique_test_id}.txt" in paths_info["test_file_str"]
        assert paths_info["test_file_name"] == f"test_{unique_test_id}.txt"
        assert paths_info["test_file_stem"] == f"test_{unique_test_id}"
        assert paths_info["test_file_suffix"] == ".txt"
        assert paths_info["pathlib_works"] is True
        
        # Verify pathlib usage appears in stdout
        assert "PATHS_INFO:" in data["stdout"]


class TestGivenComplexStringScenarios:
    """Test complex string scenarios that might break JSON serialization."""

    def test_given_complex_strings_when_executed_via_raw_api_then_handles_all_cases(
        self,
        api_client: requests.Session,
        server_health_check,
        execution_timeout: int
    ):
        """
        Test complex string scenarios using raw API execution.
        
        Given: Python code with complex string scenarios (quotes, f-strings, JSON, etc.)
        When: Executed via /api/execute-raw (no JSON escaping needed)
        Then: All string scenarios are handled correctly
        
        Args:
            api_client: HTTP client session
            server_health_check: Server health verification
            execution_timeout: Test timeout configuration
            
        Example:
            Tests strings that would break JSON.stringify():
            - F-strings with complex expressions and nested quotes
            - Triple quoted strings with mixed quote types
            - Actual JSON strings within Python
            - Regex patterns with escapes
            - SQL-like strings with quotes
            - File paths with spaces and special characters
        """
        complex_code = """
# Test ALL the complex string scenarios that break JSON serialization
import json

test_results = {}

# F-strings with complex expressions and nested quotes  
name = "Python's Amazing String Handling"
version = 3.11
complex_fstring = f"Hello {name} version {version} with math: {2**3}"
test_results["fstring"] = complex_fstring

# Triple quoted strings with ALL kinds of embedded content
triple_quoted = '''This is a triple quoted string
with single quotes, double quotes, and even embedded content
Line 3 with escape sequences: \\n \\t \\\\ 
Line 4 with unicode: ñáéíóú 🐍'''
test_results["triple_quoted"] = len(triple_quoted)

# Mixed quote scenarios that would break JSON.stringify
mixed_quotes = "Double quotes with 'single quotes' inside"
test_results["mixed_quotes"] = mixed_quotes

# Complex regex and SQL-like strings
regex_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
sql_query = "SELECT * FROM users WHERE name = 'O\\'Reilly'"
test_results["regex"] = regex_pattern
test_results["sql"] = sql_query

# Unicode and special characters
unicode_str = "🐍 Python with symbols: naïve café résumé"
test_results["unicode"] = unicode_str

# Actual JSON strings (the nightmare scenario for JSON.stringify)
actual_json = '{"key": "value with quotes", "nested": {"array": [1, 2, 3]}}'
test_results["json_string"] = actual_json

# Multi-line Python code as string
python_code_string = '''
def hello(name):
    return f"Hello {name}!"

result = hello("World")
'''
test_results["python_code"] = python_code_string.strip()

# Cross-platform file paths with spaces and special characters  
from pathlib import Path
file_paths = [
    str(Path("C:/Users/name/Documents/file with spaces.txt")),
    str(Path("/home/user/file_name.py")), 
    str(Path("server/share/folder/file.csv"))
]
test_results["file_paths"] = file_paths

print(f"COMPLEX_STRINGS_TEST: {len(test_results)} scenarios tested")
test_results
        """
        
        result = execute_python_raw(api_client, code=complex_code, timeout=execution_timeout)
        
        assert result["success"] is True
        data = result["data"]
        test_results = data["result"]
        
        # Verify all complex scenarios worked
        assert "Python's Amazing String Handling" in test_results["fstring"]
        assert "version 3.11" in test_results["fstring"]
        assert test_results["triple_quoted"] > 100
        assert "single quotes" in test_results["mixed_quotes"]
        assert "@" in test_results["regex"]
        assert "O'Reilly" in test_results["sql"]
        assert "🐍" in test_results["unicode"]
        assert "value with quotes" in test_results["json_string"]
        assert "def hello" in test_results["python_code"]
        assert len(test_results["file_paths"]) == 3
        
        # Verify output indicates success
        assert "COMPLEX_STRINGS_TEST: 9 scenarios tested" in data["stdout"]

    def test_given_json_strings_within_python_when_executed_then_no_escaping_conflicts(
        self,
        api_client: requests.Session,
        server_health_check,
        execution_timeout: int
    ):
        """
        Test JSON string handling within Python code using raw API.
        
        Given: Python code that manipulates JSON strings
        When: Executed via raw API (avoiding JSON escaping hell)
        Then: JSON operations work without escaping conflicts
        
        Args:
            api_client: HTTP client session
            server_health_check: Server health verification
            execution_timeout: Test timeout configuration
        """
        code = '''
import json

# The nightmare scenario: JSON within JSON via API
test_json_data = {
    "name": "Test User", 
    "quotes": "Text with 'single' and \\"double\\" quotes",
    "nested": {
        "array": [1, 2, 3],
        "special": "C:\\\\path\\\\to\\\\file.txt"
    }
}

# Convert to JSON string
json_string = json.dumps(test_json_data)

# Parse it back
parsed_data = json.loads(json_string)

# Verify round-trip works
roundtrip_success = parsed_data == test_json_data

results = {
    "original_name": test_json_data["name"],
    "original_quotes": test_json_data["quotes"],
    "parsed_name": parsed_data["name"],
    "parsed_quotes": parsed_data["quotes"],
    "roundtrip_success": roundtrip_success,
    "json_string_length": len(json_string)
}

print(f"JSON_TEST: roundtrip={roundtrip_success}")
results
        '''
        
        result = execute_python_raw(api_client, code, execution_timeout)
        
        assert result["success"] is True
        data = result["data"] 
        results = data["result"]
        
        # Verify JSON operations work correctly
        assert results["roundtrip_success"] is True
        assert results["original_name"] == "Test User"
        assert results["parsed_name"] == "Test User"
        assert "single" in results["original_quotes"]
        assert "double" in results["original_quotes"] 
        assert results["original_quotes"] == results["parsed_quotes"]
        assert results["json_string_length"] > 50
        
        # Verify success message in output
        assert "JSON_TEST: roundtrip=True" in data["stdout"]


class TestGivenDynamicPackageManagement:
    """Test dynamic package installation and usage scenarios."""

    def test_given_package_not_installed_when_install_then_becomes_available(
        self,
        api_client: requests.Session,
        server_health_check,
        execution_timeout: int
    ):
        """
        Test dynamic package installation and immediate availability.
        
        Given: A Python package is not yet installed
        When: Package is installed via API  
        Then: Package becomes immediately available for import and use
        
        Args:
            api_client: HTTP client session
            server_health_check: Server health verification
            execution_timeout: Test timeout configuration
        """
        # Test package installation and immediate usage in same execution
        verification_code = '''
# Install and use package in same execution context
import micropip
import asyncio

# Install package
await micropip.install("jsonschema")

# Now test immediate availability
import jsonschema

# Test basic functionality
schema = {"type": "string"}
validator = jsonschema.Draft7Validator(schema)
test_data = "hello world"

# Validate data against schema
is_valid = validator.is_valid(test_data)
package_version = getattr(jsonschema, '__version__', 'unknown')

results = {
    "package_imported": True,
    "version": package_version,
    "validation_works": is_valid,
    "validator_created": validator is not None
}

print(f"JSONSCHEMA_TEST: imported=True, version={package_version}")
results
        '''
        
        result = execute_python_raw(api_client, verification_code, execution_timeout)
        
        assert result["success"] is True
        data = result["data"]
        results = data["result"]
        
        # Verify package availability and functionality
        assert results["package_imported"] is True
        assert results["validation_works"] is True
        assert results["validator_created"] is True
        assert isinstance(results["version"], str)
        assert results["version"] != "unknown"
        
        # Verify success message
        assert "JSONSCHEMA_TEST: imported=True" in data["stdout"]

    def test_given_multiple_packages_when_used_concurrently_then_all_work_correctly(
        self,
        api_client: requests.Session,
        server_health_check,
        execution_timeout: int
    ):
        """
        Test concurrent usage of multiple packages.
        
        Given: Multiple Python packages are available
        When: Used concurrently in different execution contexts
        Then: All packages work correctly without conflicts
        
        Args:
            api_client: HTTP client session
            server_health_check: Server health verification
            execution_timeout: Test timeout configuration
        """
        def execute_package_test(package_info):
            """Execute test for a specific package."""
            package_name, test_code = package_info
            try:
                result = execute_python_raw(api_client, test_code, execution_timeout)
                return {
                    "package": package_name,
                    "success": result["success"],
                    "result": result.get("data", {}).get("result"),
                    "error": result.get("error")
                }
            except Exception as e:
                return {
                    "package": package_name,
                    "success": False,
                    "error": str(e)
                }
        
        # Define concurrent package tests
        package_tests = [
            ("numpy", '''
import numpy as np
data = np.array([1, 2, 3, 4, 5])
results = {
    "package": "numpy", 
    "sum": int(data.sum()), 
    "shape": data.shape,
    "dtype": str(data.dtype)
}
results
            '''),
            
            ("pandas", '''
import pandas as pd
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
results = {
    "package": "pandas",
    "shape": df.shape,
    "columns": list(df.columns),
    "sum_A": int(df["A"].sum())
}
results
            '''),
            
            ("matplotlib", '''
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# Test basic matplotlib functionality
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([1, 2, 3], [1, 4, 2])
ax.set_title("Concurrent Test Plot")

# Save to virtual filesystem
plot_dir = Path('/plots/matplotlib')
plot_dir.mkdir(parents=True, exist_ok=True)
plot_file = plot_dir / 'concurrent_test.png'
fig.savefig(plot_file, dpi=100)
plt.close(fig)

results = {
    "package": "matplotlib",
    "plot_created": plot_file.exists() if hasattr(plot_file, 'exists') else True,
    "backend": matplotlib.get_backend()
}
results
            ''')
        ]
        
        # Execute tests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(execute_package_test, test) for test in package_tests]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all concurrent tests succeeded
        for result in results:
            assert result["success"] is True, f"Package {result['package']} failed: {result.get('error')}"
            assert result["result"]["package"] == result["package"]
        
        # Verify specific functionality
        numpy_result = next(r for r in results if r["package"] == "numpy")
        pandas_result = next(r for r in results if r["package"] == "pandas")
        matplotlib_result = next(r for r in results if r["package"] == "matplotlib")
        
        assert numpy_result["result"]["sum"] == 15  # 1+2+3+4+5
        assert numpy_result["result"]["shape"] == [5]
        
        assert pandas_result["result"]["shape"] == [3, 2]
        assert pandas_result["result"]["columns"] == ["A", "B"]
        assert pandas_result["result"]["sum_A"] == 6  # 1+2+3
        
        assert matplotlib_result["result"]["plot_created"] is True


class TestGivenStressAndPerformanceScenarios:
    """Test stress scenarios and performance edge cases."""

    def test_given_large_code_block_when_executed_then_handles_without_issues(
        self,
        api_client: requests.Session,
        server_health_check
    ):
        """
        Test execution of large, complex code blocks.
        
        Given: Large Python code block with multiple functions and operations
        When: Executed via raw API
        Then: Executes successfully without performance issues
        
        Args:
            api_client: HTTP client session
            server_health_check: Server health verification
        """
        large_code = f'''
import numpy as np
import pandas as pd
from pathlib import Path

def generate_test_data(size={TestConfig.STRESS_TEST_SIZE}):
    """Generate test data for stress testing."""
    return {{
        'numbers': list(range(size)),
        'squares': [i**2 for i in range(size)],
        'strings': [f"item_{{i}}_test_data" for i in range(size)],
        'floats': [i * 1.5 for i in range(size)]
    }}

def process_test_data(data):
    """Process test data using pandas."""
    df = pd.DataFrame(data)
    
    # Perform various operations
    numeric_summary = {{
        'total_rows': len(df),
        'sum_numbers': int(df['numbers'].sum()),
        'mean_numbers': float(df['numbers'].mean()),
        'max_square': int(df['squares'].max()),
        'sum_floats': float(df['floats'].sum())
    }}
    
    # String operations
    string_analysis = {{
        'first_string': df['strings'].iloc[0] if len(df) > 0 else None,
        'last_string': df['strings'].iloc[-1] if len(df) > 0 else None,
        'total_string_length': sum(len(s) for s in df['strings'])
    }}
    
    return numeric_summary, string_analysis

def complex_string_operations():
    """Perform complex string operations."""
    complex_strings = []
    for i in range({TestConfig.STRING_TEST_COUNT}):
        # Create strings with various complexities
        base_str = f"Complex string {{i}} with unicode: ñáéíóú"
        json_like = f'{{"id": {{i}}, "value": "string with quotes and \\\\'escapes\\\\'"}}' 
        file_path = str(Path(f"/data/files/file_{{i}}_with_spaces.txt"))
        
        complex_strings.extend([base_str, json_like, file_path])
    
    return {{
        'total_strings': len(complex_strings),
        'total_length': sum(len(s) for s in complex_strings),
        'sample_string': complex_strings[0] if complex_strings else None,
        'unicode_count': sum(1 for s in complex_strings if 'ñáéíóú' in s)
    }}

# Execute the stress test
print("Starting large code block stress test...")

test_data = generate_test_data()
numeric_summary, string_analysis = process_test_data(test_data)
string_ops = complex_string_operations()

final_results = {{
    "data_processing": numeric_summary,
    "string_analysis": string_analysis, 
    "complex_strings": string_ops,
    "stress_test_completed": True,
    "total_operations": len(test_data['numbers']) + len(string_ops)
}}

print(f"STRESS_TEST_COMPLETE: {{final_results['stress_test_completed']}}")
final_results
        '''
        
        # Use longer timeout for stress test
        result = execute_python_raw(api_client, large_code, TestConfig.LONG_TIMEOUT)
        
        assert result["success"] is True
        data = result["data"]
        final_results = data["result"]
        
        # Verify stress test results
        assert final_results["stress_test_completed"] is True
        assert final_results["data_processing"]["total_rows"] == TestConfig.STRESS_TEST_SIZE
        assert final_results["data_processing"]["sum_numbers"] == sum(range(TestConfig.STRESS_TEST_SIZE))
        assert final_results["complex_strings"]["total_strings"] == TestConfig.STRING_TEST_COUNT * 3
        assert final_results["complex_strings"]["unicode_count"] == TestConfig.STRING_TEST_COUNT
        
        # Verify performance is reasonable
        execution_time = data["executionTime"]
        assert execution_time < TestConfig.LONG_TIMEOUT * 1000  # Convert to milliseconds
        
        # Verify success message
        assert "STRESS_TEST_COMPLETE: True" in data["stdout"]

    def test_given_concurrent_users_when_executing_simultaneously_then_isolation_maintained(
        self,
        api_client: requests.Session,
        server_health_check,
        execution_timeout: int
    ):
        """
        Test concurrent user execution with proper isolation.
        
        Given: Multiple concurrent users executing Python code
        When: Each user has unique data and operations
        Then: All executions succeed with proper data isolation
        
        Args:
            api_client: HTTP client session
            server_health_check: Server health verification
            execution_timeout: Test timeout configuration
        """
        def simulate_user_execution(user_id):
            """Simulate individual user execution with unique data."""
            user_code = f'''
import numpy as np
import pandas as pd
from pathlib import Path

# User-specific data - each user has unique values
user_id = {user_id}
user_data = np.array([{user_id}, {user_id * 2}, {user_id * 3}])
user_df = pd.DataFrame({{
    "user": [user_id],
    "values": [user_data.tolist()],
    "timestamp": ["{time.time()}"]
}})

# Process user-specific data  
user_results = {{
    "user_id": user_id,
    "data_sum": int(user_data.sum()),
    "data_product": int(user_data.prod()),
    "df_shape": user_df.shape,
    "unique_marker": f"user_{{user_id}}_processed_successfully",
    "expected_sum": user_id * 6  # user_id + user_id*2 + user_id*3 = user_id * 6
}}

# Create unique file path for user
user_file = Path(f"/data/user_{{user_id}}_data.txt")
user_results["user_file_path"] = str(user_file)

print(f"USER_{{user_id}}_COMPLETE: {{user_results['data_sum']}}")
user_results
            '''
            
            try:
                result = execute_python_raw(api_client, user_code, execution_timeout)
                return result
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "data": {"result": {"user_id": user_id}}
                }
        
        # Execute multiple users concurrently
        user_ids = [101, 202, 303, 404, 505]
        with concurrent.futures.ThreadPoolExecutor(max_workers=TestConfig.CONCURRENT_USERS) as executor:
            futures = [executor.submit(simulate_user_execution, uid) for uid in user_ids]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all executions succeeded with proper isolation
        successful_results = []
        for result in results:
            assert result["success"] is True, f"User execution failed: {result.get('error')}"
            
            user_data = result["data"]["result"]
            user_id = user_data["user_id"]
            
            # Verify user-specific calculations are correct
            expected_sum = user_id * 6  # user_id + user_id*2 + user_id*3
            assert user_data["data_sum"] == expected_sum, f"User {user_id}: expected sum {expected_sum}, got {user_data['data_sum']}"
            assert user_data["expected_sum"] == expected_sum
            
            # Verify user isolation markers
            assert f"user_{user_id}_processed_successfully" in user_data["unique_marker"]
            assert f"user_{user_id}_data.txt" in user_data["user_file_path"]
            
            successful_results.append(user_data)
        
        # Verify all 5 users completed successfully
        assert len(successful_results) == 5
        processed_user_ids = {r["user_id"] for r in successful_results}
        assert processed_user_ids == set(user_ids)


class TestGivenEdgeCasesAndErrorScenarios:
    """Test edge cases and error handling scenarios."""

    def test_given_extreme_string_scenarios_when_executed_then_all_handled_correctly(
        self,
        api_client: requests.Session,
        server_health_check,
        execution_timeout: int
    ):
        """
        Test extreme string edge cases that might break execution.
        
        Given: Extreme string scenarios (backslashes, quotes, unicode, etc.)
        When: Executed via raw API
        Then: All scenarios are handled without breaking execution
        
        Args:
            api_client: HTTP client session
            server_health_check: Server health verification
            execution_timeout: Test timeout configuration
        """
        extreme_code = r'''
# Test extreme string cases that might break execution
from pathlib import Path
import json

extreme_cases = {}

# Backslash heavy strings (Windows paths, regex, etc.)
extreme_cases["backslashes"] = str(Path("C:/server/path/file.txt"))

# Regex-like patterns with escapes
extreme_cases["regex_pattern"] = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

# SQL-like strings with quotes and escapes
extreme_cases["sql_like"] = "SELECT * FROM table WHERE name = 'O\\'Reilly' AND status = \"active\""

# Multi-line with various content and quotes
extreme_cases["multiline"] = """Line 1 with "double quotes"
Line 2 with 'single quotes'  
Line 3 with \\n escape sequences
Line 4 with unicode: ñáéíóú 🐍 αβγδε"""

# Cross-platform file paths using pathlib  
extreme_cases["file_path"] = str(Path("C:/Users/name/Documents/file with spaces & symbols!.txt"))

# JSON strings within Python (nested quote nightmare)
extreme_cases["json_nightmare"] = json.dumps({
    "quotes": "Text with 'single' and \"double\" quotes", 
    "escapes": "Line\\nBreak\\tTab\\\\Backslash",
    "unicode": "Iñtërnâtiônàlizætiøn 🚀"
})

# Verify all cases are valid
total_cases = len(extreme_cases)
all_valid = all(isinstance(v, str) for v in extreme_cases.values())

# Test specific properties
sample_backslash = "server" in extreme_cases["backslashes"]  
sample_multiline_length = len(extreme_cases["multiline"])
file_path_valid = "Documents" in extreme_cases["file_path"]

verification_results = {
    "total_cases": total_cases,
    "all_valid": all_valid,
    "sample_backslash": sample_backslash,
    "multiline_length": sample_multiline_length, 
    "file_path_valid": file_path_valid,
    "extreme_test_passed": True
}

print(f"EXTREME_CASES_TEST: total={total_cases}, all_valid={all_valid}")
verification_results
        '''
        
        result = execute_python_raw(api_client, extreme_code, execution_timeout)
        
        assert result["success"] is True
        data = result["data"]
        verification_results = data["result"]
        
        # Verify all extreme cases were handled
        assert verification_results["total_cases"] == 6
        assert verification_results["all_valid"] is True
        assert verification_results["sample_backslash"] is True
        assert verification_results["multiline_length"] > 50
        assert verification_results["file_path_valid"] is True
        assert verification_results["extreme_test_passed"] is True
        
        # Verify success message
        assert "EXTREME_CASES_TEST:" in data["stdout"]
        assert "all_valid=True" in data["stdout"]

    def test_given_timeout_scenario_when_long_execution_then_proper_error_handling(
        self,
        api_client: requests.Session,
        server_health_check
    ):
        """
        Test timeout handling for long-running code.
        
        Given: Python code that takes longer than timeout
        When: Executed with short timeout
        Then: Returns proper timeout error structure
        
        Args:
            api_client: HTTP client session  
            server_health_check: Server health verification
        """
        # Code that will take longer than our short timeout
        long_running_code = '''
import time
print("Starting long operation...")
time.sleep(5)  # This will exceed our 2-second timeout
print("This should not appear in output")
result = "Should not complete"
result
        '''
        
        # Use very short timeout to trigger timeout scenario
        short_timeout = 2  # seconds
        
        result = execute_python_raw(api_client, long_running_code, short_timeout)
        
        # Should return error structure for timeout
        assert result["success"] is False
        assert result["data"] is None
        assert result["error"] is not None
        assert isinstance(result["error"], str)
        
        # Error should indicate timeout
        error_msg = result["error"].lower()
        assert "timeout" in error_msg or "time" in error_msg or "exceed" in error_msg


# ========================================
# CUSTOM PYTEST MARKERS AND CONFIGURATION
# ========================================

# Mark tests that require server to be running
pytestmark = pytest.mark.integration

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests requiring running server"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow running (longer than 30 seconds)"
    )


# ========================================
# MAIN EXECUTION 
# ========================================

if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v",
        "--tb=short",
        "--durations=10"
    ])
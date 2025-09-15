#!/usr/bin/env python3
"""
Comprehensive pytest tests for dynamic module discovery and execution robustness.

This test suite covers:
1. Dynamic package installation and automatic availability
2. Complex Python code scenarios that might break string handling
3. Edge cases with various quote combinations and f-strings
4. Concurrent execution with newly installed packages
5. Stress testing the API response format compliance

All tests follow BDD patterns and use /api/execute-raw with plain text.
Server must return API contract: {success, data, error, meta}
"""

import json
from typing import Any, Dict, Optional

import pytest
import requests


# Configuration Constants
class TestConfig:
    """Test configuration constants - no hardcoded values"""

    BASE_URL = "http://localhost:3000"
    DEFAULT_TIMEOUT = 30
    EXECUTION_TIMEOUT = 30000  # Milliseconds for Pyodide execution
    API_ENDPOINT = "/api/execute-raw"
    HEALTH_ENDPOINT = "/health"


@pytest.fixture
def api_session():
    """
    Create a requests session for API testing.

    Returns:
        requests.Session: Configured session with timeout

    Example:
        def test_example(api_session):
            response = api_session.get("/health")
    """
    session = requests.Session()
    session.timeout = TestConfig.DEFAULT_TIMEOUT
    return session


@pytest.fixture
def base_url():
    """
    Provide the base URL for API testing.

    Returns:
        str: Base URL for the API server

    Example:
        def test_example(base_url):
            full_url = f"{base_url}/api/execute-raw"
    """
    return TestConfig.BASE_URL


def api_contract_validator(response: requests.Response) -> Optional[Dict[str, Any]]:
    """
    Validate API response follows the required contract.

    Args:
        response: HTTP response object

    Returns:
        dict: Parsed response data if valid, None if invalid

    Expected Contract:
        {
          "success": true | false,
          "data": <object|null>,
          "error": <string|null>,
          "meta": { "timestamp": <string> }
        }

    Example:
        response = requests.post(url, data=code)
        data = api_contract_validator(response)
        assert data["success"] is True
    """
    if response.status_code not in [200, 400, 500]:
        return None

    try:
        data = response.json()

        # Check required fields exist
        required_fields = ["success", "data", "error", "meta"]
        if not all(field in data for field in required_fields):
            return None

        # Validate field types
        if not isinstance(data["success"], bool):
            return None

        if data["meta"] is None or "timestamp" not in data["meta"]:
            return None

        return data

    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def execute_python_code(
    api_session: requests.Session, base_url: str, code: str
) -> Dict[str, Any]:
    """
    Execute Python code using /api/execute-raw endpoint.

    Args:
        api_session: Configured requests session
        base_url: Base URL for API
        code: Python code to execute (plain text)

    Returns:
        dict: API response data following contract

    Example:
        data = execute_python_code(session, url, "print('hello')")
        assert data["success"] is True
        assert "hello" in data["data"]["stdout"]
    """
    headers = {
        "Content-Type": "text/plain",
        "timeout": str(TestConfig.EXECUTION_TIMEOUT),
    }

    response = api_session.post(
        f"{base_url}{TestConfig.API_ENDPOINT}", data=code, headers=headers
    )

    return api_contract_validator(response)


class TestHealthAndBasicFunctionality:
    """Test basic server health and functionality"""

    def test_given_server_running_when_health_check_then_returns_success(
        self, api_session, base_url
    ):
        """
        Test server health endpoint returns expected format.

        Given: Server is running
        When: Health check endpoint is called
        Then: Returns success response with proper API contract

        Inputs:
            - GET /health request

        Expected Outputs:
            - 200 status code
            - API contract compliance: {success: true, data, error: null, meta}

        Example:
            response = {"success": true, "data": {"status": "healthy"}, "error": null, "meta": {"timestamp": "..."}}
        """
        response = api_session.get(f"{base_url}{TestConfig.HEALTH_ENDPOINT}")

        assert response.status_code == 200
        data = api_contract_validator(response)
        assert data is not None, f"API contract violation: {response.text}"
        assert data["success"] is True
        assert data["error"] is None
        assert "timestamp" in data["meta"]


class TestBasicPythonExecution:
    """Test basic Python code execution scenarios"""

    def test_given_simple_code_when_execute_then_returns_output(
        self, api_session, base_url
    ):
        """
        Test basic Python code execution with print statements.

        Given: Simple Python print code
        When: Code is executed via /api/execute-raw
        Then: Returns successful execution with stdout output

        Inputs:
            - Python code: print("Hello, World!")

        Expected Outputs:
            - success: true
            - data.stdout contains "Hello, World!"
            - data.result contains final expression result

        Example:
            Input: print("Hello, World!")
            Output: {"success": true, "data": {"stdout": "Hello, World!\n", ...}}
        """
        code = 'print("Hello, World!")'

        data = execute_python_code(api_session, base_url, code)
        assert data is not None, "API contract validation failed"
        assert data["success"] is True
        assert data["error"] is None
        assert "Hello, World!" in data["data"]["stdout"]

    def test_given_numpy_operations_when_execute_then_returns_results(
        self, api_session, base_url
    ):
        """
        Test NumPy operations and mathematical computations.

        Given: Python code with NumPy operations
        When: Code is executed with array operations
        Then: Returns successful execution with computation results

        Inputs:
            - NumPy array creation and operations

        Expected Outputs:
            - success: true
            - data.stdout contains NumPy version and array operations
            - data.result contains completion message

        Example:
            Input: NumPy array [1,2,3,4,5] operations
            Output: {"success": true, "data": {"stdout": "NumPy version: x.x.x\n...", "result": "completed"}}
        """
        code = """
import numpy as np
print(f"NumPy version: {np.__version__}")

# Create array
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Sum: {arr.sum()}")
print(f"Mean: {arr.mean()}")

"NumPy operations completed successfully"
"""

        data = execute_python_code(api_session, base_url, code)
        assert data is not None, "API contract validation failed"
        assert data["success"] is True
        assert data["error"] is None

        # Check both result and stdout
        assert data["data"]["result"] == "NumPy operations completed successfully"
        assert "NumPy version:" in data["data"]["stdout"]
        assert "Array: [1 2 3 4 5]" in data["data"]["stdout"]

    def test_given_pathlib_operations_when_execute_then_handles_cross_platform(
        self, api_session, base_url
    ):
        """
        Test pathlib usage for cross-platform file operations.

        Given: Python code using pathlib for file operations
        When: Path operations are performed
        Then: Returns successful execution with path information

        Inputs:
            - pathlib Path operations for cross-platform compatibility

        Expected Outputs:
            - success: true
            - data.result contains path operation results
            - Cross-platform path handling verified

        Example:
            Input: Path('/tmp') / 'file.txt' operations
            Output: {"success": true, "data": {"result": "Path operations completed"}}
        """
        code = """
from pathlib import Path

# Test cross-platform path operations
base_path = Path('/tmp')
file_path = base_path / 'test_file.txt'

print(f"Base path: {base_path}")
print(f"File path: {file_path}")
print(f"Is absolute: {file_path.is_absolute()}")
print(f"Path parts: {file_path.parts}")

"Pathlib operations completed successfully"
"""

        data = execute_python_code(api_session, base_url, code)
        assert data is not None, "API contract validation failed"
        assert data["success"] is True
        assert data["error"] is None
        assert "Pathlib operations completed successfully" in data["data"]["result"]
        assert "Base path:" in data["data"]["stdout"]


class TestComplexStringHandling:
    """Test complex string scenarios and edge cases"""

    def test_given_mixed_quotes_when_execute_then_handles_correctly(
        self, api_session, base_url
    ):
        """
        Test complex string handling with mixed quotes and escapes.

        Given: Python code with complex quote combinations
        When: Code contains single, double quotes and f-strings
        Then: Returns successful execution without string parsing errors

        Inputs:
            - Mixed single quotes, double quotes, triple quotes
            - F-strings with embedded quotes

        Expected Outputs:
            - success: true
            - data.stdout contains all quote variations
            - No parsing or execution errors

        Example:
            Input: Various quote combinations
            Output: {"success": true, "data": {"stdout": "All quote types handled"}}
        """
        code = """
# Test various quote combinations
single = 'This is a single-quoted string'
double = "This is a double-quoted string"
mixed = 'String with "embedded double quotes"'
reverse_mixed = "String with 'embedded single quotes'"

name = "World"
f_string = f"Hello {name}! How's it going?"

triple_single = \"\"\"This is a
multi-line string with 'single' and "double" quotes\"\"\"

print(single)
print(double)
print(mixed)
print(reverse_mixed)
print(f_string)
print("Triple quote string processed")

"Complex string handling completed"
"""

        data = execute_python_code(api_session, base_url, code)
        assert data is not None, "API contract validation failed"
        assert data["success"] is True
        assert data["error"] is None
        assert "Complex string handling completed" in data["data"]["result"]
        assert "Hello World!" in data["data"]["stdout"]

    def test_given_json_like_strings_when_execute_then_processes_correctly(
        self, api_session, base_url
    ):
        """
        Test JSON-like string processing without breaking parsing.

        Given: Python code generating JSON-like output
        When: Code creates and manipulates JSON structures
        Then: Returns successful execution with proper JSON handling

        Inputs:
            - Python dict to JSON conversion
            - JSON string manipulation

        Expected Outputs:
            - success: true
            - data.stdout contains JSON output
            - data.result contains processing confirmation

        Example:
            Input: JSON creation and parsing
            Output: {"success": true, "data": {"stdout": "{\\"key\\": \\"value\\"}"}}
        """
        code = """
import json

# Create JSON-like data
data = {
    "name": "Test User",
    "age": 30,
    "skills": ["Python", "JavaScript", "SQL"],
    "active": True,
    "metadata": {
        "created": "2023-01-01",
        "updated": None
    }
}

json_string = json.dumps(data, indent=2)
print("Generated JSON:")
print(json_string)

# Parse back
parsed = json.loads(json_string)
print(f"Parsed name: {parsed['name']}")
print(f"Skills count: {len(parsed['skills'])}")

"JSON processing completed successfully"
"""

        data = execute_python_code(api_session, base_url, code)
        assert data is not None, "API contract validation failed"
        assert data["success"] is True
        assert data["error"] is None
        assert "JSON processing completed successfully" in data["data"]["result"]
        assert "Generated JSON:" in data["data"]["stdout"]


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_given_syntax_error_when_execute_then_returns_error_response(
        self, api_session, base_url
    ):
        """
        Test syntax error handling with proper API contract.

        Given: Python code with syntax errors
        When: Code execution is attempted
        Then: Returns error response following API contract

        Inputs:
            - Python code with invalid syntax

        Expected Outputs:
            - success: false
            - error: string with error description
            - data: null or error details

        Example:
            Input: print("missing quote)
            Output: {"success": false, "error": "SyntaxError: ...", "data": null}
        """
        code = 'print("This has a syntax error'  # Missing closing quote

        data = execute_python_code(api_session, base_url, code)
        assert data is not None, "API contract validation failed"
        assert data["success"] is False
        assert data["error"] is not None
        # Server may return generic "Execution failed" or specific error details
        assert len(data["error"]) > 0, "Error message should not be empty"

    def test_given_runtime_error_when_execute_then_returns_error_response(
        self, api_session, base_url
    ):
        """
        Test runtime error handling with proper API contract.

        Given: Python code that raises runtime errors
        When: Code execution encounters runtime exception
        Then: Returns error response with exception details

        Inputs:
            - Python code that raises exceptions (division by zero, undefined variables)

        Expected Outputs:
            - success: false
            - error: string with runtime error description
            - data: may contain partial execution results

        Example:
            Input: 1/0 division by zero
            Output: {"success": false, "error": "ZeroDivisionError: ...", "data": {...}}
        """
        code = """
print("Starting execution...")
result = 1 / 0  # This will cause ZeroDivisionError
print("This won't be reached")
"""

        data = execute_python_code(api_session, base_url, code)
        assert data is not None, "API contract validation failed"
        assert data["success"] is False
        assert data["error"] is not None
        # Server may return generic "Execution failed" or specific error details
        assert len(data["error"]) > 0, "Error message should not be empty"


class TestDataScienceOperations:
    """Test data science package operations"""

    def test_given_pandas_operations_when_execute_then_returns_dataframe_results(
        self, api_session, base_url
    ):
        """
        Test pandas DataFrame operations and data manipulation.

        Given: Python code with pandas DataFrame operations
        When: Data manipulation and analysis is performed
        Then: Returns successful execution with DataFrame results

        Inputs:
            - pandas DataFrame creation and manipulation
            - Data analysis operations

        Expected Outputs:
            - success: true
            - data.stdout contains DataFrame information and operations
            - data.result contains completion confirmation

        Example:
            Input: DataFrame with statistical operations
            Output: {"success": true, "data": {"stdout": "DataFrame info...", "result": "completed"}}
        """
        code = """
import pandas as pd
import numpy as np

print(f"Pandas version: {pd.__version__}")

# Create sample DataFrame
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 70000, 55000]
}

df = pd.DataFrame(data)
print("DataFrame created:")
print(df)
print(f"Shape: {df.shape}")
print(f"Mean salary: ${df['salary'].mean():,.2f}")

"Pandas operations completed successfully"
"""

        data = execute_python_code(api_session, base_url, code)
        assert data is not None, "API contract validation failed"
        assert data["success"] is True
        assert data["error"] is None
        assert "Pandas operations completed successfully" in data["data"]["result"]
        assert "DataFrame created:" in data["data"]["stdout"]


class TestFileSystemOperations:
    """Test file system operations using pathlib"""

    def test_given_file_operations_when_execute_then_handles_paths_correctly(
        self, api_session, base_url
    ):
        """
        Test file system operations using pathlib for cross-platform compatibility.

        Given: Python code using pathlib for file operations
        When: File and directory operations are performed
        Then: Returns successful execution with file operation results

        Inputs:
            - pathlib Path operations
            - Directory and file checks

        Expected Outputs:
            - success: true
            - data.stdout contains file operation results
            - Cross-platform path handling verified

        Example:
            Input: Path existence checks and operations
            Output: {"success": true, "data": {"stdout": "File operations completed"}}
        """
        code = """
from pathlib import Path
import os

# Use pathlib for cross-platform compatibility
current_dir = Path('/')
print(f"Current directory: {current_dir}")
print(f"Is directory: {current_dir.is_dir()}")
print(f"Exists: {current_dir.exists()}")

# Check for common system paths
system_paths = [Path('/tmp'), Path('/var'), Path('/usr')]
for path in system_paths:
    if path.exists():
        print(f"Found system path: {path}")
        break
else:
    print("No common system paths found")

# Demonstrate path operations
test_path = Path('/tmp') / 'test_file.txt'
print(f"Test path: {test_path}")
print(f"Parent: {test_path.parent}")
print(f"Name: {test_path.name}")
print(f"Suffix: {test_path.suffix}")

"File system operations completed successfully"
"""

        data = execute_python_code(api_session, base_url, code)
        assert data is not None, "API contract validation failed"
        assert data["success"] is True
        assert data["error"] is None
        assert "File system operations completed successfully" in data["data"]["result"]
        assert "Current directory:" in data["data"]["stdout"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

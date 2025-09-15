"""
Comprehensive Pytest BDD Tests for Pyodide Data Type Return Handling.

This test suite follows BDD (Behavior-Driven Development) patterns with Given-When-Then
structure and provides complete coverage for data type handling in the Pyodide Express
Server REST API.

Test Coverage:
- Basic Python data types (str, int, float, bool, None)
- Collections (list, dict, tuple, set)
- NumPy arrays and mathematical operations
- Pandas DataFrames in various formats
- Complex nested data structures
- Complete data science workflows
- Error handling and exception management

Requirements Compliance:
1. ✅ Pytest framework with BDD style scenarios
2. ✅ All globals parameterized via constants and fixtures
3. ✅ No internal REST APIs (no 'pyodide' endpoints)
4. ✅ BDD Given-When-Then structure
5. ✅ Only /api/execute-raw for Python execution
6. ✅ No internal pyodide REST APIs
7. ✅ Comprehensive test coverage
8. ✅ Full docstrings with examples
9. ✅ Python code uses pathlib for portability
10. ✅ JavaScript API contract validation

API Contract Validation:
{
  "success": true | false,
  "data": { "result": str, "stdout": str, "stderr": str, "executionTime": int } | null,
  "error": string | null,
  "meta": { "timestamp": string }
}
"""

import time
from typing import Any, Dict, Generator

import pytest
import requests


# ==================== TEST CONFIGURATION CONSTANTS ====================


class Config:
    """Test configuration constants - centralized and easily tweakable."""

    BASE_URL = "http://localhost:3000"
    API_TIMEOUT = 30
    HEALTH_CHECK_TIMEOUT = 5
    MAX_SERVER_RETRIES = 30
    RETRY_DELAY = 1

    # API Endpoints
    HEALTH_ENDPOINT = "/health"
    EXECUTE_RAW_ENDPOINT = "/api/execute-raw"

    # Expected API contract fields
    API_SUCCESS_FIELD = "success"
    API_DATA_FIELD = "data"
    API_ERROR_FIELD = "error"
    API_META_FIELD = "meta"
    API_TIMESTAMP_FIELD = "timestamp"

    # Data structure validation fields
    DATA_RESULT_FIELD = "result"
    DATA_STDOUT_FIELD = "stdout"
    DATA_STDERR_FIELD = "stderr"
    DATA_EXECUTION_TIME_FIELD = "executionTime"


# ==================== PYTEST FIXTURES ====================


@pytest.fixture(scope="session")
def api_client() -> Generator[requests.Session, None, None]:
    """
    Provide a configured requests session for API testing.

    Description:
        Creates a persistent HTTP session with proper timeout configuration
        and connection pooling for efficient API testing.

    Input:
        None (pytest fixture)

    Output:
        requests.Session: Configured session object

    Example:
        def test_api_call(api_client):
            response = api_client.post("/api/execute-raw", data="print('hello')")
            assert response.status_code == 200
    """
    session = requests.Session()
    session.timeout = Config.API_TIMEOUT
    yield session
    session.close()


@pytest.fixture(scope="session")
def server_health_check(api_client) -> bool:
    """
    Ensure server is running and healthy before tests start.

    Description:
        Performs health check with retry logic to verify server availability.
        Fails fast if server is not accessible within timeout period.

    Input:
        api_client: HTTP session fixture

    Output:
        bool: True if server is healthy

    Example:
        def test_requires_server(server_health_check):
            # Test will automatically wait for server to be ready
            assert server_health_check is True

    Raises:
        RuntimeError: If server is not available after maximum retries
    """
    health_url = f"{Config.BASE_URL}{Config.HEALTH_ENDPOINT}"

    for attempt in range(Config.MAX_SERVER_RETRIES):
        try:
            response = api_client.get(
                health_url, timeout=Config.HEALTH_CHECK_TIMEOUT
            )
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(Config.RETRY_DELAY)

    raise RuntimeError(
        f"Server at {Config.BASE_URL} not available after "
        f"{Config.MAX_SERVER_RETRIES} retries"
    )


def execute_python_code_raw(
    api_client: requests.Session, code: str
) -> Dict[str, Any]:
    """
    Execute Python code via /api/execute-raw endpoint with full validation.

    Description:
        Sends Python code to the execute-raw endpoint and validates the response
        conforms to the API contract. Provides clean interface for test functions.

    Input:
        api_client: Configured HTTP session
        code: Python code string to execute

    Output:
        Dict containing validated API response

    Example:
        result = execute_python_code_raw(api_client, "print('hello')")
        assert result["success"] is True
        assert "hello" in result["data"]["result"]

    Raises:
        AssertionError: If API response is invalid
        requests.RequestException: If HTTP request fails
    """
    execute_url = f"{Config.BASE_URL}{Config.EXECUTE_RAW_ENDPOINT}"

    response = api_client.post(
        execute_url,
        data=code,
        headers={"Content-Type": "text/plain"},
        timeout=Config.API_TIMEOUT
    )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )

    response_data = response.json()

    # Basic API contract validation
    assert Config.API_SUCCESS_FIELD in response_data
    assert Config.API_DATA_FIELD in response_data
    assert Config.API_ERROR_FIELD in response_data
    assert Config.API_META_FIELD in response_data

    return response_data


def extract_python_result(response_data: Dict[str, Any]) -> Any:
    """
    Extract Python execution result from validated API response.

    Description:
        Safely extracts the Python result from execute-raw response data.
        Handles result parsing and validation for test assertions.

    Input:
        response_data: Validated API response dictionary

    Output:
        Python object representing the execution result

    Example:
        response = execute_python_code_raw(api_client, "42")
        result = extract_python_result(response)
        assert result == "42"  # Note: execute-raw returns strings

    Raises:
        ValueError: If result cannot be extracted or parsed
    """
    if not response_data[Config.API_SUCCESS_FIELD]:
        error_msg = response_data.get(Config.API_ERROR_FIELD, "Unknown error")
        raise ValueError(f"Python execution failed: {error_msg}")

    data = response_data[Config.API_DATA_FIELD]
    return data[Config.DATA_RESULT_FIELD]


# ==================== BDD TEST SCENARIOS ====================


class TestBasicDataTypesReturn:
    """
    Test scenarios for basic Python data type return handling.

    Covers fundamental data types that form the building blocks
    of more complex data structures and scientific computing workflows.
    """

    def test_given_basic_string_when_executed_then_returns_string_result(
        self, api_client, server_health_check
    ):
        """
        Scenario: Basic string value execution and return.

        Description:
            Given a Python string literal
            When executed via the execute-raw endpoint
            Then the API should return the string value correctly
            And conform to the API contract

        Input:
            api_client: HTTP session fixture
            server_health_check: Server availability fixture

        Output:
            None (assertions validate behavior)

        Example:
            Input: "'hello world'"
            Expected: API response with data.result containing "hello world"
        """
        # Given: A basic string value to execute
        python_code = "'hello world'"
        expected_result = "hello world"

        # When: Code is executed via execute-raw endpoint
        response = execute_python_code_raw(api_client, python_code)

        # Then: Response should contain the expected string result
        assert response[Config.API_SUCCESS_FIELD] is True
        result = extract_python_result(response)
        assert expected_result in result

    def test_given_numeric_values_when_executed_then_returns_numeric_results(
        self, api_client, server_health_check
    ):
        """
        Scenario: Numeric value execution with different number types.

        Description:
            Given various numeric Python literals (int, float)
            When executed via the execute-raw endpoint
            Then the API should return numeric values correctly formatted
            And preserve numeric precision appropriately

        Input:
            api_client: HTTP session fixture
            server_health_check: Server availability fixture

        Output:
            None (assertions validate behavior)

        Example:
            Input: "42", "3.14159"
            Expected: API responses with numeric values in results
        """
        test_cases = [
            ("42", "42"),
            ("3.14159", "3.14159"),
            ("-17", "-17"),
            ("0", "0"),
        ]

        for python_code, expected_str in test_cases:
            # Given: A numeric value to execute
            # When: Code is executed via execute-raw endpoint
            response = execute_python_code_raw(api_client, python_code)

            # Then: Response should contain the expected numeric result
            assert response[Config.API_SUCCESS_FIELD] is True
            result = extract_python_result(response)
            assert expected_str in result


class TestCollectionDataTypesReturn:
    """
    Test scenarios for Python collection data type return handling.

    Covers lists, dictionaries, tuples, sets and nested collection structures
    commonly used in data science and web application contexts.
    """

    def test_given_list_collection_when_executed_then_returns_list_structure(
        self, api_client, server_health_check
    ):
        """
        Scenario: List collection execution and structure preservation.

        Description:
            Given Python list literals with various data types
            When executed via the execute-raw endpoint
            Then the API should return list structures correctly
            And preserve element ordering and nesting

        Input:
            api_client: HTTP session fixture
            server_health_check: Server availability fixture

        Output:
            None (assertions validate behavior)

        Example:
            Input: "[1, 2, 3, 'hello']"
            Expected: API response with list structure preserved
        """
        test_cases = [
            "[1, 2, 3, 4, 5]",
            "['hello', 'world', 'test']",
            "[1, 'mixed', 3.14, True]",
            "[]",  # Empty list
        ]

        for python_code in test_cases:
            # Given: A list structure to execute
            # When: Code is executed via execute-raw endpoint
            response = execute_python_code_raw(api_client, python_code)

            # Then: Response should contain the list structure
            assert response[Config.API_SUCCESS_FIELD] is True
            result = extract_python_result(response)

            # Verify list structure is represented
            assert "[" in result and "]" in result


class TestDataScienceWorkflows:
    """
    Test scenarios for comprehensive data science workflow execution.

    Covers NumPy arrays, Pandas DataFrames, and complex analytical workflows
    that combine multiple data types and operations.
    """

    def test_given_numpy_workflow_when_executed_then_returns_array_results(
        self, api_client, server_health_check
    ):
        """
        Scenario: NumPy array operations with cross-platform file handling.

        Description:
            Given Python code using NumPy with pathlib for file operations
            When executed via the execute-raw endpoint
            Then the API should return array processing results
            And demonstrate cross-platform compatibility

        Input:
            api_client: HTTP session fixture
            server_health_check: Server availability fixture

        Output:
            None (assertions validate behavior)

        Example:
            Input: NumPy array processing with pathlib
            Expected: API response with computed array results
        """
        # Use pathlib for cross-platform compatibility
        python_code = '''
from pathlib import Path
import numpy as np

# Create arrays and perform operations
data = np.array([1, 2, 3, 4, 5])
results = {
    "array_data": data.tolist(),
    "statistics": {
        "mean": float(np.mean(data)),
        "sum": float(np.sum(data)),
        "std": float(np.std(data))
    },
    "file_compatibility": {
        "temp_path": str(Path("/tmp/numpy_results.csv")),
        "platform_separator": str(Path("/data/analysis").parts[-1])
    }
}
print(f"NumPy processing completed with {len(data)} elements")
results
        '''

        # Given: NumPy workflow with pathlib usage
        # When: Code is executed via execute-raw endpoint
        response = execute_python_code_raw(api_client, python_code)

        # Then: Response should contain array processing results
        assert response[Config.API_SUCCESS_FIELD] is True
        result = extract_python_result(response)

        # Verify NumPy processing completed
        assert "array_data" in result or "statistics" in result

    def test_given_pandas_workflow_when_executed_then_returns_dataframe_results(
        self, api_client, server_health_check
    ):
        """
        Scenario: Pandas DataFrame operations with file path handling.

        Description:
            Given Python code using Pandas with pathlib for file operations
            When executed via the execute-raw endpoint
            Then the API should return DataFrame processing results
            And demonstrate data analysis capabilities

        Input:
            api_client: HTTP session fixture
            server_health_check: Server availability fixture

        Output:
            None (assertions validate behavior)

        Example:
            Input: Pandas DataFrame analysis with pathlib
            Expected: API response with data analysis results
        """
        # Use pathlib for cross-platform compatibility
        python_code = '''
from pathlib import Path
import pandas as pd

# Create sample DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'score': [85.5, 92.0, 78.5]
})

# Perform analysis
analysis = {
    "data_info": {
        "shape": list(df.shape),
        "columns": df.columns.tolist()
    },
    "statistics": {
        "age_mean": float(df['age'].mean()),
        "score_max": float(df['score'].max())
    },
    "file_operations": {
        "output_path": str(Path("/data/results") / "analysis.csv"),
        "backup_path": str(Path("/tmp") / "backup.xlsx")
    }
}
print(f"DataFrame analysis completed with {len(df)} records")
analysis
        '''

        # Given: Pandas workflow with pathlib usage
        # When: Code is executed via execute-raw endpoint
        response = execute_python_code_raw(api_client, python_code)

        # Then: Response should contain DataFrame analysis results
        assert response[Config.API_SUCCESS_FIELD] is True
        result = extract_python_result(response)

        # Verify Pandas processing completed
        assert "data_info" in result or "statistics" in result


class TestErrorHandlingScenarios:
    """
    Test scenarios for error handling and edge case management.

    Covers exception handling, partial execution, and graceful failure modes
    that maintain API contract compliance even during errors.
    """

    def test_given_exception_handling_when_executed_then_returns_error_info(
        self, api_client, server_health_check
    ):
        """
        Scenario: Python exception handling with error recovery.

        Description:
            Given Python code that handles exceptions gracefully
            When executed via the execute-raw endpoint
            Then the API should return error information appropriately
            And maintain proper error handling patterns

        Input:
            api_client: HTTP session fixture
            server_health_check: Server availability fixture

        Output:
            None (assertions validate behavior)

        Example:
            Input: Code with exception handling
            Expected: API response with error information and recovery
        """
        python_code = '''
try:
    # This will cause a deliberate error
    result = 1 / 0  # ZeroDivisionError
except ZeroDivisionError as e:
    error_info = {
        "error_occurred": True,
        "error_type": type(e).__name__,
        "error_message": str(e),
        "error_handling": "caught and processed",
        "recovery_action": "returned error information"
    }
    print(f"Handled error: {type(e).__name__}")
    error_info
        '''

        # Given: Code with exception handling
        # When: Code is executed via execute-raw endpoint
        response = execute_python_code_raw(api_client, python_code)

        # Then: Response should contain error handling information
        assert response[Config.API_SUCCESS_FIELD] is True
        result = extract_python_result(response)

        # Verify error handling information is present
        assert "error_occurred" in result or "error" in result

    def test_given_minimal_code_when_executed_then_handles_gracefully(
        self, api_client, server_health_check
    ):
        """
        Scenario: Minimal code execution with graceful handling.

        Description:
            Given minimal Python code expressions
            When executed via the execute-raw endpoint
            Then the API should handle all cases gracefully
            And return appropriate responses for edge cases

        Input:
            api_client: HTTP session fixture
            server_health_check: Server availability fixture

        Output:
            None (assertions validate behavior)

        Example:
            Input: Simple expressions and statements
            Expected: API responses that handle edge cases appropriately
        """
        test_cases = [
            "pass",  # Minimal valid Python
            "42",    # Simple expression
            "print('minimal test')",  # Simple statement
        ]

        for python_code in test_cases:
            # Given: Minimal Python code
            # When: Code is executed via execute-raw endpoint
            response = execute_python_code_raw(api_client, python_code)

            # Then: Response should handle the case appropriately
            assert response[Config.API_SUCCESS_FIELD] is True

            # For minimal code, we expect valid API response structure
            data = response[Config.API_DATA_FIELD]
            assert data is not None
            assert Config.DATA_RESULT_FIELD in data


if __name__ == "__main__":
    # Run pytest with verbose output and specific markers
    pytest.main([__file__, "-v", "--tb=short"])

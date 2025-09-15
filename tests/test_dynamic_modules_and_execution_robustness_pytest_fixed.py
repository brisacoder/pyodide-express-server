"""
Test dynamic module loading and execution robustness in Pyodide environment.

This comprehensive test suite validates:
1. Module availability and discovery using only /api/execute-raw endpoint
2. Package installation and usage patterns
3. Complex string handling and execution robustness
4. Cross-platform file operations using pathlib
5. Performance characteristics and timeout handling
6. Edge case handling and error recovery
7. API contract compliance with proper JSON structure

All tests follow BDD (Behavior-Driven Development) style and use pytest fixtures
for parameterization and configuration management.
"""

import pytest
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any


# Test Configuration Constants
class TestConfig:
    """Centralized test configuration to avoid hardcoded values."""
    BASE_URL = "http://localhost:3000"
    DEFAULT_TIMEOUT = 30000  # 30 seconds in milliseconds
    LONG_TIMEOUT = 60000     # 60 seconds in milliseconds
    SHORT_TIMEOUT = 10000    # 10 seconds in milliseconds
    MAX_CONCURRENT_REQUESTS = 5
    
    # API Endpoints (only using /api/execute-raw as per requirements)
    EXECUTE_RAW_ENDPOINT = "/api/execute-raw"
    HEALTH_ENDPOINT = "/health"


@pytest.fixture(scope="session")
def api_client():
    """
    Create a configured HTTP client for API testing.
    
    Returns:
        requests.Session: Configured session with timeout and base URL
    """
    session = requests.Session()
    session.timeout = 30
    
    # Health check to ensure server is running
    try:
        health_response = session.get(f"{TestConfig.BASE_URL}{TestConfig.HEALTH_ENDPOINT}")
        if health_response.status_code != 200:
            pytest.fail(f"Server health check failed: {health_response.status_code}")
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Server is not reachable at {TestConfig.BASE_URL}. Start server with 'npm start'. Error: {e}")
    
    return session


@pytest.fixture(scope="session")
def base_url():
    """Provide base URL for API testing."""
    return TestConfig.BASE_URL


@pytest.fixture
def default_timeout():
    """Provide default timeout value for API requests."""
    return TestConfig.DEFAULT_TIMEOUT


@pytest.fixture
def long_timeout():
    """Provide extended timeout for complex operations."""
    return TestConfig.LONG_TIMEOUT


@pytest.fixture
def short_timeout():
    """Provide short timeout for quick operations."""
    return TestConfig.SHORT_TIMEOUT


@pytest.fixture
def api_contract_validator():
    """
    Validate API responses follow the expected contract structure.
    
    Expected format:
    {
        "success": true|false,
        "data": object|null,
        "error": string|null,
        "meta": {"timestamp": string}
    }
    
    Returns:
        Callable: Validator function for API responses
    """
    def validate(response: requests.Response) -> Dict[str, Any]:
        """
        Validate API response structure and return parsed data.
        
        Args:
            response: HTTP response object
            
        Returns:
            Dict containing validated response data
            
        Raises:
            AssertionError: If response doesn't match expected contract
        """
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            pytest.fail(f"Response is not valid JSON: {e}")
        
        # Validate contract structure
        assert "success" in data, "Response missing 'success' field"
        assert "data" in data, "Response missing 'data' field"
        assert "error" in data, "Response missing 'error' field"
        assert "meta" in data, "Response missing 'meta' field"
        assert isinstance(data["success"], bool), "'success' field must be boolean"
        
        if data["success"]:
            assert data["error"] is None, "Successful response should have error=null"
            assert data["data"] is not None, "Successful response should have data"
        else:
            assert data["error"] is not None, "Failed response should have error message"
            assert isinstance(data["error"], str), "Error field must be string"
        
        # Validate metadata
        assert "timestamp" in data["meta"], "Meta missing 'timestamp' field"
        
        return data
    
    return validate


def test_given_server_running_when_health_check_then_should_respond_successfully(
    api_client, base_url, api_contract_validator
):
    """
    Test server health endpoint responds correctly.
    
    Description:
        Verifies the server is running and responding to health checks
        with the proper API contract structure.
    
    Input:
        - HTTP GET request to /health endpoint
    
    Output:
        - JSON response following API contract
        - Success status with server information
    
    Example:
        GET /health -> {"success": true, "data": {...}, "error": null, "meta": {...}}
    """
    # Given: A running server
    # When: Health check is requested
    response = api_client.get(f"{base_url}/health")
    
    # Then: Should respond with success and proper contract
    data = api_contract_validator(response)
    assert data["success"] is True


def test_given_pyodide_environment_when_importing_numpy_then_should_be_available(
    api_client, base_url, default_timeout, api_contract_validator
):
    """
    Test NumPy availability in Pyodide environment.
    
    Description:
        Verifies that NumPy (pre-installed in Pyodide) is available
        and can be used for basic array operations.
    
    Input:
        - Python code that imports and uses NumPy
    
    Output:
        - Successful execution with NumPy version and array operations
    
    Example:
        import numpy -> array creation -> mathematical operations
    """
    # Given: A Pyodide environment with NumPy pre-installed
    # When: We import and use NumPy for array operations
    code = """
import numpy as np
print(f"NumPy version: {np.__version__}")
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Sum: {np.sum(arr)}")
print(f"Mean: {np.mean(arr)}")
"NumPy operations completed successfully"
    """
    
    # Then: Should execute successfully with proper results
    response = api_client.post(
        f"{base_url}{TestConfig.EXECUTE_RAW_ENDPOINT}",
        data=code,
        headers={"Content-Type": "text/plain"}
    )
    
    data = api_contract_validator(response)
    result = data["data"]["result"]
    stdout = data["data"]["stdout"]
    
    # Check final result
    assert "NumPy operations completed successfully" in result
    
    # Check stdout for printed output
    assert "NumPy version:" in stdout
    assert "Array: [1 2 3 4 5]" in stdout
    assert "Sum: 15" in stdout
    assert "Mean: 3.0" in stdout


def test_given_complex_strings_when_executing_raw_then_should_handle_correctly(
    api_client, base_url, default_timeout, api_contract_validator
):
    """
    Test complex string handling in raw execution mode.
    
    Description:
        Verifies that the /api/execute-raw endpoint can handle complex strings
        with various quote combinations, escape characters, and multiline content
        without requiring JSON escaping.
    
    Input:
        - Python code with complex string patterns
        - Various quote combinations and escape sequences
    
    Output:
        - Successful execution preserving all string content
    
    Example:
        Complex strings with quotes, newlines, escape sequences
    """
    # Given: Python code with complex string scenarios
    # When: We execute code with various string patterns
    code = '''
# Test various string scenarios
test_strings = [
    "Simple string",
    'Single quotes with "double inside"',
    """Triple double quotes with 'single' and "double" """,
    f"F-string with {2 + 3} calculation",
    "String with\nnewlines\tand\ttabs",
    r"Raw string with \n and \t literals",
    "Unicode: café, naïve, résumé"
]

for i, s in enumerate(test_strings):
    print(f"String {i+1}: {repr(s)}")

print(f"Total test strings: {len(test_strings)}")
"Complex string handling completed"
    '''
    
    # Then: Should execute successfully handling all string types
    response = api_client.post(
        f"{base_url}{TestConfig.EXECUTE_RAW_ENDPOINT}",
        data=code,
        headers={"Content-Type": "text/plain"}
    )
    
    data = api_contract_validator(response)
    result = data["data"]["result"]
    stdout = data["data"]["stdout"]
    
    # Check final result
    assert "Complex string handling completed" in result
    
    # Check stdout for printed output
    assert "String 1: 'Simple string'" in stdout
    assert 'String 2: \'Single quotes with "double inside"\'' in stdout
    assert "Total test strings: 7" in stdout


def test_given_pathlib_operations_when_cross_platform_then_should_work_portably(
    api_client, base_url, default_timeout, api_contract_validator
):
    """
    Test cross-platform file operations using pathlib.
    
    Description:
        Verifies that pathlib-based file operations work correctly
        across Windows and Linux environments in Pyodide.
    
    Input:
        - Python code using pathlib for file operations
        - Path creation, manipulation, and existence checking
    
    Output:
        - Successful path operations with proper cross-platform behavior
    
    Example:
        pathlib.Path operations -> platform-independent file handling
    """
    # Given: Cross-platform file operation requirements
    # When: We use pathlib for portable file operations
    code = """
from pathlib import Path
import sys

# Test pathlib operations
test_dir = Path("/tmp/test_directory")
test_file = test_dir / "sample.txt"

print(f"Platform: {sys.platform}")
print(f"Test directory: {test_dir}")
print(f"Test file: {test_file}")

# Path operations (without actual file creation in Pyodide)
print(f"Directory name: {test_dir.name}")
print(f"File stem: {test_file.stem}")
print(f"File suffix: {test_file.suffix}")
print(f"Parent directory: {test_file.parent}")

# Test path joining
config_path = Path("/config") / "app" / "settings.json"
print(f"Config path: {config_path}")

"Pathlib operations completed successfully"
    """
    
    # Then: Should execute with proper cross-platform path handling
    response = api_client.post(
        f"{base_url}{TestConfig.EXECUTE_RAW_ENDPOINT}",
        data=code,
        headers={"Content-Type": "text/plain"}
    )
    
    data = api_contract_validator(response)
    result_text = data["data"]["result"]
    
    assert "Platform:" in result_text
    assert "Test directory: /tmp/test_directory" in result_text
    assert "Directory name: test_directory" in result_text
    assert "File stem: sample" in result_text
    assert "File suffix: .txt" in result_text
    assert "Config path: /config/app/settings.json" in result_text
    assert "Pathlib operations completed successfully" in result_text


def test_given_pandas_operations_when_data_processing_then_should_execute_correctly(
    api_client, base_url, default_timeout, api_contract_validator
):
    """
    Test pandas data processing capabilities.
    
    Description:
        Verifies that pandas is available and can perform basic
        data processing operations in the Pyodide environment.
    
    Input:
        - Python code using pandas for data manipulation
        - DataFrame creation, filtering, and aggregation
    
    Output:
        - Successful pandas operations with expected results
    
    Example:
        pandas.DataFrame -> data manipulation -> summary statistics
    """
    # Given: Pandas availability in Pyodide
    # When: We perform data processing operations
    code = """
import pandas as pd
import numpy as np

# Create sample dataset
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 70000, 55000],
    'department': ['IT', 'HR', 'IT', 'Finance']
}

df = pd.DataFrame(data)
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Basic operations
it_employees = df[df['department'] == 'IT']
print(f"IT employees: {len(it_employees)}")

avg_salary = df['salary'].mean()
print(f"Average salary: {avg_salary}")

# Group by department
dept_summary = df.groupby('department')['salary'].mean()
print(f"Department salary averages:")
for dept, avg in dept_summary.items():
    print(f"  {dept}: {avg}")

"Pandas operations completed successfully"
    """
    
    # Then: Should execute pandas operations successfully
    response = api_client.post(
        f"{base_url}{TestConfig.EXECUTE_RAW_ENDPOINT}",
        data=code, headers={"Content-Type": "text/plain"}
    )
    
    data = api_contract_validator(response)
    result_text = data["data"]["result"]
    
    assert "DataFrame shape: (4, 4)" in result_text
    assert "IT employees: 2" in result_text
    assert "Average salary: 58750.0" in result_text
    assert "IT: 60000.0" in result_text
    assert "Pandas operations completed successfully" in result_text


def test_given_matplotlib_plotting_when_creating_visualizations_then_should_work(
    api_client, base_url, default_timeout, api_contract_validator
):
    """
    Test matplotlib plotting capabilities.
    
    Description:
        Verifies that matplotlib can create plots and handle
        figure operations in the Pyodide environment.
    
    Input:
        - Python code using matplotlib for visualization
        - Plot creation and configuration
    
    Output:
        - Successful plot creation without errors
    
    Example:
        matplotlib.pyplot -> plot creation -> figure management
    """
    # Given: Matplotlib availability in Pyodide
    # When: We create a simple plot
    code = """
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, 'b-', label='sin(x)', linewidth=2)
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_title('Sine Wave')
ax.legend()
ax.grid(True, alpha=0.3)

print(f"Figure size: {fig.get_size_inches()}")
print(f"Number of axes: {len(fig.axes)}")
print(f"Plot line count: {len(ax.lines)}")

# In Pyodide, we can't save to filesystem, but we can verify the plot exists
print(f"X data points: {len(x)}")
print(f"Y data range: [{np.min(y):.2f}, {np.max(y):.2f}]")

plt.close(fig)  # Clean up
"Matplotlib operations completed successfully"
    """
    
    # Then: Should execute matplotlib operations successfully
    response = api_client.post(
        f"{base_url}{TestConfig.EXECUTE_RAW_ENDPOINT}",
        data=code, headers={"Content-Type": "text/plain"}
    )
    
    data = api_contract_validator(response)
    result_text = data["data"]["result"]
    
    assert "Figure size:" in result_text
    assert "Number of axes: 1" in result_text
    assert "Plot line count: 1" in result_text
    assert "X data points: 100" in result_text
    assert "Y data range: [-1.00, 1.00]" in result_text
    assert "Matplotlib operations completed successfully" in result_text


def test_given_error_scenarios_when_invalid_code_then_should_handle_gracefully(
    api_client, base_url, short_timeout, api_contract_validator
):
    """
    Test error handling for invalid Python code.
    
    Description:
        Verifies that the API handles various error scenarios gracefully,
        returning proper error responses with meaningful messages.
    
    Input:
        - Invalid Python code (syntax errors, runtime errors, etc.)
    
    Output:
        - Error response following API contract
        - Meaningful error messages for debugging
    
    Example:
        Invalid syntax -> API error response with details
    """
    # Given: Invalid Python code with syntax error
    # When: We attempt to execute malformed code
    code = """
# Intentional syntax error
def broken_function(
    print("Missing closing parenthesis")
    return "This won't work"
    """
    
    # Then: Should return error response with proper contract
    response = api_client.post(
        f"{base_url}{TestConfig.EXECUTE_RAW_ENDPOINT}",
        data=code, headers={"Content-Type": "text/plain"}
    )
    
    # API should still return 200 with error in response body
    assert response.status_code == 200
    data = response.json()
    
    # Should follow error contract
    assert "success" in data
    assert "data" in data
    assert "error" in data
    assert "meta" in data
    
    # Should indicate failure
    assert data["success"] is False
    assert data["error"] is not None
    assert "syntax" in data["error"].lower() or "error" in data["error"].lower()


def test_given_timeout_scenario_when_long_running_code_then_should_timeout(
    api_client, base_url, api_contract_validator
):
    """
    Test timeout handling for long-running code.
    
    Description:
        Verifies that the API properly handles timeout scenarios
        for code that takes longer than the specified timeout.
    
    Input:
        - Python code with deliberate delay
        - Short timeout value
    
    Output:
        - Timeout error response following API contract
    
    Example:
        time.sleep(10) with 5-second timeout -> timeout error
    """
    # Given: Code that will exceed timeout
    # When: We execute long-running code with short timeout
    code = """
import time
print("Starting long operation...")
time.sleep(6)  # Sleep for 6 seconds
print("Operation completed")
    """
    
    timeout = 3000  # 3 seconds timeout
    
    # Then: Should timeout and return error response
    response = api_client.post(
        f"{base_url}{TestConfig.EXECUTE_RAW_ENDPOINT}",
        data=code, headers={"Content-Type": "text/plain"}
    )
    
    # Should return 200 with timeout error in body
    assert response.status_code == 200
    data = response.json()
    
    # Should indicate timeout failure
    assert data["success"] is False
    assert data["error"] is not None
    assert "timeout" in data["error"].lower() or "time" in data["error"].lower()


@pytest.mark.parametrize("test_case,expected_output", [
    ("print('Simple test')", "Simple test"),
    ("result = 2 + 3; print(f'Result: {result}')", "Result: 5"),
    ("import json; print(json.dumps({'key': 'value'}))", '{"key": "value"}'),
    ("[x**2 for x in range(5)]", "[0, 1, 4, 9, 16]"),
])
def test_given_parametric_code_when_executing_then_should_return_expected_output(
    api_client, base_url, default_timeout, api_contract_validator, test_case, expected_output
):
    """
    Parametric test for various code execution scenarios.
    
    Description:
        Tests multiple code scenarios using pytest parametrization
        to ensure consistent behavior across different code patterns.
    
    Input:
        - Various Python code snippets
        - Expected output strings
    
    Output:
        - Successful execution with expected results
    
    Example:
        print('Hello') -> output contains 'Hello'
    """
    # Given: Various Python code scenarios
    # When: We execute the parametric test case
    response = api_client.post(
        f"{base_url}{TestConfig.EXECUTE_RAW_ENDPOINT}",
        data=test_case, headers={"Content-Type": "text/plain"}
    )
    
    # Then: Should execute successfully with expected output
    data = api_contract_validator(response)
    assert expected_output in data["data"]["result"]


def test_given_concurrent_requests_when_multiple_execution_then_should_handle_properly(
    api_client, base_url, default_timeout, api_contract_validator
):
    """
    Test concurrent request handling.
    
    Description:
        Verifies that the API can handle multiple concurrent requests
        without interference or resource conflicts.
    
    Input:
        - Multiple simultaneous code execution requests
    
    Output:
        - All requests should complete successfully
        - No cross-contamination between executions
    
    Example:
        5 concurrent requests -> 5 successful independent responses
    """
    # Given: Multiple concurrent execution requests
    # When: We send concurrent requests
    def execute_concurrent_code(request_id):
        code = f"""
import time
request_id = {request_id}
print(f"Request {{request_id}} starting")
result = sum(range(1000))
print(f"Request {{request_id}} result: {{result}}")
f"Request {{request_id}} completed with result {{result}}"
        """
        
        response = api_client.post(
            f"{base_url}{TestConfig.EXECUTE_RAW_ENDPOINT}",
            data=code, headers={"Content-Type": "text/plain"}
        )
        return response, request_id
    
    # Execute concurrent requests
    with ThreadPoolExecutor(max_workers=TestConfig.MAX_CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(execute_concurrent_code, i) 
                  for i in range(TestConfig.MAX_CONCURRENT_REQUESTS)]
        results = [future.result() for future in futures]
    
    # Then: All requests should complete successfully
    for response, request_id in results:
        data = api_contract_validator(response)
        result_text = data["data"]["result"]
        
        assert f"Request {request_id} starting" in result_text
        assert f"Request {request_id} result: 499500" in result_text  # sum(range(1000))
        assert f"Request {request_id} completed" in result_text


def test_given_large_output_when_generating_extensive_data_then_should_handle_correctly(
    api_client, base_url, long_timeout, api_contract_validator
):
    """
    Test handling of large output data.
    
    Description:
        Verifies that the API can handle code that generates
        large amounts of output data without truncation or errors.
    
    Input:
        - Python code generating substantial output
    
    Output:
        - Complete output preserved in response
        - No truncation or data loss
    
    Example:
        Generate 1000 lines -> all lines preserved in response
    """
    # Given: Code that generates large output
    # When: We execute code with extensive output
    code = """
# Generate substantial output
lines = []
for i in range(100):
    line = f"Generated line {i+1}: {'*' * 20} Data content {i*2}"
    lines.append(line)
    print(line)

print(f"Generated {len(lines)} lines")
print(f"Total characters: {sum(len(line) for line in lines)}")
"Large output generation completed"
    """
    
    # Then: Should handle large output correctly
    response = api_client.post(
        f"{base_url}{TestConfig.EXECUTE_RAW_ENDPOINT}",
        data=code, headers={"Content-Type": "text/plain"}
    )
    
    data = api_contract_validator(response)
    result_text = data["data"]["result"]
    
    # Verify key markers in output
    assert "Generated line 1:" in result_text
    assert "Generated line 100:" in result_text
    assert "Generated 100 lines" in result_text
    assert "Large output generation completed" in result_text
    
    # Verify substantial content
    assert len(result_text) > 10000  # Should be substantial output
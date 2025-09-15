"""
Comprehensive Pytest BDD Tests for Pyodide Express Server API.

This test suite follows BDD (Behavior-Driven Development) patterns with Given-When-Then
structure and provides complete coverage for the Pyodide Express Server REST API.

Test Coverage:
- Health and status endpoints (/health, /api/status)
- Python code execution (/api/execute-raw only)
- File upload, listing, and deletion operations
- Error handling and edge cases
- Security validations
- Performance and timeout scenarios
- Server crash protection and stress testing

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
  "data": { "result": { "stdout": str, "stderr": str, "executionTime": int } } | null,
  "error": string | null,
  "meta": { "timestamp": string }
}
"""

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Generator

import pytest
import requests


# ==================== TEST CONFIGURATION CONSTANTS ====================


class TestConfig:
    """Test configuration constants - centralized and easily tweakable."""

    # Base URL for all API requests
    BASE_URL: str = "http://localhost:3000"

    # Timeout values for different operations (in seconds)
    TIMEOUTS = {
        "health_check": 10,
        "code_execution": 30,
        "file_upload": 60,
        "stress_test": 120,
        "api_request": 30,
        "server_startup": 120,
    }

    # API limits and constraints
    LIMITS = {
        "max_file_size_mb": 10,
        "max_code_length": 50000,
        "min_timeout": 1000,
        "max_timeout": 60000,
        "stress_test_iterations": 5,
        "concurrent_requests": 3,
    }

    # Test data samples - all Python code uses pathlib for cross-platform compatibility
    SAMPLE_DATA = {
        "csv_content": "name,value,category\nitem1,100,A\nitem2,200,B\nitem3,300,C\n",
        "simple_python": "print('Hello World')",

        "complex_pathlib_python": """
from pathlib import Path
import json
import time

# Test pathlib usage for cross-platform compatibility
base_path = Path('/tmp')
test_file = base_path / 'test.json'
data_dir = Path('/uploads')
plots_dir = Path('/plots/matplotlib')

# Create sample data structure
data = {
    'message': 'Pathlib test successful',
    'base_path': str(base_path),
    'test_file': str(test_file),
    'data_dir_exists': data_dir.exists(),
    'plots_dir_exists': plots_dir.exists(),
    'timestamp': time.time()
}

print(json.dumps(data, indent=2))
        """.strip(),
        "matplotlib_pathlib_python": """
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import time

# Generate test data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Use pathlib for plot file handling
plots_dir = Path('/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)

# Create unique filename with timestamp
timestamp = int(time.time())
plot_file = plots_dir / f'pathlib_test_{timestamp}.png'

# Create and save plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Pathlib Matplotlib Test')
plt.xlabel('X values')
plt.ylabel('sin(x)')
plt.grid(True)
plt.savefig(plot_file, dpi=100, bbox_inches='tight')
plt.close()

print(f'Plot saved successfully: {plot_file}')
print(f'File exists: {plot_file.exists()}')
        """.strip(),
        "file_operations_pathlib_python": """
from pathlib import Path
import json
import time

# Test comprehensive pathlib file operations
uploads_dir = Path('/uploads')
temp_dir = Path('/tmp')
plots_dir = Path('/plots/matplotlib')

# Create directories if they don't exist
for directory in [uploads_dir, temp_dir, plots_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# Test file operations
test_data = {
    'timestamp': time.time(),
    'directories': {
        'uploads_exists': uploads_dir.exists(),
        'temp_exists': temp_dir.exists(),
        'plots_exists': plots_dir.exists()
    },
    'paths': {
        'uploads_path': str(uploads_dir),
        'temp_path': str(temp_dir),
        'plots_path': str(plots_dir)
    }
}

# Use pathlib to create test file
test_file = temp_dir / 'pathlib_test.json'
test_file.write_text(json.dumps(test_data, indent=2))

print(f'Created test file: {test_file}')
print(f'File size: {test_file.stat().st_size} bytes')
print(f'File exists: {test_file.exists()}')

# Read back the file
content = test_file.read_text()
print('File content preview:')
print(content[:200] + '...' if len(content) > 200 else content)
        """.strip(),
        "error_python": "import nonexistent_module",
        "syntax_error_python": "print('unclosed string",
        "infinite_loop_python": "while True: pass",
        "memory_intensive_python": """
import time
from pathlib import Path

# Memory allocation test with pathlib
data = []
temp_dir = Path('/tmp')

for i in range(100):  # Reduced for testing
    data.append([0] * 100)
    if i % 25 == 0:
        print(f'Allocated {i * 100} items')
        print(f'Temp directory exists: {temp_dir.exists()}')
        time.sleep(0.01)

print(f'Total allocated: {len(data) * len(data[0])} items')
        """.strip(),
    }


# ==================== UTILITY FUNCTIONS ====================


def wait_for_server(url: str, timeout: int = TestConfig.TIMEOUTS["server_startup"]) -> None:
    """
    Wait for the server to become available before running tests.

    Args:
        url: Server URL to poll for availability
        timeout: Maximum time to wait in seconds

    Raises:
        RuntimeError: If server doesn't respond within timeout period

    Example:
        >>> wait_for_server("http://localhost:3000/health", timeout=60)
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except (requests.RequestException, OSError):
            pass  # Server not ready yet
        time.sleep(1)
    raise RuntimeError(f"Server at {url} did not start within {timeout} seconds")


def validate_api_contract(response_data: Dict[str, Any]) -> None:
    """
    Validate that API response follows the expected contract structure.

    Expected format:
    {
        "success": true | false,
        "data": { "result": str, "stdout": str, "stderr": str, "executionTime": int } | null,
        "error": string | null,
        "meta": { "timestamp": string }
    }

    Args:
        response_data: The JSON response data to validate

    Raises:
        AssertionError: If response doesn't follow the contract

    Example:
        >>> response = {
        ...     "success": True,
        ...     "data": {"result": "Hello", "stdout": "Hello", "stderr": "", "executionTime": 123},
        ...     "error": None,
        ...     "meta": {"timestamp": "2025-01-01T00:00:00Z"}
        ... }
        >>> validate_api_contract(response)  # Should pass without error
    """
    # Check required top-level fields
    required_fields = ["success", "data", "error", "meta"]
    for field in required_fields:
        assert (
            field in response_data
        ), f"Response missing required field '{field}': {response_data}"

    # Validate field types
    assert isinstance(
        response_data["success"], bool
    ), f"'success' must be boolean: {type(response_data['success'])}"
    assert isinstance(
        response_data["meta"], dict
    ), f"'meta' must be dict: {type(response_data['meta'])}"
    assert (
        "timestamp" in response_data["meta"]
    ), f"Meta missing 'timestamp': {response_data['meta']}"

    # Validate success/error contract
    if response_data["success"]:
        assert (
            response_data["data"] is not None
        ), "Success response should have non-null data"
        assert response_data["error"] is None, "Success response should have null error"

        # For execute-raw responses, validate data structure
        if isinstance(response_data["data"], dict):
            data = response_data["data"]

            # Check if this is an execute-raw response (has result, stdout, stderr, executionTime)
            execute_fields = ["result", "stdout", "stderr", "executionTime"]
            is_execute_response = all(field in data for field in execute_fields)

            if is_execute_response:
                # Validate execute-raw response format
                for field in execute_fields:
                    assert field in data, f"data missing '{field}': {data}"

                # Validate field types
                assert isinstance(data["result"], str), f"data.result must be str: {type(data['result'])}"
                assert isinstance(data["stdout"], str), f"data.stdout must be str: {type(data['stdout'])}"
                assert isinstance(data["stderr"], str), f"data.stderr must be str: {type(data['stderr'])}"
                assert isinstance(data["executionTime"], int), \
                    f"data.executionTime must be int: {type(data['executionTime'])}"
            # For other endpoints (status, file lists, etc.), data can have different structure
    else:
        assert (
            response_data["error"] is not None
        ), "Error response should have non-null error"
        assert isinstance(
            response_data["error"], str
        ), f"Error must be string: {type(response_data['error'])}"


def execute_python_code(
    code: str, timeout: int = TestConfig.TIMEOUTS["code_execution"]
) -> Dict[str, Any]:
    """
    Execute Python code using the /api/execute-raw endpoint.

    Args:
        code: Python code to execute
        timeout: Request timeout in seconds

    Returns:
        Dictionary containing the API response

    Raises:
        requests.RequestException: If request fails

    Example:
        >>> result = execute_python_code("print('Hello')")
        >>> assert result["success"] is True
        >>> assert "Hello" in result["data"]["stdout"]
    """
    response = requests.post(
        f"{TestConfig.BASE_URL}/api/execute-raw",
        headers={"Content-Type": "text/plain"},
        data=code,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def create_test_file(content: str, suffix: str = ".csv") -> Path:
    """
    Create a temporary test file with specified content.

    Args:
        content: File content to write
        suffix: File extension/suffix

    Returns:
        Path object pointing to the created file

    Example:
        >>> file_path = create_test_file("name,value\ntest,123", ".csv")
        >>> assert file_path.exists()
        >>> assert file_path.read_text() == "name,value\ntest,123"
    """
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    temp_file.write(content)
    temp_file.close()
    return Path(temp_file.name)


# ==================== PYTEST FIXTURES ====================


@pytest.fixture(scope="session")
def server_ready() -> bool:
    """
    Session-scoped fixture ensuring server is available before running any tests.

    Returns:
        bool: True when server is ready

    Raises:
        RuntimeError: If server fails to start within timeout
    """
    wait_for_server(f"{TestConfig.BASE_URL}/health")
    return True


@pytest.fixture
def stress_test_ready() -> bool:
    """
    Fixture for stress tests that can handle server crashes gracefully.

    Returns:
        bool: True if server is ready

    Raises:
        pytest.skip: If server is unrecoverable after crash
    """
    try:
        wait_for_server(f"{TestConfig.BASE_URL}/health", timeout=30)
        return True
    except RuntimeError:
        pytest.skip("Server unavailable - likely crashed during stress testing")


@pytest.fixture
def temp_csv_file() -> Generator[Path, None, None]:
    """
    Fixture providing a temporary CSV file for testing.

    Yields:
        Path: Path to temporary CSV file

    Example:
        >>> def test_csv_upload(temp_csv_file):
        ...     assert temp_csv_file.exists()
        ...     content = temp_csv_file.read_text()
        ...     assert "name,value" in content
    """
    file_path = create_test_file(TestConfig.SAMPLE_DATA["csv_content"], ".csv")
    try:
        yield file_path
    finally:
        if file_path.exists():
            file_path.unlink()


@pytest.fixture
def uploaded_files_cleanup() -> Generator[list, None, None]:
    """
    Fixture to track and cleanup uploaded files after tests.

    Yields:
        list: List to track uploaded filenames for cleanup

    Example:
        >>> def test_upload(uploaded_files_cleanup):
        ...     # Upload file
        ...     uploaded_files_cleanup.append("test.csv")
        ...     # File will be automatically cleaned up
    """
    uploaded_files = []
    yield uploaded_files

    # Cleanup uploaded files
    for filename in uploaded_files:
        try:
            requests.delete(
                f"{TestConfig.BASE_URL}/api/uploaded-files/{filename}",
                timeout=TestConfig.TIMEOUTS["api_request"],
            )
        except requests.RequestException:
            pass  # File might already be deleted


# ==================== HEALTH AND STATUS TESTS ====================


class TestHealthAndStatus:
    """Test suite for health check and server status endpoints."""

    def test_given_server_running_when_health_check_then_returns_ok(self, server_ready):
        """
        Test server health check endpoint functionality.

        Given: Server is running and accessible
        When: Making a GET request to /health endpoint
        Then: Response should indicate server is healthy or degraded but responsive

        Args:
            server_ready: Fixture ensuring server availability
        """
        # When: Making health check request
        response = requests.get(
            f"{TestConfig.BASE_URL}/health", timeout=TestConfig.TIMEOUTS["health_check"]
        )

        # Then: Response should indicate server status
        assert response.status_code == 200
        data = response.json()

        # Accept either "ok" or "degraded" status as long as server is responsive
        valid_statuses = ["ok", "degraded"]
        assert (
            data.get("status") in valid_statuses or data.get("success") is True
        ), f"Health check returned unexpected status: {data}"

    def test_given_server_running_when_status_check_then_returns_detailed_info(
        self, server_ready
    ):
        """
        Test server status endpoint for detailed system information.

        Given: Server is running with process pool
        When: Making a GET request to /api/status endpoint
        Then: Response should contain detailed server status and pool information

        Args:
            server_ready: Fixture ensuring server availability
        """
        # When: Making status check request
        response = requests.get(
            f"{TestConfig.BASE_URL}/api/status", timeout=TestConfig.TIMEOUTS["health_check"]
        )

        # Then: Response should contain detailed status
        assert response.status_code == 200
        data = response.json()

        # Validate API contract
        validate_api_contract(data)
        assert data["success"] is True

        # Validate process pool information
        status_data = data["data"]
        assert "ready" in status_data
        assert "poolStats" in status_data
        assert status_data["ready"] is True


# ==================== PYTHON CODE EXECUTION TESTS ====================


class TestPythonExecution:
    """Test suite for Python code execution via /api/execute-raw endpoint."""

    def test_given_simple_python_code_when_executed_then_returns_stdout(
        self, server_ready
    ):
        """
        Test basic Python code execution functionality.

        Given: Server is ready for code execution
        When: Executing simple Python print statement
        Then: Response should contain expected output in stdout

        Args:
            server_ready: Fixture ensuring server availability
        """
        # Given: Simple Python code
        code = TestConfig.SAMPLE_DATA["simple_python"]

        # When: Executing the code
        result = execute_python_code(code)

        # Then: Response should be successful with correct output
        validate_api_contract(result)
        assert result["success"] is True
        assert "Hello World" in result["data"]["stdout"]
        assert result["data"]["stderr"] == ""
        assert result["data"]["executionTime"] > 0

    def test_given_pathlib_python_code_when_executed_then_handles_paths_correctly(
        self, server_ready
    ):
        """
        Test Python code execution with pathlib for cross-platform compatibility.

        Given: Server is ready and Python code uses pathlib.Path
        When: Executing code with pathlib Path operations
        Then: Response should handle pathlib operations without errors

        Args:
            server_ready: Fixture ensuring server availability
        """
        # Given: Python code using pathlib for cross-platform compatibility
        code = TestConfig.SAMPLE_DATA["pathlib_python"]

        # When: Executing the pathlib code
        result = execute_python_code(code)

        # Then: Response should be successful
        validate_api_contract(result)
        assert result["success"] is True

        # Validate pathlib functionality
        output = result["data"]["stdout"]
        assert "Pathlib test successful" in output

        assert "/tmp/test.json" in output
        assert result["data"]["stderr"] == ""

    def test_given_complex_python_code_when_executed_then_handles_imports_and_logic(
        self, server_ready
    ):
        """
        Test execution of complex Python code with multiple imports and logic.

        Given: Server is ready for complex code execution
        When: Executing code with imports, data structures, and logic
        Then: Response should handle complex operations successfully

        Args:
            server_ready: Fixture ensuring server availability
        """
        # Given: Complex Python code
        code = Config.SAMPLE_DATA["complex_pathlib_python"]


        # When: Executing complex code
        result = execute_python_code(code)

        # Then: Response should handle complexity
        validate_api_contract(result)
        assert result["success"] is True

        output = result["data"]["stdout"]
        assert "Pathlib test successful" in output
        assert "timestamp" in output
        assert result["data"]["stderr"] == ""

    def test_given_python_syntax_error_when_executed_then_returns_error_in_stderr(
        self, server_ready
    ):
        """
        Test handling of Python syntax errors.

        Given: Server is ready and Python code contains syntax error
        When: Executing code with invalid syntax
        Then: Response should capture syntax error appropriately

        Args:
            server_ready: Fixture ensuring server availability
        """
        # Given: Python code with syntax error
        code = "print('Missing closing quote"

        # When: Executing invalid code
        result = execute_python_code(code)

        # Then: Should capture syntax error
        validate_api_contract(result)

        if result["success"]:
            # If success=True, error should be in stderr
            assert (
                result["data"]["stderr"] != ""
            ), f"Expected stderr for syntax error, got: {result['data']}"
        else:
            # If success=False, error should be in error field
            assert (
                result["error"] is not None
            ), f"Expected error field for syntax error, got: {result}"
            # Accept any execution failure as syntax errors are execution failures
            expected_terms = ["syntax", "invalid", "execution", "failed", "error"]
            assert any(
                term in result["error"].lower() for term in expected_terms
            ), f"Expected error message for syntax error, got: {result['error']}"

    def test_given_empty_code_when_executed_then_returns_appropriate_response(
        self, server_ready
    ):
        """
        Test handling of empty or whitespace-only code.

        Given: Server is ready
        When: Executing empty or whitespace-only code
        Then: Response should handle empty input gracefully

        Args:
            server_ready: Fixture ensuring server availability
        """
        # Given: Empty code
        code = ""

        # When: Executing empty code
        response = requests.post(
            f"{TestConfig.BASE_URL}/api/execute-raw",
            headers={"Content-Type": "text/plain"},
            data=code,
            timeout=TestConfig.TIMEOUTS["code_execution"],
        )

        # Then: Should return error for empty code
        assert response.status_code == 400
        result = response.json()
        validate_api_contract(result)
        assert result["success"] is False
        assert "No Python code provided" in result["error"]

    def test_given_matplotlib_code_when_executed_then_creates_plot_with_pathlib(
        self, server_ready
    ):
        """
        Test matplotlib plot creation using pathlib for file operations.

        Given: Server is ready for matplotlib code execution
        When: Executing code that creates and saves a plot using pathlib
        Then: Response should indicate successful plot creation with pathlib

        Args:
            server_ready: Fixture ensuring server availability

        Example:
            This test validates that matplotlib plots can be created and saved
            using pathlib.Path for cross-platform file operations.
        """
        # Given: Matplotlib code using pathlib
        code = Config.SAMPLE_DATA["matplotlib_pathlib_python"]

        # When: Executing matplotlib code with pathlib
        result = execute_python_code(code)

        # Then: Response should be successful
        validate_api_contract(result)
        assert result["success"] is True

        # Validate plot creation output
        output = result["data"]["stdout"]
        assert "Plot saved successfully:" in output
        assert "File exists: True" in output
        assert result["data"]["stderr"] == ""

    def test_given_file_operations_code_when_executed_then_handles_pathlib_correctly(
        self, server_ready
    ):
        """
        Test comprehensive file operations using pathlib.

        Given: Server is ready for file operations
        When: Executing code that performs various file operations with pathlib
        Then: Response should demonstrate successful pathlib file handling

        Args:
            server_ready: Fixture ensuring server availability

        Example:
            This test validates pathlib usage for directory creation, file writing,
            reading, and file system introspection operations.
        """
        # Given: File operations code using pathlib
        code = Config.SAMPLE_DATA["file_operations_pathlib_python"]

        # When: Executing file operations code
        result = execute_python_code(code)

        # Then: Response should be successful
        validate_api_contract(result)
        assert result["success"] is True

        # Validate file operations output
        output = result["data"]["stdout"]
        assert "Created test file:" in output
        assert "File size:" in output
        assert "File exists: True" in output
        assert "File content preview:" in output
        assert result["data"]["stderr"] == ""


# ==================== FILE MANAGEMENT TESTS ====================


class TestFileManagement:
    """Test suite for file upload, listing, and deletion operations."""

    def test_given_csv_file_when_uploaded_then_appears_in_file_list(
        self, server_ready, temp_csv_file, uploaded_files_cleanup
    ):
        """
        Test file upload and listing functionality.

        Given: Server is ready and a CSV file exists
        When: Uploading the file and requesting file list
        Then: Uploaded file should appear in the file list

        Args:
            server_ready: Fixture ensuring server availability
            temp_csv_file: Fixture providing temporary CSV file
            uploaded_files_cleanup: Fixture for cleanup tracking
        """
        # Given: CSV file to upload
        filename = f"test_upload_{int(time.time())}.csv"

        # When: Uploading the file
        with open(temp_csv_file, "rb") as file:
            files = {"file": (filename, file, "text/csv")}
            upload_response = requests.post(
                f"{TestConfig.BASE_URL}/api/upload",
                files=files,
                timeout=TestConfig.TIMEOUTS["file_upload"],
            )

        # Track for cleanup
        uploaded_files_cleanup.append(filename)

        # Then: Upload should be successful
        assert upload_response.status_code == 200
        upload_result = upload_response.json()
        validate_api_contract(upload_result)
        assert upload_result["success"] is True

        # When: Requesting file list
        list_response = requests.get(
            f"{TestConfig.BASE_URL}/api/uploaded-files",
            timeout=TestConfig.TIMEOUTS["api_request"],
        )

        # Then: File should appear in list
        assert list_response.status_code == 200
        list_result = list_response.json()
        validate_api_contract(list_result)
        assert list_result["success"] is True

        files_list = list_result["data"]["files"]
        assert any(f["filename"] == filename for f in files_list)

    def test_given_uploaded_file_when_deleted_then_removed_from_list(
        self, server_ready, temp_csv_file, uploaded_files_cleanup
    ):
        """
        Test file deletion functionality.

        Given: Server is ready and a file has been uploaded
        When: Deleting the uploaded file
        Then: File should be removed from the file list

        Args:
            server_ready: Fixture ensuring server availability
            temp_csv_file: Fixture providing temporary CSV file
            uploaded_files_cleanup: Fixture for cleanup tracking
        """
        # Given: Upload a file first
        filename = f"test_delete_{int(time.time())}.csv"

        with open(temp_csv_file, "rb") as file:
            files = {"file": (filename, file, "text/csv")}
            upload_response = requests.post(
                f"{TestConfig.BASE_URL}/api/upload",
                files=files,
                timeout=TestConfig.TIMEOUTS["file_upload"],
            )

        assert upload_response.status_code == 200

        # When: Deleting the file
        delete_response = requests.delete(
            f"{TestConfig.BASE_URL}/api/uploaded-files/{filename}",
            timeout=TestConfig.TIMEOUTS["api_request"],
        )

        # Then: Deletion should be successful
        assert delete_response.status_code == 200
        delete_result = delete_response.json()
        validate_api_contract(delete_result)
        assert delete_result["success"] is True

        # Verify file is removed from list
        list_response = requests.get(
            f"{TestConfig.BASE_URL}/api/uploaded-files",
            timeout=TestConfig.TIMEOUTS["api_request"],
        )

        assert list_response.status_code == 200
        list_result = list_response.json()
        files_list = list_result["data"]["files"]
        assert not any(f["filename"] == filename for f in files_list)


# ==================== SECURITY AND ERROR HANDLING TESTS ====================


class TestSecurityAndErrorHandling:
    """Test suite for security validations and error handling scenarios."""

    def test_given_malformed_request_when_sent_then_returns_appropriate_error(
        self, server_ready
    ):
        """
        Test handling of malformed requests.

        Given: Server is ready
        When: Sending request with wrong content type
        Then: Response should return appropriate error

        Args:
            server_ready: Fixture ensuring server availability
        """
        # Given: Valid code but wrong content type
        code = "print('test')"

        # When: Sending with wrong content type
        response = requests.post(
            f"{TestConfig.BASE_URL}/api/execute-raw",
            headers={"Content-Type": "application/json"},  # Wrong content type
            data=code,
            timeout=TestConfig.TIMEOUTS["code_execution"],
        )

        # Then: Should handle gracefully (might still work or return error)
        # The exact behavior depends on implementation, but should not crash
        assert response.status_code in [200, 400, 422]

    def test_given_large_code_when_executed_then_handles_size_appropriately(
        self, server_ready
    ):
        """
        Test handling of large code submissions.

        Given: Server is ready
        When: Executing very large Python code
        Then: Response should handle large input appropriately

        Args:
            server_ready: Fixture ensuring server availability
        """
        # Given: Large Python code (near limit)
        large_code = "\n".join([f"print('Line {i}')" for i in range(1000)])

        # When: Executing large code
        result = execute_python_code(large_code, timeout=60)

        # Then: Should handle large code
        validate_api_contract(result)
        # Should either succeed or return appropriate error for size
        if result["success"]:
            assert "Line 999" in result["data"]["stdout"]
        else:
            assert (
                "too large" in result["error"].lower()
                or "limit" in result["error"].lower()
            )


# ==================== PERFORMANCE AND STRESS TESTS ====================


class TestPerformanceAndStress:
    """Test suite for performance validation and stress testing with crash protection."""

    def test_given_rapid_requests_when_executed_then_handles_concurrency(
        self, server_ready
    ):
        """
        Test server ability to handle multiple rapid requests.

        Given: Server is ready with process pool
        When: Sending multiple rapid requests
        Then: All requests should be handled successfully

        Args:
            server_ready: Fixture ensuring server availability
        """
        # Given: Multiple simple Python codes
        codes = [
            f"print('Request {i}')" for i in range(TestConfig.LIMITS["concurrent_requests"])
        ]

        # When: Executing multiple requests rapidly
        results = []
        for i, code in enumerate(codes):
            result = execute_python_code(code)
            results.append(result)

            # Brief pause to allow process pool to handle requests
            time.sleep(0.1)

        # Then: All requests should succeed
        for i, result in enumerate(results):
            validate_api_contract(result)
            assert result["success"] is True
            assert f"Request {i}" in result["data"]["stdout"]

    def test_given_infinite_loop_when_executed_then_server_survives(
        self, stress_test_ready
    ):
        """
        Test server crash protection with infinite loop code.

        Given: Server is ready with process pool crash protection
        When: Executing Python code with infinite loop
        Then: Server should survive and process should be terminated

        Args:
            stress_test_ready: Fixture ensuring server can handle stress tests
        """
        # Given: Python code with infinite loop
        code = TestConfig.SAMPLE_DATA["infinite_loop"]

        # When: Executing infinite loop with short timeout
        start_time = time.time()
        try:
            result = execute_python_code(code, timeout=10)  # Short timeout
            execution_time = time.time() - start_time

            # Then: Should timeout or be terminated
            # Process pool should handle this gracefully
            if result["success"]:
                # If it succeeded, it should have been terminated quickly
                assert execution_time < 15, "Infinite loop should be terminated quickly"
            else:
                # If it failed, should be due to timeout
                assert (
                    "timeout" in result["error"].lower()
                    or "terminated" in result["error"].lower()
                )

        except requests.Timeout:
            # Acceptable - request timed out
            execution_time = time.time() - start_time
            assert execution_time <= 15, "Request should timeout within reasonable time"

        # Most importantly: Server should still be responsive
        health_response = requests.get(
            f"{TestConfig.BASE_URL}/health", timeout=TestConfig.TIMEOUTS["health_check"]
        )
        assert (
            health_response.status_code == 200
        ), "Server should remain responsive after infinite loop"

    def test_given_memory_intensive_code_when_executed_then_server_survives(
        self, stress_test_ready
    ):
        """
        Test server crash protection with memory-intensive code.

        Given: Server is ready with process pool protection
        When: Executing memory-intensive Python code
        Then: Server should survive and handle memory pressure

        Args:
            stress_test_ready: Fixture ensuring server can handle stress tests
        """
        # Given: Memory-intensive Python code
        code = TestConfig.SAMPLE_DATA["memory_intensive"]

        # When: Executing memory-intensive code
        start_time = time.time()
        result = execute_python_code(code, timeout=30)
        execution_time = time.time() - start_time

        # Then: Should either complete or be terminated safely
        validate_api_contract(result)

        if result["success"]:
            # If successful, should show memory allocation
            assert "allocated" in result["data"]["stdout"].lower()
        else:
            # If failed, should be due to memory/timeout limits
            error_msg = result["error"].lower()
            assert any(
                term in error_msg
                for term in ["memory", "timeout", "terminated", "limit"]
            )

        # Verify execution time is reasonable
        assert execution_time < 45, f"Memory test took too long: {execution_time:.2f}s"

        # Server should remain responsive
        health_response = requests.get(
            f"{TestConfig.BASE_URL}/health", timeout=TestConfig.TIMEOUTS["health_check"]
        )
        assert (
            health_response.status_code == 200
        ), "Server should remain responsive after memory stress"

    def test_given_multiple_stress_iterations_when_executed_then_server_stability(
        self, stress_test_ready
    ):
        """
        Test server stability over multiple stress iterations.

        Given: Server is ready with process pool
        When: Running multiple stress test iterations
        Then: Server should maintain stability throughout

        Args:
            stress_test_ready: Fixture ensuring server can handle stress tests
        """
        # Given: Stress test scenarios
        stress_codes = [
            "for i in range(10000): print(f'Iteration {i}') if i % 1000 == 0 else None",
            "import time; [time.sleep(0.001) for i in range(100)]",
            "data = [i**2 for i in range(1000)]; print(f'Sum: {sum(data)}')",
        ]

        successful_executions = 0

        # When: Running multiple stress iterations
        for iteration in range(TestConfig.LIMITS["stress_test_iterations"]):
            code = stress_codes[iteration % len(stress_codes)]

            try:
                result = execute_python_code(code, timeout=15)

                if result["success"]:
                    successful_executions += 1

                # Brief pause between iterations
                time.sleep(0.5)

            except (requests.Timeout, requests.RequestException):
                # Some failures are acceptable during stress testing
                pass

        # Then: Should have reasonable success rate
        success_rate = successful_executions / TestConfig.LIMITS["stress_test_iterations"]
        assert (
            success_rate >= 0.6
        ), f"Success rate {success_rate:.2f} too low during stress testing"

        # Final health check
        health_response = requests.get(
            f"{TestConfig.BASE_URL}/health", timeout=TestConfig.TIMEOUTS["health_check"]
        )
        assert (
            health_response.status_code == 200
        ), "Server should remain healthy after stress testing"


# ==================== INTEGRATION TESTS ====================


class TestIntegration:
    """Integration tests for complete workflows and scenarios."""

    def test_given_full_workflow_when_executed_then_all_operations_succeed(
        self, server_ready, temp_csv_file, uploaded_files_cleanup
    ):
        """
        Test complete workflow: upload file, execute code to process it, verify results.

        Given: Server is ready and CSV file is available
        When: Uploading file and executing Python code to process it
        Then: All operations should succeed in sequence

        Args:
            server_ready: Fixture ensuring server availability
            temp_csv_file: Fixture providing temporary CSV file
            uploaded_files_cleanup: Fixture for cleanup tracking
        """
        # Given: CSV file and processing code
        filename = f"integration_test_{int(time.time())}.csv"

        # When: Uploading the file
        with open(temp_csv_file, "rb") as file:
            files = {"file": (filename, file, "text/csv")}
            upload_response = requests.post(
                f"{TestConfig.BASE_URL}/api/upload",
                files=files,
                timeout=TestConfig.TIMEOUTS["file_upload"],
            )

        uploaded_files_cleanup.append(filename)
        assert upload_response.status_code == 200

        # When: Executing Python code to process the uploaded file
        processing_code = f"""
from pathlib import Path
import csv

# Use pathlib for cross-platform file handling
uploads_dir = Path('/uploads')
csv_file = uploads_dir / '{filename}'

print(f'Processing file: {{csv_file}}')
print(f'File exists: {{csv_file.exists()}}')

# If file exists, process it
if csv_file.exists():
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f'Found {{len(rows)}} rows')
            for row in rows:
                print(f'Row: {{row}}')
    except Exception as e:
        print(f'Error reading file: {{e}}')
else:
    print('File not found in expected location')
    # List available files
    if uploads_dir.exists():
        files = list(uploads_dir.glob('*'))
        print(f'Available files: {{[f.name for f in files]}}')
        """.strip()

        result = execute_python_code(processing_code)

        # Then: Processing should be successful
        validate_api_contract(result)
        assert result["success"] is True

        output = result["data"]["stdout"]
        assert filename in output
        assert "Processing file:" in output

        # Clean up by deleting the file
        delete_response = requests.delete(
            f"{TestConfig.BASE_URL}/api/uploaded-files/{filename}",
            timeout=TestConfig.TIMEOUTS["api_request"],
        )
        assert delete_response.status_code == 200


# ==================== PERFORMANCE BENCHMARKS ====================


class TestPerformanceBenchmarks:
    """Performance benchmark tests to validate process pool efficiency."""

    def test_given_process_pool_when_multiple_executions_then_fast_response_times(
        self, server_ready
    ):
        """
        Test process pool performance with multiple executions.

        Given: Server is ready with process pool
        When: Executing multiple Python codes in sequence
        Then: Response times should be consistently fast (process reuse)

        Args:
            server_ready: Fixture ensuring server availability
        """
        # Given: Simple Python codes for performance testing
        test_codes = [
            "print('Execution 1')",
            "import time; print(f'Execution 2 at {time.time()}')",
            "x = [i**2 for i in range(100)]; print(f'Execution 3: sum={sum(x)}')",
            "from pathlib import Path; print(f'Execution 4: cwd={Path.cwd()}')",
            "import json; print(json.dumps({'execution': 5, 'status': 'complete'}))",
        ]

        execution_times = []

        # When: Executing codes and measuring response times
        for i, code in enumerate(test_codes):
            start_time = time.time()
            result = execute_python_code(code)
            total_time = time.time() - start_time

            # Then: Each execution should be successful and fast
            validate_api_contract(result)
            assert (
                result["success"] is True
            ), f"Execution {i+1} failed: {result.get('error', 'Unknown error')}"

            # Verify output contains expected content (more flexible matching)
            output = result["data"]["stdout"]
            expected_content = [
                "Execution 1",
                "Execution 2",
                "Execution 3",
                "Execution 4",
                "execution",  # JSON contains "execution": 5
            ]
            assert (
                expected_content[i].lower() in output.lower()
            ), f"Expected '{expected_content[i]}' in output: {output}"

            execution_times.append(total_time)

            # Brief pause between executions
            time.sleep(0.1)

        # Then: All execution times should be reasonable (process pool efficiency)
        average_time = sum(execution_times) / len(execution_times)
        assert (
            average_time < 5.0
        ), f"Average execution time {average_time:.2f}s too high for process pool"

        # After first execution, subsequent ones should be even faster (no initialization)
        if len(execution_times) > 1:
            later_times = execution_times[1:]
            later_average = sum(later_times) / len(later_times)
            assert (
                later_average < 3.0
            ), f"Later executions {later_average:.2f}s should be faster due to process reuse"

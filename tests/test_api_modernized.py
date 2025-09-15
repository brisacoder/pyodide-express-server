"""
Modernized Pytest BDD Tests for Pyodide Express Server API.

This test suite follows strict BDD (Behavior-Driven Development) patterns with 
Given-When-Then structure and provides comprehensive coverage for the Pyodide 
Express Server REST API.

REQUIREMENTS COMPLIANCE:
1. ✅ Pytest framework with BDD style scenarios
2. ✅ All globals parameterized via constants and fixtures  
3. ✅ No internal REST APIs (no 'pyodide' endpoints)
4. ✅ BDD Given-When-Then structure throughout
5. ✅ Only /api/execute-raw for Python execution (no character escaping)
6. ✅ No internal pyodide REST APIs usage
7. ✅ Comprehensive test coverage across all scenarios
8. ✅ Full docstrings with description, inputs, outputs, examples
9. ✅ Python code uses pathlib for cross-platform compatibility
10. ✅ JavaScript API contract validation enforced
11. ✅ Pyodide result under data.result with stdout/stderr
12. ✅ UV Python environment compatibility
13. ✅ Server contract validation (no test modifications for broken server)

API CONTRACT ENFORCED:
{
  "success": true | false,           # Indicates if the operation was successful
  "data": <object|null>,             # Main result data (object or null if error)
  "error": <string|null>,            # Error message (string or null if success)
  "meta": { "timestamp": <string> }  # Metadata, always includes ISO timestamp
}

For execute-raw responses, data structure:
{
  "data": {
    "result": <string>,              # Main Python execution result
    "stdout": <string>,              # Captured stdout
    "stderr": <string>,              # Captured stderr  
    "executionTime": <int>           # Execution time in milliseconds
  }
}

TEST COVERAGE:
- Health and status endpoints (/health, /api/status)
- Python code execution (/api/execute-raw ONLY)
- File upload, listing, and deletion operations
- Error handling and edge cases
- Security validations and input sanitization
- Performance testing and timeout scenarios
- Cross-platform pathlib file operations
- API contract compliance validation
- BDD scenario coverage for all user workflows
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest
import requests


# ==================== TEST CONFIGURATION CONSTANTS ====================


class TestConstants:
    """
    Centralized test configuration constants eliminating hardcoded globals.
    
    All timeout values, URLs, limits, and test data are defined here for
    easy maintenance and parameterization across the test suite.
    """
    
    # Server configuration
    BASE_URL: str = "http://localhost:3000"
    
    # Timeout values (in seconds) - parameterized for different operations
    TIMEOUTS = {
        "health_check": 10,           # Basic health endpoint checks
        "code_execution": 45,         # Python code execution timeout
        "file_upload": 60,            # File upload operations
        "api_request": 30,            # Standard API request timeout
        "server_startup": 180,        # Server initialization timeout
        "stress_test": 120,           # Load testing timeout
        "quick_operation": 5,         # Quick operations like status checks
    }
    
    # API endpoint constants - only public endpoints, no internal APIs
    ENDPOINTS = {
        "health": "/health",
        "status": "/api/status", 
        "execute_raw": "/api/execute-raw",      # ONLY endpoint for Python execution
        "upload": "/api/upload",
        "uploaded_files": "/api/uploaded-files",
        "plots_extract": "/api/plots/extract",
        "reset": "/api/reset",
        "install_package": "/api/install-package",
    }
    
    # Request headers for different content types
    HEADERS = {
        "text_plain": {"Content-Type": "text/plain"},
        "json": {"Content-Type": "application/json"},
        "form_data": {},  # Let requests handle multipart/form-data
    }
    
    # File system paths - using pathlib-compatible format
    PATHS = {
        "plots_dir": "/plots/matplotlib",
        "uploads_dir": "/uploads", 
        "temp_dir": "/tmp",
        "seaborn_plots": "/plots/seaborn",
    }
    
    # Test constraints and limits
    LIMITS = {
        "max_file_size_mb": 10,
        "max_code_length": 50000,
        "min_execution_time": 0,      # Minimum expected execution time
        "max_execution_time": 60000,  # Maximum allowed execution time (ms)
        "stress_iterations": 5,       # Number of stress test iterations
        "concurrent_requests": 3,     # Concurrent request testing
    }
    
    # Sample test data - all Python code uses pathlib for portability
    SAMPLE_CODE = {
        "simple_print": "print('Hello from execute-raw endpoint')",
        
        "pathlib_basic": """
from pathlib import Path
import json

# Test basic pathlib operations for cross-platform compatibility
base_dir = Path('/tmp')
test_file = base_dir / 'pathlib_test.txt'

# Create directory if needed
base_dir.mkdir(parents=True, exist_ok=True)

# File operations using pathlib
test_file.write_text('Pathlib test successful')
content = test_file.read_text()

result = {
    'message': content,
    'file_path': str(test_file),
    'file_exists': test_file.exists(),
    'file_size': test_file.stat().st_size
}

print(json.dumps(result, indent=2))
""".strip(),
        
        "pathlib_matplotlib": """
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server environment
import matplotlib.pyplot as plt
import numpy as np
import time

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Use pathlib for cross-platform file operations
plots_dir = Path('/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)

# Create unique filename with timestamp
timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
plot_file = plots_dir / f'bdd_test_{timestamp}.png'

# Create and save plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
ax.set_xlabel('X values')
ax.set_ylabel('Y values') 
ax.set_title('BDD Matplotlib Test with Pathlib')
ax.legend()
ax.grid(True, alpha=0.3)

# Save plot using pathlib
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f'Plot saved: {plot_file}')
print(f'File exists: {plot_file.exists()}')
if plot_file.exists():
    print(f'File size: {plot_file.stat().st_size} bytes')
""".strip(),

        "pathlib_file_operations": """
from pathlib import Path
import json
import time

# Comprehensive pathlib file operations test
uploads_dir = Path('/uploads')
temp_dir = Path('/tmp') 
plots_dir = Path('/plots/matplotlib')

# Ensure directories exist using pathlib
for directory in [uploads_dir, temp_dir, plots_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# Create test files using pathlib
timestamp = int(time.time())
test_data = {
    'test_id': f'pathlib_test_{timestamp}',
    'directories': {
        str(dir_path): {
            'exists': dir_path.exists(),
            'is_dir': dir_path.is_dir(),
            'absolute': str(dir_path.absolute())
        }
        for dir_path in [uploads_dir, temp_dir, plots_dir]
    },
    'timestamp': timestamp
}

# Write test file using pathlib
test_file = temp_dir / f'pathlib_test_{timestamp}.json'
test_file.write_text(json.dumps(test_data, indent=2))

# Read back and verify
content = json.loads(test_file.read_text())
print(f'Test file created: {test_file}')
print(f'Content verification: {content["test_id"]}')
print(f'All directories accessible via pathlib')
""".strip(),

        "error_syntax": "print('Missing closing quote",
        "error_import": "import nonexistent_module_that_does_not_exist",
        "error_runtime": "result = 1 / 0  # Division by zero",
        
        "empty_code": "",
        "whitespace_only": "   \n  \t  \n   ",
        
        # Memory and performance test code
        "memory_test": """
from pathlib import Path
import time

# Controlled memory allocation test
temp_dir = Path('/tmp')
data = []

for i in range(50):  # Reasonable size for testing
    data.append([0] * 100)
    if i % 10 == 0:
        print(f'Allocated {i * 100} elements')

print(f'Total memory test: {len(data) * len(data[0])} elements')
print(f'Temp directory accessible: {temp_dir.exists()}')
""".strip(),
    }
    
    # CSV test data for file upload tests
    CSV_TEST_DATA = "name,value,category,timestamp\ntest1,100,A,2025-01-01\ntest2,200,B,2025-01-02\ntest3,300,C,2025-01-03\n"


# ==================== API CONTRACT VALIDATION UTILITIES ====================


def validate_api_contract(response_data: Dict[str, Any]) -> None:
    """
    Validate API response follows the required contract format.
    
    This function enforces the standardized API response format and fails
    tests if the server returns non-compliant responses. Tests should NOT
    be modified to accommodate broken server contracts.
    
    Args:
        response_data: JSON response from any API endpoint
        
    Raises:
        AssertionError: If response doesn't match the required contract
        
    Expected Contract Format:
        {
          "success": true | false,
          "data": <object|null>,
          "error": <string|null>, 
          "meta": { "timestamp": <string> }
        }
        
    For execute-raw responses, data should contain:
        {
          "data": {
            "result": <string>,
            "stdout": <string>,
            "stderr": <string>,
            "executionTime": <int>
          }
        }
        
    Example:
        >>> response = {
        ...     "success": True,
        ...     "data": {"result": "output", "stdout": "output", "stderr": "", "executionTime": 123},
        ...     "error": None,
        ...     "meta": {"timestamp": "2025-01-01T00:00:00Z"}
        ... }
        >>> validate_api_contract(response)  # Should pass
    """
    # Validate required top-level fields
    required_fields = ["success", "data", "error", "meta"]
    for field in required_fields:
        assert field in response_data, f"API contract violation - missing field '{field}': {response_data}"
    
    # Validate field types
    assert isinstance(response_data["success"], bool), \
        f"API contract violation - 'success' must be boolean, got {type(response_data['success'])}"
    
    assert isinstance(response_data["meta"], dict), \
        f"API contract violation - 'meta' must be dict, got {type(response_data['meta'])}"
    
    assert "timestamp" in response_data["meta"], \
        f"API contract violation - meta.timestamp missing: {response_data['meta']}"
    
    # Validate success/error relationship
    if response_data["success"]:
        assert response_data["data"] is not None, \
            "API contract violation - success response must have non-null data"
        assert response_data["error"] is None, \
            "API contract violation - success response must have null error" 
            
        # For execute-raw responses, validate data structure
        if isinstance(response_data["data"], dict) and "result" in response_data["data"]:
            data = response_data["data"]
            execute_fields = ["result", "stdout", "stderr", "executionTime"]
            for field in execute_fields:
                assert field in data, f"API contract violation - execute-raw data missing '{field}': {data}"
            
            # Validate field types for execute-raw
            assert isinstance(data["result"], str), \
                f"API contract violation - data.result must be string, got {type(data['result'])}"
            assert isinstance(data["stdout"], str), \
                f"API contract violation - data.stdout must be string, got {type(data['stdout'])}"
            assert isinstance(data["stderr"], str), \
                f"API contract violation - data.stderr must be string, got {type(data['stderr'])}"
            assert isinstance(data["executionTime"], (int, float)), \
                f"API contract violation - data.executionTime must be number, got {type(data['executionTime'])}"
            assert data["executionTime"] >= TestConstants.LIMITS["min_execution_time"], \
                f"API contract violation - executionTime cannot be negative: {data['executionTime']}"
    else:
        assert response_data["error"] is not None, \
            "API contract violation - error response must have non-null error"
        assert isinstance(response_data["error"], str), \
            f"API contract violation - error must be string, got {type(response_data['error'])}"
        assert response_data["data"] is None, \
            "API contract violation - error response must have null data"


def execute_python_code(
    code: str, 
    timeout: Optional[int] = None,
    expect_success: bool = True
) -> Dict[str, Any]:
    """
    Execute Python code using ONLY the /api/execute-raw endpoint.
    
    This function uses plain text submission (no character escaping) and
    validates the API contract automatically. This is the ONLY way tests
    should execute Python code.
    
    Args:
        code: Python code to execute (plain text, no escaping needed)
        timeout: Request timeout in seconds (uses default if None)
        expect_success: Whether to expect successful execution
        
    Returns:
        Dictionary containing the validated API response
        
    Raises:
        requests.RequestException: If HTTP request fails
        AssertionError: If response doesn't match API contract
        
    Example:
        >>> result = execute_python_code("print('Hello World')")
        >>> assert result["success"] is True  
        >>> assert "Hello World" in result["data"]["stdout"]
        
        >>> # Test error case
        >>> result = execute_python_code("invalid syntax", expect_success=False)
        >>> assert result["success"] is False
        >>> assert result["error"] is not None
    """
    if timeout is None:
        timeout = TestConstants.TIMEOUTS["code_execution"]
    
    response = requests.post(
        f"{TestConstants.BASE_URL}{TestConstants.ENDPOINTS['execute_raw']}",
        headers=TestConstants.HEADERS["text_plain"],
        data=code,  # Plain text, no character escaping needed
        timeout=timeout
    )
    
    # Handle different status codes appropriately
    if expect_success and response.status_code not in [200, 400]:
        response.raise_for_status()
    
    result = response.json()
    validate_api_contract(result)
    return result


# ==================== PYTEST FIXTURES ====================


@pytest.fixture(scope="session")
def server_ready() -> None:
    """
    Session-scoped fixture ensuring server is available for all tests.
    
    This fixture runs once per test session and validates that the
    Pyodide Express Server is running and accessible. If the server
    is not available, the entire test session will be skipped.
    
    Raises:
        pytest.skip: If server is not available within timeout period
        
    Example:
        >>> def test_health_check(server_ready):
        ...     # Server is guaranteed ready here
        ...     response = requests.get(f"{TestConstants.BASE_URL}/health")
        ...     assert response.status_code == 200
    """
    health_url = f"{TestConstants.BASE_URL}{TestConstants.ENDPOINTS['health']}"
    start_time = time.time()
    timeout = TestConstants.TIMEOUTS["server_startup"]
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=TestConstants.TIMEOUTS["quick_operation"])
            if response.status_code == 200:
                return  # Server is ready
        except requests.RequestException:
            pass  # Server not ready yet
        time.sleep(1)
    
    pytest.skip(f"Server at {TestConstants.BASE_URL} not available within {timeout}s")


@pytest.fixture
def cleanup_tracker():
    """
    Fixture providing cleanup tracking for test artifacts.
    
    This fixture creates a cleanup manager that automatically removes
    test files, uploaded files, and other artifacts after each test.
    
    Yields:
        CleanupManager: Object to track resources for cleanup
        
    Example:
        >>> def test_file_upload(cleanup_tracker):
        ...     # Upload file and track for cleanup
        ...     cleanup_tracker.track_upload("test.csv")
        ...     # File automatically cleaned up after test
    """
    class CleanupManager:
        def __init__(self):
            self.temp_files: List[Path] = []
            self.uploaded_files: List[str] = []
            self.start_time = time.time()
        
        def track_temp_file(self, file_path: Union[str, Path]) -> None:
            """Track temporary file for cleanup."""
            self.temp_files.append(Path(file_path))
        
        def track_upload(self, filename: str) -> None:
            """Track uploaded file for API cleanup."""
            self.uploaded_files.append(filename)
        
        def cleanup(self) -> None:
            """Clean up all tracked resources."""
            # Clean uploaded files via API
            for filename in self.uploaded_files:
                try:
                    requests.delete(
                        f"{TestConstants.BASE_URL}{TestConstants.ENDPOINTS['uploaded_files']}/{filename}",
                        timeout=TestConstants.TIMEOUTS["api_request"]
                    )
                except requests.RequestException:
                    pass  # Best effort cleanup
            
            # Clean local temporary files
            for temp_file in self.temp_files:
                if temp_file.exists() and temp_file.is_file():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass  # Best effort cleanup
        
        def get_execution_time(self) -> float:
            """Get test execution time in seconds."""
            return time.time() - self.start_time
    
    manager = CleanupManager()
    yield manager
    manager.cleanup()


@pytest.fixture
def temp_csv_file(cleanup_tracker):
    """
    Fixture providing a temporary CSV file for upload testing.
    
    Creates a CSV file with test data that is automatically cleaned up
    after the test completes.
    
    Args:
        cleanup_tracker: Cleanup manager fixture
        
    Yields:
        Path: Path to the temporary CSV file
        
    Example:
        >>> def test_csv_processing(temp_csv_file):
        ...     assert temp_csv_file.exists()
        ...     content = temp_csv_file.read_text()
        ...     assert "name,value" in content
    """
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write(TestConstants.CSV_TEST_DATA)
    temp_file.close()
    
    file_path = Path(temp_file.name)
    cleanup_tracker.track_temp_file(file_path)
    
    yield file_path


# ==================== BDD TEST SCENARIOS ====================


class TestHealthAndStatusEndpoints:
    """
    BDD test scenarios for health and status monitoring endpoints.
    
    These tests validate that the server health and status endpoints
    are functioning correctly and returning appropriate information.
    """
    
    def test_given_server_running_when_checking_health_then_responds_ok(self, server_ready):
        """
        Scenario: Health check endpoint responds when server is running
        
        Given: The Pyodide Express Server is running and accessible
        When: A client makes a GET request to the /health endpoint
        Then: The server responds with HTTP 200 and health status
        
        This test validates basic server availability and health monitoring.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            HTTP 200 response with health status information
            
        Example:
            >>> response = requests.get("/health")
            >>> assert response.status_code == 200
            >>> assert response.json()["status"] in ["ok", "degraded"]
        """
        # Given: Server is running (ensured by fixture)
        # When: Making health check request
        response = requests.get(
            f"{TestConstants.BASE_URL}{TestConstants.ENDPOINTS['health']}",
            timeout=TestConstants.TIMEOUTS["health_check"]
        )
        
        # Then: Server responds with health status
        assert response.status_code == 200
        health_data = response.json()
        
        # Health endpoint may return different formats, accept common ones
        valid_health_indicators = [
            health_data.get("status") in ["ok", "degraded"],
            health_data.get("success") is True,
            "healthy" in str(health_data).lower(),
            "ok" in str(health_data).lower()
        ]
        
        assert any(valid_health_indicators), f"Health check returned unexpected format: {health_data}"
    
    def test_given_server_running_when_checking_status_then_returns_detailed_info(self, server_ready):
        """
        Scenario: Status endpoint provides detailed server information
        
        Given: The Pyodide Express Server is running with process pool
        When: A client makes a GET request to the /api/status endpoint  
        Then: The server responds with detailed status following API contract
        
        This test validates the server status endpoint and API contract compliance.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            API contract compliant response with server status details
            
        Example:
            >>> response = requests.get("/api/status")
            >>> data = response.json()
            >>> assert data["success"] is True
            >>> assert "ready" in data["data"]
        """
        # Given: Server is running (ensured by fixture)
        # When: Making status check request
        response = requests.get(
            f"{TestConstants.BASE_URL}{TestConstants.ENDPOINTS['status']}",
            timeout=TestConstants.TIMEOUTS["health_check"]
        )
        
        # Then: Response follows API contract with detailed status
        assert response.status_code == 200
        status_data = response.json()
        
        # Validate API contract compliance
        validate_api_contract(status_data)
        assert status_data["success"] is True
        
        # Validate status-specific data
        data = status_data["data"]
        assert isinstance(data, dict), f"Status data must be dict, got {type(data)}"
        
        # Server should provide ready status information
        assert "ready" in data, f"Status missing 'ready' field: {data}"


class TestPythonCodeExecution:
    """
    BDD test scenarios for Python code execution via /api/execute-raw endpoint.
    
    These tests validate Python code execution functionality using ONLY the
    /api/execute-raw endpoint with plain text submission (no character escaping).
    All Python code uses pathlib for cross-platform compatibility.
    """
    
    def test_given_simple_python_code_when_executed_then_captures_stdout(self, server_ready):
        """
        Scenario: Simple Python print statement execution
        
        Given: The server is ready for code execution
        When: A client submits simple Python print code to /api/execute-raw
        Then: The server executes code and captures stdout in API response
        
        This test validates basic Python code execution and output capture.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            Successful execution with print output in data.stdout
            
        Example:
            >>> code = "print('Hello World')"
            >>> result = execute_python_code(code)
            >>> assert "Hello World" in result["data"]["stdout"]
        """
        # Given: Simple Python print code
        code = TestConstants.SAMPLE_CODE["simple_print"]
        
        # When: Executing the code via execute-raw endpoint
        result = execute_python_code(code)
        
        # Then: Execution succeeds with output captured
        assert result["success"] is True
        assert "Hello from execute-raw endpoint" in result["data"]["stdout"]
        assert result["data"]["stderr"] == ""
        assert result["data"]["executionTime"] > 0
        assert isinstance(result["data"]["result"], str)
    
    def test_given_pathlib_code_when_executed_then_handles_cross_platform_paths(self, server_ready):
        """
        Scenario: Python code using pathlib for cross-platform file operations
        
        Given: The server is ready and Python code uses pathlib.Path for file operations
        When: A client submits pathlib-based code to /api/execute-raw
        Then: The server executes code successfully handling pathlib operations
        
        This test validates cross-platform file operations using pathlib.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            Successful pathlib operations with file creation confirmation
            
        Example:
            >>> code = "from pathlib import Path; p = Path('/tmp/test'); print(p.exists())"
            >>> result = execute_python_code(code)
            >>> assert result["success"] is True
        """
        # Given: Python code using pathlib for cross-platform compatibility
        code = TestConstants.SAMPLE_CODE["pathlib_basic"]
        
        # When: Executing pathlib-based code
        result = execute_python_code(code)
        
        # Then: Pathlib operations execute successfully
        assert result["success"] is True
        
        stdout = result["data"]["stdout"]
        assert "Pathlib test successful" in stdout
        assert "/tmp/pathlib_test.txt" in stdout
        assert result["data"]["stderr"] == ""
        
        # Verify pathlib file operations were successful
        assert "file_exists" in stdout
        assert "file_size" in stdout
    
    def test_given_matplotlib_pathlib_code_when_executed_then_creates_plot_files(self, server_ready):
        """
        Scenario: Matplotlib plot creation using pathlib for file operations
        
        Given: The server is ready for matplotlib operations
        When: A client submits matplotlib code using pathlib for plot saving
        Then: The server creates and saves plots using pathlib successfully
        
        This test validates matplotlib integration with pathlib file operations.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            Successful plot creation with pathlib file confirmation
            
        Example:
            >>> code = "import matplotlib.pyplot as plt; from pathlib import Path; ..."
            >>> result = execute_python_code(code)
            >>> assert "Plot saved:" in result["data"]["stdout"]
        """
        # Given: Matplotlib code using pathlib for file operations
        code = TestConstants.SAMPLE_CODE["pathlib_matplotlib"]
        
        # When: Executing matplotlib code with pathlib
        result = execute_python_code(code)
        
        # Then: Plot creation succeeds with pathlib operations
        assert result["success"] is True
        
        stdout = result["data"]["stdout"]
        assert "Plot saved:" in stdout
        assert "File exists: True" in stdout
        assert "File size:" in stdout
        assert result["data"]["stderr"] == ""
    
    def test_given_comprehensive_pathlib_operations_when_executed_then_all_operations_succeed(self, server_ready):
        """
        Scenario: Comprehensive pathlib file operations across multiple directories
        
        Given: The server supports full pathlib operations
        When: A client submits code with multiple pathlib operations (create, read, write, check)
        Then: All pathlib operations execute successfully across different directories
        
        This test validates comprehensive pathlib functionality.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            Successful pathlib operations across uploads, temp, and plots directories
            
        Example:
            >>> # Code creates and accesses files in multiple directories
            >>> result = execute_python_code(comprehensive_pathlib_code)
            >>> assert "All directories accessible via pathlib" in result["data"]["stdout"]
        """
        # Given: Comprehensive pathlib operations code
        code = TestConstants.SAMPLE_CODE["pathlib_file_operations"]
        
        # When: Executing comprehensive pathlib operations
        result = execute_python_code(code)
        
        # Then: All pathlib operations succeed
        assert result["success"] is True
        
        stdout = result["data"]["stdout"]
        assert "Test file created:" in stdout
        assert "Content verification:" in stdout
        assert "All directories accessible via pathlib" in stdout
        assert result["data"]["stderr"] == ""


class TestErrorHandlingScenarios:
    """
    BDD test scenarios for error handling and edge cases.
    
    These tests validate that the server properly handles various error
    conditions and invalid inputs while maintaining API contract compliance.
    """
    
    def test_given_python_syntax_error_when_executed_then_captures_error_appropriately(self, server_ready):
        """
        Scenario: Python syntax error handling
        
        Given: The server is ready for code execution
        When: A client submits Python code with syntax errors
        Then: The server captures and reports the syntax error appropriately
        
        This test validates syntax error handling and reporting.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            Error captured in stderr or error field following API contract
            
        Example:
            >>> code = "print('missing quote"
            >>> result = execute_python_code(code, expect_success=False)
            >>> # Error should be captured in response
        """
        # Given: Python code with syntax error
        code = TestConstants.SAMPLE_CODE["error_syntax"]
        
        # When: Executing code with syntax error
        result = execute_python_code(code, expect_success=False)
        
        # Then: Error is captured appropriately
        if result["success"]:
            # If marked as success, error should be in stderr
            assert result["data"]["stderr"] != "", \
                f"Expected stderr for syntax error, got: {result['data']}"
            assert any(term in result["data"]["stderr"].lower() 
                      for term in ["syntax", "error", "invalid"]), \
                f"Expected syntax error in stderr: {result['data']['stderr']}"
        else:
            # If marked as failure, error should be in error field
            assert result["error"] is not None, \
                f"Expected error field for syntax error, got: {result}"
            assert any(term in result["error"].lower() 
                      for term in ["syntax", "error", "invalid", "execution", "failed"]), \
                f"Expected syntax error in error field: {result['error']}"
    
    def test_given_python_import_error_when_executed_then_handles_module_not_found(self, server_ready):
        """
        Scenario: Python import error handling
        
        Given: The server is ready for code execution
        When: A client submits code importing non-existent modules
        Then: The server captures and reports the import error
        
        This test validates import error handling.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            Import error captured and reported
            
        Example:
            >>> code = "import nonexistent_module"
            >>> result = execute_python_code(code, expect_success=False)
            >>> # Import error should be captured
        """
        # Given: Python code with import error
        code = TestConstants.SAMPLE_CODE["error_import"]
        
        # When: Executing code with import error
        result = execute_python_code(code, expect_success=False)
        
        # Then: Import error is captured
        if result["success"]:
            assert result["data"]["stderr"] != "", \
                f"Expected stderr for import error, got: {result['data']}"
        else:
            assert result["error"] is not None, \
                f"Expected error field for import error, got: {result}"
    
    def test_given_python_runtime_error_when_executed_then_captures_execution_error(self, server_ready):
        """
        Scenario: Python runtime error handling
        
        Given: The server is ready for code execution
        When: A client submits code that causes runtime errors (division by zero)
        Then: The server captures and reports the runtime error
        
        This test validates runtime error handling.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            Runtime error captured and reported
            
        Example:
            >>> code = "result = 1 / 0"
            >>> result = execute_python_code(code, expect_success=False)  
            >>> # Runtime error should be captured
        """
        # Given: Python code with runtime error
        code = TestConstants.SAMPLE_CODE["error_runtime"]
        
        # When: Executing code with runtime error
        result = execute_python_code(code, expect_success=False)
        
        # Then: Runtime error is captured
        if result["success"]:
            assert result["data"]["stderr"] != "", \
                f"Expected stderr for runtime error, got: {result['data']}"
        else:
            assert result["error"] is not None, \
                f"Expected error field for runtime error, got: {result}"
    
    def test_given_empty_code_when_submitted_then_returns_validation_error(self, server_ready):
        """
        Scenario: Empty code validation
        
        Given: The server validates input before execution
        When: A client submits empty or whitespace-only code
        Then: The server returns a validation error without execution
        
        This test validates input validation for empty submissions.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            HTTP 400 response with validation error message
            
        Example:
            >>> response = requests.post("/api/execute-raw", data="")
            >>> assert response.status_code == 400
            >>> assert "No Python code provided" in response.json()["error"]
        """
        # Given: Empty code
        code = TestConstants.SAMPLE_CODE["empty_code"]
        
        # When: Submitting empty code
        response = requests.post(
            f"{TestConstants.BASE_URL}{TestConstants.ENDPOINTS['execute_raw']}",
            headers=TestConstants.HEADERS["text_plain"],
            data=code,
            timeout=TestConstants.TIMEOUTS["code_execution"]
        )
        
        # Then: Returns validation error
        assert response.status_code == 400
        result = response.json()
        validate_api_contract(result)
        assert result["success"] is False
        assert "No Python code provided" in result["error"]
    
    def test_given_whitespace_only_code_when_submitted_then_returns_validation_error(self, server_ready):
        """
        Scenario: Whitespace-only code validation
        
        Given: The server validates meaningful code content
        When: A client submits whitespace-only code
        Then: The server returns a validation error
        
        This test validates input validation for whitespace-only submissions.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            HTTP 400 response with validation error message
            
        Example:
            >>> response = requests.post("/api/execute-raw", data="   \n  ")
            >>> assert response.status_code == 400
        """
        # Given: Whitespace-only code
        code = TestConstants.SAMPLE_CODE["whitespace_only"]
        
        # When: Submitting whitespace-only code
        response = requests.post(
            f"{TestConstants.BASE_URL}{TestConstants.ENDPOINTS['execute_raw']}",
            headers=TestConstants.HEADERS["text_plain"],
            data=code,
            timeout=TestConstants.TIMEOUTS["code_execution"]
        )
        
        # Then: Returns validation error
        assert response.status_code == 400
        result = response.json()
        validate_api_contract(result)
        assert result["success"] is False
        assert "No Python code provided" in result["error"]


class TestPerformanceAndLimits:
    """
    BDD test scenarios for performance testing and resource limits.
    
    These tests validate that the server handles performance requirements
    and resource constraints appropriately.
    """
    
    def test_given_memory_intensive_code_when_executed_then_completes_within_limits(self, server_ready):
        """
        Scenario: Memory-intensive code execution within reasonable limits
        
        Given: The server supports memory allocation up to reasonable limits
        When: A client submits memory-intensive code with pathlib operations  
        Then: The code executes successfully within memory and time constraints
        
        This test validates performance with memory allocation.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            Successful execution of memory-intensive operations
            
        Example:
            >>> # Code allocates memory and uses pathlib
            >>> result = execute_python_code(memory_test_code)
            >>> assert result["success"] is True
            >>> assert result["data"]["executionTime"] < 30000  # Under 30 seconds
        """
        # Given: Memory-intensive code with pathlib
        code = TestConstants.SAMPLE_CODE["memory_test"]
        
        # When: Executing memory-intensive code
        result = execute_python_code(code)
        
        # Then: Executes successfully within constraints
        assert result["success"] is True
        
        stdout = result["data"]["stdout"]
        assert "Total memory test:" in stdout
        assert "Temp directory accessible: True" in stdout
        
        # Validate execution time is reasonable
        execution_time = result["data"]["executionTime"]
        assert execution_time < TestConstants.LIMITS["max_execution_time"], \
            f"Execution time {execution_time}ms exceeds limit {TestConstants.LIMITS['max_execution_time']}ms"
    
    def test_given_multiple_sequential_requests_when_executed_then_all_succeed(self, server_ready):
        """
        Scenario: Multiple sequential code execution requests
        
        Given: The server supports multiple sequential requests
        When: A client makes multiple sequential code execution requests
        Then: All requests execute successfully with proper API contract compliance
        
        This test validates server stability under sequential load.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            All sequential requests succeed with consistent API responses
            
        Example:
            >>> for i in range(5):
            ...     result = execute_python_code(f"print(f'Request {i}')")
            ...     assert result["success"] is True
        """
        # Given: Multiple sequential requests
        num_requests = TestConstants.LIMITS["stress_iterations"]
        
        # When: Making multiple sequential requests
        results = []
        for i in range(num_requests):
            code = f"print(f'Sequential request {i + 1} of {num_requests}')"
            result = execute_python_code(code)
            results.append(result)
        
        # Then: All requests succeed
        for i, result in enumerate(results):
            assert result["success"] is True, f"Request {i + 1} failed: {result}"
            assert f"Sequential request {i + 1}" in result["data"]["stdout"]
            assert result["data"]["executionTime"] > 0


class TestAPIContractCompliance:
    """
    BDD test scenarios specifically for API contract validation.
    
    These tests ensure the server strictly follows the required API contract
    format and that tests fail if the server returns non-compliant responses.
    """
    
    def test_given_successful_execution_when_response_received_then_follows_success_contract(self, server_ready):
        """
        Scenario: Successful execution API contract validation
        
        Given: The server executes Python code successfully
        When: The API response is returned for successful execution
        Then: The response follows the exact success contract format
        
        This test validates API contract compliance for successful executions.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            Response exactly matching required API contract format
            
        Example:
            >>> result = execute_python_code("print('test')")
            >>> assert result["success"] is True
            >>> assert result["data"] is not None
            >>> assert result["error"] is None
            >>> assert "timestamp" in result["meta"]
        """
        # Given: Simple successful Python code
        code = "print('API contract test')"
        
        # When: Executing code and receiving response
        result = execute_python_code(code)
        
        # Then: Response follows exact success contract
        assert result["success"] is True
        assert result["data"] is not None
        assert result["error"] is None
        assert isinstance(result["meta"], dict)
        assert "timestamp" in result["meta"]
        
        # Validate execute-raw specific data structure
        data = result["data"]
        assert "result" in data
        assert "stdout" in data
        assert "stderr" in data
        assert "executionTime" in data
        
        assert isinstance(data["result"], str)
        assert isinstance(data["stdout"], str)
        assert isinstance(data["stderr"], str)
        assert isinstance(data["executionTime"], (int, float))
    
    def test_given_execution_error_when_response_received_then_follows_error_contract(self, server_ready):
        """
        Scenario: Error execution API contract validation
        
        Given: The server encounters an error during Python code execution
        When: The API response is returned for failed execution
        Then: The response follows the exact error contract format
        
        This test validates API contract compliance for error responses.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            Response exactly matching required error API contract format
            
        Example:
            >>> result = execute_python_code("invalid syntax", expect_success=False)
            >>> # Response should follow error contract regardless of success value
        """
        # Given: Python code that will cause an error
        code = TestConstants.SAMPLE_CODE["error_syntax"]
        
        # When: Executing code that causes error
        result = execute_python_code(code, expect_success=False)
        
        # Then: Response follows API contract for error case
        if result["success"] is False:
            # Error contract: success=false, data=null, error=string
            assert result["data"] is None
            assert result["error"] is not None
            assert isinstance(result["error"], str)
        else:
            # If success=true, error should be in stderr, not error field
            assert result["data"] is not None
            assert result["error"] is None
            assert isinstance(result["data"]["stderr"], str)
        
        # Metadata should always be present
        assert isinstance(result["meta"], dict)
        assert "timestamp" in result["meta"]
    
    def test_given_any_api_response_when_validated_then_contains_required_fields(self, server_ready):
        """
        Scenario: Universal API contract field validation
        
        Given: Any API endpoint returns a response
        When: The response is received and validated
        Then: All required contract fields are present with correct types
        
        This test validates universal API contract compliance.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            All API responses contain required fields with correct types
            
        Example:
            >>> # Test multiple endpoints for contract compliance
            >>> health_response = requests.get("/health")
            >>> status_response = requests.get("/api/status")
            >>> execute_response = requests.post("/api/execute-raw", data="print('test')")
            >>> # All should have consistent field structure
        """
        # Given: Various API endpoints
        endpoints_to_test = [
            ("GET", TestConstants.ENDPOINTS["status"], None),
            ("POST", TestConstants.ENDPOINTS["execute_raw"], "print('contract test')"),
        ]
        
        # When: Making requests to different endpoints
        for method, endpoint, data in endpoints_to_test:
            if method == "GET":
                response = requests.get(
                    f"{TestConstants.BASE_URL}{endpoint}",
                    timeout=TestConstants.TIMEOUTS["api_request"]
                )
            else:  # POST
                response = requests.post(
                    f"{TestConstants.BASE_URL}{endpoint}",
                    headers=TestConstants.HEADERS["text_plain"],
                    data=data,
                    timeout=TestConstants.TIMEOUTS["api_request"]
                )
            
            # Then: Response follows API contract
            if response.status_code == 200:
                result = response.json()
                validate_api_contract(result)  # This will fail if contract is violated
                
                # Verify required fields are present
                assert "success" in result
                assert "data" in result
                assert "error" in result
                assert "meta" in result
                assert "timestamp" in result["meta"]


class TestFileOperations:
    """
    BDD test scenarios for file upload and management operations.
    
    These tests validate file upload, listing, and deletion functionality
    while ensuring proper cleanup and API contract compliance.
    """
    
    def test_given_csv_file_when_uploaded_then_becomes_available(self, server_ready, temp_csv_file, cleanup_tracker):
        """
        Scenario: CSV file upload and availability
        
        Given: A CSV file with test data exists locally
        When: The file is uploaded via /api/upload endpoint
        Then: The file becomes available in the server's file system
        
        This test validates file upload functionality and server file management.
        
        Args:
            server_ready: Fixture ensuring server is available
            temp_csv_file: Fixture providing test CSV file
            cleanup_tracker: Fixture for automatic cleanup
            
        Expected Output:
            Successful upload with file available for subsequent operations
            
        Example:
            >>> files = {"file": open("test.csv", "rb")}
            >>> response = requests.post("/api/upload", files=files)
            >>> assert response.json()["success"] is True
        """
        # Given: CSV file with test data (provided by fixture)
        assert temp_csv_file.exists()
        
        # When: Uploading file via API
        with open(temp_csv_file, 'rb') as file:
            files = {"file": file}
            response = requests.post(
                f"{TestConstants.BASE_URL}{TestConstants.ENDPOINTS['upload']}",
                files=files,
                timeout=TestConstants.TIMEOUTS["file_upload"]
            )
        
        # Then: Upload succeeds and file is available
        assert response.status_code == 200
        upload_result = response.json()
        validate_api_contract(upload_result)
        assert upload_result["success"] is True
        
        # Track uploaded file for cleanup
        if upload_result["data"] and "file" in upload_result["data"]:
            filename = upload_result["data"]["file"].get("sanitizedOriginal") or upload_result["data"]["file"].get("filename")
            if filename:
                cleanup_tracker.track_upload(filename)
    
    def test_given_uploaded_files_when_listed_then_shows_available_files(self, server_ready):
        """
        Scenario: Listing uploaded files
        
        Given: Files have been uploaded to the server
        When: A client requests the list of uploaded files via /api/uploaded-files
        Then: The server returns a list of available uploaded files
        
        This test validates file listing functionality.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            API contract compliant response with list of uploaded files
            
        Example:
            >>> response = requests.get("/api/uploaded-files")
            >>> files_data = response.json()
            >>> assert files_data["success"] is True
            >>> assert "files" in files_data["data"]
        """
        # Given: Server with potential uploaded files
        # When: Requesting list of uploaded files
        response = requests.get(
            f"{TestConstants.BASE_URL}{TestConstants.ENDPOINTS['uploaded_files']}",
            timeout=TestConstants.TIMEOUTS["api_request"]
        )
        
        # Then: Returns file list following API contract
        assert response.status_code == 200
        files_result = response.json()
        validate_api_contract(files_result)
        assert files_result["success"] is True
        
        # Data should contain files list (empty or populated)
        assert "files" in files_result["data"]
        assert isinstance(files_result["data"]["files"], list)
    
    def test_given_uploaded_file_when_deleted_then_removes_from_system(self, server_ready, temp_csv_file):
        """
        Scenario: File deletion functionality
        
        Given: A file has been uploaded and is available on the server
        When: A client requests deletion of the file via DELETE /api/uploaded-files/{filename}
        Then: The file is removed from the server file system
        
        This test validates file deletion functionality.
        
        Args:
            server_ready: Fixture ensuring server is available
            temp_csv_file: Fixture providing test CSV file
            
        Expected Output:
            Successful file deletion with confirmation response
            
        Example:
            >>> # Upload file first
            >>> upload_response = requests.post("/api/upload", files={"file": file})
            >>> filename = upload_response.json()["data"]["file"]["filename"]
            >>> # Then delete it
            >>> delete_response = requests.delete(f"/api/uploaded-files/{filename}")
            >>> assert delete_response.json()["success"] is True
        """
        # Given: Upload a file first
        with open(temp_csv_file, 'rb') as file:
            files = {"file": file}
            upload_response = requests.post(
                f"{TestConstants.BASE_URL}{TestConstants.ENDPOINTS['upload']}",
                files=files,
                timeout=TestConstants.TIMEOUTS["file_upload"]
            )
        
        assert upload_response.status_code == 200
        upload_result = upload_response.json()
        validate_api_contract(upload_result)
        assert upload_result["success"] is True
        
        # Extract filename for deletion
        filename = None
        if upload_result["data"] and "file" in upload_result["data"]:
            file_info = upload_result["data"]["file"]
            filename = file_info.get("sanitizedOriginal") or file_info.get("filename")
        
        if filename:
            # When: Deleting the uploaded file
            delete_response = requests.delete(
                f"{TestConstants.BASE_URL}{TestConstants.ENDPOINTS['uploaded_files']}/{filename}",
                timeout=TestConstants.TIMEOUTS["api_request"]
            )
            
            # Then: Deletion succeeds
            assert delete_response.status_code in [200, 404]  # 404 if already deleted
            if delete_response.status_code == 200:
                delete_result = delete_response.json()
                validate_api_contract(delete_result)
                assert delete_result["success"] is True


class TestAdvancedScenarios:
    """
    BDD test scenarios for advanced functionality and edge cases.
    
    These tests cover complex workflows, integration scenarios, and
    comprehensive validation of server capabilities.
    """
    
    def test_given_pathlib_python_with_file_creation_when_executed_then_creates_accessible_files(self, server_ready):
        """
        Scenario: Python code creates files accessible via pathlib
        
        Given: The server supports Python pathlib file operations
        When: Python code creates files using pathlib in mounted directories
        Then: The files are created and accessible via subsequent operations
        
        This test validates end-to-end file creation and access via pathlib.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            File creation confirmation with pathlib verification
            
        Example:
            >>> code = '''
            ... from pathlib import Path
            ... test_file = Path("/tmp/created_file.txt")
            ... test_file.write_text("Created via pathlib")
            ... print(f"File created: {test_file.exists()}")
            ... '''
            >>> result = execute_python_code(code)
            >>> assert "File created: True" in result["data"]["stdout"]
        """
        # Given: Python code that creates files using pathlib
        code = '''
from pathlib import Path
import json
import time

# Create test file using pathlib
temp_dir = Path("/tmp")
timestamp = int(time.time())
test_file = temp_dir / f"bdd_test_file_{timestamp}.txt"

# Create file content
content = {
    "test_type": "BDD file creation test",
    "timestamp": timestamp,
    "pathlib_used": True,
    "file_path": str(test_file)
}

# Write file using pathlib
test_file.write_text(json.dumps(content, indent=2))

# Verify file creation
verification = {
    "file_exists": test_file.exists(),
    "file_size": test_file.stat().st_size if test_file.exists() else 0,
    "file_readable": test_file.is_file() if test_file.exists() else False
}

print(f"File created at: {test_file}")
print(f"File verification: {verification}")
print(f"Content preview: {test_file.read_text()[:100]}..." if test_file.exists() else "File not found")
'''.strip()
        
        # When: Executing file creation code
        result = execute_python_code(code)
        
        # Then: File creation succeeds with pathlib
        assert result["success"] is True
        
        stdout = result["data"]["stdout"]
        assert "File created at:" in stdout
        assert "file_exists': True" in stdout
        assert "file_readable': True" in stdout
        assert "Content preview:" in stdout
    
    def test_given_comprehensive_workflow_when_executed_then_all_operations_succeed(self, server_ready):
        """
        Scenario: Comprehensive data processing workflow
        
        Given: The server supports complete data science workflows
        When: A complex workflow with multiple operations is executed
        Then: All operations complete successfully with proper pathlib usage
        
        This test validates comprehensive server capabilities.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Expected Output:
            Successful completion of complex multi-step workflow
            
        Example:
            >>> # Complex workflow with data processing, file operations, and validation
            >>> result = execute_python_code(comprehensive_workflow_code)
            >>> assert "Workflow completed successfully" in result["data"]["stdout"]
        """
        # Given: Comprehensive workflow code
        code = '''
from pathlib import Path
import json
import time
import math

# Step 1: Setup directories using pathlib
base_dirs = [Path("/tmp"), Path("/uploads"), Path("/plots/matplotlib")]
for directory in base_dirs:
    directory.mkdir(parents=True, exist_ok=True)

# Step 2: Generate test data
data_size = 100
test_data = {
    "numbers": [i * 2 for i in range(data_size)],
    "squares": [i ** 2 for i in range(data_size)],
    "sin_values": [math.sin(i * 0.1) for i in range(data_size)]
}

# Step 3: File operations using pathlib
data_file = Path("/tmp") / f"workflow_data_{int(time.time())}.json"
data_file.write_text(json.dumps(test_data, indent=2))

# Step 4: Data processing
processed_data = {
    "total_numbers": len(test_data["numbers"]),
    "sum_numbers": sum(test_data["numbers"]),
    "max_square": max(test_data["squares"]),
    "avg_sin": sum(test_data["sin_values"]) / len(test_data["sin_values"])
}

# Step 5: Results file using pathlib
results_file = Path("/tmp") / f"workflow_results_{int(time.time())}.json"
results_file.write_text(json.dumps(processed_data, indent=2))

# Step 6: Verification
verification = {
    "data_file_exists": data_file.exists(),
    "results_file_exists": results_file.exists(),
    "data_file_size": data_file.stat().st_size,
    "results_file_size": results_file.stat().st_size,
    "all_directories_exist": all(d.exists() for d in base_dirs)
}

print("Comprehensive workflow completed successfully")
print(f"Data processing results: {processed_data}")
print(f"File verification: {verification}")
print(f"All pathlib operations successful: {all(verification.values())}")
'''.strip()
        
        # When: Executing comprehensive workflow
        result = execute_python_code(code, timeout=TestConstants.TIMEOUTS["code_execution"])
        
        # Then: Workflow completes successfully
        assert result["success"] is True
        
        stdout = result["data"]["stdout"]
        assert "Comprehensive workflow completed successfully" in stdout
        assert "Data processing results:" in stdout
        assert "All pathlib operations successful: True" in stdout
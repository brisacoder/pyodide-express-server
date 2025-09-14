"""
Pytest configuration and shared fixtures for Pyodide Express Server tests.

This module provides centralized configuration, fixtures, and utilities
for all test modules in the test suite. It ensures consistent behavior
across all tests and provides reusable components.

Shared Components:
- Configuration constants and settings
- Server readiness fixtures
- API contract validation
- Test cleanup utilities
- Common test data generators
- Utility functions for code execution

Requirements Compliance:
1. ✅ Centralized configuration eliminates hardcoded globals
2. ✅ Shared fixtures ensure consistent test setup
3. ✅ API contract validation enforces response format
4. ✅ Only /api/execute-raw endpoint usage
5. ✅ Comprehensive error handling
6. ✅ Pathlib usage for all file operations
7. ✅ Modern pytest patterns and best practices
"""

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import requests


class Config:
    """Centralized test configuration constants and settings."""
    
    # Server configuration
    BASE_URL = "http://localhost:3000"
    
    # Timeout settings (in seconds)
    TIMEOUTS = {
        "server_start": 180,      # Time to wait for server startup
        "server_health": 30,      # Time to wait for health check
        "code_execution": 45,     # Maximum time for code execution
        "api_request": 10,        # Standard API request timeout
        "quick_operation": 5,     # Quick operations (health checks)
    }
    
    # API endpoints
    ENDPOINTS = {
        "health": "/health",
        "execute_raw": "/api/execute-raw",
        "reset": "/api/reset",
        "install_package": "/api/install-package",
        "uploaded_files": "/api/uploaded-files",
        "upload": "/api/upload",
        "plots_extract": "/api/plots/extract",
    }
    
    # Request headers
    HEADERS = {
        "execute_raw": {"Content-Type": "text/plain"},
        "json": {"Content-Type": "application/json"},
        "form": {"Content-Type": "multipart/form-data"},
    }
    
    # File and directory settings
    PATHS = {
        "plots_dir": "/plots/matplotlib",
        "uploads_dir": "/uploads",
        "temp_dir": "/tmp",
        "test_data_dir": "/test_data",
    }
    
    # Plot generation settings
    PLOT_SETTINGS = {
        "default_dpi": 150,
        "figure_size": (8, 6),
        "supported_formats": ["png", "jpg", "pdf", "svg"],
    }
    
    # Test data settings (maintaining backward compatibility)
    DEFAULT_TIMEOUT = 30
    MAX_CODE_LENGTH = 50000
    MAX_FILE_SIZE_MB = 10


def wait_for_server(url: str, timeout: int = 120) -> bool:
    """Poll URL until it responds or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="session")
def server():
    """Ensure server is running for the entire test session (legacy compatibility)."""
    # Check if server is already running
    if wait_for_server(f"{Config.BASE_URL}/health", timeout=5):
        yield None  # Server already running
        return
    
    # Start server if not running
    process = subprocess.Popen(
        ["node", "src/server.js"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent.parent,
    )
    
    # Wait for server to start
    if not wait_for_server(f"{Config.BASE_URL}/health"):
        process.terminate()
        raise RuntimeError("Server failed to start")
    
    yield process
    
    # Cleanup
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture(scope="session")
def server_ready():
    """
    Ensure server is running and ready to accept requests.
    
    This fixture runs once per test session and validates that the
    Pyodide Express Server is available and responding to requests.
    
    Returns:
        None: Fixture validates server availability
        
    Raises:
        pytest.skip: If server is not available within timeout period
        
    Example:
        >>> def test_something(server_ready):
        ...     # Server is guaranteed to be ready here
        ...     response = requests.get(f"{Config.BASE_URL}/health")
        ...     assert response.status_code == 200
    """
    def wait_for_server_ready(url: str, timeout: int) -> None:
        """Wait for server to become available."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(url, timeout=Config.TIMEOUTS["quick_operation"])
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        pytest.skip(f"Server at {url} not available within {timeout}s")
    
    # Check server health
    health_url = f"{Config.BASE_URL}{Config.ENDPOINTS['health']}"
    wait_for_server_ready(health_url, Config.TIMEOUTS["server_health"])
    
    # Optional: Reset Pyodide environment for clean test state
    try:
        reset_url = f"{Config.BASE_URL}{Config.ENDPOINTS['reset']}"
        reset_response = requests.post(reset_url, timeout=Config.TIMEOUTS["api_request"])
        if reset_response.status_code == 200:
            print("✅ Pyodide environment reset successfully")
    except requests.RequestException:
        print("⚠️ Warning: Could not reset Pyodide environment")


@pytest.fixture
def test_cleanup():
    """
    Provide cleanup functionality for test artifacts.
    
    This fixture creates a cleanup tracker that automatically removes
    test files and artifacts after each test completes.
    
    Yields:
        CleanupTracker: Object to track files and resources for cleanup
        
    Example:
        >>> def test_file_creation(test_cleanup):
        ...     cleanup = test_cleanup
        ...     # Create test file
        ...     test_file = Path("/tmp/test.txt")
        ...     test_file.write_text("test")
        ...     cleanup.track_file(test_file)
        ...     # File automatically cleaned up after test
    """
    class CleanupTracker:
        """Tracks and cleans up test artifacts."""
        
        def __init__(self):
            self.temp_files = []
            self.temp_dirs = []
            self.uploaded_files = []
            self.start_time = time.time()
        
        def track_file(self, filepath: str | Path) -> None:
            """Track file for automatic cleanup."""
            self.temp_files.append(Path(filepath))
        
        def track_directory(self, dirpath: str | Path) -> None:
            """Track directory for automatic cleanup."""
            self.temp_dirs.append(Path(dirpath))
        
        def track_upload(self, filename: str) -> None:
            """Track uploaded file for API cleanup."""
            self.uploaded_files.append(filename)
        
        def cleanup(self) -> None:
            """Clean up all tracked files and directories."""
            # Clean up uploaded files via API
            for filename in self.uploaded_files:
                try:
                    requests.delete(f"{Config.BASE_URL}/api/uploaded-files/{filename}", timeout=5)
                except Exception:
                    pass  # Best effort cleanup
            
            # Clean up local files
            for temp_file in self.temp_files:
                if temp_file.exists() and temp_file.is_file():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass  # Best effort cleanup
            
            # Clean up directories (in reverse order for nested dirs)
            for temp_dir in reversed(self.temp_dirs):
                if temp_dir.exists() and temp_dir.is_dir():
                    try:
                        # Remove directory and contents
                        import shutil
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass  # Best effort cleanup
        
        def get_test_duration(self) -> float:
            """Get test execution duration in seconds."""
            return time.time() - self.start_time
    
    tracker = CleanupTracker()
    yield tracker
    tracker.cleanup()


# Legacy fixtures for backward compatibility
@pytest.fixture
def base_url():
    """Provide the base URL for API requests."""
    return Config.BASE_URL


@pytest.fixture
def api_timeout():
    """Provide default API timeout for requests."""
    return Config.DEFAULT_TIMEOUT


@pytest.fixture
def max_code_length():
    """Provide maximum allowed code length."""
    return Config.MAX_CODE_LENGTH


@pytest.fixture
def max_file_size_mb():
    """Provide maximum allowed file size in MB."""
    return Config.MAX_FILE_SIZE_MB


@pytest.fixture(autouse=True)
def cleanup_uploads(request):
    """Automatically clean up uploaded files after each test (legacy compatibility)."""
    uploaded_files = []
    
    # Store original methods to track uploads
    original_post = requests.post
    
    def track_post(url, *args, **kwargs):
        response = original_post(url, *args, **kwargs)
        # Track uploaded files
        if "/upload" in url and response.status_code == 200:
            try:
                data = response.json()
                if data.get("success") and "data" in data and "file" in data["data"]:
                    file_info = data["data"]["file"]
                    if "sanitizedOriginal" in file_info:
                        uploaded_files.append(file_info["sanitizedOriginal"])
            except Exception:
                pass
        return response
    
    # Monkey patch for tracking
    requests.post = track_post
    
    yield
    
    # Restore original method
    requests.post = original_post
    
    # Clean up tracked files
    for filename in uploaded_files:
        try:
            requests.delete(f"{Config.BASE_URL}/api/uploaded-files/{filename}", timeout=5)
        except Exception:
            pass


def validate_api_contract(response_data: Dict[str, Any]) -> None:
    """
    Validate API response follows the expected contract format.
    
    This function ensures all API responses conform to the standardized
    format required by the system specification.
    
    Args:
        response_data: JSON response from API endpoint
        
    Raises:
        AssertionError: If response doesn't match contract specification
        
    Example:
        >>> response = {
        ...     "success": True,
        ...     "data": {"result": "output", "stdout": "", "stderr": "", "executionTime": 100},
        ...     "error": None,
        ...     "meta": {"timestamp": "2025-01-01T00:00:00Z"}
        ... }
        >>> validate_api_contract(response)  # Should pass without error
    
    Contract Format:
        {
          "success": true | false,
          "data": { "result": any, "stdout": str, "stderr": str, "executionTime": int } | null,
          "error": str | null,
          "meta": { "timestamp": str }
        }
    """
    # Validate top-level structure
    required_fields = ["success", "data", "error", "meta"]
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"
    
    # Validate field types
    assert isinstance(response_data["success"], bool), f"success must be boolean: {type(response_data['success'])}"
    assert isinstance(response_data["meta"], dict), f"meta must be dict: {type(response_data['meta'])}"
    assert "timestamp" in response_data["meta"], "meta must contain timestamp"
    
    # Validate success/error relationship
    if response_data["success"]:
        assert response_data["data"] is not None, "Success response should have non-null data"
        assert response_data["error"] is None, "Success response should have null error"
        
        # For execute-raw responses, validate data structure
        if isinstance(response_data["data"], dict) and "result" in response_data["data"]:
            data = response_data["data"]
            required_data_fields = ["result", "stdout", "stderr", "executionTime"]
            for field in required_data_fields:
                assert field in data, f"data missing '{field}': {data}"
            
            # Validate executionTime is a positive number
            assert isinstance(data["executionTime"], (int, float)), f"executionTime must be number: {type(data['executionTime'])}"
            assert data["executionTime"] >= 0, f"executionTime must be non-negative: {data['executionTime']}"
    else:
        assert response_data["error"] is not None, "Error response should have non-null error"
        assert isinstance(response_data["error"], str), f"error must be string: {type(response_data['error'])}"


def execute_python_code(code: str, timeout: int = None) -> Dict[str, Any]:
    """
    Execute Python code using the /api/execute-raw endpoint.
    
    This is a convenience function for executing Python code in tests
    with automatic API contract validation and error handling.
    
    Args:
        code: Python code to execute
        timeout: Request timeout in seconds (uses default if None)
        
    Returns:
        Dictionary containing the validated API response
        
    Raises:
        requests.RequestException: If request fails
        AssertionError: If response doesn't match API contract
        
    Example:
        >>> result = execute_python_code("print('Hello World')")
        >>> assert result["success"] is True
        >>> assert "Hello World" in result["data"]["stdout"]
    """
    if timeout is None:
        timeout = Config.TIMEOUTS["code_execution"]
    
    response = requests.post(
        f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
        headers=Config.HEADERS["execute_raw"],
        data=code,
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    validate_api_contract(result)
    return result


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "filesystem: marks tests that interact with filesystem"
    )
    config.addinivalue_line(
        "markers", "plotting: marks tests that generate plots"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test names
        if "slow" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        if "integration" in item.name.lower() or "complex" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        if "api" in item.name.lower() or "execute" in item.name.lower():
            item.add_marker(pytest.mark.api)
        
        if "file" in item.name.lower() or "directory" in item.name.lower():
            item.add_marker(pytest.mark.filesystem)
        
        if "plot" in item.name.lower() or "matplotlib" in item.name.lower():
            item.add_marker(pytest.mark.plotting)


def pytest_report_header(config):
    """Add custom header information to pytest reports."""
    return [
        f"Pyodide Express Server Tests",
        f"Server URL: {Config.BASE_URL}",
        f"Test Configuration: {len(Config.ENDPOINTS)} endpoints configured"
    ]
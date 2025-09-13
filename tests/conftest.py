"""Pytest configuration and fixtures for BDD-style test suite.

This module provides shared configuration, fixtures, and utilities
for all test files following BDD (Behavior-Driven Development) patterns.
"""

import pytest
import subprocess
import tempfile
import time
import requests
from pathlib import Path
from typing import Generator

# Global Configuration Constants
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30  # Default timeout for API requests
EXECUTE_TIMEOUT = 30000  # Default timeout for code execution (30 seconds)
LONG_TIMEOUT = 60000  # Extended timeout for complex operations (60 seconds)


def wait_for_server(url: str, timeout: int = 120):
    """Poll ``url`` until it responds or timeout expires."""
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
    """Ensure server is running for the entire test session."""
    # Check if server is already running
    if wait_for_server(f"{BASE_URL}/health", timeout=5):
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
    if not wait_for_server(f"{BASE_URL}/health"):
        process.terminate()
        raise RuntimeError("Server failed to start")
    
    yield process
    
    # Cleanup
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture
def base_url():
    """Provide the base URL for API requests."""
    return BASE_URL


@pytest.fixture
def default_timeout() -> int:
    """Provide the default timeout for API requests."""
    return DEFAULT_TIMEOUT


@pytest.fixture
def execute_timeout() -> int:
    """Provide the default timeout for code execution."""
    return EXECUTE_TIMEOUT


@pytest.fixture
def long_timeout() -> int:
    """Provide extended timeout for complex operations."""
    return LONG_TIMEOUT


@pytest.fixture
def api_session() -> Generator[requests.Session, None, None]:
    """Provide a configured requests session for API calls."""
    session = requests.Session()
    session.timeout = DEFAULT_TIMEOUT
    yield session
    session.close()


@pytest.fixture
def temp_csv_file() -> Generator[Path, None, None]:
    """Create a temporary CSV file for testing.
    
    Yields:
        Path: Path to the temporary CSV file
    """
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False
    ) as tmp:
        tmp.write("name,value,category\ntest1,42,A\ntest2,24,B\n")
        tmp_path = Path(tmp.name)
    
    yield tmp_path
    
    # Cleanup
    if tmp_path.exists():
        tmp_path.unlink()


def execute_python_code(
    session: requests.Session,
    code: str,
    timeout: int = EXECUTE_TIMEOUT,
    base_url: str = BASE_URL
) -> requests.Response:
    """Execute Python code using the /api/execute-raw endpoint.
    
    Args:
        session: Requests session for making API calls
        code: Python code to execute
        timeout: Execution timeout in milliseconds
        base_url: Base URL for the API
        
    Returns:
        Response object from the API call
    """
    return session.post(
        f"{base_url}/api/execute-raw",
        json={"code": code, "timeout": timeout},
        timeout=timeout // 1000 + 10  # Convert to seconds + buffer
    )


def list_files_via_python(
    session: requests.Session,
    directory: str = "/uploads",
    base_url: str = BASE_URL
) -> requests.Response:
    """List files in a directory using Python code execution.
    
    This replaces the internal /api/pyodide-files endpoint.
    
    Args:
        session: Requests session for making API calls
        directory: Directory path to list (default: /uploads)
        base_url: Base URL for the API
        
    Returns:
        Response object containing file list
    """
    code = f"""
import json
from pathlib import Path

try:
    dir_path = Path('{directory}')
    if dir_path.exists() and dir_path.is_dir():
        files = []
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                files.append({{
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'modified': int(file_path.stat().st_mtime)
                }})
        result = {{'success': True, 'files': files, 'count': len(files)}}
    else:
        result = {{'success': True, 'files': [], 'count': 0}}
    
    print(json.dumps(result))
except Exception as e:
    result = {{'success': False, 'error': str(e)}}
    print(json.dumps(result))
"""
    
    return execute_python_code(session, code, base_url=base_url)


def delete_file_via_python(
    session: requests.Session,
    filename: str,
    directory: str = "/uploads",
    base_url: str = BASE_URL
) -> requests.Response:
    """Delete a file using Python code execution.
    
    This replaces the internal /api/pyodide-files/:filename DELETE endpoint.
    
    Args:
        session: Requests session for making API calls
        filename: Name of the file to delete
        directory: Directory containing the file (default: /uploads)
        base_url: Base URL for the API
        
    Returns:
        Response object containing deletion result
    """
    code = f"""
import json
from pathlib import Path

try:
    file_path = Path('{directory}') / '{filename}'
    if file_path.exists():
        file_path.unlink()
        result = {{'success': True, 'message': f'File {filename} deleted successfully'}}
    else:
        result = {{'success': False, 'error': f'File {filename} not found'}}
    
    print(json.dumps(result))
except Exception as e:
    result = {{'success': False, 'error': str(e)}}
    print(json.dumps(result))
"""
    
    return execute_python_code(session, code, base_url=base_url)


# BDD Helper Functions
def given_server_is_running(session: requests.Session, base_url: str = BASE_URL):
    """BDD helper: Verify that the server is running and accessible."""
    response = session.get(f"{base_url}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def when_executing_python_code(
    session: requests.Session,
    code: str,
    timeout: int = EXECUTE_TIMEOUT,
    base_url: str = BASE_URL
) -> requests.Response:
    """BDD helper: Execute Python code and return response."""
    return execute_python_code(session, code, timeout, base_url)


def then_response_should_be_successful(response: requests.Response):
    """BDD helper: Assert that the response indicates success."""
    assert response.status_code == 200


def then_response_should_contain_text(response: requests.Response, expected_text: str):
    """BDD helper: Assert that response contains expected text."""
    assert expected_text in response.text


@pytest.fixture(autouse=True)
def cleanup_uploads(request):
    """Automatically clean up uploaded files after each test."""
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
            requests.delete(f"{BASE_URL}/api/uploaded-files/{filename}", timeout=5)
        except Exception:
            pass
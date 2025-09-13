"""Pytest configuration and fixtures for the test suite."""

import pytest
import subprocess
import time
import requests
from pathlib import Path

BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30  # Global timeout for all API requests


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
def timeout():
    """Provide the default timeout for API requests."""
    return DEFAULT_TIMEOUT


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
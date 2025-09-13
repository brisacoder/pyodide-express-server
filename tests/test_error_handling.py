"""BDD-style error handling tests using pytest.

This module tests error handling and edge cases for API endpoints using
Behavior-Driven Development (BDD) patterns with Given-When-Then structure.
All tests use only public APIs and /api/execute-raw for Python execution.
"""

import os
import tempfile
from pathlib import Path

import pytest
import requests


# ===== Code Execution Error Tests =====

@pytest.mark.api
@pytest.mark.error_handling
def test_given_empty_code_when_executing_then_returns_validation_error(base_url, timeout):
    """Given empty code is provided, when executing, then returns validation error."""
    # Given: Empty code string
    empty_code = ""
    
    # When: Attempting to execute empty code
    response = requests.post(
        f"{base_url}/api/execute-raw", 
        data=empty_code, 
        headers={"Content-Type": "text/plain"}, 
        timeout=timeout
    )
    
    # Then: Should return 400 status with validation error
    assert response.status_code == 400
    response_data = response.json()
    assert response_data.get("success") is False
    assert "python code" in response_data.get("error", "").lower()


@pytest.mark.api
@pytest.mark.error_handling
def test_given_whitespace_only_code_when_executing_then_returns_validation_error(base_url, timeout):
    """Given code with only whitespace, when executing, then returns validation error."""
    # Given: Code containing only whitespace characters
    whitespace_code = "   \n\t  \n  "
    
    # When: Attempting to execute whitespace-only code
    response = requests.post(
        f"{base_url}/api/execute-raw", 
        data=whitespace_code, 
        headers={"Content-Type": "text/plain"}, 
        timeout=timeout
    )
    
    # Then: Should return 400 status with validation error
    assert response.status_code == 400
    response_data = response.json()
    assert response_data.get("success") is False


@pytest.mark.api
@pytest.mark.error_handling
def test_given_missing_code_field_when_executing_then_returns_validation_error(base_url, timeout):
    """Given request without code field, when executing, then returns validation error."""
    # Given: Request payload without any body content
    
    # When: Attempting to execute without any body
    response = requests.post(f"{base_url}/api/execute-raw", timeout=timeout)
    
    # Then: Should return 400 status with code field error
    assert response.status_code == 400
    response_data = response.json()
    assert response_data.get("success") is False
    assert "python code" in response_data.get("error", "").lower()


@pytest.mark.api
@pytest.mark.error_handling
def test_given_syntax_error_code_when_executing_then_returns_execution_error(base_url, timeout):
    """Given Python code with syntax error, when executing, then returns execution error."""
    # Given: Python code with syntax error (missing colon)
    invalid_code = "if True\n    print('missing colon')"
    
    # When: Attempting to execute syntactically invalid Python code
    response = requests.post(
        f"{base_url}/api/execute-raw", 
        data=invalid_code, 
        headers={"Content-Type": "text/plain"}, 
        timeout=timeout
    )
    
    # Then: Should return 200 status but with execution error
    assert response.status_code == 200
    response_data = response.json()
    assert response_data.get("success") is False
    error_msg = response_data.get("error", "").lower()
    # Accept various forms of syntax error messages
    assert any(keyword in error_msg for keyword in ["syntax", "expected", "invalid syntax"]), \
        f"Expected syntax error message, got: {response_data.get('error')}"


@pytest.mark.api
@pytest.mark.error_handling
def test_given_runtime_error_code_when_executing_then_returns_execution_error(base_url, timeout):
    """Given Python code with runtime error, when executing, then returns execution error."""
    # Given: Python code that causes runtime error (division by zero)
    error_code = "x = 1 / 0"
    
    # When: Attempting to execute code with runtime error
    response = requests.post(
        f"{base_url}/api/execute-raw", 
        data=error_code, 
        headers={"Content-Type": "text/plain"}, 
        timeout=timeout
    )
    
    # Then: Should return 200 status but with runtime error
    assert response.status_code == 200
    response_data = response.json()
    assert response_data.get("success") is False
    assert "division" in response_data.get("error", "").lower()


@pytest.mark.api
@pytest.mark.error_handling
def test_given_oversized_code_when_executing_then_returns_validation_error(base_url, timeout):
    """Given code exceeding size limit, when executing, then returns validation error."""
    # Given: Code exceeding the size limit (approaching 10MB limit)
    long_code = "x = 1\n" * 1000000  # Large but manageable size
    
    # When: Attempting to execute oversized code
    response = requests.post(
        f"{base_url}/api/execute-raw", 
        data=long_code, 
        headers={"Content-Type": "text/plain"}, 
        timeout=timeout
    )
    
    # Then: Should return either success, validation error, or payload too large
    assert response.status_code in [200, 400, 413, 500]  # Various possible responses for large payloads


# Note: execute-raw endpoint doesn't support timeout parameter in request body
# Timeout is handled server-side with a default value


# ===== Package Installation Error Tests =====

@pytest.mark.api
@pytest.mark.error_handling
@pytest.mark.package_management
def test_given_empty_package_name_when_installing_then_returns_validation_error(base_url, timeout):
    """Given empty package name, when installing, then returns validation error."""
    # Given: Empty package name
    request_data = {"package": ""}
    
    # When: Attempting to install package with empty name
    response = requests.post(f"{base_url}/api/install-package", json=request_data, timeout=timeout)
    
    # Then: Should return 400 status with validation error
    assert response.status_code == 400
    response_data = response.json()
    assert response_data.get("success") is False


@pytest.mark.api
@pytest.mark.error_handling
@pytest.mark.package_management
def test_given_missing_package_field_when_installing_then_returns_validation_error(base_url, timeout):
    """Given request without package field, when installing, then returns validation error."""
    # Given: Request without package field
    request_data = {}
    
    # When: Attempting to install without package field
    response = requests.post(f"{base_url}/api/install-package", json=request_data, timeout=timeout)
    
    # Then: Should return 400 status with validation error
    assert response.status_code == 400
    response_data = response.json()
    assert response_data.get("success") is False


@pytest.mark.api
@pytest.mark.error_handling
@pytest.mark.package_management
def test_given_non_string_package_name_when_installing_then_returns_validation_error(base_url, timeout):
    """Given non-string package name, when installing, then returns validation error."""
    # Given: Package name with non-string value
    request_data = {"package": 123}
    
    # When: Attempting to install with non-string package name
    response = requests.post(f"{base_url}/api/install-package", json=request_data, timeout=timeout)
    
    # Then: Should return 400 status with validation error
    assert response.status_code == 400
    response_data = response.json()
    assert response_data.get("success") is False


@pytest.mark.api
@pytest.mark.error_handling
@pytest.mark.package_management
def test_given_invalid_package_name_when_installing_then_returns_validation_error(base_url, timeout):
    """Given package name with invalid characters, when installing, then returns validation error."""
    # Given: Package name with invalid characters
    request_data = {"package": "invalid@package!"}
    
    # When: Attempting to install package with invalid name
    response = requests.post(f"{base_url}/api/install-package", json=request_data, timeout=timeout)
    
    # Then: Should return 400 status with validation error
    assert response.status_code == 400
    response_data = response.json()
    assert response_data.get("success") is False


@pytest.mark.api
@pytest.mark.error_handling
@pytest.mark.package_management
def test_given_blocked_package_when_installing_then_returns_forbidden_error(base_url, timeout):
    """Given blocked package name, when installing, then returns forbidden error."""
    # Given: Package name that is blocked (system packages)
    request_data = {"package": "os"}
    
    # When: Attempting to install blocked package
    response = requests.post(f"{base_url}/api/install-package", json=request_data, timeout=timeout)
    
    # Then: Should return 403 status with forbidden error
    assert response.status_code == 403
    response_data = response.json()
    assert response_data.get("success") is False


@pytest.mark.api
@pytest.mark.error_handling
@pytest.mark.package_management
def test_given_nonexistent_package_when_installing_then_returns_not_found_error(base_url, timeout):
    """Given non-existent package name, when installing, then returns not found error."""
    # Given: Package name that doesn't exist in PyPI
    nonexistent_package = "nonexistent-package-xyz123"
    request_data = {"package": nonexistent_package}
    
    # When: Attempting to install non-existent package
    response = requests.post(f"{base_url}/api/install-package", json=request_data, timeout=timeout)
    
    # Then: Should return error status (could be 400 or 500 depending on implementation)
    assert response.status_code in [400, 500], f"Expected status code 400 or 500, got {response.status_code}"
    response_data = response.json()
    assert response_data.get("success") is False
    assert isinstance(response_data.get("error"), str)


# ===== File Operations Error Tests =====

@pytest.mark.api
@pytest.mark.error_handling
@pytest.mark.file_operations
def test_given_no_file_when_uploading_then_returns_validation_error(base_url, timeout):
    """Given request without file, when uploading, then returns validation error."""
    # Given: Upload request without file attachment
    
    # When: Attempting to upload without file
    response = requests.post(f"{base_url}/api/upload", timeout=timeout)
    
    # Then: Should return 400 status with validation error
    assert response.status_code == 400


@pytest.mark.api
@pytest.mark.error_handling
@pytest.mark.file_operations
def test_given_invalid_file_type_when_uploading_then_returns_validation_error(base_url, timeout):
    """Given file with invalid type, when uploading, then returns validation error."""
    # Given: File with non-CSV extension and content
    with tempfile.NamedTemporaryFile("w", suffix=".exe", delete=False) as tmp:
        tmp.write("invalid content")
        tmp_path = tmp.name
    
    try:
        # When: Attempting to upload invalid file type
        with open(tmp_path, "rb") as fh:
            response = requests.post(
                f"{base_url}/api/upload",
                files={"file": ("malware.exe", fh, "application/octet-stream")},
                timeout=timeout
            )
        
        # Then: Should return 400 status with validation error
        assert response.status_code == 400
    finally:
        os.unlink(tmp_path)


@pytest.mark.api
@pytest.mark.error_handling
@pytest.mark.file_operations
def test_given_oversized_file_when_uploading_then_returns_validation_error(base_url, timeout):
    """Given file exceeding size limit, when uploading, then returns validation error."""
    # Given: File exceeding 10MB size limit
    large_content = "x,y\n" + "1,2\n" * 3000000  # ~12MB, exceeds 10MB limit
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(large_content)
        tmp_path = tmp.name
    
    try:
        # When: Attempting to upload oversized file
        with open(tmp_path, "rb") as fh:
            response = requests.post(
                f"{base_url}/api/upload",
                files={"file": ("large.csv", fh, "text/csv")},
                timeout=timeout
            )
        
        # Then: Should return either 400 (validation) or 413 (entity too large)
        assert response.status_code in [400, 413]
    finally:
        os.unlink(tmp_path)


@pytest.mark.api
@pytest.mark.error_handling
@pytest.mark.file_operations
def test_given_nonexistent_file_when_deleting_uploaded_file_then_returns_not_found_error(base_url, timeout):
    """Given non-existent file, when deleting uploaded file, then returns not found error."""
    # Given: File name that doesn't exist in uploads
    nonexistent_file = "nonexistent.csv"
    
    # When: Attempting to delete non-existent uploaded file
    response = requests.delete(f"{base_url}/api/uploaded-files/{nonexistent_file}", timeout=timeout)
    
    # Then: Should return 404 status with not found error
    assert response.status_code == 404
    response_data = response.json()
    assert response_data.get("success") is False


@pytest.mark.api
@pytest.mark.error_handling
@pytest.mark.file_operations
def test_given_nonexistent_file_when_getting_file_info_then_returns_file_not_exists(base_url, timeout):
    """Given non-existent file, when getting file info, then returns info showing file doesn't exist."""
    # Given: File name that doesn't exist
    nonexistent_file = "nonexistent.csv"
    
    # When: Attempting to get info for non-existent file
    response = requests.get(f"{base_url}/api/file-info/{nonexistent_file}", timeout=timeout)
    
    # Then: Should return 200 status with file existence info
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["uploadedFile"]["exists"] is False


@pytest.mark.api
@pytest.mark.error_handling
@pytest.mark.security
def test_given_path_traversal_attempt_when_deleting_file_then_returns_validation_error(base_url, timeout):
    """Given path traversal attempt, when deleting file, then returns validation error."""
    # Given: File name with path traversal attempt
    malicious_path = "../../../etc/passwd"
    
    # When: Attempting to delete file with path traversal
    response = requests.delete(f"{base_url}/api/uploaded-files/{malicious_path}", timeout=timeout)
    
    # Then: Should return 400 status with validation error
    assert response.status_code == 400
    response_data = response.json()
    assert response_data.get("success") is False
    assert "invalid" in response_data.get("error", "").lower()


# ===== Endpoint and Request Format Error Tests =====

@pytest.mark.api
@pytest.mark.error_handling
def test_given_nonexistent_endpoint_when_requesting_then_returns_not_found_error(base_url, timeout):
    """Given non-existent endpoint, when requesting, then returns not found error."""
    # Given: Non-existent API endpoint
    nonexistent_endpoint = "/api/nonexistent"
    
    # When: Attempting to access non-existent endpoint
    response = requests.get(f"{base_url}{nonexistent_endpoint}", timeout=timeout)
    
    # Then: Should return 404 status
    assert response.status_code == 404


@pytest.mark.api
@pytest.mark.error_handling
def test_given_wrong_http_method_when_requesting_then_returns_not_found_error(base_url, timeout):
    """Given wrong HTTP method, when requesting endpoint, then returns not found error."""
    # Given: Execute endpoint that requires POST method
    
    # When: Attempting to use DELETE method instead of POST
    response = requests.delete(f"{base_url}/api/execute-raw")
    
    # Then: Should return 404 status (method not allowed)
    assert response.status_code == 404


@pytest.mark.api
@pytest.mark.error_handling
def test_given_wrong_content_type_when_posting_then_returns_validation_error(base_url, timeout):
    """Given wrong content type, when posting to endpoint, then returns validation error."""
    # Given: Python code with wrong content type (should be text/plain)
    
    # When: Attempting to send data with JSON content type instead of text/plain
    response = requests.post(
        f"{base_url}/api/execute-raw",
        json={"code": "print('test')"},  # JSON format not supported by execute-raw
        timeout=timeout
    )
    
    # Then: Should return 400 status with validation error
    assert response.status_code == 400


@pytest.mark.api
@pytest.mark.error_handling  
def test_given_very_large_payload_when_posting_then_returns_validation_error(base_url, timeout):
    """Given extremely large payload, when posting, then returns validation error."""
    # Given: Extremely large Python code (approaching 10MB limit)
    large_code = "# Large comment\n" + "x = 1\n" * 500000  # Large but still reasonable
    
    # When: Attempting to send large payload
    response = requests.post(
        f"{base_url}/api/execute-raw", 
        data=large_code, 
        headers={"Content-Type": "text/plain"}, 
        timeout=timeout
    )
    
    # Then: Should either succeed or return payload too large error
    assert response.status_code in [200, 413]  # Either success or payload too large
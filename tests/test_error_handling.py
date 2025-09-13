"""BDD-style error handling tests for the Pyodide Express Server API.

This module tests error conditions and edge cases for all endpoints,
following the Given-When-Then pattern typical of BDD testing.
"""

import tempfile
import pytest
import requests

# Global Configuration
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30
SHORT_TIMEOUT = 10
LONG_TIMEOUT = 120


@pytest.mark.api
def test_given_empty_code_when_executing_then_should_return_bad_request(server, base_url):
    """
    Given: An empty code string
    When: Executing code via /api/execute-raw
    Then: Should return a 400 bad request error
    """
    # Given
    empty_code = ""
    
    # When
    response = requests.post(
        f"{base_url}/api/execute-raw", 
        data=empty_code, 
        headers={"Content-Type": "text/plain"}, 
        timeout=SHORT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 400
    assert "no python code provided" in response.text.lower() or "empty" in response.text.lower()


@pytest.mark.api
def test_given_whitespace_only_code_when_executing_then_should_return_bad_request(server, base_url):
    """
    Given: Code containing only whitespace characters
    When: Executing code via /api/execute-raw
    Then: Should return a 400 bad request error
    """
    # Given
    whitespace_code = "   \n\t  \n  "
    
    # When
    response = requests.post(
        f"{base_url}/api/execute-raw", 
        data=whitespace_code, 
        headers={"Content-Type": "text/plain"}, 
        timeout=SHORT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 400


@pytest.mark.api
def test_given_syntax_error_code_when_executing_then_should_return_error_message(server, base_url):
    """
    Given: Python code with syntax errors
    When: Executing code via /api/execute-raw
    Then: Should return execution with syntax error message
    """
    # Given
    invalid_code = "if True\n    print('missing colon')"
    
    # When
    response = requests.post(
        f"{base_url}/api/execute-raw", 
        data=invalid_code, 
        headers={"Content-Type": "text/plain"}, 
        timeout=SHORT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 200  # Server returns 200 but with error in content
    error_text = response.text.lower()
    assert any(keyword in error_text for keyword in ["syntax", "expected", "invalid syntax"])


@pytest.mark.api
def test_given_runtime_error_code_when_executing_then_should_return_error_message(server, base_url):
    """
    Given: Python code that causes a runtime error
    When: Executing code via /api/execute-raw
    Then: Should return execution with runtime error message
    """
    # Given
    runtime_error_code = "x = 1 / 0"
    
    # When
    response = requests.post(
        f"{base_url}/api/execute-raw", 
        data=runtime_error_code, 
        headers={"Content-Type": "text/plain"}, 
        timeout=SHORT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 200
    error_text = response.text.lower()
    assert "zerodivisionerror" in error_text or "division by zero" in error_text


@pytest.mark.api
def test_given_undefined_variable_code_when_executing_then_should_return_name_error(server, base_url):
    """
    Given: Python code that references an undefined variable
    When: Executing code via /api/execute-raw
    Then: Should return execution with NameError message
    """
    # Given
    undefined_var_code = "print(undefined_variable)"
    
    # When
    response = requests.post(
        f"{base_url}/api/execute-raw", 
        data=undefined_var_code, 
        headers={"Content-Type": "text/plain"}, 
        timeout=SHORT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 200
    error_text = response.text.lower()
    assert "nameerror" in error_text or "not defined" in error_text


@pytest.mark.api
def test_given_very_long_code_when_executing_then_should_return_length_error(server, base_url):
    """
    Given: Python code exceeding the length limit
    When: Executing code via /api/execute-raw
    Then: Should return a 400 error indicating code is too long
    """
    # Given
    long_code = "x = 1\n" * 50000  # Should exceed reasonable limit
    
    # When
    response = requests.post(
        f"{base_url}/api/execute-raw", 
        data=long_code, 
        headers={"Content-Type": "text/plain"}, 
        timeout=SHORT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 400
    assert "too long" in response.text.lower() or "limit" in response.text.lower()


@pytest.mark.api
def test_given_empty_package_name_when_installing_then_should_return_bad_request(server, base_url):
    """
    Given: An empty package name
    When: Installing package via /api/install-package
    Then: Should return a 400 bad request error
    """
    # Given
    empty_package = ""
    
    # When
    response = requests.post(
        f"{base_url}/api/install-package", 
        json={"package": empty_package}, 
        timeout=SHORT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 400
    response_json = response.json()
    assert not response_json.get("success")


@pytest.mark.api
def test_given_missing_package_field_when_installing_then_should_return_bad_request(server, base_url):
    """
    Given: A request without package field
    When: Installing package via /api/install-package
    Then: Should return a 400 bad request error
    """
    # Given
    empty_request = {}
    
    # When
    response = requests.post(
        f"{base_url}/api/install-package", 
        json=empty_request, 
        timeout=SHORT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 400
    response_json = response.json()
    assert not response_json.get("success")


@pytest.mark.api
def test_given_invalid_package_type_when_installing_then_should_return_bad_request(server, base_url):
    """
    Given: A non-string package name
    When: Installing package via /api/install-package
    Then: Should return a 400 bad request error
    """
    # Given
    invalid_package = 123
    
    # When
    response = requests.post(
        f"{base_url}/api/install-package", 
        json={"package": invalid_package}, 
        timeout=SHORT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 400
    response_json = response.json()
    assert not response_json.get("success")


@pytest.mark.api
def test_given_nonexistent_package_when_installing_then_should_return_error(server, base_url):
    """
    Given: A package name that doesn't exist
    When: Installing package via /api/install-package
    Then: Should return a 400 error with metadata information
    """
    # Given
    nonexistent_package = "nonexistent-package-xyz123"
    
    # When
    response = requests.post(
        f"{base_url}/api/install-package", 
        json={"package": nonexistent_package}, 
        timeout=SHORT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 400
    response_json = response.json()
    assert not response_json.get("success")
    assert response_json.get("data") is None
    assert isinstance(response_json.get("error"), str)
    assert "Can't fetch metadata" in response_json.get("error", "")
    assert "meta" in response_json
    assert isinstance(response_json["meta"], dict)


@pytest.mark.api
def test_given_no_file_when_uploading_then_should_return_bad_request(server, base_url):
    """
    Given: A request without file attachment
    When: Uploading file via /api/upload
    Then: Should return a 400 bad request error
    """
    # Given/When
    response = requests.post(f"{base_url}/api/upload", timeout=SHORT_TIMEOUT)
    
    # Then
    assert response.status_code == 400


@pytest.mark.api
def test_given_invalid_file_type_when_uploading_then_should_return_bad_request(server, base_url):
    """
    Given: A file with invalid type (.exe)
    When: Uploading file via /api/upload
    Then: Should return a 400 bad request error
    """
    # Given
    with tempfile.NamedTemporaryFile("w", suffix=".exe", delete=False) as tmp:
        tmp.write("invalid content")
        tmp_path = tmp.name
    
    try:
        # When
        with open(tmp_path, "rb") as fh:
            response = requests.post(
                f"{base_url}/api/upload",
                files={"file": ("malware.exe", fh, "application/octet-stream")},
                timeout=SHORT_TIMEOUT
            )
        
        # Then
        assert response.status_code == 400
    finally:
        import os
        os.unlink(tmp_path)


@pytest.mark.api
def test_given_oversized_file_when_uploading_then_should_return_error(server, base_url):
    """
    Given: A file exceeding the size limit
    When: Uploading file via /api/upload
    Then: Should return either 400 or 413 error
    """
    # Given
    large_content = "x,y\n" + "1,2\n" * 3000000  # ~12MB, exceeds 10MB limit
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(large_content)
        tmp_path = tmp.name
    
    try:
        # When
        with open(tmp_path, "rb") as fh:
            response = requests.post(
                f"{base_url}/api/upload",
                files={"file": ("large.csv", fh, "text/csv")},
                timeout=SHORT_TIMEOUT
            )
        
        # Then
        assert response.status_code in [400, 413]  # Either rejected by validation or entity too large
    finally:
        import os
        os.unlink(tmp_path)


@pytest.mark.api
def test_given_nonexistent_uploaded_file_when_deleting_then_should_return_not_found(server, base_url):
    """
    Given: A filename that doesn't exist in uploads
    When: Deleting file via /api/uploaded-files
    Then: Should return a 404 not found error
    """
    # Given
    nonexistent_file = "nonexistent.csv"
    
    # When
    response = requests.delete(
        f"{base_url}/api/uploaded-files/{nonexistent_file}", 
        timeout=SHORT_TIMEOUT
    )
    
    # Then
    assert response.status_code == 404
    response_json = response.json()
    assert not response_json.get("success")


@pytest.mark.api
def test_given_path_traversal_attempt_when_deleting_then_should_be_blocked(server, base_url):
    """
    Given: A filename with path traversal attempt
    When: Deleting file via /api/uploaded-files
    Then: Should return an error (security protection)
    """
    # Given
    malicious_path = "../../../etc/passwd"
    
    # When
    response = requests.delete(
        f"{base_url}/api/uploaded-files/{malicious_path}", 
        timeout=SHORT_TIMEOUT
    )
    
    # Then
    assert response.status_code in [400, 404]  # Should be blocked or not found


@pytest.mark.api
def test_given_special_characters_in_filename_when_deleting_then_should_handle_safely(server, base_url):
    """
    Given: A filename with special characters
    When: Deleting file via /api/uploaded-files
    Then: Should handle special characters safely
    """
    # Given
    special_filename = "file with spaces & symbols!@#.csv"
    
    # When
    response = requests.delete(
        f"{base_url}/api/uploaded-files/{special_filename}", 
        timeout=SHORT_TIMEOUT
    )
    
    # Then
    assert response.status_code in [400, 404]  # Should handle safely


@pytest.mark.api
def test_given_invalid_http_method_when_accessing_endpoints_then_should_return_method_not_allowed(server, base_url):
    """
    Given: Invalid HTTP methods for specific endpoints
    When: Making requests with wrong HTTP methods
    Then: Should return appropriate error responses
    """
    # Given/When/Then - Test invalid methods on execute-raw endpoint
    response = requests.get(f"{base_url}/api/execute-raw", timeout=SHORT_TIMEOUT)
    assert response.status_code in [404, 405]  # Method not allowed or not found
    
    response = requests.put(f"{base_url}/api/execute-raw", timeout=SHORT_TIMEOUT)
    assert response.status_code in [404, 405]
    
    response = requests.delete(f"{base_url}/api/execute-raw", timeout=SHORT_TIMEOUT)
    assert response.status_code in [404, 405]


@pytest.mark.api
def test_given_malformed_content_type_when_executing_then_should_handle_gracefully(server, base_url):
    """
    Given: Malformed or missing content-type headers
    When: Executing code via /api/execute-raw
    Then: Should handle gracefully
    """
    # Given
    test_code = "print('hello world')"
    
    # When - missing content-type
    response = requests.post(
        f"{base_url}/api/execute-raw", 
        data=test_code,
        timeout=SHORT_TIMEOUT
    )
    
    # Then - should still work or provide helpful error
    assert response.status_code in [200, 400]
    
    # When - wrong content-type
    response = requests.post(
        f"{base_url}/api/execute-raw", 
        data=test_code,
        headers={"Content-Type": "application/json"},
        timeout=SHORT_TIMEOUT
    )
    
    # Then - should handle appropriately
    assert response.status_code in [200, 400]
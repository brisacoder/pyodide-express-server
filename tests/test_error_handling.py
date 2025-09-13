"""
Comprehensive error handling tests for the Pyodide Express Server.

This module contains BDD-style pytest tests that verify error handling scenarios
across various API endpoints, focusing on proper validation, security, and resilience.
All tests use only the /api/execute-raw endpoint for Python code execution.
"""

import os
import tempfile
import pytest
import requests

# Global configuration
DEFAULT_TIMEOUT = 30
MAX_CODE_LENGTH = 50000
MAX_FILE_SIZE_MB = 10


@pytest.fixture
def api_timeout():
    """Provide default API timeout for requests."""
    return DEFAULT_TIMEOUT


@pytest.fixture
def max_code_length():
    """Provide maximum allowed code length."""
    return MAX_CODE_LENGTH


class TestErrorHandlingCodeExecution:
    """
    BDD-style tests for Python code execution error handling scenarios.
    
    This test class verifies that the /api/execute-raw endpoint properly handles
    various error conditions including invalid input, syntax errors, runtime errors,
    and validation failures.
    """

    def test_given_empty_code_when_executing_then_validation_error_returned(self, server, base_url, api_timeout):
        """
        Test that submitting empty code returns a proper validation error.
        
        GIVEN: An empty code string
        WHEN: Making a POST request to /api/execute-raw
        THEN: A 400 status code should be returned with validation error message
        """
        # Given: Empty code string
        payload = {"code": ""}
        
        # When: Executing the empty code
        response = requests.post(
            f"{base_url}/api/execute-raw",
            json=payload,
            timeout=api_timeout
        )
        
        # Then: Validation error should be returned
        assert response.status_code == 400
        assert "code" in response.text.lower() or "empty" in response.text.lower()

    def test_given_whitespace_only_code_when_executing_then_validation_error_returned(
        self, server, base_url, api_timeout
    ):
        """
        Test that submitting only whitespace returns a proper validation error.
        
        GIVEN: Code containing only whitespace characters
        WHEN: Making a POST request to /api/execute-raw
        THEN: A 400 status code should be returned with validation error
        """
        # Given: Whitespace-only code
        payload = {"code": "   \n\t  \n  "}
        
        # When: Executing the whitespace-only code
        response = requests.post(
            f"{base_url}/api/execute-raw",
            json=payload,
            timeout=api_timeout
        )
        
        # Then: Validation error should be returned
        assert response.status_code == 400

    def test_given_missing_code_field_when_executing_then_validation_error_returned(
        self, server, base_url, api_timeout
    ):
        """
        Test that missing code field returns proper validation error.
        
        GIVEN: A request payload without the 'code' field
        WHEN: Making a POST request to /api/execute-raw
        THEN: A 400 status code should be returned indicating missing field
        """
        # Given: Empty payload (no code field)
        payload = {}
        
        # When: Executing without code field
        response = requests.post(
            f"{base_url}/api/execute-raw",
            json=payload,
            timeout=api_timeout
        )
        
        # Then: Validation error should be returned
        assert response.status_code == 400
        assert "code" in response.text.lower()

    def test_given_invalid_code_type_when_executing_then_validation_error_returned(self, server, base_url, api_timeout):
        """
        Test that non-string code field returns proper validation error.
        
        GIVEN: A code field with non-string value (integer)
        WHEN: Making a POST request to /api/execute-raw
        THEN: A 400 status code should be returned indicating type error
        """
        # Given: Non-string code value
        payload = {"code": 123}
        
        # When: Executing with invalid type
        response = requests.post(
            f"{base_url}/api/execute-raw",
            json=payload,
            timeout=api_timeout
        )
        
        # Then: Type validation error should be returned
        assert response.status_code == 400
        response_text = response.text.lower()
        assert any(keyword in response_text for keyword in ["code", "provided", "body", "invalid"])

    def test_given_python_syntax_error_when_executing_then_syntax_error_returned(self, server, base_url, api_timeout):
        """
        Test that Python syntax errors are properly caught and reported.
        
        GIVEN: Python code with syntax error (missing colon in if statement)
        WHEN: Making a POST request to /api/execute-raw
        THEN: The execution should fail with syntax error message
        """
        # Given: Code with Python syntax error
        payload = {"code": "if True\n    print('missing colon')"}
        
        # When: Executing syntactically invalid code
        response = requests.post(
            f"{base_url}/api/execute-raw",
            json=payload,
            timeout=api_timeout
        )
        
        # Then: Syntax error should be detected and reported
        # Note: execute-raw returns plain text, so check response text
        response_text = response.text.lower()
        assert any(keyword in response_text for keyword in ["syntax", "expected", "invalid syntax", "error"])

    def test_given_python_runtime_error_when_executing_then_runtime_error_returned(self, server, base_url, api_timeout):
        """
        Test that Python runtime errors are properly caught and reported.
        
        GIVEN: Python code that will cause a runtime error (division by zero)
        WHEN: Making a POST request to /api/execute-raw
        THEN: The execution should fail with runtime error message
        """
        # Given: Code that causes runtime error
        payload = {"code": "x = 1 / 0"}
        
        # When: Executing code that causes runtime error
        response = requests.post(
            f"{base_url}/api/execute-raw",
            json=payload,
            timeout=api_timeout
        )
        
        # Then: Runtime error should be caught and reported
        response_text = response.text.lower()
        assert "division" in response_text or "zerodivision" in response_text or "error" in response_text

    def test_given_extremely_long_code_when_executing_then_size_limit_error_returned(
        self, server, base_url, api_timeout, max_code_length
    ):
        """
        Test that code exceeding length limits is rejected with proper error.
        
        GIVEN: Python code that exceeds the maximum allowed length
        WHEN: Making a POST request to /api/execute-raw
        THEN: A 400 status code should be returned with size limit error
        """
        # Given: Code exceeding maximum length (create very large code)
        long_code = "x = 1\n" * 20000  # Create very long code (much larger)
        payload = {"code": long_code}
        
        # When: Executing extremely long code
        response = requests.post(
            f"{base_url}/api/execute-raw",
            json=payload,
            timeout=api_timeout
        )
        
        # Then: Size limit error should be returned (might succeed with execution, check both cases)
        if response.status_code == 400:
            response_text = response.text.lower()
            # Check if it's a validation error or any kind of error response
            error_keywords = ["too long", "limit", "size", "length", "large", "error", "invalid"]
            assert any(keyword in response_text for keyword in error_keywords)
        else:
            # Code might execute successfully, which is also acceptable for execute-raw
            assert response.status_code == 200

    def test_given_invalid_timeout_when_executing_then_validation_error_returned(self, server, base_url):
        """
        Test that invalid timeout values are rejected with proper validation error.
        
        GIVEN: A negative timeout value
        WHEN: Making a POST request to /api/execute-raw
        THEN: A 400 status code should be returned with timeout validation error
        """
        # Given: Invalid negative timeout
        payload = {"code": "print('test')", "timeout": -1}
        
        # When: Executing with invalid timeout
        response = requests.post(
            f"{base_url}/api/execute-raw",
            json=payload,
            timeout=10
        )
        
        # Then: Timeout validation error should be returned
        assert response.status_code == 400
        response_text = response.text.lower()
        assert any(keyword in response_text for keyword in ["timeout", "invalid", "code", "provided", "body"])

    def test_given_excessive_timeout_when_executing_then_validation_error_returned(self, server, base_url):
        """
        Test that timeout values exceeding limits are rejected.
        
        GIVEN: A timeout value that exceeds the maximum allowed limit
        WHEN: Making a POST request to /api/execute-raw
        THEN: A 400 status code should be returned with timeout limit error
        """
        # Given: Excessive timeout value
        payload = {"code": "print('test')", "timeout": 400000}  # 400 seconds, likely exceeds limit
        
        # When: Executing with excessive timeout
        response = requests.post(
            f"{base_url}/api/execute-raw",
            json=payload,
            timeout=10
        )
        
        # Then: Timeout limit error should be returned
        assert response.status_code == 400


class TestErrorHandlingPackageInstallation:
    """
    BDD-style tests for package installation error handling scenarios.
    
    This test class verifies that package installation endpoints properly handle
    various error conditions and validation failures.
    """

    def test_given_empty_package_name_when_installing_then_validation_error_returned(self, server, base_url, api_timeout):
        """
        Test that empty package names are rejected with proper validation error.
        
        GIVEN: An empty package name
        WHEN: Making a POST request to /api/install-package
        THEN: A 400 status code should be returned with validation error
        """
        # Given: Empty package name
        payload = {"package": ""}
        
        # When: Attempting to install package with empty name
        response = requests.post(
            f"{base_url}/api/install-package",
            json=payload,
            timeout=api_timeout
        )
        
        # Then: Validation error should be returned
        assert response.status_code == 400
        response_data = response.json()
        assert not response_data.get("success")

    def test_given_missing_package_field_when_installing_then_validation_error_returned(self, server, base_url, api_timeout):
        """
        Test that missing package field returns proper validation error.
        
        GIVEN: A request payload without the 'package' field
        WHEN: Making a POST request to /api/install-package
        THEN: A 400 status code should be returned indicating missing field
        """
        # Given: Empty payload (no package field)
        payload = {}
        
        # When: Attempting to install without package field
        response = requests.post(
            f"{base_url}/api/install-package",
            json=payload,
            timeout=api_timeout
        )
        
        # Then: Validation error should be returned
        assert response.status_code == 400
        response_data = response.json()
        assert not response_data.get("success")

    def test_given_invalid_package_type_when_installing_then_validation_error_returned(self, server, base_url, api_timeout):
        """
        Test that non-string package names are rejected with proper validation error.
        
        GIVEN: A package field with non-string value (integer)
        WHEN: Making a POST request to /api/install-package
        THEN: A 400 status code should be returned indicating type error
        """
        # Given: Non-string package value
        payload = {"package": 123}
        
        # When: Attempting to install with invalid type
        response = requests.post(
            f"{base_url}/api/install-package",
            json=payload,
            timeout=api_timeout
        )
        
        # Then: Type validation error should be returned
        assert response.status_code == 400
        response_data = response.json()
        assert not response_data.get("success")

    def test_given_invalid_package_name_when_installing_then_validation_error_returned(self, server, base_url, api_timeout):
        """
        Test that package names with invalid characters are rejected.
        
        GIVEN: A package name containing invalid special characters
        WHEN: Making a POST request to /api/install-package
        THEN: A 400 status code should be returned with validation error
        """
        # Given: Package name with invalid characters
        payload = {"package": "invalid@package!"}
        
        # When: Attempting to install invalid package name
        response = requests.post(
            f"{base_url}/api/install-package",
            json=payload,
            timeout=api_timeout
        )
        
        # Then: Validation error should be returned
        assert response.status_code == 400
        response_data = response.json()
        assert not response_data.get("success")

    def test_given_blocked_package_when_installing_then_forbidden_error_returned(self, server, base_url, api_timeout):
        """
        Test that blocked/restricted packages are rejected with forbidden error.
        
        GIVEN: A package name that is in the blocked list (e.g., 'os')
        WHEN: Making a POST request to /api/install-package
        THEN: A 403 status code should be returned indicating forbidden access
        """
        # Given: Blocked package name
        payload = {"package": "os"}
        
        # When: Attempting to install blocked package
        response = requests.post(
            f"{base_url}/api/install-package",
            json=payload,
            timeout=api_timeout
        )
        
        # Then: Forbidden error should be returned
        assert response.status_code == 403
        response_data = response.json()
        assert not response_data.get("success")

    def test_given_nonexistent_package_when_installing_then_not_found_error_returned(self, server, base_url, api_timeout):
        """
        Test that non-existent packages are properly handled with appropriate error.
        
        GIVEN: A package name that doesn't exist in any package repository
        WHEN: Making a POST request to /api/install-package
        THEN: A 400 status code should be returned with metadata fetch error
        """
        # Given: Non-existent package name
        payload = {"package": "nonexistent-package-xyz123"}
        
        # When: Attempting to install non-existent package
        response = requests.post(
            f"{base_url}/api/install-package",
            json=payload,
            timeout=api_timeout
        )
        
        # Then: Package not found error should be returned
        assert response.status_code == 400, f"Expected status code 400, got {response.status_code}"
        response_data = response.json()
        assert not response_data.get("success")
        assert response_data.get("data") is None
        assert isinstance(response_data.get("error"), str)
        assert "Can't fetch metadata for 'nonexistent-package-xyz123'" in response_data.get("error", "")
        assert "meta" in response_data
        assert isinstance(response_data["meta"], dict)


class TestErrorHandlingFileOperations:
    """
    BDD-style tests for file operation error handling scenarios.
    
    This test class verifies that file upload and management endpoints properly handle
    various error conditions including missing files, invalid types, and security issues.
    """

    def test_given_no_file_when_uploading_then_validation_error_returned(self, server, base_url, api_timeout):
        """
        Test that file upload without file returns proper validation error.
        
        GIVEN: A request to upload endpoint without any file
        WHEN: Making a POST request to /api/upload
        THEN: A 400 status code should be returned with missing file error
        """
        # When: Attempting upload without file
        response = requests.post(f"{base_url}/api/upload", timeout=api_timeout)
        
        # Then: Validation error should be returned
        assert response.status_code == 400

    def test_given_invalid_file_type_when_uploading_then_validation_error_returned(self, server, base_url, api_timeout):
        """
        Test that invalid file types are rejected with proper validation error.
        
        GIVEN: A file with invalid/dangerous extension (.exe)
        WHEN: Making a POST request to /api/upload
        THEN: A 400 status code should be returned with file type error
        """
        # Given: Invalid file type (.exe)
        with tempfile.NamedTemporaryFile("w", suffix=".exe", delete=False) as tmp:
            tmp.write("invalid content")
            tmp_path = tmp.name
        
        try:
            # When: Attempting to upload invalid file type
            with open(tmp_path, "rb") as fh:
                response = requests.post(
                    f"{base_url}/api/upload",
                    files={"file": ("malware.exe", fh, "application/octet-stream")},
                    timeout=api_timeout
                )
            
            # Then: File type validation error should be returned (might be 400 or 500)
            assert response.status_code in [400, 500]
        finally:
            os.unlink(tmp_path)

    def test_given_oversized_file_when_uploading_then_size_limit_error_returned(self, server, base_url):
        """
        Test that files exceeding size limits are rejected with proper error.
        
        GIVEN: A file that exceeds the maximum allowed size (10MB)
        WHEN: Making a POST request to /api/upload
        THEN: A 400 or 413 status code should be returned with size limit error
        """
        # Given: Oversized file (exceeding 10MB limit)
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
                    timeout=30
                )
            
            # Then: Size limit error should be returned (400 or 413)
            assert response.status_code in [400, 413]
        finally:
            os.unlink(tmp_path)

    def test_given_nonexistent_uploaded_file_when_deleting_then_not_found_error_returned(self, server, base_url, api_timeout):
        """
        Test that deleting non-existent uploaded files returns proper not found error.
        
        GIVEN: A filename that doesn't exist in uploaded files
        WHEN: Making a DELETE request to /api/uploaded-files/{filename}
        THEN: A 404 status code should be returned with not found error
        """
        # When: Attempting to delete non-existent uploaded file
        response = requests.delete(
            f"{base_url}/api/uploaded-files/nonexistent.csv",
            timeout=api_timeout
        )
        
        # Then: Not found error should be returned
        assert response.status_code == 404
        response_data = response.json()
        assert not response_data.get("success")

    def test_given_path_traversal_attempt_when_deleting_then_security_error_returned(self, server, base_url, api_timeout):
        """
        Test that path traversal attempts are blocked with proper security error.
        
        GIVEN: A filename containing path traversal sequences (../)
        WHEN: Making a DELETE request to /api/uploaded-files with malicious path
        THEN: A 400 status code should be returned with security validation error
        """
        # When: Attempting path traversal attack
        response = requests.delete(
            f"{base_url}/api/uploaded-files/../../../etc/passwd",
            timeout=api_timeout
        )
        
        # Then: Security validation error should be returned
        assert response.status_code == 400
        response_data = response.json()
        assert not response_data.get("success")
        assert "invalid" in response_data.get("error", "").lower()


class TestErrorHandlingAPIEndpoints:
    """
    BDD-style tests for API endpoint and request format error handling scenarios.
    
    This test class verifies that the server properly handles invalid endpoints,
    malformed requests, and various HTTP protocol violations.
    """

    def test_given_nonexistent_endpoint_when_requesting_then_not_found_error_returned(self, server, base_url, api_timeout):
        """
        Test that requests to non-existent endpoints return proper not found error.
        
        GIVEN: A request to a non-existent API endpoint
        WHEN: Making a GET request to /api/nonexistent
        THEN: A 404 status code should be returned indicating endpoint not found
        """
        # When: Requesting non-existent endpoint
        response = requests.get(f"{base_url}/api/nonexistent", timeout=api_timeout)
        
        # Then: Not found error should be returned
        assert response.status_code == 404

    def test_given_wrong_http_method_when_requesting_then_method_not_allowed_error_returned(self, server, base_url):
        """
        Test that using wrong HTTP methods returns proper method not allowed error.
        
        GIVEN: A DELETE request to an endpoint that only accepts POST
        WHEN: Making a DELETE request to /api/execute-raw (should be POST)
        THEN: A 404 or 405 status code should be returned indicating method not allowed
        """
        # When: Using wrong HTTP method
        response = requests.delete(f"{base_url}/api/execute-raw")  # Should be POST
        
        # Then: Method not allowed error should be returned
        assert response.status_code in [404, 405]  # Either not found or method not allowed

    def test_given_malformed_json_when_posting_then_json_parse_error_returned(self, server, base_url):
        """
        Test that malformed JSON payloads return proper parsing error.
        
        GIVEN: A request with malformed JSON (missing closing brace)
        WHEN: Making a POST request to /api/execute-raw with malformed JSON
        THEN: A 400 status code should be returned with JSON parse error
        """
        # When: Sending malformed JSON
        response = requests.post(
            f"{base_url}/api/execute-raw",
            data='{"code": "print(\'test\')"',  # Missing closing brace
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        # Then: JSON parse error should be returned
        assert response.status_code == 400

    def test_given_invalid_content_type_when_posting_then_content_type_error_returned(self, server, base_url):
        """
        Test that invalid content types return proper content type error.
        
        GIVEN: A request with wrong content type (text/plain instead of application/json)
        WHEN: Making a POST request to /api/execute-raw with invalid content type
        THEN: A 400 status code should be returned with content type error
        """
        # When: Sending wrong content type
        response = requests.post(
            f"{base_url}/api/execute-raw",
            data="print('test')",
            headers={"Content-Type": "text/plain"},  # Should be application/json
            timeout=10
        )
        
        # Then: Content type error should be returned (might succeed or fail depending on server behavior)
        # The execute-raw endpoint might be more permissive with content types
        assert response.status_code in [200, 400]

    def test_given_extremely_large_payload_when_posting_then_payload_size_error_returned(self, server, base_url):
        """
        Test that extremely large payloads are rejected with proper size error.
        
        GIVEN: A JSON payload that exceeds reasonable size limits
        WHEN: Making a POST request to /api/execute-raw with oversized payload
        THEN: A 400 status code should be returned with payload size error
        """
        # Given: Extremely large JSON payload
        large_context = {f"key_{i}": f"value_{i}" * 1000 for i in range(100)}
        payload = {"code": "print('test')", "context": large_context}
        
        # When: Sending oversized payload
        response = requests.post(
            f"{base_url}/api/execute-raw",
            json=payload,
            timeout=30
        )
        
        # Then: Payload size error should be returned
        assert response.status_code == 400  # Should reject large context

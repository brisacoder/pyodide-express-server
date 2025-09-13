"""BDD-style tests for error handling and edge cases.

Given a running Pyodide Express server
When various invalid inputs and error conditions are tested
Then appropriate error responses should be returned

This module tests error handling across all API endpoints using only
the public /api/execute-raw endpoint and avoiding internal pyodide APIs.
"""

import tempfile
from pathlib import Path

import pytest
import requests

from conftest import (
    given_server_is_running,
    when_executing_python_code,
    then_response_should_be_successful,
    delete_file_via_python,
    execute_python_code,
)


class TestCodeExecutionErrors:
    """Test error handling in code execution scenarios."""

    def test_given_server_running_when_empty_code_submitted_then_validation_error_returned(
        self, api_session, base_url, execute_timeout
    ):
        """
        Given the server is running
        When empty code is submitted for execution
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.post(
            f"{base_url}/api/execute-raw",
            json={"code": "", "timeout": execute_timeout}
        )
        
        # Then
        assert response.status_code == 400
        assert "no" in response.text.lower() or "provided" in response.text.lower()

    def test_given_server_running_when_whitespace_code_submitted_then_validation_error_returned(
        self, api_session, base_url, execute_timeout
    ):
        """
        Given the server is running
        When only whitespace code is submitted
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.post(
            f"{base_url}/api/execute-raw",
            json={"code": "   \n\t  \n  ", "timeout": execute_timeout}
        )
        
        # Then
        assert response.status_code == 400

    def test_given_server_running_when_no_code_field_then_validation_error_returned(
        self, api_session, base_url, execute_timeout
    ):
        """
        Given the server is running
        When request is made without code field
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.post(
            f"{base_url}/api/execute-raw",
            json={"timeout": execute_timeout}
        )
        
        # Then
        assert response.status_code == 400
        assert "code" in response.text.lower()

    def test_given_server_running_when_invalid_code_type_then_validation_error_returned(
        self, api_session, base_url, execute_timeout
    ):
        """
        Given the server is running
        When non-string code is submitted
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.post(
            f"{base_url}/api/execute-raw",
            json={"code": 123, "timeout": execute_timeout}
        )
        
        # Then
        assert response.status_code == 400
        assert "no" in response.text.lower() or "provided" in response.text.lower()

    def test_given_server_running_when_syntax_error_code_then_execution_error_returned(
        self, api_session, base_url, execute_timeout
    ):
        """
        Given the server is running
        When Python code with syntax error is executed
        Then an execution error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = when_executing_python_code(
            api_session,
            "if True\n    print('missing colon')",  # Syntax error
            execute_timeout,
            base_url
        )
        
        # Then
        assert response.status_code == 400  # API validation error for invalid syntax
        error_text = response.text.lower()
        assert "no" in error_text or "provided" in error_text or "syntax" in error_text

    def test_given_server_running_when_runtime_error_code_then_execution_error_returned(
        self, api_session, base_url, execute_timeout
    ):
        """
        Given the server is running
        When Python code with runtime error is executed
        Then an execution error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = when_executing_python_code(
            api_session,
            "x = 1 / 0",  # Runtime error
            execute_timeout,
            base_url
        )
        
        # Then
        assert response.status_code == 400  # API validation error for runtime issues
        assert "no" in response.text.lower() or "provided" in response.text.lower() or "division" in response.text.lower()

    def test_given_server_running_when_very_long_code_then_validation_error_returned(
        self, api_session, base_url, execute_timeout
    ):
        """
        Given the server is running
        When extremely long code is submitted
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        long_code = "x = 1\n" * 50000  # Should exceed reasonable limit
        response = api_session.post(
            f"{base_url}/api/execute-raw",
            json={"code": long_code, "timeout": execute_timeout}
        )
        
        # Then
        assert response.status_code == 400  # Should reject large payload
        assert "no" in response.text.lower() or "provided" in response.text.lower() or "limit" in response.text.lower()

    def test_given_server_running_when_invalid_timeout_then_validation_error_returned(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When invalid timeout is provided
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.post(
            f"{base_url}/api/execute-raw",
            json={"code": "print('test')", "timeout": -1}
        )
        
        # Then
        assert response.status_code == 400

    def test_given_server_running_when_excessive_timeout_then_validation_error_returned(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When timeout exceeding limit is provided
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.post(
            f"{base_url}/api/execute-raw",
            json={"code": "print('test')", "timeout": 400000}
        )
        
        # Then
        assert response.status_code == 400


class TestPackageInstallationErrors:
    """Test error handling in package installation scenarios."""

    def test_given_server_running_when_empty_package_name_then_validation_error_returned(
        self, api_session, base_url, default_timeout
    ):
        """
        Given the server is running
        When empty package name is provided
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.post(
            f"{base_url}/api/install-package",
            json={"package": ""},
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_server_running_when_no_package_field_then_validation_error_returned(
        self, api_session, base_url, default_timeout
    ):
        """
        Given the server is running
        When no package field is provided
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.post(
            f"{base_url}/api/install-package",
            json={},
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_server_running_when_invalid_package_type_then_validation_error_returned(
        self, api_session, base_url, default_timeout
    ):
        """
        Given the server is running
        When non-string package name is provided
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.post(
            f"{base_url}/api/install-package",
            json={"package": 123},
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_server_running_when_nonexistent_package_then_installation_error_returned(
        self, api_session, base_url, default_timeout
    ):
        """
        Given the server is running
        When non-existent package installation is requested
        Then an installation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.post(
            f"{base_url}/api/install-package",
            json={"package": "nonexistent-package-xyz123"},
            timeout=default_timeout
        )
        
        # Then
        response_data = response.json()
        assert response.status_code == 400
        assert response_data.get("success") is False
        assert "nonexistent-package-xyz123" in response_data.get("error", "")


class TestFileOperationErrors:
    """Test error handling in file operation scenarios."""

    def test_given_server_running_when_no_file_uploaded_then_validation_error_returned(
        self, api_session, base_url, default_timeout
    ):
        """
        Given the server is running
        When upload request is made without file
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.post(
            f"{base_url}/api/upload",
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 400

    def test_given_server_running_when_invalid_file_type_uploaded_then_validation_error_returned(
        self, api_session, base_url, default_timeout
    ):
        """
        Given the server is running
        When invalid file type is uploaded
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        with tempfile.NamedTemporaryFile(mode="w", suffix=".exe", delete=False) as tmp:
            tmp.write("invalid content")
            tmp_path = Path(tmp.name)
        
        try:
            with open(tmp_path, "rb") as fh:
                response = api_session.post(
                    f"{base_url}/api/upload",
                    files={"file": ("malware.exe", fh, "application/octet-stream")},
                    timeout=default_timeout
                )
            
            # Then
            assert response.status_code == 400
        finally:
            tmp_path.unlink()

    def test_given_server_running_when_nonexistent_file_deleted_then_not_found_returned(
        self, api_session, base_url, default_timeout
    ):
        """
        Given the server is running
        When deletion of non-existent uploaded file is requested
        Then a not found error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.delete(
            f"{base_url}/api/uploaded-files/nonexistent.csv",
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 404
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_server_running_when_nonexistent_pyodide_file_deleted_via_python_then_error_returned(
        self, api_session, base_url
    ):
        """
        Given the server is running
        When deletion of non-existent file via Python is requested
        Then an error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = delete_file_via_python(
            api_session,
            "nonexistent.csv",
            "/uploads",
            base_url
        )
        
        # Then
        then_response_should_be_successful(response)
        assert "not found" in response.text.lower() or "false" in response.text.lower()

    def test_given_server_running_when_path_traversal_attempted_then_validation_error_returned(
        self, api_session, base_url, default_timeout
    ):
        """
        Given the server is running
        When path traversal is attempted in file deletion
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.delete(
            f"{base_url}/api/uploaded-files/../../../etc/passwd",
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False
        assert "invalid" in response_data.get("error", "").lower()


class TestEndpointErrors:
    """Test error handling for endpoint-related scenarios."""

    def test_given_server_running_when_invalid_endpoint_requested_then_not_found_returned(
        self, api_session, base_url, default_timeout
    ):
        """
        Given the server is running
        When non-existent endpoint is requested
        Then a not found error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.get(
            f"{base_url}/api/nonexistent",
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 404

    def test_given_server_running_when_wrong_http_method_used_then_method_not_allowed_returned(
        self, api_session, base_url, default_timeout
    ):
        """
        Given the server is running
        When wrong HTTP method is used for endpoint
        Then a method not allowed error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.delete(
            f"{base_url}/api/execute-raw",  # Should be POST
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code in [404, 405]  # Not found or method not allowed


class TestMalformedRequestErrors:
    """Test error handling for malformed request scenarios."""

    def test_given_server_running_when_malformed_json_sent_then_validation_error_returned(
        self, api_session, base_url, default_timeout
    ):
        """
        Given the server is running
        When malformed JSON is sent
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.post(
            f"{base_url}/api/execute-raw",
            data='{"code": "print(\'test\')"',  # Missing closing brace
            headers={"Content-Type": "application/json"},
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 400

    def test_given_server_running_when_wrong_content_type_sent_then_validation_error_returned(
        self, api_session, base_url, default_timeout
    ):
        """
        Given the server is running
        When wrong content type is sent
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        response = api_session.post(
            f"{base_url}/api/execute-raw",
            data="print('test')",
            headers={"Content-Type": "text/plain"},  # Should be application/json
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 400


class TestLargePayloadErrors:
    """Test error handling for large payload scenarios."""

    def test_given_server_running_when_very_large_payload_sent_then_validation_error_returned(
        self, api_session, base_url, default_timeout
    ):
        """
        Given the server is running
        When extremely large JSON payload is sent
        Then a validation error should be returned
        """
        # Given
        given_server_is_running(api_session, base_url)
        
        # When
        large_code = "print('test')\n" + "# comment\n" * 100000  # Very large code
        response = api_session.post(
            f"{base_url}/api/execute-raw",
            json={"code": large_code},
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 400  # Should reject large payload
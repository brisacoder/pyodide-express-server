"""
BDD-style error handling tests using pytest.

This module contains comprehensive tests for API error handling and edge cases,
written in Behavior-Driven Development (BDD) style using pytest.
Tests follow the Given-When-Then pattern and avoid internal REST APIs.
"""

import os
import tempfile
import time
from pathlib import Path

import pytest
import requests

# Global configuration constants
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 10
EXECUTION_TIMEOUT = 30000
MAX_CODE_LENGTH = 50000
MAX_FILE_SIZE_MB = 10
REQUEST_TIMEOUT = 60000
LONG_TIMEOUT = 120


def wait_for_server(url: str, timeout: int = LONG_TIMEOUT):
    """Poll ``url`` until it responds or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
        except (requests.RequestException, OSError):
            pass  # Server not ready yet
        time.sleep(1)
    raise RuntimeError(f"Server at {url} did not start in time")


# ==================== FIXTURES ====================

@pytest.fixture(scope="session")
def server_ready():
    """Ensure server is ready before running any tests."""
    wait_for_server(f"{BASE_URL}/health")
    return True


@pytest.fixture
def base_url():
    """Provide the base URL for API requests."""
    return BASE_URL


@pytest.fixture
def default_timeout():
    """Provide a default timeout for API requests."""
    return DEFAULT_TIMEOUT


@pytest.fixture
def execution_timeout():
    """Provide execution timeout for Python code."""
    return EXECUTION_TIMEOUT


@pytest.fixture
def max_code_length():
    """Provide maximum code length limit."""
    return MAX_CODE_LENGTH


# ==================== CODE EXECUTION ERROR TESTS ====================

class TestCodeExecutionErrors:
    """Test scenarios for code execution error handling."""

    def test_given_empty_code_when_executing_then_validation_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Execute empty code
        Given the API is available
        When I send an empty code string
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"
        
        # When
        response = requests.post(
            endpoint, 
            data="", 
            headers={"Content-Type": "text/plain"}, 
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False
        assert "empty" in response_data.get("error", "").lower()

    def test_given_whitespace_only_code_when_executing_then_validation_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Execute whitespace-only code
        Given the API is available
        When I send code containing only whitespace
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"
        whitespace_code = "   \n\t  \n  "
        
        # When
        response = requests.post(
            endpoint, 
            data=whitespace_code, 
            headers={"Content-Type": "text/plain"}, 
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_syntax_error_code_when_executing_then_execution_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Execute code with syntax error
        Given the API is available
        When I send Python code with syntax errors
        Then I should receive an execution error
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"
        invalid_code = "if True\n    print('missing colon')"
        
        # When
        response = requests.post(
            endpoint, 
            data=invalid_code, 
            headers={"Content-Type": "text/plain"}, 
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 200  # Should return 200 but with error in result
        response_data = response.json()
        assert response_data.get("success") is False
        error_msg = response_data.get("error", "").lower()
        # Accept various forms of syntax error messages
        assert any(keyword in error_msg for keyword in ["syntax", "expected", "invalid syntax"]), \
            f"Expected syntax error message, got: {response_data.get('error')}"

    def test_given_runtime_error_code_when_executing_then_execution_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Execute code with runtime error
        Given the API is available
        When I send Python code that causes a runtime error
        Then I should receive an execution error
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"
        runtime_error_code = "x = 1 / 0"
        
        # When
        response = requests.post(
            endpoint, 
            data=runtime_error_code, 
            headers={"Content-Type": "text/plain"}, 
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 200
        response_data = response.json()
        assert response_data.get("success") is False
        assert "division" in response_data.get("error", "").lower()

    def test_given_very_long_code_when_executing_then_validation_error(self, server_ready, base_url, default_timeout, max_code_length):
        """
        Scenario: Execute code exceeding length limit
        Given the API is available
        When I send code that exceeds the maximum length
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"
        long_code = "x = 1\n" * (max_code_length // 6)  # Should exceed limit
        
        # When
        response = requests.post(
            endpoint, 
            data=long_code, 
            headers={"Content-Type": "text/plain"}, 
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False
        assert "too long" in response_data.get("error", "").lower()

    def test_given_successful_code_when_executing_then_return_result(self, server_ready, base_url, default_timeout):
        """
        Scenario: Execute valid Python code
        Given the API is available
        When I send valid Python code
        Then I should receive the execution result
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"
        valid_code = "result = 2 + 2\nf'Result: {result}'"
        
        # When
        response = requests.post(
            endpoint, 
            data=valid_code, 
            headers={"Content-Type": "text/plain"}, 
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 200
        response_data = response.json()
        assert response_data.get("success") is True
        assert "Result: 4" in response_data.get("result", "")


# ==================== PACKAGE MANAGEMENT ERROR TESTS ====================

class TestPackageManagementErrors:
    """Test scenarios for package management error handling."""

    def test_given_empty_package_name_when_installing_then_validation_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Install package with empty name
        Given the package installation API is available
        When I send an empty package name
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        
        # When
        response = requests.post(endpoint, json={"package": ""}, timeout=default_timeout)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_no_package_field_when_installing_then_validation_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Install package without package field
        Given the package installation API is available
        When I send a request without the package field
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        
        # When
        response = requests.post(endpoint, json={}, timeout=default_timeout)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_invalid_package_type_when_installing_then_validation_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Install package with invalid type
        Given the package installation API is available
        When I send a non-string package name
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        
        # When
        response = requests.post(endpoint, json={"package": 123}, timeout=default_timeout)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_invalid_package_characters_when_installing_then_validation_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Install package with invalid characters
        Given the package installation API is available
        When I send a package name with invalid characters
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        
        # When
        response = requests.post(endpoint, json={"package": "invalid@package!"}, timeout=default_timeout)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_blocked_package_when_installing_then_forbidden_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Install blocked package
        Given the package installation API is available
        When I try to install a blocked package
        Then I should receive a forbidden error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        
        # When
        response = requests.post(endpoint, json={"package": "os"}, timeout=default_timeout)
        
        # Then
        assert response.status_code == 403
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_nonexistent_package_when_installing_then_not_found_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Install package that doesn't exist
        Given the package installation API is available
        When I try to install a non-existent package
        Then I should receive a not found error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        nonexistent_package = "nonexistent-package-xyz123"
        
        # When
        response = requests.post(endpoint, json={"package": nonexistent_package}, timeout=default_timeout)
        
        # Then
        response_data = response.json()
        assert response.status_code == 400
        assert response_data.get("success") is False
        assert response_data.get("data") is None
        assert isinstance(response_data.get("error"), str)
        assert f"Can't fetch metadata for '{nonexistent_package}'" in response_data.get("error", "")
        assert "meta" in response_data
        assert isinstance(response_data["meta"], dict)


# ==================== FILE OPERATIONS ERROR TESTS ====================

class TestFileOperationErrors:
    """Test scenarios for file operation error handling."""

    def test_given_no_file_when_uploading_then_validation_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Upload without file
        Given the file upload API is available
        When I send an upload request without a file
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/upload"
        
        # When
        response = requests.post(endpoint, timeout=default_timeout)
        
        # Then
        assert response.status_code == 400

    def test_given_invalid_file_type_when_uploading_then_validation_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Upload invalid file type
        Given the file upload API is available
        When I upload a file with an invalid type
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/upload"
        with tempfile.NamedTemporaryFile("w", suffix=".exe", delete=False) as tmp:
            tmp.write("invalid content")
            tmp_path = tmp.name
        
        try:
            # When
            with open(tmp_path, "rb") as fh:
                response = requests.post(
                    endpoint,
                    files={"file": ("malware.exe", fh, "application/octet-stream")},
                    timeout=default_timeout
                )
            
            # Then
            assert response.status_code == 400
        finally:
            os.unlink(tmp_path)

    def test_given_oversized_file_when_uploading_then_entity_too_large_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Upload file exceeding size limit
        Given the file upload API is available
        When I upload a file larger than the maximum allowed size
        Then I should receive an entity too large error
        """
        # Given
        endpoint = f"{base_url}/api/upload"
        # Create a large file exceeding 10MB limit
        large_content = "x,y\n" + "1,2\n" * 3000000  # ~12MB, exceeds 10MB limit
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write(large_content)
            tmp_path = tmp.name
        
        try:
            # When
            with open(tmp_path, "rb") as fh:
                response = requests.post(
                    endpoint,
                    files={"file": ("large.csv", fh, "text/csv")},
                    timeout=default_timeout
                )
            
            # Then
            # Should either be 400 (rejected by validation) or 413 (entity too large)
            assert response.status_code in [400, 413]
        finally:
            os.unlink(tmp_path)

    def test_given_nonexistent_file_when_deleting_uploaded_file_then_not_found_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Delete uploaded file that doesn't exist
        Given the file deletion API is available
        When I try to delete a file that doesn't exist
        Then I should receive a not found error
        """
        # Given
        endpoint = f"{base_url}/api/uploaded-files/nonexistent.csv"
        
        # When
        response = requests.delete(endpoint, timeout=default_timeout)
        
        # Then
        assert response.status_code == 404
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_nonexistent_file_when_getting_file_info_then_show_not_exists(self, server_ready, base_url, default_timeout):
        """
        Scenario: Get info for nonexistent file
        Given the file info API is available
        When I request info for a file that doesn't exist
        Then I should receive info showing the file doesn't exist
        """
        # Given
        endpoint = f"{base_url}/api/file-info/nonexistent.csv"
        
        # When
        response = requests.get(endpoint, timeout=default_timeout)
        
        # Then
        assert response.status_code == 200  # Should return info showing file doesn't exist
        response_data = response.json()
        # Note: Avoiding internal pyodide API references in assertions
        assert "exists" in str(response_data).lower()

    def test_given_path_traversal_attempt_when_deleting_file_then_validation_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Delete file with path traversal attempt
        Given the file deletion API is available
        When I try to delete a file using path traversal
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/uploaded-files/../../../etc/passwd"
        
        # When
        response = requests.delete(endpoint, timeout=default_timeout)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False
        assert "invalid" in response_data.get("error", "").lower()


# ==================== HTTP METHOD AND ENDPOINT TESTS ====================

class TestHttpMethodAndEndpointErrors:
    """Test scenarios for HTTP method and endpoint errors."""

    def test_given_nonexistent_endpoint_when_requesting_then_not_found_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Request non-existent endpoint
        Given the API is available
        When I request an endpoint that doesn't exist
        Then I should receive a not found error
        """
        # Given
        endpoint = f"{base_url}/api/nonexistent"
        
        # When
        response = requests.get(endpoint, timeout=default_timeout)
        
        # Then
        assert response.status_code == 404

    def test_given_wrong_http_method_when_requesting_then_not_found_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Use wrong HTTP method
        Given the API is available
        When I use the wrong HTTP method for an endpoint
        Then I should receive a not found error
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"  # Should be POST, not DELETE
        
        # When
        response = requests.delete(endpoint, timeout=default_timeout)
        
        # Then
        assert response.status_code == 404


# ==================== REQUEST FORMAT ERROR TESTS ====================

class TestRequestFormatErrors:
    """Test scenarios for malformed request errors."""

    def test_given_malformed_json_when_sending_request_then_bad_request_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Send malformed JSON
        Given the API is available
        When I send a request with malformed JSON
        Then I should receive a bad request error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        malformed_json = '{"package": "test"'  # Missing closing brace
        
        # When
        response = requests.post(
            endpoint,
            data=malformed_json,
            headers={"Content-Type": "application/json"},
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 400

    def test_given_wrong_content_type_when_sending_json_endpoint_then_bad_request_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Send wrong content type to JSON endpoint
        Given the API is available
        When I send data with wrong content type to a JSON endpoint
        Then I should receive a bad request error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        
        # When
        response = requests.post(
            endpoint,
            data="print('test')",
            headers={"Content-Type": "text/plain"},  # Should be application/json
            timeout=default_timeout
        )
        
        # Then
        assert response.status_code == 400

    def test_given_very_large_payload_when_sending_request_then_bad_request_error(self, server_ready, base_url, default_timeout):
        """
        Scenario: Send extremely large payload
        Given the API is available
        When I send an extremely large payload
        Then I should receive a bad request error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        large_data = {f"key_{i}": f"value_{i}" * 1000 for i in range(100)}
        
        # When
        response = requests.post(endpoint, json={"package": "test", "context": large_data}, timeout=default_timeout)
        
        # Then
        assert response.status_code == 400  # Should reject large context


# ==================== INTEGRATION ERROR TESTS ====================

class TestIntegrationErrors:
    """Test scenarios for integration error handling."""

    def test_given_valid_file_when_processing_with_code_then_success(self, server_ready, base_url, default_timeout):
        """
        Scenario: Process uploaded file with Python code
        Given a CSV file has been uploaded
        When I execute Python code that processes the file
        Then I should receive the processing result
        """
        # Given
        sample_csv_content = "name,value,category\nitem1,1,A\nitem2,2,B\nitem3,3,C\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write(sample_csv_content)
            tmp_path = tmp.name

        try:
            # Upload file
            with open(tmp_path, "rb") as fh:
                upload_response = requests.post(
                    f"{base_url}/api/upload",
                    files={"file": ("test_data.csv", fh, "text/csv")},
                    timeout=default_timeout,
                )

            assert upload_response.status_code == 200
            upload_data = upload_response.json()
            assert upload_data.get("success") is True

            file_info = upload_data["data"]["file"]
            server_filename = Path(file_info["vfsPath"]).as_posix()

            # When - Execute code that processes the file
            code = f'''import pandas as pd
filename = "{server_filename}"
df = pd.read_csv(filename)
total = df["value"].sum()
columns = list(df.columns)
f"sum={{total}}, columns={{columns}}"'''

            response = requests.post(
                f"{base_url}/api/execute-raw",
                data=code,
                headers={"Content-Type": "text/plain"},
                timeout=default_timeout,
            )

            # Then
            assert response.status_code == 200
            result = response.json().get("result")
            assert "sum=6" in result
            assert "columns=['name', 'value', 'category']" in result

            # Cleanup
            pyodide_name = Path(file_info["vfsPath"]).name
            requests.delete(f"{base_url}/api/uploaded-files/{pyodide_name}", timeout=default_timeout)

        finally:
            os.unlink(tmp_path)

    def test_given_execution_context_isolation_when_running_separate_requests_then_variables_not_shared(self, server_ready, base_url, default_timeout):
        """
        Scenario: Execution context isolation
        Given I execute code that defines a variable
        When I execute code in a separate request
        Then the variable should not exist, proving context isolation
        """
        # Given: Define a variable in the first request
        define_code = "isolated_variable = 'hello'"
        define_response = requests.post(
            f"{base_url}/api/execute-raw",
            data=define_code,
            headers={"Content-Type": "text/plain"},
            timeout=default_timeout,
        )
        assert define_response.status_code == 200
        assert define_response.json()["success"] is True

        # When: Check for the variable in a second request
        check_code = "'defined' if 'isolated_variable' in globals() else 'undefined'"
        check_response = requests.post(
            f"{base_url}/api/execute-raw",
            data=check_code,
            headers={"Content-Type": "text/plain"},
            timeout=default_timeout,
        )
        
        # Then: The variable should be undefined
        assert check_response.status_code == 200
        result = check_response.json()
        assert result.get("result") == "undefined", "Execution context should be isolated between requests"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
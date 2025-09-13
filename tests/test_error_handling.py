"""
Error handling test scenarios in BDD style using pytest.

This module contains comprehensive tests for API error handling and edge cases,
written in Behavior-Driven Development (BDD) style using pytest.
"""

import pytest
import requests
import tempfile
import time
from pathlib import Path

# Global configuration constants
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 10
EXECUTION_TIMEOUT = 30000
MAX_CODE_LENGTH = 50000
MAX_FILE_SIZE_MB = 10
REQUEST_TIMEOUT = 60000


class TestCodeExecutionErrors:
    """Test scenarios for code execution error handling."""

    def test_given_empty_code_when_executing_then_validation_error(self, server, base_url):
        """
        Scenario: Execute empty code
        Given the API is available
        When I send an empty code string
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/execute"
        
        # When
        response = requests.post(endpoint, json={"code": ""}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False
        assert "empty" in response_data.get("error", "").lower()

    def test_given_whitespace_only_code_when_executing_then_validation_error(self, server, base_url):
        """
        Scenario: Execute whitespace-only code
        Given the API is available
        When I send code containing only whitespace
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/execute"
        whitespace_code = "   \n\t  \n  "
        
        # When
        response = requests.post(endpoint, json={"code": whitespace_code}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_missing_code_field_when_executing_then_validation_error(self, server, base_url):
        """
        Scenario: Execute without code field
        Given the API is available
        When I send a request without the code field
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/execute"
        
        # When
        response = requests.post(endpoint, json={}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False
        assert "code" in response_data.get("error", "").lower()

    def test_given_invalid_code_type_when_executing_then_validation_error(self, server, base_url):
        """
        Scenario: Execute with non-string code
        Given the API is available
        When I send code as a non-string type
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/execute"
        
        # When
        response = requests.post(endpoint, json={"code": 123}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False
        assert "string" in response_data.get("error", "").lower()

    def test_given_syntax_error_code_when_executing_then_execution_error(self, server, base_url):
        """
        Scenario: Execute code with syntax error
        Given the API is available
        When I send Python code with syntax errors
        Then I should receive an execution error
        """
        # Given
        endpoint = f"{base_url}/api/execute"
        invalid_code = "if True\n    print('missing colon')"
        
        # When
        response = requests.post(endpoint, json={"code": invalid_code}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 200  # API accepts but execution fails
        response_data = response.json()
        assert response_data.get("success") is False
        error_msg = response_data.get("error", "").lower()
        assert any(keyword in error_msg for keyword in ["syntax", "expected", "invalid syntax"])

    def test_given_runtime_error_code_when_executing_then_execution_error(self, server, base_url):
        """
        Scenario: Execute code with runtime error
        Given the API is available
        When I send Python code that causes a runtime error
        Then I should receive an execution error
        """
        # Given
        endpoint = f"{base_url}/api/execute"
        error_code = "x = 1 / 0"
        
        # When
        response = requests.post(endpoint, json={"code": error_code}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 200
        response_data = response.json()
        assert response_data.get("success") is False
        assert "division" in response_data.get("error", "").lower()

    def test_given_very_long_code_when_executing_then_validation_error(self, server, base_url):
        """
        Scenario: Execute code exceeding length limit
        Given the API is available
        When I send code exceeding the maximum length limit
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/execute"
        long_code = "x = 1\n" * 50000  # Exceeds MAX_CODE_LENGTH
        
        # When
        response = requests.post(endpoint, json={"code": long_code}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False
        assert "too long" in response_data.get("error", "").lower()

    def test_given_invalid_timeout_when_executing_then_validation_error(self, server, base_url):
        """
        Scenario: Execute with invalid timeout
        Given the API is available
        When I send a request with invalid timeout value
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/execute"
        
        # When
        response = requests.post(endpoint, json={"code": "print('test')", "timeout": -1}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_excessive_timeout_when_executing_then_validation_error(self, server, base_url):
        """
        Scenario: Execute with timeout exceeding limit
        Given the API is available
        When I send a request with timeout exceeding the maximum allowed
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/execute"
        excessive_timeout = 400000
        
        # When
        response = requests.post(endpoint, json={"code": "print('test')", "timeout": excessive_timeout}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False


class TestExecuteRawEndpoint:
    """Test scenarios for execute-raw endpoint with code execution."""

    def test_given_simple_python_code_when_using_execute_raw_then_success(self, server, base_url):
        """
        Scenario: Execute simple Python code via execute-raw
        Given the execute-raw endpoint is available
        When I send simple Python code as plain text
        Then I should receive successful execution result
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"
        simple_code = "print('Hello from execute-raw!')\nresult = 2 + 2\nprint(f'2 + 2 = {result}')"
        
        # When
        response = requests.post(
            endpoint, 
            data=simple_code, 
            headers={"Content-Type": "text/plain"}, 
            timeout=DEFAULT_TIMEOUT
        )
        
        # Then
        assert response.status_code == 200
        response_data = response.json()
        assert response_data.get("success") is True
        # Check for output in the result
        if "output" in response_data:
            assert "Hello from execute-raw!" in response_data["output"]
            assert "2 + 2 = 4" in response_data["output"]

    def test_given_file_check_code_when_using_execute_raw_then_file_operations_work(self, server, base_url):
        """
        Scenario: Check file system operations via execute-raw
        Given the execute-raw endpoint is available
        When I send code to check file system operations as plain text
        Then I should receive information about available paths
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"
        file_check_code = """
import os
from pathlib import Path

# Check available directories
available_paths = []
for path in ['/home/pyodide', '/uploads', '/plots']:
    if os.path.exists(path):
        available_paths.append(path)
        print(f"✓ Path exists: {path}")
    else:
        print(f"✗ Path missing: {path}")

print(f"Available paths: {available_paths}")
print(f"Current working directory: {os.getcwd()}")
        """
        
        # When
        response = requests.post(
            endpoint, 
            data=file_check_code, 
            headers={"Content-Type": "text/plain"}, 
            timeout=DEFAULT_TIMEOUT
        )
        
        # Then
        assert response.status_code == 200
        response_data = response.json()
        assert response_data.get("success") is True
        # Check for output in the result
        if "output" in response_data:
            assert "Available paths:" in response_data["output"]
            assert "Current working directory:" in response_data["output"]


class TestPackageInstallationErrors:
    """Test scenarios for package installation error handling."""

    def test_given_empty_package_name_when_installing_then_validation_error(self, server, base_url):
        """
        Scenario: Install package with empty name
        Given the package installation API is available
        When I send an empty package name
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        
        # When
        response = requests.post(endpoint, json={"package": ""}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_missing_package_field_when_installing_then_validation_error(self, server, base_url):
        """
        Scenario: Install package without package field
        Given the package installation API is available
        When I send a request without the package field
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        
        # When
        response = requests.post(endpoint, json={}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_invalid_package_type_when_installing_then_validation_error(self, server, base_url):
        """
        Scenario: Install package with non-string type
        Given the package installation API is available
        When I send a package name as non-string type
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        
        # When
        response = requests.post(endpoint, json={"package": 123}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_invalid_package_characters_when_installing_then_validation_error(self, server, base_url):
        """
        Scenario: Install package with invalid characters
        Given the package installation API is available
        When I send a package name with invalid characters
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        
        # When
        response = requests.post(endpoint, json={"package": "invalid@package!"}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_blocked_package_when_installing_then_forbidden_error(self, server, base_url):
        """
        Scenario: Install blocked package
        Given the package installation API is available
        When I try to install a blocked package
        Then I should receive a forbidden error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        
        # When
        response = requests.post(endpoint, json={"package": "os"}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 403
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_nonexistent_package_when_installing_then_error(self, server, base_url):
        """
        Scenario: Install package that doesn't exist
        Given the package installation API is available
        When I try to install a non-existent package
        Then I should receive an appropriate error
        """
        # Given
        endpoint = f"{base_url}/api/install-package"
        
        # When
        response = requests.post(endpoint, json={"package": "nonexistent-package-xyz123"}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        response_data = response.json()
        # Should return error for non-existent package
        assert response.status_code in [400, 500]  # Accept either based on server implementation
        assert response_data.get("success") is False
        assert isinstance(response_data.get("error"), str)


class TestFileOperationErrors:
    """Test scenarios for file operation error handling."""

    def test_given_no_file_when_uploading_then_validation_error(self, server, base_url):
        """
        Scenario: Upload without file
        Given the file upload API is available
        When I send an upload request without a file
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/upload"
        
        # When
        response = requests.post(endpoint, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 400

    def test_given_invalid_file_type_when_uploading_then_validation_error(self, server, base_url):
        """
        Scenario: Upload invalid file type
        Given the file upload API is available
        When I try to upload a file with invalid type
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
                    timeout=DEFAULT_TIMEOUT
                )
            
            # Then
            assert response.status_code in [400, 500]  # Accept either based on implementation
        finally:
            Path(tmp_path).unlink()

    def test_given_oversized_file_when_uploading_then_validation_error(self, server, base_url):
        """
        Scenario: Upload file exceeding size limit
        Given the file upload API is available
        When I try to upload a file exceeding the size limit
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/upload"
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
                    timeout=DEFAULT_TIMEOUT
                )
            
            # Then
            assert response.status_code in [400, 413]  # Either validation error or entity too large
        finally:
            Path(tmp_path).unlink()

    def test_given_nonexistent_uploaded_file_when_deleting_then_not_found_error(self, server, base_url):
        """
        Scenario: Delete uploaded file that doesn't exist
        Given the file deletion API is available
        When I try to delete a file that doesn't exist
        Then I should receive a not found error
        """
        # Given
        endpoint = f"{base_url}/api/uploaded-files/nonexistent.csv"
        
        # When
        response = requests.delete(endpoint, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 404
        response_data = response.json()
        assert response_data.get("success") is False

    def test_given_file_info_request_for_nonexistent_file_when_checking_then_file_not_exists(self, server, base_url):
        """
        Scenario: Get info for nonexistent file
        Given the file info API is available
        When I request info for a file that doesn't exist
        Then I should receive information showing the file doesn't exist
        """
        # Given
        endpoint = f"{base_url}/api/file-info/nonexistent.csv"
        
        # When
        response = requests.get(endpoint, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 200  # API succeeds but shows file doesn't exist
        response_data = response.json()
        # Check that the response indicates the file doesn't exist
        assert response_data.get("data", {}).get("exists") is False

    def test_given_path_traversal_attempt_when_deleting_then_validation_error(self, server, base_url):
        """
        Scenario: Delete file with path traversal attempt
        Given the file deletion API is available
        When I try to delete a file using path traversal
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/uploaded-files/../../../etc/passwd"
        
        # When
        response = requests.delete(endpoint, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 400
        response_data = response.json()
        assert response_data.get("success") is False
        assert "invalid" in response_data.get("error", "").lower()


class TestHttpProtocolErrors:
    """Test scenarios for HTTP protocol error handling."""

    def test_given_nonexistent_endpoint_when_requesting_then_not_found_error(self, server, base_url):
        """
        Scenario: Request non-existent endpoint
        Given the API is available
        When I request an endpoint that doesn't exist
        Then I should receive a not found error
        """
        # Given
        endpoint = f"{base_url}/api/nonexistent"
        
        # When
        response = requests.get(endpoint, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 404

    def test_given_wrong_http_method_when_requesting_then_method_not_allowed_error(self, server, base_url):
        """
        Scenario: Use wrong HTTP method
        Given the API is available
        When I use the wrong HTTP method for an endpoint
        Then I should receive a method not allowed error
        """
        # Given
        endpoint = f"{base_url}/api/execute"  # Should be POST, not DELETE
        
        # When
        response = requests.delete(endpoint, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 404  # Express returns 404 for unmatched routes

    def test_given_malformed_json_when_sending_request_then_validation_error(self, server, base_url):
        """
        Scenario: Send malformed JSON
        Given the API is available
        When I send a request with malformed JSON
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/execute"
        malformed_json = '{"code": "print(\'test\')"'  # Missing closing brace
        
        # When
        response = requests.post(
            endpoint,
            data=malformed_json,
            headers={"Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT
        )
        
        # Then
        assert response.status_code == 400

    def test_given_invalid_content_type_when_sending_request_then_validation_error(self, server, base_url):
        """
        Scenario: Send wrong content type
        Given the API is available
        When I send a request with invalid content type
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/execute"
        
        # When
        response = requests.post(
            endpoint,
            data="print('test')",
            headers={"Content-Type": "text/plain"},  # Should be application/json
            timeout=DEFAULT_TIMEOUT
        )
        
        # Then
        assert response.status_code == 400

    def test_given_very_large_json_payload_when_sending_request_then_validation_error(self, server, base_url):
        """
        Scenario: Send extremely large JSON payload
        Given the API is available
        When I send a request with an extremely large JSON payload
        Then I should receive a validation error
        """
        # Given
        endpoint = f"{base_url}/api/execute"
        large_context = {f"key_{i}": f"value_{i}" * 1000 for i in range(100)}
        
        # When
        response = requests.post(endpoint, json={"code": "print('test')", "context": large_context}, timeout=DEFAULT_TIMEOUT)
        
        # Then
        assert response.status_code == 400  # Should reject large context


class TestPyodideFileSystemOperations:
    """Test scenarios for Pyodide filesystem operations using execute-raw."""

    def test_given_file_creation_code_when_using_execute_raw_then_file_operations_work(self, server, base_url):
        """
        Scenario: Create and verify files using execute-raw
        Given the execute-raw endpoint is available
        When I create a file using Python code as plain text
        Then I should be able to verify its existence
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"
        file_operation_code = """
import os
from pathlib import Path

# Create a test file
test_content = "test,data\\n1,2\\n3,4\\n"
test_file = Path("/tmp/test_file.csv")

# Write test file
with open(test_file, "w") as f:
    f.write(test_content)

# Verify file exists
if test_file.exists():
    print(f"✓ File created successfully: {test_file}")
    print(f"File size: {test_file.stat().st_size} bytes")
    
    # Read and verify content
    with open(test_file, "r") as f:
        content = f.read()
    print(f"Content preview: {content[:50]}...")
else:
    print("✗ File creation failed")
        """
        
        # When
        response = requests.post(
            endpoint, 
            data=file_operation_code, 
            headers={"Content-Type": "text/plain"}, 
            timeout=DEFAULT_TIMEOUT
        )
        
        # Then
        assert response.status_code == 200
        response_data = response.json()
        assert response_data.get("success") is True
        # Check for output in the result
        if "output" in response_data:
            assert "✓ File created successfully" in response_data["output"]
            assert "File size:" in response_data["output"]

    def test_given_directory_listing_code_when_using_execute_raw_then_directory_operations_work(self, server, base_url):
        """
        Scenario: List directories using execute-raw
        Given the execute-raw endpoint is available
        When I list directories using Python code as plain text
        Then I should get directory information
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"
        directory_listing_code = """
import os
from pathlib import Path

# List available directories
directories_to_check = ["/", "/tmp", "/home", "/home/pyodide"]
print("Directory listing:")

for dir_path in directories_to_check:
    path = Path(dir_path)
    if path.exists():
        print(f"\\n✓ {dir_path}:")
        try:
            items = list(path.iterdir())[:5]  # Limit to first 5 items
            for item in items:
                item_type = "DIR" if item.is_dir() else "FILE"
                print(f"  {item_type}: {item.name}")
        except PermissionError:
            print("  (Permission denied)")
        except Exception as e:
            print(f"  (Error: {e})")
    else:
        print(f"✗ {dir_path}: Not found")
        """
        
        # When
        response = requests.post(
            endpoint, 
            data=directory_listing_code, 
            headers={"Content-Type": "text/plain"}, 
            timeout=DEFAULT_TIMEOUT
        )
        
        # Then
        assert response.status_code == 200
        response_data = response.json()
        assert response_data.get("success") is True
        # Check for output in the result
        if "output" in response_data:
            assert "Directory listing:" in response_data["output"]
            assert "✓" in response_data["output"]  # At least one directory should exist
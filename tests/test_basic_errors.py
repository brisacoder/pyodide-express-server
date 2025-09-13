"""Test basic error handling scenarios using pytest."""

import tempfile
from pathlib import Path

import pytest
import requests

# Global constants - import from conftest or define here
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30  # Standard timeout for API requests
QUICK_TIMEOUT = 10    # Quick timeout for simple operations


class TestBasicErrorHandling:
    """Test basic error handling scenarios without starting a new server."""

    def test_when_executing_empty_code_then_returns_validation_error(self):
        """
        Given: An API request to execute code via execute-raw
        When: The request body is empty
        Then: Should return 400 with validation error message
        """
        # When
        r = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data="",
            headers={"Content-Type": "text/plain"},
            timeout=QUICK_TIMEOUT
        )
        
        # Then
        assert r.status_code == 400
        response = r.json()
        assert response.get("success") is False
        assert "code" in response.get("error", "").lower()

    def test_when_no_data_provided_then_returns_validation_error(self):
        """
        Given: An API request to execute code via execute-raw
        When: No data is provided in the request body
        Then: Should return 400 with validation error message
        """
        # When
        r = requests.post(
            f"{BASE_URL}/api/execute-raw",
            headers={"Content-Type": "text/plain"},
            timeout=QUICK_TIMEOUT
        )
        
        # Then
        assert r.status_code == 400
        response = r.json()
        assert response.get("success") is False
        assert "code" in response.get("error", "").lower()

    def test_when_sending_non_text_content_type_then_returns_error(self):
        """
        Given: An API request to execute code
        When: The content-type is not text/plain
        Then: Should return 400 or appropriate error
        """
        # When - sending JSON instead of plain text
        r = requests.post(
            f"{BASE_URL}/api/execute-raw",
            json={"code": "print('hello')"},  # Wrong content type
            timeout=QUICK_TIMEOUT
        )
        
        # Then - should fail because execute-raw expects text/plain
        assert r.status_code in [400, 415]  # Bad request or unsupported media type
        # Note: The exact behavior depends on server middleware configuration

    def test_when_python_syntax_invalid_then_returns_syntax_error(self):
        """
        Given: An API request to execute Python code via execute-raw
        When: The code contains syntax errors
        Then: Should return 200 with error details in response body
        """
        # Given
        invalid_python_code = "if True\n    print('missing colon')"
        
        # When
        r = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=invalid_python_code,
            headers={"Content-Type": "text/plain"},
            timeout=QUICK_TIMEOUT
        )
        
        # Then
        assert r.status_code == 200  # API returns 200 but with error in body
        response = r.json()
        assert response.get("success") is False
        assert "error" in response

    def test_when_python_runtime_error_occurs_then_returns_execution_error(self):
        """
        Given: An API request to execute Python code via execute-raw
        When: The code causes a runtime error (division by zero)
        Then: Should return 200 with error details in response body
        """
        # Given
        code_with_runtime_error = "x = 1 / 0"
        
        # When
        r = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=code_with_runtime_error,
            headers={"Content-Type": "text/plain"},
            timeout=QUICK_TIMEOUT
        )
        
        # Then
        assert r.status_code == 200  # API returns 200 but with error in body
        response = r.json()
        assert response.get("success") is False
        assert "error" in response

    def test_install_empty_package(self):
        """Test installing package with empty name."""
        r = requests.post(
            f"{BASE_URL}/api/install-package", 
            json={"package": ""}, 
            timeout=QUICK_TIMEOUT
        )
        assert r.status_code == 400
        response = r.json()
        assert response.get("success") is False

    def test_install_no_package_field(self):
        """Test installing without package field."""
        r = requests.post(
            f"{BASE_URL}/api/install-package", 
            json={}, 
            timeout=QUICK_TIMEOUT
        )
        assert r.status_code == 400
        response = r.json()
        assert response.get("success") is False

    def test_delete_nonexistent_uploaded_file(self):
        """Test deleting uploaded file that doesn't exist."""
        r = requests.delete(
            f"{BASE_URL}/api/uploaded-files/nonexistent.csv", 
            timeout=QUICK_TIMEOUT
        )
        assert r.status_code == 404
        response = r.json()
        assert response.get("success") is False

    def test_file_info_nonexistent_file(self):
        """Test getting info for nonexistent file."""
        r = requests.get(
            f"{BASE_URL}/api/file-info/nonexistent.csv", 
            timeout=QUICK_TIMEOUT
        )
        # Server returns 200 for nonexistent files with exists=False
        assert r.status_code == 200
        response = r.json()
        assert response.get("success") is True
        assert "data" in response
        assert response["data"].get("exists") is False

    def test_invalid_endpoint(self):
        """Test requesting non-existent endpoint."""
        r = requests.get(
            f"{BASE_URL}/api/nonexistent", 
            timeout=QUICK_TIMEOUT
        )
        assert r.status_code == 404

    def test_upload_no_file(self):
        """Test upload without file."""
        r = requests.post(
            f"{BASE_URL}/api/upload", 
            timeout=QUICK_TIMEOUT
        )
        assert r.status_code == 400

    def test_upload_invalid_file_type(self):
        """Test upload with invalid file type."""
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("invalid content")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, "rb") as fh:
                r = requests.post(
                    f"{BASE_URL}/api/upload",
                    files={"file": ("test.txt", fh, "text/plain")},
                    timeout=QUICK_TIMEOUT
                )
            # Should either reject or handle appropriately
            assert r.status_code in [400, 200]  # Depends on validation
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# Parametrized tests for better coverage
@pytest.mark.parametrize("code,expected_error_keyword", [
    ("", "code"),  # Empty code
    ("   ", "code"),  # Whitespace only
    ("\n\n", "code"),  # Only newlines
])
def test_execute_invalid_code_variations(code, expected_error_keyword):
    """Test various invalid code inputs."""
    r = requests.post(
        f"{BASE_URL}/api/execute-raw",
        data=code,
        headers={"Content-Type": "text/plain"},
        timeout=QUICK_TIMEOUT
    )
    assert r.status_code == 400
    response = r.json()
    assert response.get("success") is False
    assert expected_error_keyword in response.get("error", "").lower()


@pytest.mark.parametrize("package_name", [
    "",  # Empty string
    "   ",  # Whitespace only
    None,  # None value (if accepted by API)
])
def test_install_invalid_package_names(package_name):
    """Test various invalid package names."""
    payload = {"package": package_name} if package_name is not None else {}
    r = requests.post(
        f"{BASE_URL}/api/install-package",
        json=payload,
        timeout=QUICK_TIMEOUT
    )
    assert r.status_code == 400
    response = r.json()
    assert response.get("success") is False

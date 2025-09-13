"""BDD-style API tests with proper test isolation and fixtures.

Each test is independent and can be run in any order.
Tests follow the Given-When-Then pattern typical of BDD.
"""

# pylint: disable=redefined-outer-name,unused-argument
# ^ Normal for pytest: fixtures are used as parameters and may not be directly referenced

import os
import tempfile
import time
from pathlib import Path
from typing import Generator, Tuple

import pytest
import requests

BASE_URL = "http://localhost:3000"


# Return API contract for Upload
# success: true,
# data: {
# file: {
#     originalName: req.file.originalname,
#     sanitizedOriginal: req.file.safeOriginalName,
#     storedFilename: req.file.filename,
#     size: req.file.size,
#     mimetype: req.file.mimetype,
#     filesystemPath: urlPath, // absolute server path
#     urlPath,                // "/uploads/<file>"
#     absoluteUrl,           // "http(s)://host/uploads/<file>"
#     userAgent: req.get('User-Agent'),
#     fileSize: req.file.size,
#     mimeType: req.file.mimetype,
#     timestamp: new Date().toISOString(),
# }
# }


def wait_for_server(url: str, timeout: int = 120):
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
def default_timeout():
    """Provide a default timeout for API requests."""
    return 30


@pytest.fixture
def sample_csv_content():
    """Provide sample CSV content for testing."""
    return "name,value,category\nitem1,1,A\nitem2,2,B\nitem3,3,C\n"


@pytest.fixture
def uploaded_file(
    sample_csv_content, default_timeout
) -> Generator[Tuple[str, str], None, None]:
    """Upload a test CSV file and return (pyodide_name, server_filename).

    This fixture handles both upload and cleanup automatically.
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(sample_csv_content)  # This is the fixture value, not the fixture
        tmp_path = tmp.name

    # Upload file
    with open(tmp_path, "rb") as fh:
        r = requests.post(
            f"{BASE_URL}/api/upload",
            files={"file": ("test_data.csv", fh, "text/csv")},
            timeout=default_timeout,
        )

    # Clean up temp file
    os.unlink(tmp_path)

    # Extract file references
    assert r.status_code == 200
    upload_data = r.json()
    assert upload_data.get("success") is True

    file_info = upload_data["data"]["file"]
    pyodide_name = Path(file_info["vfsPath"]).name
    server_filename = Path(file_info["vfsPath"]).as_posix()

    # Yield for test to use
    yield pyodide_name, server_filename

    try:
        # Delete from server
        requests.delete(f"{BASE_URL}/api/uploaded-files/{server_filename}", timeout=10)
    except (requests.RequestException, OSError):
        pass  # Ignore cleanup errors


# ==================== HEALTH CHECK TESTS ====================


class TestHealthEndpoints:
    """Test server health and status endpoints."""

    def test_basic_health_endpoint(self, server_ready):
        """Given: Server is running
        When: I check the health endpoint
        Then: It should return OK status
        """
        # When
        response = requests.get(f"{BASE_URL}/health", timeout=10)

        # Then
        assert response.status_code == 200
        assert response.json().get("status") == "ok"

    def test_api_status_endpoint(self, server_ready):
        """Given: Server is running
        When: I check the API status endpoint
        Then: It should return server readiness information
        """
        # When
        response = requests.get(f"{BASE_URL}/api/status", timeout=10)

        # Then
        assert response.status_code == 200
        payload = response.json()
        assert "data" in payload, f"Response missing 'data': {payload}"
        assert (
            "isReady" in payload["data"]
        ), f"Data missing 'isReady': {payload['data']}"

    def test_pyodide_health_endpoint(self, server_ready):
        """Given: Server has Pyodide runtime
        When: I check the Pyodide health endpoint
        Then: It should confirm Pyodide is initialized
        """
        # When
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)

        # Then
        assert response.status_code == 200
        assert response.json().get("success") is True, "Pyodide should be initialized"

    def test_statistics_endpoint(self, server_ready):
        """Given: Server is tracking statistics
        When: I request server stats
        Then: It should return uptime and metrics
        """
        # When
        response = requests.get(f"{BASE_URL}/api/stats", timeout=10)

        # Then
        assert response.status_code == 200
        stats = response.json()
        assert "uptime" in stats, f"Stats missing 'uptime': {stats}"


# ==================== PACKAGE MANAGEMENT TESTS ====================


class TestPackageManagement:
    """Test Python package installation and listing."""

    def test_install_python_package(self, server_ready):
        """Given: Pyodide environment is ready
        When: I install a Python package (beautifulsoup4)
        Then: It should install successfully
        """
        # When
        response = requests.post(
            f"{BASE_URL}/api/install-package",
            json={"package": "beautifulsoup4"},
            timeout=120,
        )

        # Then
        assert response.status_code == 200
        result = response.json()
        assert "success" in result, f"Response missing 'success': {result}"
        assert result.get("success") is True, f"Package installation failed: {result}"

    def test_list_installed_packages(self, server_ready):
        """Given: Pyodide has pre-installed packages
        When: I list installed packages
        Then: It should return package information
        """
        # When
        response = requests.get(f"{BASE_URL}/api/packages", timeout=10)

        # Then
        assert response.status_code == 200
        payload = response.json()
        assert "data" in payload, f"Response missing 'data': {payload}"

        result = payload["data"]
        assert result["success"] is True, f"Failed to get packages: {result}"

        # Handle nested result structure
        if isinstance(result, dict) and "result" in result:
            result = result["result"]

        assert result is not None, "Packages result should not be null"
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert (
            "installed_packages" in result
        ), f"Missing 'installed_packages': {result.keys()}"
        assert "total_packages" in result, f"Missing 'total_packages': {result.keys()}"
        assert isinstance(
            result["installed_packages"], list
        ), "installed_packages should be a list"
        assert result["total_packages"] > 0, "Should have packages installed"


# ==================== CODE EXECUTION TESTS ====================


class TestCodeExecution:
    """Test Python code execution endpoints."""

    def test_execute_endpoint_with_json(self, server_ready, default_timeout):
        """Given: Valid Python code
        When: I execute it via the JSON execute endpoint
        Then: It should return the result in JSON format
        """
        # Given
        code = '''name = "World"
f"Hello {name}"'''

        # When
        response = requests.post(
            f"{BASE_URL}/api/execute", json={"code": code}, timeout=default_timeout
        )

        # Then
        assert response.status_code == 200
        result = response.json()
        assert result.get("success") is True, f"Execution failed: {result}"
        assert (
            result.get("result") == "Hello World"
        ), f"Unexpected result: {result.get('result')}"

    def test_execute_raw_endpoint(self, server_ready, default_timeout):
        """Given: Valid Python code as plain text
        When: I execute it via the raw endpoint
        Then: It should return the result
        """
        # Given
        code = '''x = 3
f"{x + 3}"'''

        # When
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=default_timeout,
        )

        # Then
        assert response.status_code == 200
        result = response.json()
        assert result.get("result") == "6", f"Expected '6', got {result.get('result')}"


# ==================== FILE OPERATIONS TESTS ====================


class TestFileOperations:
    """Test file upload, listing, and deletion operations."""

    def test_upload_and_verify_pyodide_file(
        self, server_ready, sample_csv_content, default_timeout
    ):
        """Given: A CSV file to upload
        When: I upload it and verify it exists in Pyodide
        Then: The file should be accessible via Pyodide API
        """
        # Given
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write(sample_csv_content)
            tmp_path = tmp.name

        try:
            # When - Upload
            with open(tmp_path, "rb") as fh:
                upload_response = requests.post(
                    f"{BASE_URL}/api/upload",
                    files={"file": ("verify_test.csv", fh, "text/csv")},
                    timeout=default_timeout,
                )

            # Then - Verify upload
            assert upload_response.status_code == 200
            upload_data = upload_response.json()
            assert upload_data.get("success") is True

            file_info = upload_data["data"]["file"]
            pyodide_name = Path(file_info["vfsPath"]).name

            # Verify file exists in Pyodide
            list_response = requests.get(f"{BASE_URL}/api/uploaded-files", timeout=10)
            assert list_response.status_code == 200

            files = list_response.json().get("data", {}).get("files", [])
            py_file_names = [f["filename"] for f in files]
            assert (
                pyodide_name in py_file_names
            ), f"Uploaded file {pyodide_name} should be in Pyodide files"

            # Cleanup
            delete_response = requests.delete(
                f"{BASE_URL}/api/uploaded-files/{pyodide_name}", timeout=10
            )

            # Log delete response for debugging
            print(f"Delete response status: {delete_response.status_code}")
            print(f"Delete response body: {delete_response.text}")

        finally:
            os.unlink(tmp_path)

    def test_upload_csv_file(self, server_ready, sample_csv_content, default_timeout):
        """Given: A CSV file
        When: I upload it to the server
        Then: It should be stored and accessible
        """
        # Given
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write(sample_csv_content)
            tmp_path = tmp.name

        try:
            # When
            with open(tmp_path, "rb") as fh:
                response = requests.post(
                    f"{BASE_URL}/api/upload",
                    files={"file": ("data.csv", fh, "text/csv")},
                    timeout=default_timeout,
                )

            # Then
            assert response.status_code == 200
            upload_data = response.json()
            assert upload_data.get("success") is True, f"Upload failed: {upload_data}"
            assert "data" in upload_data, f"Response missing 'data': {upload_data}"
            assert (
                "file" in upload_data["data"]
            ), f"Data missing 'file': {upload_data['data']}"

            file_info = upload_data["data"]["file"]
            assert "sanitizedOriginal" in file_info
            assert "vfsPath" in file_info

            # Cleanup uploaded file
            pyodide_name = Path(file_info["vfsPath"]).name
            requests.delete(f"{BASE_URL}/api/uploaded-files/{pyodide_name}", timeout=10)

        finally:
            # Always clean up temp file
            os.unlink(tmp_path)

    def test_list_uploaded_files(self, server_ready, uploaded_file):
        """Given: A file has been uploaded
        When: I list uploaded files
        Then: The file should appear in the list
        """
        # Given
        pyodide_name, _ = uploaded_file

        # When
        response = requests.get(f"{BASE_URL}/api/uploaded-files", timeout=10)

        # Then
        assert response.status_code == 200
        upload_files = response.json()
        assert upload_files.get("success") is True, f"List failed: {upload_files}"
        assert "data" in upload_files, f"Response missing 'data': {upload_files}"
        data = upload_files.get("data", {})
        assert "files" in data, f"Data missing 'files': {data.keys()}"

        files = data["files"]
        uploaded_names = [f["filename"] for f in files]
        assert (
            pyodide_name in uploaded_names
        ), f"File {pyodide_name} not in list: {uploaded_names}"

    def test_get_file_info(self, server_ready, uploaded_file):
        """Given: A file has been uploaded
        When: I request file information
        Then: It should return file existence and metadata
        """
        # Given
        pyodide_name, _ = uploaded_file

        # When
        response = requests.get(f"{BASE_URL}/api/file-info/{pyodide_name}", timeout=10)

        # Then
        assert response.status_code == 200
        info = response.json()
        assert info["data"]["exists"] is True, "File should exist"
        assert (
            info["data"]["pyodideFile"]["exists"] is True
        ), "Pyodide file should exist"
        assert info["data"]["filename"] == pyodide_name, "Filename mismatch"

    def test_execute_with_uploaded_file(
        self, server_ready, uploaded_file, default_timeout
    ):
        """Given: A CSV file has been uploaded
        When: I execute Python code that reads the file
        Then: It should successfully process the file
        """
        # Given
        _, server_filename = uploaded_file
        code = f'''import pandas as pd
filename = "{server_filename}"
df = pd.read_csv(filename)
total = df["value"].sum()
columns = list(df.columns)
f"sum={{total}}, columns={{columns}}"'''

        # When
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=default_timeout,
        )

        # Then
        assert response.status_code == 200
        result = response.json().get("result")
        assert "sum=6" in result, f"Expected sum=6 in result: {result}"
        assert (
            "columns=['name', 'value', 'category']" in result
        ), f"Expected columns in result: {result}"

    def test_list_pyodide_files(self, server_ready, uploaded_file):
        """Given: A file has been uploaded
        When: I list files in Pyodide filesystem
        Then: The file should be accessible from Python
        """
        # Given
        pyodide_name, _ = uploaded_file

        # When
        response = requests.get(f"{BASE_URL}/api/uploaded-files", timeout=10)

        # Then
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True, f"API call failed: {data}"
        
        # The structure is response.data.files
        files = data.get("data", {}).get("files", [])
        py_file_names = [f["filename"] for f in files]
        assert (
            pyodide_name in py_file_names
        ), f"File {pyodide_name} not in uploaded files: {py_file_names}"

    def test_delete_pyodide_file(self, server_ready, uploaded_file):
        """Given: A file exists in Pyodide filesystem
        When: I delete it
        Then: It should be removed successfully
        """
        # Given
        pyodide_name, _ = uploaded_file

        # When
        response = requests.delete(
            f"{BASE_URL}/api/uploaded-files/{pyodide_name}", timeout=10
        )

        # Then
        assert response.status_code == 200
        assert response.json().get("success") is True, "Deletion should succeed"

        # Verify file is gone
        list_response = requests.get(f"{BASE_URL}/api/uploaded-files", timeout=10)
        files = list_response.json().get("data", {}).get("files", [])
        py_file_names = [f["filename"] for f in files]
        assert (
            pyodide_name not in py_file_names
        ), f"File {pyodide_name} should be deleted"


# ==================== ENVIRONMENT TESTS ====================


class TestEnvironment:
    """Test Pyodide environment management."""

    def test_execution_context_is_isolated(self, server_ready, default_timeout):
        """
        Given: I execute code that defines a variable
        When: I execute code in a separate request
        Then: The variable should not exist, proving context isolation
        """
        # Given: Define a variable in the first request
        define_code = "isolated_variable = 'hello'"
        define_response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=define_code,
            headers={"Content-Type": "text/plain"},
            timeout=default_timeout,
        )
        assert define_response.status_code == 200
        assert define_response.json()["success"] is True

        # When: Check for the variable in a second request
        check_code = "'defined' if 'isolated_variable' in globals() else 'undefined'"
        check_response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=check_code,
            headers={"Content-Type": "text/plain"},
            timeout=default_timeout,
        )
        assert check_response.status_code == 200
        result = check_response.json()

        # Then: The variable should be undefined
        assert (
            result.get("result") == "undefined"
        ), "Execution context should be isolated between requests"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

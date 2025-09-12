import os
import tempfile
import time
from pathlib import Path

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
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Server at {url} did not start in time")


class TestAPI:
    """Exercise API endpoints and report results per test."""

    server_filename: str | None = None
    pyodide_name: str | None = None

    def test_01_health(self):
        """Test the basic health endpoint to ensure server is running.
        
        Verifies that the health endpoint returns a 200 status code
        and the expected 'ok' status in the response.
        """
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        assert r.status_code == 200
        assert r.json().get("status") == "ok"

    def test_02_status(self):
        """Test the API status endpoint for detailed server information.
        
        Verifies that the status endpoint returns server readiness state
        and contains the expected data structure with isReady flag.
        """
        r = requests.get(f"{BASE_URL}/api/status", timeout=10)
        assert r.status_code == 200
        payload = r.json()
        assert "data" in payload
        assert "isReady" in payload["data"]

    def test_03_pyodide_health(self):
        """Test the Pyodide-specific health endpoint.
        
        Verifies that the Pyodide runtime is initialized and healthy
        by checking the success flag in the response.
        """
        r = requests.get(f"{BASE_URL}/api/health", timeout=10)
        assert r.status_code == 200
        assert r.json().get("success") is True

    def test_04_stats(self):
        """Test the statistics endpoint for server metrics.
        
        Verifies that the stats endpoint returns server uptime
        and other performance metrics.
        """
        r = requests.get(f"{BASE_URL}/api/stats", timeout=10)
        assert r.status_code == 200
        assert "uptime" in r.json()

    def test_05_install_package(self):
        """Test package installation functionality.
        
        Verifies that Python packages can be installed in the Pyodide
        environment by attempting to install beautifulsoup4.
        """
        r = requests.post(
            f"{BASE_URL}/api/install-package",
            json={"package": "beautifulsoup4"},
            timeout=120,
        )
        assert r.status_code == 200
        assert "success" in r.json()

    def test_06_list_packages(self):
        """Test listing installed packages in the Pyodide environment.
        
        Verifies that the packages endpoint returns a list of installed
        packages with correct structure including package names and count.
        """
        r = requests.get(f"{BASE_URL}/api/packages", timeout=10)
        assert r.status_code == 200
        payload = r.json()
        assert "data" in payload
        result = payload["data"]
        if isinstance(result, dict) and "result" in result:
            result = result["result"]
        assert result is not None, "Packages result should not be null"
        assert isinstance(result, dict), "Packages result should be a dictionary"
        assert "installed_packages" in result
        assert "total_packages" in result
        assert isinstance(result["installed_packages"], list)
        assert result["total_packages"] > 0, "Should have packages installed"

    def test_07_execute(self):
        """Test basic Python code execution through the execute endpoint.
        
        Verifies that Python code can be executed and returns the expected
        result from the last expression in JSON format.
        """
        exec_code = '''"""
basic demonstration
"""
name = "World"
f"Hello {name}"
'''
        r = requests.post(
            f"{BASE_URL}/api/execute", json={"code": exec_code}, timeout=30
        )
        assert r.status_code == 200
        assert r.json().get("result") == "Hello World"

    def test_08_execute_raw(self):
        """Test Python code execution through the execute-raw endpoint.
        
        Verifies that Python code can be executed with plain text input
        and returns the expected result from the last expression.
        """
        raw_code = '''"""
raw execution sample
"""
x = 3
f"{x + 3}"
'''
        r = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=raw_code,
            headers={"Content-Type": "text/plain"},
            timeout=30,
        )
        assert r.status_code == 200
        assert r.json().get("result") == "6"

    def test_09_upload_csv(self):
        """Test CSV file upload functionality.
        
        Creates a temporary CSV file and uploads it to the server,
        verifying the upload response structure and storing file
        references for use in subsequent tests.
        """
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write("name,value,category\nitem1,1,A\nitem2,2,B\nitem3,3,C\n")
            tmp_path = tmp.name
        with open(tmp_path, "rb") as fh:
            r = requests.post(
                f"{BASE_URL}/api/upload-csv",
                files={"file": ("data.csv", fh, "text/csv")},
                timeout=30,
            )
        os.unlink(tmp_path)
        assert r.status_code == 200
        upload_data = r.json()
        assert upload_data.get("success") is True
        assert "data" in upload_data
        assert "file" in upload_data.get("data", {})
        file = upload_data["data"]["file"]
        self.__class__.pyodide_name = file["sanitizedOriginal"]
        self.__class__.server_filename = Path(file["vfsPath"]).as_posix()

    def test_10_list_uploaded_files(self):
        """Test listing uploaded files on the server.
        
        Verifies that the uploaded files endpoint returns a list
        containing the previously uploaded CSV file.
        """
        r = requests.get(f"{BASE_URL}/api/uploaded-files", timeout=10)
        assert r.status_code == 200
        upload_files = r.json()
        assert upload_files.get("success") is True
        assert "data" in upload_files
        assert "files" in upload_files.get("data", {})
        files = upload_files["data"]["files"]
        uploaded_names = [f["filename"] for f in files]
        assert self.__class__.pyodide_name in uploaded_names

    def test_11_file_info(self):
        """Test retrieving information about a specific uploaded file.
        
        Verifies that the file info endpoint returns correct existence
        status and metadata for the previously uploaded file.
        """
        r = requests.get(
            f"{BASE_URL}/api/file-info/{self.__class__.pyodide_name}", timeout=10
        )
        assert r.status_code == 200
        info = r.json()
        # For the pyodide filename, only the pyodide file should exist
        assert info["data"]["exists"] is True  # Upload uses different name
        assert info['data']["pyodideFile"]["exists"] is True
        assert info['data']["filename"] == self.__class__.pyodide_name

    def test_12_execute_with_uploaded_file(self):
        """Test executing Python code that reads the uploaded CSV file.
        
        Verifies that the uploaded file can be accessed from within
        Python code execution, and that pandas can read and process it.
        """
        code = f'''"""
read csv and compute sum using pandas
"""
import pandas as pd
filename = "{Path(self.__class__.server_filename).as_posix() if self.__class__.server_filename else ''}"
df = pd.read_csv(filename)  # type: ignore
total = df["value"].sum()
# Verify we have multiple columns by checking column names
columns = list(df.columns)
f"sum={{total}}, columns={{columns}}"
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=30)
        assert r.status_code == 200
        result = r.json().get("result")
        assert "sum=6" in result
        assert "columns=['name', 'value', 'category']" in result

    def test_13_list_pyodide_files(self):
        """Test listing files available in the Pyodide filesystem.
        
        Verifies that the pyodide-files endpoint returns a list
        containing the uploaded file accessible from Python.
        """
        r = requests.get(f"{BASE_URL}/api/pyodide-files", timeout=10)
        assert r.status_code == 200
        py_files = [f["name"] for f in r.json().get("result", {}).get("files", [])]
        assert self.__class__.pyodide_name in py_files

    def test_14_delete_pyodide_file(self):
        """Test deleting a file from the Pyodide filesystem.
        
        Verifies that files can be deleted from the Pyodide environment
        and that the deletion operation returns success.
        """
        r = requests.delete(
            f"{BASE_URL}/api/pyodide-files/{self.__class__.pyodide_name}", timeout=10
        )
        assert r.status_code == 200
        assert r.json().get("success") is True

    def test_15_delete_uploaded_file(self):
        """Test deleting an uploaded file from the server.
        
        Verifies that uploaded files can be deleted from server storage
        and that the deletion operation returns success.
        """
        r = requests.delete(
            f"{BASE_URL}/api/uploaded-files/{self.__class__.server_filename}",
            timeout=10,
        )
        assert r.status_code == 200
        assert r.json().get("success") is True

    def test_16_reset(self):
        """Test resetting the Pyodide environment.
        
        Verifies that the reset endpoint clears the Pyodide state
        and returns the environment to its initial condition.
        """
        r = requests.post(f"{BASE_URL}/api/reset", timeout=10)
        assert r.status_code == 200
        assert r.json().get("success") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

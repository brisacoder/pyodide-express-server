import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

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


class APITestCase(unittest.TestCase):
    """Exercise API endpoints and report results per test."""

    server_filename: str | None = None
    pyodide_name: str | None = None

    @classmethod
    def setUpClass(cls):
        # Use existing server - just wait for it to be ready
        try:
            wait_for_server(f"{BASE_URL}/health")
            cls.server = None  # No server to manage
        except RuntimeError:
            # If no server is running, start one
            cls.server = subprocess.Popen(
                ["node", "src/server.js"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            wait_for_server(f"{BASE_URL}/health")

    @classmethod
    def tearDownClass(cls):
        # Only terminate if we started the server
        if cls.server is not None:
            cls.server.terminate()
            try:
                cls.server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.server.kill()

    def test_01_health(self):
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("status"), "ok")

    def test_02_status(self):
        r = requests.get(f"{BASE_URL}/api/status", timeout=10)
        self.assertEqual(r.status_code, 200)
        payload = r.json()
        self.assertIn("data", payload)
        self.assertIn("isReady", payload["data"])

    def test_03_pyodide_health(self):
        r = requests.get(f"{BASE_URL}/api/health", timeout=10)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))

    def test_04_stats(self):
        r = requests.get(f"{BASE_URL}/api/stats", timeout=10)
        self.assertEqual(r.status_code, 200)
        self.assertIn("uptime", r.json())

    def test_05_install_package(self):
        r = requests.post(
            f"{BASE_URL}/api/install-package",
            json={"package": "beautifulsoup4"},
            timeout=120,
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn("success", r.json())

    def test_06_list_packages(self):
        r = requests.get(f"{BASE_URL}/api/packages", timeout=10)
        self.assertEqual(r.status_code, 200)
        payload = r.json()
        self.assertIn("data", payload)
        result = payload["data"]
        if isinstance(result, dict) and "result" in result:
            result = result["result"]
        self.assertIsNotNone(result, "Packages result should not be null")
        self.assertIsInstance(result, dict, "Packages result should be a dictionary")
        self.assertIn("installed_packages", result)
        self.assertIn("total_packages", result)
        self.assertIsInstance(result["installed_packages"], list)
        self.assertGreater(
            result["total_packages"], 0, "Should have packages installed"
        )

    def test_07_execute(self):
        exec_code = '''"""
basic demonstration
"""
name = "World"
f"Hello {name}"
'''
        r = requests.post(
            f"{BASE_URL}/api/execute", json={"code": exec_code}, timeout=30
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("result"), "Hello World")

    def test_08_execute_raw(self):
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
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("result"), "6")

    def test_09_upload_csv(self):
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
        self.assertEqual(r.status_code, 200)
        upload_data = r.json()
        self.assertTrue(upload_data.get("success"))
        self.assertIn("data", upload_data)
        self.assertIn("file", upload_data.get("data", {}))
        file = upload_data["data"]["file"]
        self.__class__.pyodide_name = file["sanitizedOriginal"]
        self.__class__.server_filename = Path(file["vfsPath"]).as_posix()

    def test_10_list_uploaded_files(self):
        r = requests.get(f"{BASE_URL}/api/uploaded-files", timeout=10)
        self.assertEqual(r.status_code, 200)
        upload_files = r.json()
        self.assertTrue(upload_files.get("success"))
        self.assertIn("data", upload_files)
        self.assertIn("files", upload_files.get("data", {}))
        files = upload_files["data"]["files"]
        uploaded_names = [f["filename"] for f in files]
        self.assertIn(self.__class__.pyodide_name, uploaded_names)

    def test_11_file_info(self):
        r = requests.get(
            f"{BASE_URL}/api/file-info/{self.__class__.pyodide_name}", timeout=10
        )
        self.assertEqual(r.status_code, 200)
        info = r.json()
        # For the pyodide filename, only the pyodide file should exist
        self.assertTrue(info["data"]["exists"])  # Upload uses different name
        self.assertTrue(info['data']["pyodideFile"]["exists"])
        self.assertEqual(info['data']["filename"], self.__class__.pyodide_name)

    def test_12_execute_with_uploaded_file(self):
        code = f'''"""
read csv and compute sum using pandas
"""
import pandas as pd
df = pd.read_csv("{Path(self.__class__.server_filename).as_posix()}") # type: ignore
total = df["value"].sum()
# Verify we have multiple columns by checking column names
columns = list(df.columns)
f"sum={{total}}, columns={{columns}}"
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        result = r.json().get("result")
        self.assertIn("sum=6", result)
        self.assertIn("columns=['name', 'value', 'category']", result)

    def test_13_list_pyodide_files(self):
        r = requests.get(f"{BASE_URL}/api/pyodide-files", timeout=10)
        self.assertEqual(r.status_code, 200)
        py_files = [f["name"] for f in r.json().get("result", {}).get("files", [])]
        self.assertIn(self.__class__.pyodide_name, py_files)

    def test_14_delete_pyodide_file(self):
        r = requests.delete(
            f"{BASE_URL}/api/pyodide-files/{self.__class__.pyodide_name}", timeout=10
        )
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))

    def test_15_delete_uploaded_file(self):
        r = requests.delete(
            f"{BASE_URL}/api/uploaded-files/{self.__class__.server_filename}",
            timeout=10,
        )
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))

    def test_16_reset(self):
        r = requests.post(f"{BASE_URL}/api/reset", timeout=10)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))


if __name__ == "__main__":
    unittest.main()

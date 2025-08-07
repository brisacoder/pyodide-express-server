import os
import tempfile
import time
import subprocess
import requests
import unittest

BASE_URL = "http://localhost:3000"

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


class ErrorHandlingTestCase(unittest.TestCase):
    """Test error handling and edge cases for API endpoints."""

    @classmethod
    def setUpClass(cls):
        # Use existing server - just wait for it to be ready
        try:
            wait_for_server(f"{BASE_URL}/health")
            cls.server = None  # No server to manage
        except RuntimeError:
            # If no server is running, start one
            cls.server = subprocess.Popen(["node", "src/server.js"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

    # ===== Code Execution Error Tests =====
    
    def test_execute_empty_code(self):
        """Test execution with empty code"""
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": ""})
        self.assertEqual(r.status_code, 400)
        response = r.json()
        self.assertFalse(response.get("success"))
        self.assertIn("empty", response.get("error", "").lower())

    def test_execute_whitespace_only_code(self):
        """Test execution with only whitespace"""
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "   \n\t  \n  "})
        self.assertEqual(r.status_code, 400)
        response = r.json()
        self.assertFalse(response.get("success"))

    def test_execute_no_code_field(self):
        """Test execution without code field"""
        r = requests.post(f"{BASE_URL}/api/execute", json={})
        self.assertEqual(r.status_code, 400)
        response = r.json()
        self.assertFalse(response.get("success"))
        self.assertIn("code", response.get("error", "").lower())

    def test_execute_invalid_code_type(self):
        """Test execution with non-string code"""
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": 123})
        self.assertEqual(r.status_code, 400)
        response = r.json()
        self.assertFalse(response.get("success"))
        self.assertIn("string", response.get("error", "").lower())

    def test_execute_syntax_error(self):
        """Test execution with Python syntax error"""
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "if True\n    print('missing colon')"})
        self.assertEqual(r.status_code, 200)  # Should return 200 but with error in result
        response = r.json()
        self.assertFalse(response.get("success"))
        self.assertIn("syntax", response.get("error", "").lower())

    def test_execute_runtime_error(self):
        """Test execution with runtime error"""
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "x = 1 / 0"})
        self.assertEqual(r.status_code, 200)
        response = r.json()
        self.assertFalse(response.get("success"))
        self.assertIn("division", response.get("error", "").lower())

    def test_execute_very_long_code(self):
        """Test execution with code exceeding length limit"""
        long_code = "x = 1\n" * 50000  # Should exceed 100KB limit
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": long_code})
        self.assertEqual(r.status_code, 400)
        response = r.json()
        self.assertFalse(response.get("success"))
        self.assertIn("too long", response.get("error", "").lower())

    def test_execute_invalid_context(self):
        """Test execution with invalid context parameter"""
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "print('test')", "context": "invalid"})
        self.assertEqual(r.status_code, 400)
        response = r.json()
        self.assertFalse(response.get("success"))
        self.assertIn("context", response.get("error", "").lower())

    def test_execute_invalid_timeout(self):
        """Test execution with invalid timeout"""
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "print('test')", "timeout": -1})
        self.assertEqual(r.status_code, 400)
        response = r.json()
        self.assertFalse(response.get("success"))

    def test_execute_excessive_timeout(self):
        """Test execution with timeout exceeding limit"""
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "print('test')", "timeout": 400000})
        self.assertEqual(r.status_code, 400)
        response = r.json()
        self.assertFalse(response.get("success"))

    # ===== Package Installation Error Tests =====
    
    def test_install_empty_package(self):
        """Test installing package with empty name"""
        r = requests.post(f"{BASE_URL}/api/install-package", json={"package": ""})
        self.assertEqual(r.status_code, 400)
        response = r.json()
        self.assertFalse(response.get("success"))

    def test_install_no_package_field(self):
        """Test installing without package field"""
        r = requests.post(f"{BASE_URL}/api/install-package", json={})
        self.assertEqual(r.status_code, 400)
        response = r.json()
        self.assertFalse(response.get("success"))

    def test_install_invalid_package_type(self):
        """Test installing with non-string package name"""
        r = requests.post(f"{BASE_URL}/api/install-package", json={"package": 123})
        self.assertEqual(r.status_code, 400)
        response = r.json()
        self.assertFalse(response.get("success"))

    def test_install_invalid_package_name(self):
        """Test installing package with invalid characters"""
        r = requests.post(f"{BASE_URL}/api/install-package", json={"package": "invalid@package!"})
        self.assertEqual(r.status_code, 400)
        response = r.json()
        self.assertFalse(response.get("success"))

    def test_install_blocked_package(self):
        """Test installing blocked package"""
        r = requests.post(f"{BASE_URL}/api/install-package", json={"package": "os"})
        self.assertEqual(r.status_code, 403)
        response = r.json()
        self.assertFalse(response.get("success"))

    def test_install_nonexistent_package(self):
        """Test installing package that doesn't exist"""
        r = requests.post(f"{BASE_URL}/api/install-package", json={"package": "nonexistent-package-xyz123"})
        self.assertEqual(r.status_code, 200)  # Should return 200 but with error in result
        response = r.json()
        self.assertFalse(response.get("success"))

    # ===== File Operations Error Tests =====
    
    def test_upload_no_file(self):
        """Test upload without file"""
        r = requests.post(f"{BASE_URL}/api/upload-csv")
        self.assertEqual(r.status_code, 400)

    def test_upload_invalid_file_type(self):
        """Test upload with invalid file type"""
        with tempfile.NamedTemporaryFile("w", suffix=".exe", delete=False) as tmp:
            tmp.write("invalid content")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, "rb") as fh:
                r = requests.post(
                    f"{BASE_URL}/api/upload-csv",
                    files={"csvFile": ("malware.exe", fh, "application/octet-stream")},
                )
            self.assertEqual(r.status_code, 400)
        finally:
            os.unlink(tmp_path)

    def test_upload_oversized_file(self):
        """Test upload with file exceeding size limit"""
        # Create a large file (assuming 50MB limit)
        large_content = "x,y\n" + "1,2\n" * 1000000  # Should be large enough
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write(large_content)
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, "rb") as fh:
                r = requests.post(
                    f"{BASE_URL}/api/upload-csv",
                    files={"csvFile": ("large.csv", fh, "text/csv")},
                )
            # Should either be 400 (rejected by validation) or 413 (entity too large)
            self.assertIn(r.status_code, [400, 413])
        finally:
            os.unlink(tmp_path)

    def test_delete_nonexistent_uploaded_file(self):
        """Test deleting uploaded file that doesn't exist"""
        r = requests.delete(f"{BASE_URL}/api/uploaded-files/nonexistent.csv")
        self.assertEqual(r.status_code, 404)
        response = r.json()
        self.assertFalse(response.get("success"))

    def test_delete_nonexistent_pyodide_file(self):
        """Test deleting pyodide file that doesn't exist"""
        r = requests.delete(f"{BASE_URL}/api/pyodide-files/nonexistent.csv")
        self.assertEqual(r.status_code, 404)
        response = r.json()
        self.assertFalse(response.get("success"))

    def test_file_info_nonexistent_file(self):
        """Test getting info for nonexistent file"""
        r = requests.get(f"{BASE_URL}/api/file-info/nonexistent.csv")
        self.assertEqual(r.status_code, 200)  # Should return info showing file doesn't exist
        response = r.json()
        self.assertFalse(response["uploadedFile"]["exists"])
        self.assertFalse(response["pyodideFile"]["exists"])

    def test_delete_file_with_path_traversal(self):
        """Test deleting file with path traversal attempt"""
        r = requests.delete(f"{BASE_URL}/api/uploaded-files/../../../etc/passwd")
        self.assertEqual(r.status_code, 400)
        response = r.json()
        self.assertFalse(response.get("success"))
        self.assertIn("invalid", response.get("error", "").lower())

    # ===== Endpoint Not Found Tests =====
    
    def test_invalid_endpoint(self):
        """Test requesting non-existent endpoint"""
        r = requests.get(f"{BASE_URL}/api/nonexistent")
        self.assertEqual(r.status_code, 404)

    def test_invalid_method(self):
        """Test using wrong HTTP method"""
        r = requests.delete(f"{BASE_URL}/api/execute")  # Should be POST
        self.assertEqual(r.status_code, 404)

    # ===== Malformed Request Tests =====
    
    def test_malformed_json(self):
        """Test sending malformed JSON"""
        r = requests.post(
            f"{BASE_URL}/api/execute",
            data='{"code": "print(\'test\')"',  # Missing closing brace
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(r.status_code, 400)

    def test_invalid_content_type(self):
        """Test sending wrong content type"""
        r = requests.post(
            f"{BASE_URL}/api/execute",
            data="print('test')",
            headers={"Content-Type": "text/plain"}  # Should be application/json
        )
        self.assertEqual(r.status_code, 400)

    # ===== Large Payload Tests =====
    
    def test_very_large_json_payload(self):
        """Test sending extremely large JSON payload"""
        large_context = {f"key_{i}": f"value_{i}" * 1000 for i in range(100)}
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "print('test')", "context": large_context})
        self.assertEqual(r.status_code, 400)  # Should reject large context


if __name__ == "__main__":
    unittest.main()

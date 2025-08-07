import os
import tempfile
import time
import subprocess
import requests
import unittest

BASE_URL = "http://localhost:3000"


class SecurityTestCase(unittest.TestCase):
    """Test security aspects and potential vulnerabilities."""

    @classmethod
    def setUpClass(cls):
        # Start the server in a subprocess
        cls.server = subprocess.Popen(["node", "src/server.js"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for server to be ready
        start = time.time()
        while time.time() - start < 120:
            try:
                r = requests.get(f"{BASE_URL}/health", timeout=5)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Server did not start in time")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.server.kill()

    # ===== Path Traversal Tests =====
    
    def test_path_traversal_file_deletion(self):
        """Test path traversal attempts in file deletion"""
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "../../../../etc/shadow",
            "..\\..\\..\\..\\..\\..",
            "...//...//...//etc//passwd",
        ]
        
        for path in dangerous_paths:
            r = requests.delete(f"{BASE_URL}/api/uploaded-files/{path}", timeout=10)
            # Should reject with 400 (bad request) due to path validation
            self.assertEqual(r.status_code, 400)
            response = r.json()
            self.assertFalse(response.get("success"))
            self.assertIn("invalid", response.get("error", "").lower())

    def test_file_info_path_traversal(self):
        """Test path traversal in file info endpoint"""
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\windows\\system.ini",
            "....//....//etc//hosts",
        ]
        
        for path in dangerous_paths:
            r = requests.get(f"{BASE_URL}/api/file-info/{path}", timeout=10)
            # Should either reject or safely handle the request
            self.assertIn(r.status_code, [400, 404, 200])
            if r.status_code == 200:
                # If it returns 200, should show file doesn't exist
                response = r.json()
                self.assertFalse(response["uploadedFile"]["exists"])
                self.assertFalse(response["pyodideFile"]["exists"])

    # ===== Code Injection Tests =====
    
    def test_python_code_restrictions(self):
        """Test that dangerous Python operations are properly handled"""
        dangerous_codes = [
            # File system access attempts
            "import os; os.listdir('/')",
            "open('/etc/passwd', 'r').read()",
            "with open('../../sensitive_file.txt') as f: f.read()",
            
            # System command attempts
            "import subprocess; subprocess.run(['ls', '-la'])",
            "import os; os.system('whoami')",
            "__import__('os').system('id')",
            
            # Network access attempts (might be blocked by Pyodide)
            "import urllib.request; urllib.request.urlopen('http://malicious.com')",
            "import socket; socket.socket().connect(('external.com', 80))",
            
            # Exit attempts
            "import sys; sys.exit(1)",
            "exit()",
            "quit()",
            
            # Eval/exec attempts
            "exec('print(\"injected\")')",
            "eval('1+1')",
        ]
        
        for code in dangerous_codes:
            r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=30)
            # Most should return 200 but with execution failure (blocked by Pyodide sandbox)
            # Some might be caught by validation (especially dangerous patterns)
            self.assertIn(r.status_code, [200, 400])
            
            if r.status_code == 200:
                response = r.json()
                # Should either fail execution or be sandboxed
                if not response.get("success"):
                    # Execution failed (good - sandboxed)
                    self.assertIn("error", response)
                else:
                    # If it succeeded, result should be safe/limited
                    result = response.get("result", "")
                    # Shouldn't contain sensitive system information
                    self.assertNotIn("/etc/passwd", str(result))
                    self.assertNotIn("root:", str(result))

    def test_package_installation_restrictions(self):
        """Test that dangerous packages are blocked"""
        blocked_packages = [
            "os",
            "subprocess", 
            "socket",
            "urllib3",
            "requests-oauthlib"
        ]
        
        for package in blocked_packages:
            r = requests.post(f"{BASE_URL}/api/install-package", json={"package": package}, timeout=30)
            self.assertEqual(r.status_code, 403)
            response = r.json()
            self.assertFalse(response.get("success"))
            self.assertIn("not allowed", response.get("error", "").lower())

    # ===== Input Validation Tests =====
    
    def test_malicious_filenames(self):
        """Test handling of malicious filenames in uploads"""
        malicious_names = [
            "../../../etc/passwd.csv",
            "..\\..\\windows\\system.ini.csv",
            "file\x00.csv",  # Null byte injection
            "file\n.csv",    # Newline injection
            "file\r.csv",    # Carriage return injection
            "very_long_filename_" + "x" * 1000 + ".csv",  # Extremely long filename
        ]
        
        for filename in malicious_names:
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                tmp.write("col1,col2\n1,2\n")
                tmp_path = tmp.name
            
            try:
                with open(tmp_path, "rb") as fh:
                    r = requests.post(
                        f"{BASE_URL}/api/upload-csv",
                        files={"csvFile": (filename, fh, "text/csv")},
                        timeout=30
                    )
                
                # Should either reject the filename or sanitize it safely
                if r.status_code == 200:
                    # If accepted, verify the filename was sanitized
                    response = r.json()
                    stored_name = response["file"]["pyodideFilename"]
                    # Should not contain path traversal components
                    self.assertNotIn("..", stored_name)
                    self.assertNotIn("/", stored_name)
                    self.assertNotIn("\\", stored_name)
                    self.assertNotIn("\x00", stored_name)
                    
                    # Clean up if file was created
                    try:
                        server_filename = os.path.basename(response["file"]["tempPath"])
                        requests.delete(f"{BASE_URL}/api/uploaded-files/{server_filename}", timeout=10)
                        requests.delete(f"{BASE_URL}/api/pyodide-files/{stored_name}", timeout=10)
                    except:
                        pass
                        
            finally:
                os.unlink(tmp_path)

    def test_json_injection(self):
        """Test JSON injection attempts"""
        injection_attempts = [
            '{"code": "print(\\"test\\")", "extra": {"__proto__": {"injected": true}}}',
            '{"code": "print(\\"test\\")", "constructor": {"prototype": {"injected": true}}}',
        ]
        
        for json_str in injection_attempts:
            r = requests.post(
                f"{BASE_URL}/api/execute",
                data=json_str,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            # Should either parse safely or reject
            if r.status_code == 200:
                response = r.json()
                # Should not have prototype pollution
                self.assertNotIn("injected", str(response))

    # ===== Rate Limiting Tests =====
    
    def test_rate_limiting_behavior(self):
        """Test that rate limiting works as expected"""
        # Send many requests quickly to trigger rate limiting
        responses = []
        for i in range(10):
            r = requests.post(f"{BASE_URL}/api/execute", json={"code": f"print({i})"}, timeout=30)
            responses.append(r.status_code)
        
        # All should succeed if rate limit is reasonable
        # If rate limiting is aggressive, some might return 429
        success_count = sum(1 for status in responses if status == 200)
        rate_limited_count = sum(1 for status in responses if status == 429)
        
        # Should have mostly successful requests for normal usage
        self.assertGreaterEqual(success_count, 5)  # At least half should succeed

    # ===== Content Type Security =====
    
    def test_content_type_validation(self):
        """Test proper content type validation"""
        # Try to send malicious content with wrong content type
        malicious_payloads = [
            ('<script>alert("xss")</script>', 'text/html'),
            ('<?xml version="1.0"?><!DOCTYPE root [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><root>&xxe;</root>', 'text/xml'),
            ('javascript:alert("xss")', 'text/javascript'),
        ]
        
        for payload, content_type in malicious_payloads:
            r = requests.post(
                f"{BASE_URL}/api/execute",
                data=payload,
                headers={"Content-Type": content_type},
                timeout=30
            )
            # Should reject non-JSON content types for JSON endpoints
            self.assertIn(r.status_code, [400, 415])  # Bad Request or Unsupported Media Type

    # ===== File Upload Security =====
    
    def test_malicious_file_content(self):
        """Test handling of files with malicious content"""
        malicious_contents = [
            # CSV with embedded JavaScript
            'name,script\ntest,"<script>alert(1)</script>"\n',
            # CSV with SQL injection attempts
            'id,name\n1,"Robert\'); DROP TABLE users; --"\n',
            # CSV with extremely long fields
            'name,value\n' + '"' + 'A' * 100000 + '",1\n',
            # CSV with many columns (potential DoS)
            ','.join([f'col{i}' for i in range(1000)]) + '\n' + ','.join(['1'] * 1000) + '\n',
        ]
        
        for content in malicious_contents:
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                with open(tmp_path, "rb") as fh:
                    r = requests.post(
                        f"{BASE_URL}/api/upload-csv",
                        files={"csvFile": ("malicious.csv", fh, "text/csv")},
                        timeout=30
                    )
                
                # Should either reject or handle safely
                if r.status_code == 200:
                    response = r.json()
                    # If accepted, try to process it safely
                    pyodide_name = response["file"]["pyodideFilename"]
                    
                    code = f'''
import pandas as pd
try:
    df = pd.read_csv("{pyodide_name}")
    "processed_safely"
except Exception as e:
    f"error: {{type(e).__name__}}"
'''
                    r2 = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=30)
                    self.assertEqual(r2.status_code, 200)
                    
                    # Clean up
                    try:
                        server_filename = os.path.basename(response["file"]["tempPath"])
                        requests.delete(f"{BASE_URL}/api/uploaded-files/{server_filename}", timeout=10)
                        requests.delete(f"{BASE_URL}/api/pyodide-files/{pyodide_name}", timeout=10)
                    except:
                        pass
                        
            finally:
                os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()

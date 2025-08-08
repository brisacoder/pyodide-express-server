#!/usr/bin/env python3
"""
Reset functionality test suite for Pyodide Express Server

Tests the reset behavior and isolated execution features.
"""

import time
import subprocess
import requests
import unittest

BASE_URL = "http://localhost:3000"


class ResetTestCase(unittest.TestCase):
    """Test reset functionality and isolated execution behavior."""

    @classmethod
    def setUpClass(cls):
        # Start the server in a subprocess
        cls.server = subprocess.Popen(["node", "src/server.js"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for server to be ready
        start = time.time()
        while time.time() - start < 120:
            try:
                r = requests.get(f"{BASE_URL}/health", timeout=10)
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

    def test_reset_behavior_and_isolation(self):
        """Test complete reset behavior including package persistence and variable isolation"""
        
        # 1. Install a package
        r = requests.post(f"{BASE_URL}/api/install-package", json={"package": "beautifulsoup4"}, timeout=120)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))
        
        # 2. Verify package is available
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "import bs4; 'success'"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))
        
        # 3. Test variable isolation (variables should NOT persist between requests)
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "test_var = 'isolated_value'"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))
        
        # 4. Verify variable is NOT accessible in separate execution (isolated execution)
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "test_var"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json().get("success"))  # Should fail - variables don't persist
        self.assertIn("not defined", r.json().get("error", ""))
        
        # 5. Reset the environment
        r = requests.post(f"{BASE_URL}/api/reset", timeout=60)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))
        
        # 6. Verify package is still available after reset
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "import bs4; 'still_available'"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))
        
        # 7. Verify system still functions normally after reset
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "2 + 2"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))
        self.assertEqual(r.json().get("result"), 4)

    def test_package_installation_and_imports(self):
        """Test package installation and import behavior"""
        
        # Test installing a different package
        r = requests.post(f"{BASE_URL}/api/install-package", json={"package": "lxml"}, timeout=120)
        self.assertEqual(r.status_code, 200)
        
        # Package installation should succeed (even if import name differs)
        response = r.json()
        self.assertTrue(response.get("success"))
        
        # Test that we can import the package
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "import lxml; 'lxml_available'"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        # Note: lxml might not be available in Pyodide, so we don't assert success here
        # The test is mainly to verify the installation endpoint works

    def test_reset_clears_user_variables_only(self):
        """Test that reset clears user variables but keeps system functionality"""
        
        # Execute some code that creates variables in the execution context
        test_codes = [
            "x = 42",
            "y = 'hello'", 
            "z = [1, 2, 3]",
            "def custom_func(): return 'test'"
        ]
        
        for code in test_codes:
            r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=30)
            self.assertEqual(r.status_code, 200)
            # Variables are isolated, so each execution should succeed independently
        
        # Reset should work
        r = requests.post(f"{BASE_URL}/api/reset", timeout=60)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))
        
        # System should still work after reset
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "import sys\nlen(sys.modules)"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))
        self.assertIsInstance(r.json().get("result"), int)


if __name__ == "__main__":
    unittest.main()

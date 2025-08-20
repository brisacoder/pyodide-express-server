#!/usr/bin/env python3
"""
Simple filesystem persistence test that works with a running server.

This test assumes the server is already running and tests basic file operations.
"""

import unittest
import requests
import time
from pathlib import Path


BASE_URL = "http://localhost:3000"


class SimpleFilesystemTestCase(unittest.TestCase):
    """Simple filesystem tests that work with an existing server."""

    def test_file_creation_and_local_persistence(self):
        """Test that files created in mounted directories appear locally."""
        
        # Create unique filename
        timestamp = int(time.time() * 1000)
        filename = f"simple_test_{timestamp}.txt"
        
        # Python code to create files
        create_code = f'''
from pathlib import Path

# Create file in plots directory (should appear locally)  
plots_file = Path("/plots/matplotlib/{filename}")
plots_file.parent.mkdir(parents=True, exist_ok=True)
plots_file.write_text("Test content from Pyodide")

# Return result dictionary
result = {{
    "filename": "{filename}",
    "plots_file_exists": plots_file.exists(),
    "file_content": plots_file.read_text(),
    "timestamp": {timestamp}
}}
result
'''
        
        # Execute code via RAW API to avoid f-string formatting issues
        response = requests.post(f"{BASE_URL}/api/execute-raw", 
                               data=create_code,
                               headers={"Content-Type": "text/plain"})
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        self.assertTrue(result.get("success"), f"API call failed: {result}")
        
        data = result.get("result")
        self.assertIsNotNone(data, f"No result data returned: {result}")
        self.assertTrue(data["plots_file_exists"], "File should exist in Pyodide")
        self.assertEqual(data["file_content"], "Test content from Pyodide")
        
        # Check if file appears in local filesystem
        local_file = Path(__file__).parent.parent / "plots" / "matplotlib" / filename
        self.assertTrue(local_file.exists(), "File should appear in local filesystem")
        
        # Verify content matches
        local_content = local_file.read_text()
        self.assertEqual(local_content, "Test content from Pyodide")
        
        # Clean up
        local_file.unlink()
        
        print(f"✅ Test passed: File {filename} created and persisted locally")

    def test_tmp_filesystem_behavior(self):
        """Test behavior of /tmp filesystem (virtual, non-persistent)."""
        
        timestamp = int(time.time() * 1000)
        filename = f"tmp_test_{timestamp}.txt"
        
        create_code = f'''
from pathlib import Path

# Create file in /tmp (virtual filesystem)
tmp_file = Path("/tmp/{filename}")
tmp_file.write_text("Temporary content")

# Return result dictionary
result = {{
    "tmp_exists": tmp_file.exists(),
    "tmp_content": tmp_file.read_text(),
    "filename": "{filename}"
}}
result
'''
        
        response = requests.post(f"{BASE_URL}/api/execute-raw",
                               data=create_code,
                               headers={"Content-Type": "text/plain"})
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        self.assertTrue(result.get("success"))
        
        data = result.get("result")
        self.assertIsNotNone(data, f"No result data returned: {result}")
        self.assertTrue(data["tmp_exists"], "/tmp file should exist in virtual filesystem")
        self.assertEqual(data["tmp_content"], "Temporary content")
        
        # /tmp files should NOT appear in local filesystem
        local_tmp = Path(__file__).parent.parent / "tmp" / filename
        self.assertFalse(local_tmp.exists(), "/tmp files should not appear locally")
        
        print(f"✅ Test passed: /tmp file exists in Pyodide but not locally")

    def test_server_health(self):
        """Test that server is responding."""
        response = requests.get(f"{BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        
        health_data = response.json()
        self.assertEqual(health_data["status"], "ok")
        self.assertTrue(health_data["pyodide"]["isReady"])
        
        print("✅ Server health check passed")


if __name__ == "__main__":
    print("🧪 Simple Filesystem Persistence Tests")
    print("=" * 50)
    print("NOTE: These tests require a running server on localhost:3000")
    print("Start server with: node src/server.js")
    print("=" * 50)
    
    # Quick server check
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running - starting tests...")
            unittest.main(verbosity=2)
        else:
            print("❌ Server responded but not healthy")
    except requests.exceptions.RequestException:
        print("❌ Server is not running. Please start it first:")
        print("   node src/server.js")
        print("   Then run this test again.")

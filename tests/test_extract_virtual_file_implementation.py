import time
import unittest
import requests

BASE_URL = "http://localhost:3000"


def wait_for_server(url: str, timeout: int = 180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"Server at {url} did not start in time")


class TestExtractVirtualFileImplementationTestCase(unittest.TestCase):
    """Test to verify the extract virtual file implementation bug."""

    @classmethod
    def setUpClass(cls):
        try:
            wait_for_server(f"{BASE_URL}/health", timeout=30)
            cls.server = None
        except RuntimeError:
            raise unittest.SkipTest("Server is not running on localhost:3000")

    def test_python_code_syntax_in_extract_virtual_file(self):
        """Test the Python syntax used in extractVirtualFile method."""
        # This simulates the exact Python code used in extractVirtualFile
        code = '''
import os
import base64

virtual_path = '/plots/matplotlib/debug_filesystem.png'
try:
    # Check if file exists in virtual filesystem
    if os.path.exists(virtual_path):
        # Read the file content from virtual filesystem
        with open(virtual_path, 'rb') as f:
            file_content = f.read()
        
        # Return file content as base64 for transfer
        result = {
            'success': True,
            'file_exists': True,
            'content_b64': base64.b64encode(file_content).decode('utf-8'),
            'file_size': len(file_content)
        }
    else:
        result = {
            'success': False,
            'file_exists': False,
            'error': f'File {virtual_path} does not exist in virtual filesystem'
        }
except Exception as e:
    result = {
        'success': False,
        'file_exists': False,
        'error': str(e)
    }

result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        
        print(f"Extract virtual file test result: {result}")
        
        if result.get("success"):
            self.assertTrue(result.get("file_exists"), "File should exist")
            self.assertIn("content_b64", result, "Should have base64 content")
            self.assertGreater(result.get("file_size", 0), 0, "File should have content")
            print(f"✅ Virtual file extraction working - file size: {result.get('file_size')} bytes")
        else:
            print(f"❌ Virtual file extraction failed: {result.get('error')}")
            self.fail(f"Virtual file extraction failed: {result.get('error')}")


if __name__ == "__main__":
    unittest.main()

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


class JavaScriptModuleTestCase(unittest.TestCase):
    """Test JavaScript module access and Pyodide functionality."""

    @classmethod
    def setUpClass(cls):
        # Check if server is already running
        try:
            wait_for_server(f"{BASE_URL}/health", timeout=30)
            cls.server = None
        except RuntimeError:
            raise unittest.SkipTest("Server is not running on localhost:3000")

    @classmethod
    def tearDownClass(cls):
        # We don't start our own server, so no cleanup needed
        pass

    def test_js_module_available(self):
        """Test that the js module is available in Pyodide."""
        code = '''
import js
print("js module imported successfully")
result = {"js_module_available": True, "js_type": str(type(js))}
result
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        self.assertTrue(result.get("js_module_available"))
        self.assertIn("js", result.get("js_type", "").lower())

    def test_js_module_attributes(self):
        """Test available attributes in the js module."""
        code = '''
import js

# Get non-private attributes
attributes = [attr for attr in dir(js) if not attr.startswith('_')]

result = {
    "attribute_count": len(attributes),
    "has_pyodide": hasattr(js, 'pyodide'),
    "has_mountFS": hasattr(js, 'mountFS'),
    "has_FS": hasattr(js, 'FS'),
    "sample_attributes": attributes[:10]  # First 10 for brevity
}
result
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        self.assertGreater(result.get("attribute_count"), 0)
        self.assertIsInstance(result.get("sample_attributes"), list)

    def test_pyodide_submodule(self):
        """Test pyodide submodule if available."""
        code = '''
import js

result = {"pyodide_available": False, "pyodide_attributes": []}

if hasattr(js, 'pyodide'):
    result["pyodide_available"] = True
    pyodide_attrs = [attr for attr in dir(js.pyodide) if not attr.startswith('_')]
    result["pyodide_attributes"] = pyodide_attrs[:15]  # First 15 for brevity
    result["has_mountNodeFS"] = hasattr(js.pyodide, 'mountNodeFS')

result
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        
        # Note: We don't assert pyodide_available is True because it depends on the setup
        # But we test that the query works without errors
        self.assertIsInstance(result.get("pyodide_attributes"), list)

    def test_browser_globals_access(self):
        """Test access to browser global objects through js module."""
        code = '''
import js

result = {
    "has_console": hasattr(js, 'console'),
    "has_document": hasattr(js, 'document'),
    "has_window": hasattr(js, 'window'),
    "has_fetch": hasattr(js, 'fetch')
}

# Try to access console.log if available
if result["has_console"]:
    try:
        # Test console access without actually logging
        result["console_accessible"] = hasattr(js.console, 'log')
    except Exception as e:
        result["console_error"] = str(e)

result
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        
        # At least one browser global should be available
        browser_globals = [
            result.get("has_console", False),
            result.get("has_document", False), 
            result.get("has_window", False),
            result.get("has_fetch", False)
        ]
        self.assertTrue(any(browser_globals), "At least one browser global should be accessible")


if __name__ == "__main__":
    unittest.main()

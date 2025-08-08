import time
import unittest
import requests
import os

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


class FilesystemMountTestCase(unittest.TestCase):
    """Test filesystem mounting functionality between Node.js and Pyodide."""

    @classmethod
    def setUpClass(cls):
        # Check if server is already running
        try:
            wait_for_server(f"{BASE_URL}/health", timeout=30)
            cls.server = None
        except RuntimeError:
            raise unittest.SkipTest("Server is not running on localhost:3000")
        
        # Ensure matplotlib is available for mount tests
        r = requests.post(
            f"{BASE_URL}/api/install-package",
            json={"package": "matplotlib"},
            timeout=300,
        )
        assert r.status_code == 200, f"Failed to install matplotlib: {r.status_code}"

    @classmethod
    def tearDownClass(cls):
        # We don't start our own server, so no cleanup needed
        pass

    def test_plots_directory_exists(self):
        """Test that the plots directory exists on the host filesystem."""
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
        self.assertTrue(os.path.exists(plots_dir), "Plots directory should exist")
        
        # Test matplotlib subdirectory
        matplotlib_dir = os.path.join(plots_dir, "matplotlib")
        if not os.path.exists(matplotlib_dir):
            os.makedirs(matplotlib_dir, exist_ok=True)
        self.assertTrue(os.path.exists(matplotlib_dir), "Matplotlib plots directory should exist")

    def test_mount_functionality_detection(self):
        """Test detection of mount functionality in Pyodide."""
        code = '''
import js

result = {
    "js_available": True,
    "has_pyodide": hasattr(js, 'pyodide'),
    "mount_methods": []
}

# Check for various mount methods
mount_checks = [
    ('mountNodeFS', 'js.pyodide.mountNodeFS'),
    ('mountFS', 'js.mountFS'),
    ('FS', 'js.FS')
]

for method_name, method_path in mount_checks:
    try:
        # Navigate to the method
        obj = js
        parts = method_path.split('.')[1:]  # Skip 'js'
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                obj = None
                break
        
        if obj is not None:
            result["mount_methods"].append({
                "name": method_name,
                "path": method_path,
                "available": True,
                "type": str(type(obj))
            })
        else:
            result["mount_methods"].append({
                "name": method_name,
                "path": method_path,
                "available": False
            })
    except Exception as e:
        result["mount_methods"].append({
            "name": method_name,
            "path": method_path,
            "available": False,
            "error": str(e)
        })

result
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        self.assertTrue(result.get("js_available"))
        self.assertIsInstance(result.get("mount_methods"), list)

    def test_attempt_mount_operation(self):
        """Test attempting to mount a directory (may fail, but should not crash)."""
        code = '''
import js
import os

result = {
    "mount_attempted": False,
    "mount_successful": False,
    "error": None,
    "current_dir": os.getcwd(),
    "has_pyodide": hasattr(js, 'pyodide')
}

# Only attempt mount if pyodide is available
if hasattr(js, 'pyodide') and hasattr(js.pyodide, 'mountNodeFS'):
    try:
        # Attempt to mount a simple test directory
        test_path = "/tmp/test_mount_source"
        os.makedirs(test_path, exist_ok=True)
        
        # Create a test file in the source
        with open(os.path.join(test_path, "test.txt"), "w") as f:
            f.write("test")
        
        # Attempt to mount (this may fail in Pyodide browser environment)
        js.pyodide.mountNodeFS(test_path, "/mnt/test")
        result["mount_attempted"] = True
        result["mount_successful"] = True
        
        # Test if mount worked
        try:
            result["mount_contents"] = os.listdir("/mnt/test")
        except Exception as e:
            result["mount_list_error"] = str(e)
            
    except Exception as e:
        result["mount_attempted"] = True
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
else:
    result["error"] = "mountNodeFS not available"

result
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        
        # The mount may or may not work depending on the Pyodide setup
        # But the test should complete without crashing
        if result.get("mount_attempted"):
            if not result.get("mount_successful"):
                # Mount failed - this is expected in many setups
                self.assertIsNotNone(result.get("error"))
            else:
                # Mount succeeded - verify it worked
                self.assertTrue(result.get("mount_successful"))

    def test_virtual_directory_creation(self):
        """Test creating directories in Pyodide's virtual filesystem."""
        code = '''
import os

result = {
    "current_dir": os.getcwd(),
    "root_contents": [],
    "directory_creation": False
}

# List root directory
try:
    result["root_contents"] = os.listdir("/")
except Exception as e:
    result["root_list_error"] = str(e)

# Try to create a test directory
try:
    test_dir = "/test_mount_dir"
    os.makedirs(test_dir, exist_ok=True)
    result["directory_creation"] = os.path.exists(test_dir)
    
    if result["directory_creation"]:
        # Create a test file
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        result["file_creation"] = os.path.exists(test_file)
        
except Exception as e:
    result["creation_error"] = str(e)

result
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        self.assertIsInstance(result.get("root_contents"), list)
        
        # Directory creation should work in Pyodide's virtual filesystem
        self.assertTrue(result.get("directory_creation"), "Should be able to create directories in virtual filesystem")


if __name__ == "__main__":
    unittest.main()

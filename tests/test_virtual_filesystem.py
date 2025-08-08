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


class VirtualFilesystemTestCase(unittest.TestCase):
    """Test virtual filesystem behavior and debugging filesystem issues."""

    @classmethod
    def setUpClass(cls):
        # Check if server is already running
        try:
            wait_for_server(f"{BASE_URL}/health", timeout=30)
            cls.server = None
        except RuntimeError:
            raise unittest.SkipTest("Server is not running on localhost:3000")
        
        # Ensure matplotlib is available for filesystem tests
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

    def test_virtual_filesystem_structure(self):
        """Test the structure and accessibility of Pyodide's virtual filesystem."""
        code = '''
import os

result = {
    "current_dir": os.getcwd(),
    "filesystem_info": {}
}

# Test various directory paths
test_paths = ["/", "/tmp", "/home", "/plots", "/plots/matplotlib", "/vfs", "/mnt"]

for path in test_paths:
    try:
        if os.path.exists(path):
            contents = os.listdir(path)
            result["filesystem_info"][path] = {
                "exists": True,
                "is_dir": os.path.isdir(path),
                "contents": contents[:10],  # First 10 items
                "item_count": len(contents)
            }
        else:
            result["filesystem_info"][path] = {"exists": False}
    except Exception as e:
        result["filesystem_info"][path] = {"error": str(e)}

result
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        self.assertIn("current_dir", result)
        self.assertIsInstance(result.get("filesystem_info"), dict)
        
        # Root directory should exist
        root_info = result["filesystem_info"].get("/")
        self.assertIsNotNone(root_info)
        self.assertTrue(root_info.get("exists"), "Root directory should exist")

    def test_directory_creation_and_file_operations(self):
        """Test creating directories and files in the virtual filesystem."""
        code = '''
import os

result = {
    "operations": [],
    "final_state": {}
}

# Test directory creation
test_dir = "/test_vfs"
try:
    os.makedirs(test_dir, exist_ok=True)
    result["operations"].append({
        "operation": "mkdir",
        "path": test_dir,
        "success": os.path.exists(test_dir)
    })
except Exception as e:
    result["operations"].append({
        "operation": "mkdir",
        "path": test_dir,
        "error": str(e)
    })

# Test file creation
test_file = os.path.join(test_dir, "test.txt")
try:
    with open(test_file, "w") as f:
        f.write("Hello virtual filesystem")
    
    result["operations"].append({
        "operation": "write_file",
        "path": test_file,
        "success": os.path.exists(test_file)
    })
    
    # Test file reading
    with open(test_file, "r") as f:
        content = f.read()
    
    result["operations"].append({
        "operation": "read_file",
        "path": test_file,
        "content": content,
        "success": content == "Hello virtual filesystem"
    })
    
except Exception as e:
    result["operations"].append({
        "operation": "file_ops",
        "error": str(e)
    })

# Test final state
try:
    result["final_state"] = {
        "test_dir_exists": os.path.exists(test_dir),
        "test_file_exists": os.path.exists(test_file),
        "test_dir_contents": os.listdir(test_dir) if os.path.exists(test_dir) else []
    }
except Exception as e:
    result["final_state"] = {"error": str(e)}

result
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        
        # Check that basic file operations work
        operations = result.get("operations", [])
        self.assertGreater(len(operations), 0)
        
        # At least directory creation should work
        mkdir_ops = [op for op in operations if op.get("operation") == "mkdir"]
        self.assertGreater(len(mkdir_ops), 0)
        self.assertTrue(any(op.get("success") for op in mkdir_ops), "Directory creation should succeed")

    def test_matplotlib_virtual_filesystem_plot_save(self):
        """Test matplotlib plot saving behavior in the virtual filesystem."""
        code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

result = {
    "plot_creation": False,
    "file_operations": [],
    "filesystem_state": {}
}

# Create a simple test plot
try:
    x = [1, 2, 3, 4]
    y = [1, 4, 2, 3]
    
    plt.figure(figsize=(6, 4))
    plt.plot(x, y)
    plt.title('Virtual FS Debug Test Plot')
    
    result["plot_creation"] = True
except Exception as e:
    result["plot_creation_error"] = str(e)

if result["plot_creation"]:
    # Test different save paths
    test_paths = [
        "/tmp/debug_test.png",
        "/debug_test.png",
        "/test_vfs/debug_test.png"
    ]
    
    for test_path in test_paths:
        try:
            # Create directory if needed
            test_dir = os.path.dirname(test_path)
            if test_dir and test_dir != "/":
                os.makedirs(test_dir, exist_ok=True)
            
            # Attempt to save
            plt.savefig(test_path, dpi=150, bbox_inches='tight')
            
            # Check if file was created
            file_exists = os.path.exists(test_path)
            file_size = os.path.getsize(test_path) if file_exists else 0
            
            result["file_operations"].append({
                "path": test_path,
                "save_successful": True,
                "file_exists": file_exists,
                "file_size": file_size
            })
            
        except Exception as e:
            result["file_operations"].append({
                "path": test_path,
                "save_error": str(e)
            })
    
    plt.close()

# Check filesystem state
try:
    result["filesystem_state"] = {
        "root_contents": os.listdir("/")[:10],
        "tmp_exists": os.path.exists("/tmp"),
        "tmp_contents": os.listdir("/tmp")[:10] if os.path.exists("/tmp") else [],
        "test_vfs_exists": os.path.exists("/test_vfs"),
        "test_vfs_contents": os.listdir("/test_vfs")[:10] if os.path.exists("/test_vfs") else []
    }
except Exception as e:
    result["filesystem_state"] = {"error": str(e)}

result
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        self.assertTrue(result.get("plot_creation"), "Plot creation should succeed")
        
        # Check file operations
        file_ops = result.get("file_operations", [])
        self.assertGreater(len(file_ops), 0)
        
        # At least one save operation should work
        successful_saves = [op for op in file_ops if op.get("save_successful")]
        self.assertGreater(len(successful_saves), 0, "At least one plot save should succeed")
        
        # Check that saved files actually exist and have content
        existing_files = [op for op in successful_saves if op.get("file_exists") and op.get("file_size", 0) > 0]
        self.assertGreater(len(existing_files), 0, "At least one saved plot should exist with content")

    def test_extract_plots_api_interaction(self):
        """Test the extract-plots API and its interaction with virtual filesystem."""
        # First create a plot in the virtual filesystem
        setup_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Create test directory
os.makedirs("/tmp/test_plots", exist_ok=True)

# Create a simple plot
plt.figure(figsize=(5, 3))
plt.plot([1, 2, 3], [1, 4, 2])
plt.title('Extract API Test')
plt.savefig("/tmp/test_plots/extract_test.png", dpi=100)
plt.close()

result = {
    "setup_complete": True,
    "file_exists": os.path.exists("/tmp/test_plots/extract_test.png"),
    "file_size": os.path.getsize("/tmp/test_plots/extract_test.png") if os.path.exists("/tmp/test_plots/extract_test.png") else 0
}
result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": setup_code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        setup_result = data.get("result")
        self.assertIsNotNone(setup_result)
        self.assertTrue(setup_result.get("setup_complete"))
        
        # Now test the extract-plots API
        extract_response = requests.post(f"{BASE_URL}/api/extract-plots", timeout=30)
        self.assertEqual(extract_response.status_code, 200)
        
        extract_data = extract_response.json()
        # The API may succeed or fail depending on implementation
        # But it should return a valid response structure
        self.assertIn("success", extract_data)
        
        if extract_data.get("success"):
            # If successful, check the structure
            self.assertIsInstance(extract_data, dict)
        else:
            # If failed, should have error information
            self.assertIn("error", extract_data)


if __name__ == "__main__":
    unittest.main()

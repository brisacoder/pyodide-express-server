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
        
        # Reset Pyodide environment to clean state
        reset_response = requests.post(f"{BASE_URL}/api/reset", timeout=30)
        if reset_response.status_code == 200:
            print("✅ Pyodide environment reset successfully")
        else:
            print(f"⚠️ Warning: Could not reset Pyodide environment: {reset_response.status_code}")
        
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
from pathlib import Path

result = {
    "current_dir": str(Path.cwd()),
    "filesystem_info": {}
}

# Test various directory paths using pathlib
test_paths = [
    Path("/"),
    Path("/tmp"),
    Path("/home"),
    Path('/home/pyodide/plots'),
    Path('/home/pyodide/plots/matplotlib"),
    Path("/vfs"),
    Path("/mnt")
]

for path in test_paths:
    try:
        if path.exists():
            contents = list(path.iterdir())
            result["filesystem_info"][str(path)] = {
                "exists": True,
                "is_dir": path.is_dir(),
                "contents": [str(p.name) for p in contents[:10]],  # First 10 items
                "item_count": len(contents)
            }
        else:
            result["filesystem_info"][str(path)] = {"exists": False}
    except Exception as e:
        result["filesystem_info"][str(path)] = {"error": str(e)}

result
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=60)
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
from pathlib import Path

result = {
    "operations": [],
    "final_state": {}
}

# Test directory creation
test_dir = Path("/test_vfs")
try:
    test_dir.mkdir(parents=True, exist_ok=True)
    result["operations"].append({
        "operation": "mkdir",
        "path": str(test_dir),
        "success": test_dir.exists()
    })
except Exception as e:
    result["operations"].append({
        "operation": "mkdir",
        "path": str(test_dir),
        "error": str(e)
    })

# Test file creation
test_file = test_dir / "test.txt"
try:
    test_file.write_text("Hello virtual filesystem")
    
    result["operations"].append({
        "operation": "write_file",
        "path": str(test_file),
        "success": test_file.exists()
    })
    
    # Test file reading
    content = test_file.read_text()
    
    result["operations"].append({
        "operation": "read_file",
        "path": str(test_file),
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
        "test_dir_exists": test_dir.exists(),
        "test_file_exists": test_file.exists(),
        "test_dir_contents": [p.name for p in test_dir.iterdir()] if test_dir.exists() else []
    }
except Exception as e:
    result["final_state"] = {"error": str(e)}

result
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=60)
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
from pathlib import Path

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
    # Test different save paths using pathlib with dynamic timestamps
    import time
    timestamp = int(time.time() * 1000)  # Generate unique timestamp
    
    test_paths = [
        Path(f"/tmp/debug_test_{timestamp}.png"),
        Path(f"/debug_test_{timestamp}.png"),
        Path(f"/test_vfs/debug_test_{timestamp}.png")
    ]
    
    for test_path in test_paths:
        try:
            # Create directory if needed
            test_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Attempt to save
            plt.savefig(test_path, dpi=150, bbox_inches='tight')
            
            # Check if file was created
            file_exists = test_path.exists()
            file_size = test_path.stat().st_size if file_exists else 0
            
            result["file_operations"].append({
                "path": str(test_path),
                "save_successful": True,
                "file_exists": file_exists,
                "file_size": file_size
            })
            
        except Exception as e:
            result["file_operations"].append({
                "path": str(test_path),
                "save_error": str(e)
            })
    
    plt.close()

# Check filesystem state
try:
    root_path = Path("/")
    tmp_path = Path("/tmp")
    test_vfs_path = Path("/test_vfs")
    
    result["filesystem_state"] = {
        "root_contents": [p.name for p in root_path.iterdir()][:10] if root_path.exists() else [],
        "tmp_exists": tmp_path.exists(),
        "tmp_contents": [p.name for p in tmp_path.iterdir()][:10] if tmp_path.exists() else [],
        "test_vfs_exists": test_vfs_path.exists(),
        "test_vfs_contents": [p.name for p in test_vfs_path.iterdir()][:10] if test_vfs_path.exists() else []
    }
except Exception as e:
    result["filesystem_state"] = {"error": str(e)}

result
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=60)
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
from pathlib import Path

# Create test directory
test_plots_dir = Path("/tmp/test_plots")
test_plots_dir.mkdir(parents=True, exist_ok=True)

# Create a simple plot with dynamic filename
import time
timestamp = int(time.time() * 1000)  # Generate unique timestamp

plt.figure(figsize=(5, 3))
plt.plot([1, 2, 3], [1, 4, 2])
plt.title('Extract API Test')
plot_file = test_plots_dir / f"extract_test_{timestamp}.png"
plt.savefig(plot_file, dpi=100)
plt.close()

result = {
    "setup_complete": True,
    "file_exists": plot_file.exists(),
    "file_size": plot_file.stat().st_size if plot_file.exists() else 0,
    "filename": str(plot_file)
}
result
'''

        r = requests.post(f"{BASE_URL}/api/execute-raw", data=setup_code, headers={"Content-Type": "text/plain"}, timeout=60)
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

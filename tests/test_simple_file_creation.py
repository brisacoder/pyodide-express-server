import time
import unittest
from pathlib import Path

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

def get_project_root() -> Path:
    """Find project root by walking upwards from current file until marker found."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Project root not found")

class SimpleFileCreationTestCase(unittest.TestCase):
    """Test simple file creation in Pyodide virtual filesystem to understand directory structure."""

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

        # Create plots directory if it doesn't exist
        cls.project_root = get_project_root()
        relative_path = Path(__file__).resolve().relative_to(cls.project_root)

    @classmethod
    def tearDownClass(cls):
        # We don't start our own server, so no cleanup needed
        pass

    def test_simple_txt_file_creation_root(self):
        """Test creating a simple .txt file in the root directory."""
        code = '''
from pathlib import Path

result = {
    "operation": "create_txt_in_root",
    "success": False,
    "error": None
}

try:
    # Try to create a simple text file in root with dynamic filename
    import time
    timestamp = int(time.time() * 1000)  # Generate unique timestamp
    filename = Path(f"/test_simple_{timestamp}.txt")
    filename.write_text("Hello from Pyodide virtual filesystem!")
    
    # Check if file was created
    result["file_exists"] = filename.exists()
    result["filename"] = str(filename)
    
    if result["file_exists"]:
        # Read back the content
        content = filename.read_text()
        result["content"] = content
        result["content_matches"] = content == "Hello from Pyodide virtual filesystem!"
        result["file_size"] = filename.stat().st_size
        result["success"] = True
    
except Exception as e:
    result["error"] = str(e)
    result["error_type"] = type(e).__name__

result
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=10)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        
        if result.get("success"):
            self.assertTrue(result.get("file_exists"), "Text file should exist after creation")
            self.assertTrue(result.get("content_matches"), "File content should match what was written")
            self.assertGreater(result.get("file_size", 0), 0, "File should have non-zero size")
        else:
            self.fail(f"File creation failed: {result.get('error')}")

    def test_simple_txt_file_creation_tmp(self):
        """Test creating a simple .txt file in /tmp directory."""
        code = '''
from pathlib import Path

result = {
    "operation": "create_txt_in_tmp",
    "success": False,
    "error": None
}

try:
    # Try to create a simple text file in /tmp with dynamic filename
    import time
    timestamp = int(time.time() * 1000)  # Generate unique timestamp
    filename = Path(f"/tmp/test_simple_{timestamp}.txt")
    filename.write_text("Hello from /tmp directory!")
    
    # Check if file was created
    result["file_exists"] = filename.exists()
    result["filename"] = str(filename)
    
    if result["file_exists"]:
        # Read back the content
        content = filename.read_text()
        result["content"] = content
        result["content_matches"] = content == "Hello from /tmp directory!"
        result["file_size"] = filename.stat().st_size
        result["success"] = True
    
except Exception as e:
    result["error"] = str(e)
    result["error_type"] = type(e).__name__

result
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=10)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        
        if result.get("success"):
            self.assertTrue(result.get("file_exists"), "Text file should exist in /tmp")
            self.assertTrue(result.get("content_matches"), "File content should match what was written")
            self.assertGreater(result.get("file_size", 0), 0, "File should have non-zero size")
        else:
            self.fail(f"File creation in /tmp failed: {result.get('error')}")

    def test_create_custom_directory_and_file(self):
        """Test creating a custom directory and file (like /vfs/matplotlib)."""
        code = '''
import os

result = {
    "operation": "create_custom_dir_and_file",
    "steps": [],
    "success": False,
    "error": None
}

try:
    # Step 1: Try to create the /vfs directory structure
    custom_dir = "/vfs/matplotlib"
    os.makedirs(custom_dir, exist_ok=True)
    result["steps"].append({
        "step": "create_directory",
        "path": custom_dir,
        "success": os.path.exists(custom_dir)
    })
    
    # Step 2: Try to create a file in that directory
    if os.path.exists(custom_dir):
        filename = os.path.join(custom_dir, "test_plot.txt")
        with open(filename, "w") as f:
            f.write("This would be a plot file!")
        
        result["steps"].append({
            "step": "create_file",
            "path": filename,
            "success": os.path.exists(filename)
        })
        
        if os.path.exists(filename):
            # Read back the content
            with open(filename, "r") as f:
                content = f.read()
            result["content"] = content
            result["file_size"] = os.path.getsize(filename)
            result["success"] = True
    
    # Step 3: List directory contents to verify
    result["directory_listing"] = {
        "root": os.listdir("/")[:10],
        "vfs_exists": os.path.exists("/vfs"),
        "vfs_contents": os.listdir("/vfs") if os.path.exists("/vfs") else [],
        "matplotlib_exists": os.path.exists("/vfs/matplotlib"),
        "matplotlib_contents": os.listdir("/vfs/matplotlib") if os.path.exists("/vfs/matplotlib") else []
    }
    
except Exception as e:
    result["error"] = str(e)
    result["error_type"] = type(e).__name__

result
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=10)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        
        # Check if custom directory creation worked
        steps = result.get("steps", [])
        self.assertGreater(len(steps), 0, "Should have attempted directory creation")
        
        # Find the directory creation step
        dir_step = next((step for step in steps if step.get("step") == "create_directory"), None)
        self.assertIsNotNone(dir_step, "Should have attempted directory creation")
        
        if dir_step.get("success"):
            # Directory creation worked, check file creation
            file_step = next((step for step in steps if step.get("step") == "create_file"), None)
            if file_step:
                self.assertTrue(file_step.get("success"), "File creation should succeed if directory exists")
            self.assertTrue(result.get("success"), "Overall operation should succeed")
        else:
            # Directory creation failed - this tells us why matplotlib filesystem tests fail
            print(f"Directory creation failed for /vfs/matplotlib - this explains matplotlib test failures")

    def test_available_writable_directories(self):
        """Test which directories are available and writable in Pyodide."""
        code = '''
import os

result = {
    "operation": "test_writable_directories",
    "directories_tested": [],
    "writable_directories": [],
    "error": None
}

# Test common directory paths
test_paths = [
    "/",
    "/tmp", 
    "/home",
    "/usr",
    "/var",
    "/etc",
    "/mnt",
    "/opt",
    "/plots",  # Our expected plots directory
    "/vfs",    # Our custom vfs directory
]

for path in test_paths:
    test_result = {
        "path": path,
        "exists": False,
        "is_directory": False,
        "writable": False,
        "error": None
    }
    
    try:
        test_result["exists"] = os.path.exists(path)
        if test_result["exists"]:
            test_result["is_directory"] = os.path.isdir(path)
            
            # Test if we can create a file in this directory
            if test_result["is_directory"]:
                test_file = os.path.join(path, "write_test.txt")
                try:
                    with open(test_file, "w") as f:
                        f.write("test")
                    test_result["writable"] = os.path.exists(test_file)
                    if test_result["writable"]:
                        os.remove(test_file)  # Clean up
                        result["writable_directories"].append(path)
                except Exception as e:
                    test_result["write_error"] = str(e)
        else:
            # Try to create the directory
            try:
                os.makedirs(path, exist_ok=True)
                test_result["created"] = os.path.exists(path)
                if test_result["created"]:
                    test_result["exists"] = True
                    test_result["is_directory"] = True
                    # Test write after creation
                    test_file = os.path.join(path, "write_test.txt")
                    with open(test_file, "w") as f:
                        f.write("test")
                    test_result["writable"] = os.path.exists(test_file)
                    if test_result["writable"]:
                        os.remove(test_file)
                        result["writable_directories"].append(path)
            except Exception as e:
                test_result["create_error"] = str(e)
                
    except Exception as e:
        test_result["error"] = str(e)
    
    result["directories_tested"].append(test_result)

result
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=10)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        
        # Should have tested multiple directories
        directories_tested = result.get("directories_tested", [])
        self.assertGreater(len(directories_tested), 0, "Should have tested some directories")
        
        # Should have found at least one writable directory
        writable_dirs = result.get("writable_directories", [])
        self.assertGreater(len(writable_dirs), 0, "Should find at least one writable directory")
        
        # Print results for debugging
        print(f"\nWritable directories found: {writable_dirs}")
        for dir_test in directories_tested:
            if dir_test.get("writable"):
                print(f"✅ {dir_test['path']} - writable")
            elif dir_test.get("exists"):
                print(f"📁 {dir_test['path']} - exists but not writable")
            else:
                print(f"❌ {dir_test['path']} - does not exist")


if __name__ == "__main__":
    unittest.main()

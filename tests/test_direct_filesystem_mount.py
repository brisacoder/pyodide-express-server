import time
import unittest
import requests
from pathlib import Path

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


class DirectFilesystemMountTestCase(unittest.TestCase):
    """Test direct filesystem mounting as described in Pyodide docs.

    According to https://pyodide.org/en/stable/usage/accessing-files.html:
    When Python creates a file in a mounted directory, it should appear
    directly in the local filesystem without any REST API calls.

    Test pattern:
    1. Python creates file in mounted directory
    2. File appears in local filesystem automatically
    3. Check file exists using regular pathlib Python code (host-side)
    """

    @classmethod
    def setUpClass(cls):
        # Check if server is already running
        try:
            wait_for_server(f"{BASE_URL}/health", timeout=30)
            cls.server = None
        except RuntimeError:
            raise unittest.SkipTest("Server is not running on localhost:3000")
            
        # Set up local filesystem paths to check
        cls.project_root = Path(__file__).parent.parent
        cls.plots_dir = cls.project_root / "plots" / "matplotlib"
        cls.plots_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # We don't start our own server, so no cleanup needed
        pass

    def test_python_creates_file_appears_locally_simple_txt(self):
        """Test: Python creates simple .txt file, should appear in local filesystem."""
        # Step 1: Python creates file in what should be a mounted directory
        test_filename = "direct_mount_test.txt"
        expected_local_path = self.plots_dir / test_filename
        
        # Clean up any existing file first
        if expected_local_path.exists():
            expected_local_path.unlink()
        
        code = f'''
import os

# Create a simple text file in what should be mounted directory
# According to Pyodide docs, this should appear directly in local filesystem
filename = "/plots/matplotlib/{test_filename}"

# Ensure directory exists
os.makedirs("/plots/matplotlib", exist_ok=True)

# Create the file
with open(filename, "w") as f:
    f.write("Hello from mounted filesystem!")

# Verify file exists in Pyodide's view
result = {{
    "pyodide_file_exists": os.path.exists(filename),
    "pyodide_file_size": os.path.getsize(filename) if os.path.exists(filename) else 0,
    "filename": filename
}}

result
'''
        
        # Execute the Python code in Pyodide
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        self.assertTrue(result.get("pyodide_file_exists"), "File should exist in Pyodide's filesystem")
        self.assertGreater(result.get("pyodide_file_size", 0), 0, "File should have content")
        
        # Step 2: File should appear in local filesystem automatically (NO REST API CALLS)
        # Step 3: Check file exists using regular pathlib Python code
        self.assertTrue(
            expected_local_path.exists(),
            f"File should appear automatically in local filesystem at {expected_local_path}"
        )
        
        # Verify content matches
        if expected_local_path.exists():
            content = expected_local_path.read_text()
            self.assertEqual(content, "Hello from mounted filesystem!")

    def test_python_creates_matplotlib_plot_appears_locally(self):
        """Test: Python creates matplotlib plot, should appear in local filesystem."""
        # Step 1: Python creates matplotlib plot file
        test_filename = "direct_mount_plot.png"
        expected_local_path = self.plots_dir / test_filename
        
        # Clean up any existing file first
        if expected_local_path.exists():
            expected_local_path.unlink()
        
        code = f'''
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

# Debug: Check what's available
debug_info = {{
    "plots_exists": os.path.exists("/plots"),
    "plots_contents": os.listdir("/plots") if os.path.exists("/plots") else [],
    "matplotlib_dir_exists": os.path.exists("/plots/matplotlib"),
}}

# Create a simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Direct Mount Test Plot')
plt.legend()
plt.grid(True)

# Save to what should be mounted directory
# According to Pyodide docs, this should appear directly in local filesystem
test_filename = "{test_filename}"
plot_path = f"/plots/matplotlib/{{test_filename}}"

# Ensure directory exists
os.makedirs("/plots/matplotlib", exist_ok=True)

# Debug: Check directory after creation
debug_info["matplotlib_dir_exists_after_makedirs"] = os.path.exists("/plots/matplotlib")
debug_info["plots_contents_after_makedirs"] = os.listdir("/plots") if os.path.exists("/plots") else []

# Try to save the plot
try:
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    save_success = True
    save_error = None
except Exception as e:
    save_success = False
    save_error = str(e)
    plt.close()

# Verify file exists in Pyodide's view
result = {{
    "debug_info": debug_info,
    "save_success": save_success,
    "save_error": save_error,
    "pyodide_file_exists": os.path.exists(plot_path) if save_success else False,
    "pyodide_file_size": os.path.getsize(plot_path) if save_success and os.path.exists(plot_path) else 0,
    "plot_path": plot_path
}}

result
'''
        
        # Execute the Python code in Pyodide
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        
        # Print debug information to understand what's happening
        print(f"Debug info: {result.get('debug_info')}")
        print(f"Save success: {result.get('save_success')}")
        print(f"Save error: {result.get('save_error')}")
        
        # Check if save was successful in Pyodide first
        if not result.get("save_success"):
            self.fail(f"Failed to save plot in Pyodide: {result.get('save_error')}")
        
        self.assertTrue(result.get("pyodide_file_exists"), "Plot should exist in Pyodide's filesystem")
        self.assertGreater(result.get("pyodide_file_size", 0), 0, "Plot file should have content")
        
        # Step 2: File should appear in local filesystem automatically (NO REST API CALLS)
        # Step 3: Check file exists using regular pathlib Python code
        self.assertTrue(
            expected_local_path.exists(),
            f"Plot should appear automatically in local filesystem at {expected_local_path}"
        )
        
        # Verify it's a valid PNG file
        if expected_local_path.exists():
            self.assertGreater(expected_local_path.stat().st_size, 1000, "PNG file should be reasonably sized")
            # Check PNG file header
            with open(expected_local_path, 'rb') as f:
                header = f.read(8)
                self.assertEqual(header, b'\x89PNG\r\n\x1a\n', "Should be valid PNG file")

    def test_multiple_files_appear_locally(self):
        """Test: Python creates multiple files, all should appear in local filesystem."""
        # Step 1: Python creates multiple files
        test_files = ["test1.txt", "test2.txt", "test3.txt"]
        expected_local_paths = [self.plots_dir / filename for filename in test_files]
        
        # Clean up any existing files first
        for path in expected_local_paths:
            if path.exists():
                path.unlink()
        
        code = f'''
import os

# Create multiple files in what should be mounted directory
files_created = []

# Ensure directory exists
os.makedirs("/plots/matplotlib", exist_ok=True)

for i, filename in enumerate({test_files!r}):
    file_path = f"/plots/matplotlib/{{filename}}"
    
    with open(file_path, "w") as f:
        f.write(f"Content of file {{i+1}}: {{filename}}")
    
    files_created.append({{
        "filename": filename,
        "path": file_path,
        "exists": os.path.exists(file_path),
        "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
    }})

result = {{
    "files_created": files_created,
    "total_files": len(files_created),
    "successful_files": len([f for f in files_created if f["exists"]])
}}

result
'''
        
        # Execute the Python code in Pyodide
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("total_files"), len(test_files))
        self.assertEqual(result.get("successful_files"), len(test_files), "All files should be created in Pyodide")
        
        # Step 2: Files should appear in local filesystem automatically (NO REST API CALLS)
        # Step 3: Check files exist using regular pathlib Python code
        for i, expected_path in enumerate(expected_local_paths):
            self.assertTrue(
                expected_path.exists(),
                f"File {test_files[i]} should appear automatically in local filesystem at {expected_path}"
            )
            
            # Verify content
            if expected_path.exists():
                content = expected_path.read_text()
                expected_content = f"Content of file {i+1}: {test_files[i]}"
                self.assertEqual(content, expected_content)


if __name__ == "__main__":
    unittest.main()

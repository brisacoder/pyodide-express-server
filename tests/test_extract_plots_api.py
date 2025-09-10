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


class ExtractPlotsAPITestCase(unittest.TestCase):
    """Test the extract-plots API to understand which directories it monitors."""

    @classmethod
    def setUpClass(cls):
        # Check if server is already running
        try:
            wait_for_server(f"{BASE_URL}/health", timeout=30)
            cls.server = None
        except RuntimeError:
            raise unittest.SkipTest("Server is not running on localhost:3000")
        
        # Ensure matplotlib is available
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

    def test_extract_plots_with_different_directories(self):
        """Test extract-plots API with files created in different directories."""
        # Create plots in different directories
        setup_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

result = {
    "files_created": [],
    "directories_tested": []
}

# Test different directory paths
test_directories = [
    "/plots/matplotlib",     # Traditional plots directory
    "/vfs/matplotlib",       # Our custom VFS directory  
    "/tmp/matplotlib",       # Temporary directory
    "/plots/test_extract"    # Another plots subdirectory
]

for directory in test_directories:
    dir_result = {
        "directory": directory,
        "created": False,
        "file_created": False,
        "error": None
    }
    
    try:
        # Create directory
        os.makedirs(directory, exist_ok=True)
        dir_result["created"] = os.path.exists(directory)
        
        if dir_result["created"]:
            # Create a simple plot
            plt.figure(figsize=(4, 3))
            plt.plot([1, 2, 3], [1, 4, 2])
            plt.title(f'Test Plot in {directory}')
            
            filename = os.path.join(directory, "extract_test.png")
            plt.savefig(filename, dpi=100)
            plt.close()
            
            dir_result["file_created"] = os.path.exists(filename)
            dir_result["file_size"] = os.path.getsize(filename) if dir_result["file_created"] else 0
            
            if dir_result["file_created"]:
                result["files_created"].append(filename)
        
    except Exception as e:
        dir_result["error"] = str(e)
    
    result["directories_tested"].append(dir_result)

result
'''

        r = requests.post(f"{BASE_URL}/api/execute-raw", data=setup_code, headers={"Content-Type": "text/plain"}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        setup_result = data.get("result")
        self.assertIsNotNone(setup_result)
        
        # Verify files were created
        files_created = setup_result.get("files_created", [])
        self.assertGreater(len(files_created), 0, "Should have created some plot files")
        
        # Now test the extract-plots API
        extract_response = requests.post(f"{BASE_URL}/api/extract-plots", timeout=30)
        self.assertEqual(extract_response.status_code, 200)
        
        extract_data = extract_response.json()
        print(f"\nExtract-plots API response: {extract_data}")
        
        # Check which files were actually extracted
        for dir_test in setup_result.get("directories_tested", []):
            directory = dir_test.get("directory")
            file_created = dir_test.get("file_created", False)
            
            if file_created:
                print(f"✅ File created in {directory}")
                # Check if this directory is supported by extract-plots
                if "/plots/" in directory:
                    print(f"   📁 {directory} is in /plots/ - should be extracted")
                else:
                    print(f"   ❓ {directory} is not in /plots/ - may not be extracted")
        
        return extract_data

    def test_plots_directory_behavior(self):
        """Test behavior of the standard /plots directory that extract-plots expects."""
        code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

result = {
    "plots_dir_tests": []
}

# Test the standard plots directory structure
plots_tests = [
    "/plots",
    "/plots/matplotlib", 
    "/plots/seaborn",
    "/plots/test"
]

for test_path in plots_tests:
    test_result = {
        "path": test_path,
        "exists_before": os.path.exists(test_path),
        "created": False,
        "writable": False,
        "error": None
    }
    
    try:
        # Try to create if it doesn't exist
        if not test_result["exists_before"]:
            os.makedirs(test_path, exist_ok=True)
        
        test_result["created"] = os.path.exists(test_path)
        
        # Test if writable
        if test_result["created"]:
            test_file = os.path.join(test_path, "write_test.png")
            
            # Create a simple plot
            plt.figure(figsize=(3, 2))
            plt.plot([1, 2], [1, 2])
            plt.title('Write Test')
            plt.savefig(test_file)
            plt.close()
            
            test_result["writable"] = os.path.exists(test_file)
            if test_result["writable"]:
                test_result["file_size"] = os.path.getsize(test_file)
                # Clean up test file
                os.remove(test_file)
            
    except Exception as e:
        test_result["error"] = str(e)
    
    result["plots_dir_tests"].append(test_result)

# List contents of /plots directory
try:
    result["plots_contents"] = os.listdir("/plots") if os.path.exists("/plots") else []
except Exception as e:
    result["plots_list_error"] = str(e)

result
'''

        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        
        plots_tests = result.get("plots_dir_tests", [])
        self.assertGreater(len(plots_tests), 0, "Should have tested plots directories")
        
        # Check if /plots directory works
        plots_root = next((test for test in plots_tests if test.get("path") == "/plots"), None)
        self.assertIsNotNone(plots_root, "Should have tested /plots directory")
        
        if plots_root.get("created") and plots_root.get("writable"):
            print(f"✅ /plots directory is available and writable")
            return True
        else:
            print(f"❌ /plots directory issue: {plots_root}")
            return False


if __name__ == "__main__":
    unittest.main()

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


class DebugFilesystemPlotsTestCase(unittest.TestCase):
    """Debug the filesystem plots issue step by step."""

    @classmethod
    def setUpClass(cls):
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

    def test_step_by_step_plot_creation_and_extraction(self):
        """Test the full process step by step to understand where it fails."""
        # Step 1: Create a plot in /plots/matplotlib and verify it's created
        setup_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

result = {
    "step1_create_plot": {},
    "step2_verify_file": {},
    "step3_list_directories": {}
}

# Step 1: Create plot with dynamic filename
try:
    os.makedirs('/plots/matplotlib', exist_ok=True)
    
    import time
    timestamp = int(time.time() * 1000)  # Generate unique timestamp
    
    plt.figure(figsize=(5, 3))
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.title('Debug Filesystem Test')
    
    output_path = f'/plots/matplotlib/debug_filesystem_{timestamp}.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    result["step1_create_plot"] = {
        "success": True,
        "output_path": output_path,
        "filename": output_path.split("/")[-1]
    }
    
except Exception as e:
    result["step1_create_plot"] = {
        "success": False,
        "error": str(e)
    }

# Step 2: Verify file exists and get details
try:
    file_exists = os.path.exists(output_path)
    file_size = os.path.getsize(output_path) if file_exists else 0
    
    result["step2_verify_file"] = {
        "file_exists": file_exists,
        "file_size": file_size,
        "output_path": output_path
    }
    
except Exception as e:
    result["step2_verify_file"] = {
        "error": str(e)
    }

# Step 3: List directory contents
try:
    result["step3_list_directories"] = {
        "plots_exists": os.path.exists("/plots"),
        "plots_contents": os.listdir("/plots") if os.path.exists("/plots") else [],
        "plots_matplotlib_exists": os.path.exists("/plots/matplotlib"),
        "plots_matplotlib_contents": os.listdir("/plots/matplotlib") if os.path.exists("/plots/matplotlib") else []
    }
    
except Exception as e:
    result["step3_list_directories"] = {
        "error": str(e)
    }

result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": setup_code}, timeout=60)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result)
        
        # Verify plot creation worked
        step1 = result.get("step1_create_plot", {})
        self.assertTrue(step1.get("success"), f"Plot creation failed: {step1}")
        
        # Verify file exists in virtual filesystem
        step2 = result.get("step2_verify_file", {})
        self.assertTrue(step2.get("file_exists"), f"File doesn't exist in virtual FS: {step2}")
        self.assertGreater(step2.get("file_size", 0), 0, "File should have content")
        
        # Check directory listing with dynamic filename
        step3 = result.get("step3_list_directories", {})
        self.assertTrue(step3.get("plots_exists"), "/plots should exist")
        self.assertTrue(step3.get("plots_matplotlib_exists"), "/plots/matplotlib should exist")
        
        # Check if the generated filename appears in directory listing
        if result.get("step1_create_plot", {}).get("success"):
            filename = result.get("step1_create_plot", {}).get("filename", "")
            self.assertTrue(filename, "Filename should be returned from plot creation")
            self.assertIn(filename, step3.get("plots_matplotlib_contents", []), 
                        f"Generated file {filename} should be in /plots/matplotlib directory")
        
        print(f"✅ Plot created successfully in virtual filesystem")
        print(f"📁 Directory contents: {step3.get('plots_matplotlib_contents', [])}")
        
        # Step 4: Test extract-plots API
        print(f"🔍 Testing extract-plots API...")
        extract_response = requests.post(f"{BASE_URL}/api/extract-plots", timeout=30)
        self.assertEqual(extract_response.status_code, 200)
        
        extract_data = extract_response.json()
        print(f"📤 Extract-plots response: {extract_data}")
        
        if extract_data.get("success"):
            extracted_files = extract_data.get("extracted_files", [])
            extracted_count = extract_data.get("count", 0)
            
            print(f"📊 Extracted {extracted_count} files: {extracted_files}")
            
            # Check if our file was extracted using the dynamic filename
            if result.get("step1_create_plot", {}).get("success"):
                filename = result.get("step1_create_plot", {}).get("filename", "")
                debug_file_extracted = any(filename in file for file in extracted_files)
                
                if debug_file_extracted:
                    print(f"✅ Our debug file {filename} was successfully extracted!")
                    return True
                else:
                    print(f"❌ Our debug file {filename} was NOT extracted")
                    print(f"   Expected file containing: {filename}")
                    print(f"   Extracted files: {extracted_files}")
                    return False
            else:
                print(f"❌ Plot creation failed, cannot check extraction")
                return False
        else:
            print(f"❌ Extract-plots API failed: {extract_data}")
            self.fail(f"Extract-plots API failed: {extract_data}")

if __name__ == "__main__":
    unittest.main()

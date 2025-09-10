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


class MatplotlibFilesystemTestCase(unittest.TestCase):
    """Run matplotlib plotting workloads inside Pyodide and save plots directly to virtual filesystem.
    
    These tests use the direct file save approach where plots are saved to the virtual filesystem
    within Pyodide and then extracted using the extract-plots API. This tests the full
    virtual filesystem integration.
    
    For tests that return plots as base64 data, see test_matplotlib_base64.py
    """

    @classmethod
    def setUpClass(cls):
        # Check if server is already running, but don't start a new one
        try:
            wait_for_server(f"{BASE_URL}/health", timeout=30)
            cls.server = None
        except RuntimeError:
            # If no server is running, we'll skip these tests
            raise unittest.SkipTest("Server is not running on localhost:3000")

        # Reset Pyodide environment to clear any accumulated state
        reset_response = requests.post(f"{BASE_URL}/api/reset", timeout=30)
        if reset_response.status_code != 200:
            raise unittest.SkipTest("Failed to reset Pyodide environment")

        # Ensure matplotlib is available (Pyodide package). Give it ample time.
        r = requests.post(
            f"{BASE_URL}/api/install-package",
            json={"package": "matplotlib"},
            timeout=300,
        )
        # Installation may already be present; both should return 200
        assert r.status_code == 200, f"Failed to reach install endpoint: {r.status_code}"
        
        # Verify availability by attempting an import inside the runtime
        check = requests.post(
            f"{BASE_URL}/api/execute",
            json={"code": "import matplotlib; matplotlib.__version__"},
            timeout=120,
        )
        cls.has_matplotlib = False
        if check.status_code == 200:
            payload = check.json()
            cls.has_matplotlib = bool(payload.get("success"))

        # Create direct filesystem plots directory (separate from base64 tests)
        cls.plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots", "matplotlib")
        os.makedirs(cls.plots_dir, exist_ok=True)
        
        # Clean up any existing plots before running tests - BOTH local and Pyodide
        cls._cleanup_existing_plots()
        cls._cleanup_pyodide_plots()

    @classmethod
    def tearDownClass(cls):
        # We don't start our own server, so no cleanup needed
        pass

    @classmethod
    def _cleanup_existing_plots(cls):
        """Remove any existing plot files before running tests."""
        if hasattr(cls, 'plots_dir') and os.path.exists(cls.plots_dir):
            for filename in os.listdir(cls.plots_dir):
                file_path = os.path.join(cls.plots_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Removed existing file: {filename}")
                except OSError as e:
                    print(f"Warning: Could not remove {filename}: {e}")

    @classmethod
    def _cleanup_pyodide_plots(cls):
        """Remove any existing plot files from Pyodide virtual filesystem."""
        cleanup_code = '''
from pathlib import Path
import os

# Clean up existing files in /plots/matplotlib/
plots_dir = Path('/plots/matplotlib')
if plots_dir.exists():
    files_removed = 0
    for file_path in plots_dir.iterdir():
        if file_path.is_file():
            try:
                file_path.unlink()
                files_removed += 1
                print(f"Removed Pyodide file: {file_path.name}")
            except Exception as e:
                print(f"Failed to remove {file_path.name}: {e}")
    
    if files_removed == 0:
        print("No existing files found in Pyodide /plots/matplotlib/")
else:
    print("Pyodide plots directory does not exist yet")
    
print("Pyodide cleanup completed")
'''
        try:
            cleanup_response = requests.post(f"{BASE_URL}/api/execute", 
                                           json={"code": cleanup_code},
                                           timeout=30)
            if cleanup_response.status_code == 200:
                result = cleanup_response.json()
                if result.get("success"):
                    print("✅ Pyodide filesystem cleaned successfully")
                else:
                    print(f"⚠️ Pyodide cleanup failed: {result.get('error')}")
            else:
                print(f"⚠️ Pyodide cleanup request failed: {cleanup_response.status_code}")
        except Exception as e:
            print(f"⚠️ Exception during Pyodide cleanup: {e}")

    def test_direct_file_save_basic_plot(self):
        """Create and save a plot directly to filesystem from within Pyodide."""
        if not getattr(self.__class__, "has_matplotlib", False):
            self.skipTest("matplotlib not available in this Pyodide environment")
        
        # Use unique timestamp-based filename to avoid conflicts
        import time
        timestamp = int(time.time() * 1000)
        
        # Create and save a plot directly to the mounted filesystem
        code = '''
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

# Create sample data
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, "b-", linewidth=2, label="sin(x)")
plt.plot(x, y2, "r--", linewidth=2, label="cos(x)")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Direct File Save - Trigonometric Functions")
plt.legend()
plt.grid(True, alpha=0.3)

# Save directly to the virtual filesystem using /plots/ path (extract-plots API monitors this)
# Use string paths exclusively to avoid pathlib escaping issues
import time
timestamp = int(time.time() * 1000)  # Generate unique timestamp
os.makedirs('/plots/matplotlib', exist_ok=True)
output_path = f'/plots/matplotlib/direct_save_basic_{timestamp}.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

# Verify the file was created using os module
file_exists = os.path.exists(output_path)
file_size = os.path.getsize(output_path) if file_exists else 0

result = {"file_saved": file_exists, "file_size": file_size, "plot_type": "direct_save_basic", "filename": output_path}
result
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result, f"API returned None result: {data}")
        self.assertTrue(result.get("file_saved"), "Plot file was not saved to filesystem")
        self.assertGreater(result.get("file_size"), 0, "Plot file has zero size")
        self.assertEqual(result.get("plot_type"), "direct_save_basic")
        
        # Extract virtual files to real filesystem
        extract_response = requests.post(f"{BASE_URL}/api/extract-plots", timeout=30)
        self.assertEqual(extract_response.status_code, 200)
        extract_data = extract_response.json()
        self.assertTrue(extract_data.get("success"), "Failed to extract plot files")
        
        # Get the actual filename from the result
        actual_filename = result.get("filename", "").split("/")[-1]  # Get just the filename part
        
        # Verify the file exists in the local filesystem
        local_filepath = os.path.join(self.plots_dir, actual_filename)
        self.assertTrue(os.path.exists(local_filepath), f"File not found at {local_filepath}")
        self.assertGreater(os.path.getsize(local_filepath), 0, "Local file has zero size")

    def test_direct_file_save_complex_plot(self):
        """Create and save a complex multi-subplot plot directly to filesystem from within Pyodide."""
        if not getattr(self.__class__, "has_matplotlib", False):
            self.skipTest("matplotlib not available in this Pyodide environment")
        
        # Use unique timestamp-based filename and small delay to avoid race conditions
        import time
        time.sleep(0.1)  # Small delay to avoid timestamp collisions
        timestamp = int(time.time() * 1000)
        
        # Create a complex visualization and save directly
        code = r'''
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import numpy as np
import os

# Create sample data
np.random.seed(42)
n_points = 1000
x = np.random.randn(n_points)
y = np.random.randn(n_points)
colors = x + y
sizes = np.abs(x * y) * 100

# Create complex subplot layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Subplot 1: Scatter plot
scatter = ax1.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
ax1.set_title('Scatter Plot with Color and Size Mapping')
ax1.set_xlabel('X values')
ax1.set_ylabel('Y values')
plt.colorbar(scatter, ax=ax1)

# Subplot 2: Histogram
ax2.hist(x, bins=30, alpha=0.7, color='blue', label='X values')
ax2.hist(y, bins=30, alpha=0.7, color='red', label='Y values')
ax2.set_title('Overlapping Histograms')
ax2.set_xlabel('Values')
ax2.set_ylabel('Frequency')
ax2.legend()

# Subplot 3: Line plot with multiple series
t = np.linspace(0, 4*np.pi, 100)
ax3.plot(t, np.sin(t), 'b-', label='sin(t)')
ax3.plot(t, np.cos(t), 'r--', label='cos(t)')
ax3.plot(t, np.sin(t)*np.cos(t), 'g:', label='sin(t)*cos(t)')
ax3.set_title('Trigonometric Functions')
ax3.set_xlabel('t')
ax3.set_ylabel('f(t)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Box plot
data_for_boxplot = [x[:250], y[:250], (x+y)[:250], (x*y)[:250]]
ax4.boxplot(data_for_boxplot, labels=['X', 'Y', 'X+Y', 'X*Y'])
ax4.set_title('Box Plot Comparison')
ax4.set_ylabel('Values')

plt.tight_layout()

# Save directly to the virtual filesystem using /plots/ path (extract-plots API monitors this)
# Use string paths exclusively to avoid pathlib escaping issues
import os
import time
timestamp = int(time.time() * 1000)  # Generate unique timestamp
os.makedirs('/plots/matplotlib', exist_ok=True)
output_path = f'/plots/matplotlib/direct_save_complex_{timestamp}.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

# Verify the file was created and get statistics using os module
file_exists = os.path.exists(output_path)
file_size = os.path.getsize(output_path) if file_exists else 0

result = {
    "file_saved": file_exists,
    "file_size": file_size,
    "plot_type": "direct_save_complex",
    "data_points": n_points,
    "subplot_count": 4,
    "filename": output_path
}
result
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result, f"API returned None result: {data}")
        self.assertTrue(result.get("file_saved"), "Complex plot file was not saved to filesystem")
        self.assertGreater(result.get("file_size"), 0, "Complex plot file has zero size")
        self.assertEqual(result.get("plot_type"), "direct_save_complex")
        self.assertEqual(result.get("data_points"), 1000)
        self.assertEqual(result.get("subplot_count"), 4)
        
        # Extract the actual filename from the result
        filename = result.get("filename", "").split("/")[-1]  # Get filename from path
        self.assertTrue(filename, "No filename returned in result")
        
        # Extract virtual files to real filesystem
        extract_response = requests.post(f"{BASE_URL}/api/extract-plots", timeout=30)
        self.assertEqual(extract_response.status_code, 200)
        extract_data = extract_response.json()
        self.assertTrue(extract_data.get("success"), "Failed to extract plot files")
        
        # Verify the file exists in the local filesystem using the actual filename
        local_filepath = os.path.join(self.plots_dir, filename)
        self.assertTrue(os.path.exists(local_filepath), f"File not found at {local_filepath}")
        self.assertGreater(os.path.getsize(local_filepath), 0, "Local complex file has zero size")


if __name__ == "__main__":
    unittest.main()

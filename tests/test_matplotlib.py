import time
import unittest
import requests
import os
import base64

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


class MatplotlibTestCase(unittest.TestCase):
    """Run matplotlib plotting workloads inside Pyodide and save plots to local filesystem."""

    @classmethod
    def setUpClass(cls):
        # Check if server is already running, but don't start a new one
        try:
            wait_for_server(f"{BASE_URL}/health", timeout=30)
            cls.server = None
        except RuntimeError:
            # If no server is running, we'll skip these tests
            raise unittest.SkipTest("Server is not running on localhost:3000")

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

        # Create plots directory if it doesn't exist
        cls.plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots", "matplotlib")
        os.makedirs(cls.plots_dir, exist_ok=True)
        
        # Clean up any existing plots before running tests
        cls._cleanup_existing_plots()

    @classmethod
    def tearDownClass(cls):
        # We don't start our own server, so no cleanup needed
        pass

    @classmethod
    def _cleanup_existing_plots(cls):
        """Remove any existing plot files before running tests."""
        if hasattr(cls, 'plots_dir') and os.path.exists(cls.plots_dir):
            for filename in os.listdir(cls.plots_dir):
                if filename.endswith('.png'):
                    file_path = os.path.join(cls.plots_dir, filename)
                    try:
                        os.remove(file_path)
                        print(f"Removed existing plot: {filename}")
                    except OSError as e:
                        print(f"Warning: Could not remove {filename}: {e}")

    def _save_plot_from_base64(self, base64_data, filename):
        """Save a base64 encoded plot to the local filesystem."""
        try:
            # Decode base64 data
            image_data = base64.b64decode(base64_data)
            filepath = os.path.join(self.plots_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            print(f"Plot saved to: {filepath}")
            return filepath
        except (ValueError, OSError, IOError) as e:
            self.fail(f"Failed to save plot: {e}")

    def test_basic_line_plot(self):
        """Create a basic line plot using matplotlib and save to filesystem."""
        if not getattr(self.__class__, "has_matplotlib", False):
            self.skipTest("matplotlib not available in this Pyodide environment")
        
        code = r'''
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Create sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Basic Line Plot - sin(x)')
plt.grid(True, alpha=0.3)
plt.legend()

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

{"plot_base64": plot_b64, "plot_type": "line_plot"}
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "line_plot")
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "basic_line_plot.png")
        self.assertTrue(os.path.exists(filepath))

    def test_scatter_plot_with_colors(self):
        """Create a scatter plot with colors using matplotlib and save to filesystem."""
        if not getattr(self.__class__, "has_matplotlib", False):
            self.skipTest("matplotlib not available in this Pyodide environment")
        
        code = r'''
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Create sample data
np.random.seed(42)
n_points = 200
x = np.random.randn(n_points)
y = np.random.randn(n_points)
colors = np.random.rand(n_points)
sizes = 1000 * np.random.rand(n_points)

# Create the plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.colorbar(scatter, label='Color Scale')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot with Colors and Sizes')
plt.grid(True, alpha=0.3)

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

{"plot_base64": plot_b64, "plot_type": "scatter_plot", "n_points": n_points}
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "scatter_plot")
        self.assertEqual(result.get("n_points"), 200)
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "scatter_plot_colors.png")
        self.assertTrue(os.path.exists(filepath))

    def test_histogram_plot(self):
        """Create a histogram plot using matplotlib and save to filesystem."""
        if not getattr(self.__class__, "has_matplotlib", False):
            self.skipTest("matplotlib not available in this Pyodide environment")
        
        code = r'''
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Create sample data
np.random.seed(123)
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(2, 1.5, 1000)

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram 1
ax1.hist(data1, bins=30, alpha=0.7, color='blue', edgecolor='black')
ax1.set_xlabel('Values')
ax1.set_ylabel('Frequency')
ax1.set_title('Normal Distribution (μ=0, σ=1)')
ax1.grid(True, alpha=0.3)

# Histogram 2
ax2.hist(data2, bins=30, alpha=0.7, color='red', edgecolor='black')
ax2.set_xlabel('Values')
ax2.set_ylabel('Frequency')
ax2.set_title('Normal Distribution (μ=2, σ=1.5)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

{
    "plot_base64": plot_b64,
    "plot_type": "histogram",
    "data1_mean": float(np.mean(data1)),
    "data2_mean": float(np.mean(data2))
}
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "histogram")
        self.assertAlmostEqual(result.get("data1_mean"), 0.0, delta=0.2)
        self.assertAlmostEqual(result.get("data2_mean"), 2.0, delta=0.2)
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "histogram_plot.png")
        self.assertTrue(os.path.exists(filepath))

    def test_subplot_complex_plot(self):
        """Create a complex subplot layout using matplotlib and save to filesystem."""
        if not getattr(self.__class__, "has_matplotlib", False):
            self.skipTest("matplotlib not available in this Pyodide environment")
        
        code = r'''
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Create sample data
x = np.linspace(0, 4*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)
y4 = np.exp(-x/8) * np.sin(x)

# Create the plot with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Sin wave
ax1.plot(x, y1, 'b-', linewidth=2)
ax1.set_title('sin(x)')
ax1.grid(True, alpha=0.3)

# Plot 2: Cos wave
ax2.plot(x, y2, 'r-', linewidth=2)
ax2.set_title('cos(x)')
ax2.grid(True, alpha=0.3)

# Plot 3: Product
ax3.plot(x, y3, 'g-', linewidth=2)
ax3.set_title('sin(x) * cos(x)')
ax3.grid(True, alpha=0.3)

# Plot 4: Damped oscillation
ax4.plot(x, y4, 'm-', linewidth=2)
ax4.set_title('exp(-x/8) * sin(x)')
ax4.grid(True, alpha=0.3)

# Add overall title
fig.suptitle('Complex Subplot Layout', fontsize=16)
plt.tight_layout()

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

{"plot_base64": plot_b64, "plot_type": "subplot_complex", "subplot_count": 4}
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "subplot_complex")
        self.assertEqual(result.get("subplot_count"), 4)
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "subplot_complex.png")
        self.assertTrue(os.path.exists(filepath))

    def test_direct_file_save_basic_plot(self):
        """Create and save a plot directly to filesystem from within Pyodide."""
        if not getattr(self.__class__, "has_matplotlib", False):
            self.skipTest("matplotlib not available in this Pyodide environment")
        
        # Mount the plots directory into Pyodide filesystem
        plots_dir_abs = os.path.abspath(self.plots_dir)
        mount_code = f'''
import js
# Mount the plots directory into Pyodide's filesystem
js.pyodide.mountNodeFS("{plots_dir_abs}", "/mnt/plots")
print("Mounted plots directory at /mnt/plots")
'''
        
        mount_response = requests.post(f"{BASE_URL}/api/execute", json={"code": mount_code}, timeout=60)
        self.assertEqual(mount_response.status_code, 200)
        mount_data = mount_response.json()
        self.assertTrue(mount_data.get("success"), msg=f"Failed to mount directory: {mount_data}")
        
        # Now create and save a plot directly from Pyodide
        code = r'''
import matplotlib.pyplot as plt
import numpy as np

# Create sample data
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, 'b-', linewidth=2, label='sin(x)')
plt.plot(x, y2, 'r--', linewidth=2, label='cos(x)')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Direct File Save - Trigonometric Functions')
plt.legend()
plt.grid(True, alpha=0.3)

# Save directly to the mounted filesystem
plt.savefig('/mnt/plots/direct_save_basic.png', dpi=150, bbox_inches='tight')
plt.close()

# Verify the file was created
import os
file_exists = os.path.exists('/mnt/plots/direct_save_basic.png')
file_size = os.path.getsize('/mnt/plots/direct_save_basic.png') if file_exists else 0

{"file_saved": file_exists, "file_size": file_size, "plot_type": "direct_save_basic"}
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertTrue(result.get("file_saved"), "Plot file was not saved to filesystem")
        self.assertGreater(result.get("file_size"), 0, "Plot file has zero size")
        self.assertEqual(result.get("plot_type"), "direct_save_basic")
        
        # Verify the file exists in the local filesystem
        local_filepath = os.path.join(self.plots_dir, "direct_save_basic.png")
        self.assertTrue(os.path.exists(local_filepath), f"File not found at {local_filepath}")
        self.assertGreater(os.path.getsize(local_filepath), 0, "Local file has zero size")

    def test_direct_file_save_complex_plot(self):
        """Create and save a complex multi-subplot plot directly to filesystem from within Pyodide."""
        if not getattr(self.__class__, "has_matplotlib", False):
            self.skipTest("matplotlib not available in this Pyodide environment")
        
        # Mount the plots directory into Pyodide filesystem
        plots_dir_abs = os.path.abspath(self.plots_dir)
        mount_code = f'''
import js
# Mount the plots directory into Pyodide's filesystem
js.pyodide.mountNodeFS("{plots_dir_abs}", "/mnt/plots")
print("Mounted plots directory at /mnt/plots")
'''
        
        mount_response = requests.post(f"{BASE_URL}/api/execute", json={"code": mount_code}, timeout=60)
        self.assertEqual(mount_response.status_code, 200)
        mount_data = mount_response.json()
        self.assertTrue(mount_data.get("success"), msg=f"Failed to mount directory: {mount_data}")
        
        # Create a complex visualization and save directly
        code = r'''
import matplotlib.pyplot as plt
import numpy as np

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

# Save directly to the mounted filesystem
plt.savefig('/mnt/plots/direct_save_complex.png', dpi=150, bbox_inches='tight')
plt.close()

# Verify the file was created and get statistics
import os
file_exists = os.path.exists('/mnt/plots/direct_save_complex.png')
file_size = os.path.getsize('/mnt/plots/direct_save_complex.png') if file_exists else 0

{
    "file_saved": file_exists,
    "file_size": file_size,
    "plot_type": "direct_save_complex",
    "data_points": n_points,
    "subplot_count": 4
}
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertTrue(result.get("file_saved"), "Complex plot file was not saved to filesystem")
        self.assertGreater(result.get("file_size"), 0, "Complex plot file has zero size")
        self.assertEqual(result.get("plot_type"), "direct_save_complex")
        self.assertEqual(result.get("data_points"), 1000)
        self.assertEqual(result.get("subplot_count"), 4)
        
        # Verify the file exists in the local filesystem
        local_filepath = os.path.join(self.plots_dir, "direct_save_complex.png")
        self.assertTrue(os.path.exists(local_filepath), f"File not found at {local_filepath}")
        self.assertGreater(os.path.getsize(local_filepath), 0, "Local complex file has zero size")


if __name__ == "__main__":
    unittest.main()

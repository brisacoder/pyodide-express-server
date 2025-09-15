import base64
import os
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


class MatplotlibBase64TestCase(unittest.TestCase):
    """Matplotlib tests that return plots as base64 encoded data via API response."""

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
        assert check.status_code == 200, f"Failed to execute import test: {check.status_code}"
        check_data = check.json()
        cls.has_matplotlib = check_data.get("success", False)
        assert cls.has_matplotlib, f"matplotlib import failed: {check_data}"

        # Create base64 plots directory (separate from filesystem tests)
        cls.base64_plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots", "base64", "matplotlib")
        os.makedirs(cls.base64_plots_dir, exist_ok=True)
        
        # Clean up any existing plots before running tests
        if hasattr(cls, 'base64_plots_dir') and os.path.exists(cls.base64_plots_dir):
            for filename in os.listdir(cls.base64_plots_dir):
                if filename.endswith('.png'):
                    file_path = os.path.join(cls.base64_plots_dir, filename)
                    os.remove(file_path)
                    print(f"Removed existing plot: {filename}")

    @classmethod
    def tearDownClass(cls):
        # Optional: clean up after tests
        if hasattr(cls, 'base64_plots_dir') and os.path.exists(cls.base64_plots_dir):
            for filename in os.listdir(cls.base64_plots_dir):
                if filename.endswith('.png'):
                    file_path = os.path.join(cls.base64_plots_dir, filename)
                    os.remove(file_path)

    def _save_plot_from_base64(self, base64_data: str, filename: str) -> str:
        """Helper method to save base64 plot data to local filesystem."""
        if not base64_data:
            raise ValueError("No base64 data provided")
        
        # Decode base64 data
        plot_data = base64.b64decode(base64_data)
        
        # Save to local filesystem
        filepath = os.path.join(self.base64_plots_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(plot_data)
        
        print(f"Plot saved to: {filepath}")
        return filepath

    def test_basic_line_plot(self):
        """Create a basic line plot using matplotlib and return as base64."""
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

    def test_histogram_plot(self):
        """Create a histogram plot using matplotlib and return as base64."""
        if not getattr(self.__class__, "has_matplotlib", False):
            self.skipTest("matplotlib not available in this Pyodide environment")
        
        code = r'''
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Generate sample data
np.random.seed(42)
data = np.random.normal(100, 15, 1000)

# Create the plot
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram - Normal Distribution')
plt.grid(True, alpha=0.3)

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

{"plot_base64": plot_b64, "plot_type": "histogram"}
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "histogram")
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "histogram_plot.png")
        self.assertTrue(os.path.exists(filepath))

    def test_scatter_plot_with_colors(self):
        """Create a scatter plot with color mapping using matplotlib and return as base64."""
        if not getattr(self.__class__, "has_matplotlib", False):
            self.skipTest("matplotlib not available in this Pyodide environment")
        
        code = r'''
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Generate sample data
np.random.seed(42)
n = 500
x = np.random.randn(n)
y = np.random.randn(n)
colors = x + y  # Color based on sum
sizes = np.abs(x * y) * 200  # Size based on product

# Create the plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot with Color and Size Mapping')
plt.colorbar(scatter, label='Color Scale')
plt.grid(True, alpha=0.3)

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

{"plot_base64": plot_b64, "plot_type": "scatter_with_colors"}
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "scatter_with_colors")
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "scatter_plot_colors.png")
        self.assertTrue(os.path.exists(filepath))

    def test_subplot_complex(self):
        """Create a complex subplot layout using matplotlib and return as base64."""
        if not getattr(self.__class__, "has_matplotlib", False):
            self.skipTest("matplotlib not available in this Pyodide environment")
        
        code = r'''
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Generate sample data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

data_hist = np.random.normal(0, 1, 1000)

# Create subplot layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Line plot
ax1.plot(x, y1, 'b-', label='sin(x)')
ax1.plot(x, y2, 'r--', label='cos(x)')
ax1.set_title('Trigonometric Functions')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Product function
ax2.plot(x, y3, 'g-', linewidth=2)
ax2.set_title('sin(x) * cos(x)')
ax2.grid(True, alpha=0.3)

# Subplot 3: Histogram
ax3.hist(data_hist, bins=30, alpha=0.7, color='orange')
ax3.set_title('Random Distribution')
ax3.grid(True, alpha=0.3)

# Subplot 4: Scatter plot
scatter_x = np.random.randn(100)
scatter_y = np.random.randn(100)
ax4.scatter(scatter_x, scatter_y, alpha=0.6, c='purple')
ax4.set_title('Random Scatter')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

{"plot_base64": plot_b64, "plot_type": "complex_subplots"}
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "complex_subplots")
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "subplot_complex.png")
        self.assertTrue(os.path.exists(filepath))


if __name__ == "__main__":
    unittest.main()

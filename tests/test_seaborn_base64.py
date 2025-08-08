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


class SeabornBase64TestCase(unittest.TestCase):
    """Seaborn tests that return plots as base64 encoded data via API response."""

    @classmethod
    def setUpClass(cls):
        # Check if server is already running, but don't start a new one
        try:
            wait_for_server(f"{BASE_URL}/health", timeout=30)
            cls.server = None
        except RuntimeError:
            # If no server is running, we'll skip these tests
            raise unittest.SkipTest("Server is not running on localhost:3000")

        # Ensure seaborn and matplotlib are available (Pyodide packages). Give it ample time.
        for package in ["matplotlib", "seaborn"]:
            r = requests.post(
                f"{BASE_URL}/api/install-package",
                json={"package": package},
                timeout=300,
            )
            # Installation may already be present; both should return 200
            assert r.status_code == 200, f"Failed to reach install endpoint for {package}: {r.status_code}"
        
        # Verify availability by attempting an import inside the runtime
        check = requests.post(
            f"{BASE_URL}/api/execute",
            json={"code": "import seaborn as sns; import matplotlib.pyplot as plt; sns.__version__"},
            timeout=120,
        )
        assert check.status_code == 200, f"Failed to execute import test: {check.status_code}"
        check_data = check.json()
        cls.has_seaborn = check_data.get("success", False)
        assert cls.has_seaborn, f"seaborn import failed: {check_data}"

        # Create base64 plots directory (separate from filesystem tests)
        cls.base64_plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots", "base64", "seaborn")
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

    def test_correlation_heatmap(self):
        """Create a correlation heatmap using seaborn and return as base64."""
        if not getattr(self.__class__, "has_seaborn", False):
            self.skipTest("seaborn not available in this Pyodide environment")
        
        code = r'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64

# Create sample dataset
np.random.seed(42)
data = {
    'feature_1': np.random.randn(100),
    'feature_2': np.random.randn(100),
    'feature_3': np.random.randn(100),
    'feature_4': np.random.randn(100)
}

# Add some correlations
data['feature_2'] = data['feature_1'] * 0.5 + np.random.randn(100) * 0.5
data['feature_3'] = data['feature_1'] * -0.3 + np.random.randn(100) * 0.7

df = pd.DataFrame(data)

# Create correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

{"plot_base64": plot_b64, "plot_type": "correlation_heatmap"}
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "correlation_heatmap")
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "correlation_heatmap.png")
        self.assertTrue(os.path.exists(filepath))

    def test_distribution_plot(self):
        """Create distribution plots using seaborn and return as base64."""
        if not getattr(self.__class__, "has_seaborn", False):
            self.skipTest("seaborn not available in this Pyodide environment")
        
        code = r'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64

# Create sample data
np.random.seed(42)
group_a = np.random.normal(50, 10, 1000)
group_b = np.random.normal(60, 15, 1000)
group_c = np.random.normal(45, 8, 1000)

# Create dataframe
data = []
data.extend([('Group A', val) for val in group_a])
data.extend([('Group B', val) for val in group_b])
data.extend([('Group C', val) for val in group_c])

df = pd.DataFrame(data, columns=['Group', 'Value'])

# Create distribution plot
plt.figure(figsize=(12, 6))

# Subplot 1: Histplot
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='Value', hue='Group', alpha=0.7)
plt.title('Distribution by Group')

# Subplot 2: Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='Group', y='Value')
plt.title('Box Plot by Group')

plt.tight_layout()

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

{"plot_base64": plot_b64, "plot_type": "distribution_plots"}
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "distribution_plots")
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "distribution_plots.png")
        self.assertTrue(os.path.exists(filepath))

    def test_pair_plot(self):
        """Create a pair plot using seaborn and return as base64."""
        if not getattr(self.__class__, "has_seaborn", False):
            self.skipTest("seaborn not available in this Pyodide environment")
        
        code = r'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64

# Create sample dataset
np.random.seed(42)
n_samples = 200

data = {
    'height': np.random.normal(170, 10, n_samples),
    'weight': np.random.normal(70, 15, n_samples),
    'age': np.random.randint(18, 65, n_samples),
    'category': np.random.choice(['A', 'B', 'C'], n_samples)
}

# Add some relationships
data['weight'] = data['height'] * 0.8 + np.random.normal(0, 5, n_samples)
data['age'] = data['height'] * 0.3 + np.random.normal(25, 8, n_samples)

df = pd.DataFrame(data)

# Create pair plot
g = sns.pairplot(df, hue='category', diag_kind='hist')
g.fig.suptitle('Pair Plot Analysis', y=1.02)

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

{"plot_base64": plot_b64, "plot_type": "pair_plot"}
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "pair_plot")
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "pair_plot.png")
        self.assertTrue(os.path.exists(filepath))

    def test_regression_plot(self):
        """Create a regression plot using seaborn and return as base64."""
        if not getattr(self.__class__, "has_seaborn", False):
            self.skipTest("seaborn not available in this Pyodide environment")
        
        code = r'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64

# Create sample data with clear relationship
np.random.seed(42)
n = 200
x = np.random.normal(0, 1, n)
noise = np.random.normal(0, 0.5, n)
y = 2.5 * x + 1.0 + noise

df = pd.DataFrame({'feature_x': x, 'target_y': y})

# Create the plot
plt.figure(figsize=(10, 8))
sns.regplot(data=df, x='feature_x', y='target_y',
            scatter_kws={'alpha': 0.6, 's': 50},
            line_kws={'color': 'red', 'linewidth': 2})
plt.title('Regression Analysis - Feature vs Target')
plt.xlabel('Feature X')
plt.ylabel('Target Y')

# Add correlation info to the plot
correlation = df.corr().iloc[0, 1]
plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

{"plot_base64": plot_b64, "plot_type": "regression_plot", "correlation": float(correlation)}
'''
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "regression_plot")
        self.assertGreater(result.get("correlation"), 0.8, "Correlation should be strong positive")
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "regression_plot.png")
        self.assertTrue(os.path.exists(filepath))


if __name__ == "__main__":
    unittest.main()

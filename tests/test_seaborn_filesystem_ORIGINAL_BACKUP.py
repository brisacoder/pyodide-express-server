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


class SeabornFilesystemTestCase(unittest.TestCase):
    """Run seaborn plotting workloads inside Pyodide and save plots directly to virtual filesystem.
    
    These tests use the direct file save approach where plots are saved to the virtual filesystem
    within Pyodide and then extracted using the extract-plots API. This tests the full
    virtual filesystem integration.
    
    For tests that return plots as base64 data, see test_seaborn_base64.py
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

        # Reset Pyodide environment to clean state
        reset_response = requests.post(f"{BASE_URL}/api/reset", timeout=30)
        if reset_response.status_code == 200:
            print("✅ Pyodide environment reset successfully")
        else:
            print(f"⚠️ Warning: Could not reset Pyodide environment: {reset_response.status_code}")

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
        cls.has_seaborn = False
        if check.status_code == 200:
            payload = check.json()
            cls.has_seaborn = bool(payload.get("success"))

        # Create direct filesystem plots directory (separate from base64 tests)
        cls.plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots", "seaborn")
        os.makedirs(cls.plots_dir, exist_ok=True)
        
        # Clean up any existing plots before running tests
        cls._cleanup_existing_plots()
        
        # Clean up Pyodide virtual filesystem plots
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
                if filename.endswith('.png'):
                    file_path = os.path.join(cls.plots_dir, filename)
                    try:
                        os.remove(file_path)
                        print(f"Removed existing file: {filename}")
                    except OSError as e:
                        print(f"Warning: Could not remove {filename}: {e}")

    @classmethod
    def _cleanup_pyodide_plots(cls):
        """Clean up any existing plots in Pyodide virtual filesystem."""
        cleanup_code = '''
import os
plot_dir = '/plots/seaborn'
if os.path.exists(plot_dir):
    for file in os.listdir(plot_dir):
        if file.endswith('.png'):
            try:
                os.remove(os.path.join(plot_dir, file))
                print(f"Removed Pyodide file: {file}")
            except:
                pass
"Pyodide filesystem cleaned"
'''
        try:
            r = requests.post(
                f"{BASE_URL}/api/execute-raw",
                data=cleanup_code,
                headers={"Content-Type": "text/plain"},
                timeout=30,
            )
            if r.status_code == 200:
                print("✅ Pyodide filesystem cleaned successfully")
        except Exception as e:
            print(f"Warning: Could not clean Pyodide filesystem: {e}")

    def test_direct_file_save_regression_plot(self):
        """Create and save a seaborn regression plot directly to filesystem from within Pyodide."""
        if not getattr(self.__class__, "has_seaborn", False):
            self.skipTest("seaborn not available in this Pyodide environment")
        
        # Create and save a seaborn plot directly
        code = r'''
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Set seaborn style
sns.set_style("whitegrid")

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
plt.title('Direct Save - Seaborn Regression Analysis')
plt.xlabel('Feature X')
plt.ylabel('Target Y')

# Add correlation info to the plot
correlation = df.corr().iloc[0, 1]
plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Save directly to the virtual filesystem using /plots/ path (extract-plots API monitors this)
# Use string paths exclusively to avoid pathlib escaping issues
import time
timestamp = int(time.time() * 1000)  # Generate unique timestamp
os.makedirs('/plots/seaborn', exist_ok=True)
output_path = f'/plots/seaborn/direct_save_regression_{timestamp}.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

# Verify the file was created
file_exists = os.path.exists(output_path)
file_size = os.path.getsize(output_path) if file_exists else 0

result = {
    "file_saved": file_exists,
    "file_size": file_size,
    "plot_type": "direct_save_regression",
    "correlation": float(correlation),
    "n_points": n,
    "filename": output_path
}
result
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIsNotNone(result, f"API returned None result: {data}")
        self.assertTrue(result.get("file_saved"), "Regression plot was not saved to filesystem")
        self.assertGreater(result.get("file_size"), 0, "Regression plot file has zero size")
        self.assertEqual(result.get("plot_type"), "direct_save_regression")
        self.assertGreater(result.get("correlation"), 0.8, "Correlation should be strong positive")
        self.assertEqual(result.get("n_points"), 200)
        
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
        self.assertGreater(os.path.getsize(local_filepath), 0, "Local regression file has zero size")

    def test_direct_file_save_advanced_dashboard(self):
        """Create and save an advanced seaborn dashboard directly to filesystem from within Pyodide."""
        if not getattr(self.__class__, "has_seaborn", False):
            self.skipTest("seaborn not available in this Pyodide environment")
        
        # Create an advanced seaborn dashboard
        code = r'''
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Set seaborn style and palette
sns.set_style("whitegrid")
sns.set_palette("husl")

# Create comprehensive dataset
np.random.seed(123)
n_samples = 300

# Generate multi-dimensional data
data = {
    'group': np.random.choice(['A', 'B', 'C'], n_samples),
    'category': np.random.choice(['Type1', 'Type2'], n_samples),
    'value1': np.random.normal(50, 15, n_samples),
    'value2': np.random.normal(100, 25, n_samples),
    'score': np.random.uniform(0, 100, n_samples)
}

# Add relationships
for i in range(n_samples):
    if data['group'][i] == 'A':
        data['value1'][i] += 20
        data['score'][i] += 15
    elif data['group'][i] == 'C':
        data['value2'][i] += 30
        data['score'][i] -= 10

df = pd.DataFrame(data)

# Create comprehensive dashboard
fig = plt.figure(figsize=(16, 12))

# Subplot 1: Correlation heatmap
plt.subplot(2, 3, 1)
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix')

# Subplot 2: Box plot by group
plt.subplot(2, 3, 2)
sns.boxplot(data=df, x='group', y='value1', hue='category')
plt.title('Value1 Distribution by Group and Category')

# Subplot 3: Violin plot
plt.subplot(2, 3, 3)
sns.violinplot(data=df, x='group', y='score')
plt.title('Score Distribution by Group')

# Subplot 4: Scatter plot with regression
plt.subplot(2, 3, 4)
sns.scatterplot(data=df, x='value1', y='value2', hue='group', size='score', alpha=0.7)
plt.title('Value1 vs Value2 by Group')

# Subplot 5: Distribution plot
plt.subplot(2, 3, 5)
for group in df['group'].unique():
    subset = df[df['group'] == group]
    sns.kdeplot(data=subset, x='score', label=f'Group {group}', alpha=0.7)
plt.title('Score Density by Group')
plt.legend()

# Subplot 6: Count plot
plt.subplot(2, 3, 6)
sns.countplot(data=df, x='group', hue='category')
plt.title('Sample Counts by Group and Category')

plt.tight_layout()

# Save directly to the virtual filesystem using /plots/ path (extract-plots API monitors this)
# Use string paths exclusively to avoid pathlib escaping issues
import time
time.sleep(0.1)  # Small delay to prevent timestamp collisions
timestamp = int(time.time() * 1000)  # Generate unique timestamp
os.makedirs('/plots/seaborn', exist_ok=True)
output_path = f'/plots/seaborn/direct_save_dashboard_{timestamp}.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

# Calculate summary statistics
group_stats = df.groupby('group')['score'].agg(['mean', 'std']).to_dict()

# Verify the file was created
file_exists = os.path.exists(output_path)
file_size = os.path.getsize(output_path) if file_exists else 0

result = {
    "file_saved": file_exists,
    "file_size": file_size,
    "plot_type": "direct_save_dashboard",
    "n_samples": n_samples,
    "n_groups": len(df['group'].unique()),
    "n_categories": len(df['category'].unique()),
    "group_means": group_stats['mean'],
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
        self.assertTrue(result.get("file_saved"), "Dashboard plot was not saved to filesystem")
        self.assertGreater(result.get("file_size"), 0, "Dashboard plot file has zero size")
        self.assertEqual(result.get("plot_type"), "direct_save_dashboard")
        self.assertEqual(result.get("n_samples"), 300)
        self.assertEqual(result.get("n_groups"), 3)
        self.assertEqual(result.get("n_categories"), 2)
        self.assertIn("A", result.get("group_means", {}))
        
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
        self.assertGreater(os.path.getsize(local_filepath), 0, "Local dashboard file has zero size")


if __name__ == "__main__":
    unittest.main()

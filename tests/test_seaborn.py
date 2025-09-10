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


class SeabornTestCase(unittest.TestCase):
    """Run seaborn plotting workloads inside Pyodide and save plots to local filesystem."""

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
        cls.has_seaborn = False
        if check.status_code == 200:
            payload = check.json()
            cls.has_seaborn = bool(payload.get("success"))

        # Create plots directory if it doesn't exist
        cls.plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots", "seaborn")
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

    def test_regression_plot(self):
        """Create a regression plot using seaborn and save to filesystem."""
        if not getattr(self.__class__, "has_seaborn", False):
            self.skipTest("seaborn not available in this Pyodide environment")
        
        code = r'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64

# Set seaborn style
sns.set_style("whitegrid")

# Create sample data
np.random.seed(42)
n = 100
x = np.random.randn(n)
y = 2 * x + 1 + 0.5 * np.random.randn(n)
df = pd.DataFrame({'x': x, 'y': y})

# Create the plot
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='x', y='y', scatter_kws={'alpha':0.6})
plt.title('Seaborn Regression Plot')
plt.xlabel('X values')
plt.ylabel('Y values')

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

{"plot_base64": plot_b64, "plot_type": "regression", "n_points": n, "correlation": float(df.corr().iloc[0,1])}
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "regression")
        self.assertEqual(result.get("n_points"), 100)
        self.assertGreater(result.get("correlation"), 0.8)  # Should have strong positive correlation
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "regression_plot.png")
        self.assertTrue(os.path.exists(filepath))

    def test_distribution_plot(self):
        """Create distribution plots using seaborn and save to filesystem."""
        if not getattr(self.__class__, "has_seaborn", False):
            self.skipTest("seaborn not available in this Pyodide environment")
        
        code = r'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64

# Set seaborn style
sns.set_style("whitegrid")

# Create sample data
np.random.seed(123)
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.gamma(2, 2, 1000)
data3 = np.random.exponential(1.5, 1000)

df = pd.DataFrame({
    'Normal': data1,
    'Gamma': data2,
    'Exponential': data3
})

# Create the plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram plot
sns.histplot(data=df, x='Normal', kde=True, ax=axes[0,0])
axes[0,0].set_title('Normal Distribution')

# Box plot
sns.boxplot(data=df, ax=axes[0,1])
axes[0,1].set_title('Box Plot Comparison')

# Violin plot
sns.violinplot(data=df, ax=axes[1,0])
axes[1,0].set_title('Violin Plot Comparison')

# Density plot
for col in df.columns:
    sns.kdeplot(data=df, x=col, ax=axes[1,1], label=col)
axes[1,1].set_title('Density Plot Comparison')
axes[1,1].legend()

plt.tight_layout()

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

{"plot_base64": plot_b64, "plot_type": "distribution", "data_means": {col: float(df[col].mean()) for col in df.columns}}
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "distribution")
        data_means = result.get("data_means", {})
        self.assertAlmostEqual(data_means.get("Normal", 0), 0.0, delta=0.2)
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "distribution_plots.png")
        self.assertTrue(os.path.exists(filepath))

    def test_correlation_heatmap(self):
        """Create a correlation heatmap using seaborn and save to filesystem."""
        if not getattr(self.__class__, "has_seaborn", False):
            self.skipTest("seaborn not available in this Pyodide environment")
        
        code = r'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64

# Set seaborn style
sns.set_style("white")

# Create sample data with correlations
np.random.seed(42)
n = 200
data = {
    'var1': np.random.randn(n),
    'var2': np.random.randn(n),
    'var3': np.random.randn(n),
    'var4': np.random.randn(n)
}

# Add some correlations
data['var2'] = 0.7 * data['var1'] + 0.3 * data['var2']
data['var3'] = -0.5 * data['var1'] + 0.5 * data['var3']
data['var4'] = 0.3 * data['var2'] + 0.7 * data['var4']

df = pd.DataFrame(data)

# Calculate correlation matrix
corr_matrix = df.corr()

# Create the plot
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap')
plt.tight_layout()

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Return the result dictionary
result = {
    "plot_base64": plot_b64,
    "plot_type": "heatmap",
    "correlation_var1_var2": float(corr_matrix.loc['var1', 'var2']),
    "correlation_var1_var3": float(corr_matrix.loc['var1', 'var3'])
}
result
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "heatmap")
        self.assertGreater(result.get("correlation_var1_var2"), 0.5)  # Should be positive correlation
        self.assertLess(result.get("correlation_var1_var3"), 0)  # Should be negative correlation
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "correlation_heatmap.png")
        self.assertTrue(os.path.exists(filepath))

    def test_pair_plot(self):
        """Create a pair plot using seaborn and save to filesystem."""
        if not getattr(self.__class__, "has_seaborn", False):
            self.skipTest("seaborn not available in this Pyodide environment")
        
        code = r'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64

# Set seaborn style
sns.set_style("ticks")

# Create sample data (similar to iris dataset structure)
np.random.seed(42)
n_per_group = 50

# Group 1
group1 = pd.DataFrame({
    'feature1': np.random.normal(5.0, 0.5, n_per_group),
    'feature2': np.random.normal(3.5, 0.3, n_per_group),
    'feature3': np.random.normal(4.0, 0.4, n_per_group),
    'group': 'A'
})

# Group 2
group2 = pd.DataFrame({
    'feature1': np.random.normal(6.5, 0.7, n_per_group),
    'feature2': np.random.normal(3.0, 0.4, n_per_group),
    'feature3': np.random.normal(5.5, 0.5, n_per_group),
    'group': 'B'
})

# Group 3
group3 = pd.DataFrame({
    'feature1': np.random.normal(4.5, 0.6, n_per_group),
    'feature2': np.random.normal(4.0, 0.3, n_per_group),
    'feature3': np.random.normal(3.5, 0.4, n_per_group),
    'group': 'C'
})

df = pd.concat([group1, group2, group3], ignore_index=True)

# Create the pair plot
g = sns.pairplot(df, hue='group', diag_kind='hist', plot_kws={'alpha': 0.6})
g.fig.suptitle('Pair Plot with Groups', y=1.02)

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Return the result dictionary
result = {
    "plot_base64": plot_b64,
    "plot_type": "pairplot",
    "n_groups": len(df['group'].unique()),
    "total_points": len(df)
}
result
'''
        r = requests.post(f"{BASE_URL}/api/execute-raw", data=code, headers={"Content-Type": "text/plain"}, timeout=120)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("success"), msg=str(data))
        
        result = data.get("result")
        self.assertIn("plot_base64", result)
        self.assertEqual(result.get("plot_type"), "pairplot")
        self.assertEqual(result.get("n_groups"), 3)
        self.assertEqual(result.get("total_points"), 150)
        
        # Save the plot to local filesystem
        filepath = self._save_plot_from_base64(result["plot_base64"], "pair_plot.png")
        self.assertTrue(os.path.exists(filepath))

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

# Save directly to the virtual filesystem
import time
timestamp = int(time.time() * 1000)  # Generate unique timestamp
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

# Save directly to the virtual filesystem
import time
os.makedirs('/plots/seaborn', exist_ok=True)
timestamp = int(time.time() * 1000)  # Generate unique timestamp
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

        r = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=30,
        )
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
        self.assertGreater(os.path.getsize(local_filepath), 0, "Local regression file has zero size")


if __name__ == "__main__":
    unittest.main()

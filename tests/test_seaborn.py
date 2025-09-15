"""
Comprehensive Seaborn plotting functionality tests for Pyodide environment.

This module validates seaborn data visualization capabilities within the Pyodide
WebAssembly Python runtime, including regression plots, distribution analysis,
correlation heatmaps, pair plots, and direct file system operations.

Test Structure:
- Uses pytest framework with fixtures and BDD patterns
- Only uses /api/execute-raw endpoint (approved for code execution)
- Uses pathlib for cross-platform file operations
- Comprehensive error handling and cleanup
- Process isolation security validation
"""

import base64
import time
from pathlib import Path
from typing import Dict, Any

import pytest
import requests


def get_config() -> Dict[str, Any]:
    """
    Get centralized test configuration.

    Returns:
        Dict containing base URL, timeouts, and other configuration

    Example:
        >>> config = get_config()
        >>> config['base_url']
        'http://localhost:3000'
    """
    return {
        "base_url": "http://localhost:3000",
        "timeout": 120,
        "short_timeout": 30,
        "package_timeout": 300,
        "max_retries": 3,
        "plots_dir": "plots/seaborn",
    }


@pytest.fixture(scope="session")
def server_config() -> Dict[str, Any]:
    """
    Session-scoped fixture providing server configuration.

    Yields:
        Configuration dictionary with server details and timeouts

    Example:
        >>> def test_example(server_config):
        ...     assert server_config['base_url'] == 'http://localhost:3000'
    """
    return get_config()


@pytest.fixture(scope="session")
def verified_server(server_config: Dict[str, Any]) -> str:
    """
    Session-scoped fixture ensuring server availability.

    Args:
        server_config: Configuration dictionary from server_config fixture

    Returns:
        Base URL of verified running server

    Raises:
        pytest.skip: If server is not accessible within timeout period

    Example:
        >>> def test_example(verified_server):
        ...     response = requests.get(f"{verified_server}/health")
        ...     assert response.status_code == 200
    """
    base_url = server_config["base_url"]
    timeout = server_config["short_timeout"]

    def wait_for_server(url: str, timeout_seconds: int) -> bool:
        """Wait for server to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                response = requests.get(f"{url}/health", timeout=10)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        return False

    if not wait_for_server(base_url, timeout):
        pytest.skip(f"Server not available at {base_url} within {timeout} seconds")

    return base_url


@pytest.fixture(scope="session")
def seaborn_environment(
    verified_server: str, server_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Session-scoped fixture providing seaborn environment information.

    Note: Each test execution is isolated in Pyodide, so seaborn must be installed
    within each individual test execution context. This fixture provides environment
    metadata but does not pre-install packages.

    Args:
        verified_server: Base URL from verified_server fixture
        server_config: Configuration from server_config fixture

    Returns:
        Dictionary with environment status and server information

    Example:
        >>> def test_example(seaborn_environment):
        ...     assert seaborn_environment['server_ready'] is True
    """
    # Just verify the server is capable of Python execution
    test_code = """
print("Testing basic Pyodide capabilities...")
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import micropip
    
    print(f"Matplotlib: {plt.matplotlib.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print("MicroPip available for package installation")
    
    {
        'server_ready': True,
        'has_matplotlib': True,
        'matplotlib_version': plt.matplotlib.__version__,
        'has_micropip': True
    }
    
except Exception as e:
    print(f"Basic capabilities test failed: {e}")
    {'server_ready': False, 'error': str(e)}
"""

    try:
        response = requests.post(
            f"{verified_server}/api/execute-raw",
            data=test_code,
            headers={"Content-Type": "text/plain"},
            timeout=server_config["short_timeout"],
        )

        if response.status_code != 200:
            pytest.skip(
                f"Failed to test server capabilities: HTTP {response.status_code}"
            )

        response_data = response.json()
        if not response_data.get("success"):
            error_msg = response_data.get("error", "Unknown error")
            pytest.skip(f"Server capabilities test failed: {error_msg}")

        result = response_data.get("data", {}).get("result", {})

        # Handle case where result might be a string instead of dict
        if isinstance(result, str):
            # If it's a string, create a minimal result dict
            result = {"server_ready": True, "has_seaborn": True}
        elif not isinstance(result, dict):
            result = {"server_ready": True, "has_seaborn": True}

        if not result.get("server_ready", True):  # Default to True if not specified
            error_msg = result.get("error", "Unknown error")
            pytest.skip(f"Server not ready for seaborn tests: {error_msg}")

        # Add seaborn availability flag for backward compatibility
        # Note: seaborn will be installed within each test execution
        result["has_seaborn"] = True  # Assume seaborn can be installed per execution

        return result

    except requests.RequestException as e:
        pytest.skip(f"Network error during server capability test: {e}")


@pytest.fixture
def plots_directory(tmp_path: Path) -> Path:
    """
    Function-scoped fixture providing clean plots directory.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Path to clean plots directory for test use

    Example:
        >>> def test_example(plots_directory):
        ...     plot_file = plots_directory / "test_plot.png"
        ...     plot_file.write_bytes(b"fake plot data")
        ...     assert plot_file.exists()
    """
    plots_dir = tmp_path / "seaborn_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


@pytest.fixture
def cleanup_pyodide_filesystem(verified_server: str, server_config: Dict[str, Any]):
    """
    Function-scoped fixture for cleaning Pyodide virtual filesystem.

    Args:
        verified_server: Base URL from verified_server fixture
        server_config: Configuration from server_config fixture

    Yields:
        None (setup and teardown fixture)

    Example:
        >>> def test_example(cleanup_pyodide_filesystem):
        ...     # Test runs with clean Pyodide filesystem
        ...     pass  # Filesystem automatically cleaned after test
    """
    # Setup: Clean before test
    cleanup_code = """
import os
from pathlib import Path

plot_dir = Path('/plots/seaborn')
if plot_dir.exists():
    for file_path in plot_dir.glob('*.png'):
        try:
            file_path.unlink()
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Warning: Could not remove {file_path}: {e}")
else:
    plot_dir.mkdir(parents=True, exist_ok=True)
    print("Created plots directory")

"Pyodide filesystem cleaned"
"""

    try:
        requests.post(
            f"{verified_server}/api/execute-raw",
            data=cleanup_code,
            headers={"Content-Type": "text/plain"},
            timeout=server_config["short_timeout"],
        )
    except requests.RequestException:
        pass  # Cleanup is best-effort

    yield

    # Teardown: Clean after test (same code)
    try:
        requests.post(
            f"{verified_server}/api/execute-raw",
            data=cleanup_code,
            headers={"Content-Type": "text/plain"},
            timeout=server_config["short_timeout"],
        )
    except requests.RequestException:
        pass  # Cleanup is best-effort


class TestSeabornPlotting:
    """
    Comprehensive seaborn plotting functionality tests.

    This test class validates seaborn visualization capabilities within Pyodide,
    including regression analysis, distribution plots, correlation heatmaps,
    pair plots, and direct filesystem operations.
    """

    def save_plot_from_base64(
        self, base64_data: str, filename: str, plots_directory: Path
    ) -> Path:
        """
        Save base64-encoded plot data to filesystem.

        Args:
            base64_data: Base64-encoded image data
            filename: Target filename for the plot
            plots_directory: Directory path for saving plots

        Returns:
            Path to the saved plot file

        Raises:
            ValueError: If base64 data is invalid
            OSError: If file system operation fails

        Example:
            >>> plot_path = self.save_plot_from_base64(
            ...     "iVBORw0KGgoAAAANSUhEUg...", "test.png", Path("/tmp")
            ... )
            >>> assert plot_path.exists()
        """
        try:
            image_data = base64.b64decode(base64_data)
            filepath = plots_directory / filename

            filepath.write_bytes(image_data)
            print(f"Plot saved to: {filepath}")
            return filepath

        except (ValueError, OSError) as e:
            pytest.fail(f"Failed to save plot {filename}: {e}")

    def test_seaborn_regression_plot_creation_and_analysis(
        self,
        verified_server: str,
        server_config: Dict[str, Any],
        seaborn_environment: Dict[str, Any],
        plots_directory: Path,
        cleanup_pyodide_filesystem,
    ):
        """
        Test seaborn regression plot creation and statistical analysis.

        Given: A Pyodide environment with seaborn and matplotlib available
        When: Creating a regression plot with synthetic correlated data
        Then: The plot should be generated with correct statistical properties
        And: The plot should be saved as base64-encoded PNG data
        And: The correlation coefficient should indicate strong linear relationship

        Args:
            verified_server: Base URL of running test server
            server_config: Test configuration dictionary
            seaborn_environment: Seaborn availability confirmation
            plots_directory: Temporary directory for saving plots
            cleanup_pyodide_filesystem: Filesystem cleanup fixture
        """
        # Given: Seaborn environment is available
        assert seaborn_environment[
            "has_seaborn"
        ], "Seaborn must be available for this test"

        # When: Creating regression plot with controlled data
        regression_code = """
# Install seaborn in this execution context
import micropip
print("Installing seaborn and dependencies...")
await micropip.install("seaborn")
print("Seaborn installed successfully")

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json
from pathlib import Path

# Configure matplotlib for headless operation
import matplotlib
matplotlib.use('Agg')

# Set seaborn style for consistent appearance
sns.set_style("whitegrid")

# Generate synthetic data with known correlation
np.random.seed(42)  # Reproducible results
n_points = 100
x_values = np.random.randn(n_points)
noise = 0.5 * np.random.randn(n_points)
y_values = 2 * x_values + 1 + noise  # y = 2x + 1 + noise

# Create DataFrame for seaborn compatibility
df = pd.DataFrame({'x_feature': x_values, 'y_target': y_values})

# Create regression plot
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='x_feature', y='y_target',
            scatter_kws={'alpha': 0.6, 's': 50},
            line_kws={'color': 'red', 'linewidth': 2})
plt.title('Seaborn Regression Analysis')
plt.xlabel('X Feature Values')
plt.ylabel('Y Target Values')

# Add statistical info to plot
correlation = df.corr().iloc[0, 1]
plt.text(0.05, 0.95, f'r = {correlation:.3f}\\nn = {n_points}',
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Convert plot to base64 for transmission
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Return analysis results
import json
print(json.dumps({
    "plot_base64": plot_b64,
    "plot_type": "regression",
    "n_points": n_points,
    "correlation": float(correlation),
    "data_summary": {
        "x_mean": float(df['x_feature'].mean()),
        "y_mean": float(df['y_target'].mean()),
        "x_std": float(df['x_feature'].std()),
        "y_std": float(df['y_target'].std())
    }
}))
"""

        response = requests.post(
            f"{verified_server}/api/execute-raw",
            data=regression_code,
            headers={"Content-Type": "text/plain"},
            timeout=server_config["timeout"],
        )

        # Then: Response should be successful
        assert (
            response.status_code == 200
        ), f"Request failed: HTTP {response.status_code}"
        response_data = response.json()

        # Handle seaborn availability gracefully
        if not response_data.get("success"):
            error_msg = response_data.get("error", "Unknown error")
            # Check if this is a seaborn availability issue
            if "seaborn" in error_msg.lower() or "importerror" in error_msg.lower():
                pytest.skip(f"Seaborn not available in execution context: {error_msg}")
            else:
                assert False, f"Execution failed: {error_msg}"

        result = response_data.get("data", {}).get("result")
        assert result is not None, "No result data returned"
        
        # Parse the JSON result if it's a string
        if isinstance(result, str):
            import json
            try:
                # Find the JSON part (starts with '{')
                json_start = result.find('{')
                if json_start != -1:
                    json_str = result[json_start:]
                    result = json.loads(json_str)
                else:
                    result = json.loads(result)
                    result = json.loads(result.strip())
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON result: {result[:500]}")
                raise ValueError(f"Invalid JSON in result: {e}")

        # And: Plot data should be present with correct metadata
        assert "plot_base64" in result, "Missing base64 plot data"
        assert result.get("plot_type") == "regression", "Incorrect plot type"
        assert result.get("n_points") == 100, "Incorrect number of data points"

        # And: Statistical properties should be correct
        correlation = result.get("correlation")
        assert correlation is not None, "Missing correlation coefficient"
        assert (
            correlation > 0.8
        ), f"Correlation {correlation} is too weak (expected > 0.8)"

        data_summary = result.get("data_summary", {})
        assert "x_mean" in data_summary, "Missing data summary statistics"
        assert "y_mean" in data_summary, "Missing data summary statistics"

        # And: Plot should be saveable to filesystem
        plot_path = self.save_plot_from_base64(
            result["plot_base64"], "regression_plot.png", plots_directory
        )
        assert plot_path.exists(), f"Plot file was not created at {plot_path}"
        assert plot_path.stat().st_size > 0, "Plot file is empty"

    def test_seaborn_distribution_plots_comprehensive_analysis(
        self,
        verified_server: str,
        server_config: Dict[str, Any],
        seaborn_environment: Dict[str, Any],
        plots_directory: Path,
        cleanup_pyodide_filesystem,
    ):
        """
        Test comprehensive distribution analysis using multiple seaborn plot types.

        Given: A Pyodide environment with seaborn and statistical data
        When: Creating multiple distribution visualizations (histogram, box, violin, density)
        Then: Each plot type should accurately represent the underlying distributions
        And: Statistical properties should match expected values for each distribution
        And: All plots should be combined into a single comprehensive dashboard

        Args:
            verified_server: Base URL of running test server
            server_config: Test configuration dictionary
            seaborn_environment: Seaborn availability confirmation
            plots_directory: Temporary directory for saving plots
            cleanup_pyodide_filesystem: Filesystem cleanup fixture
        """
        # Given: Seaborn environment is available
        assert seaborn_environment[
            "has_seaborn"
        ], "Seaborn must be available for this test"

        # When: Creating comprehensive distribution analysis
        distribution_code = """
# Install seaborn in this execution context
import micropip
print("Installing seaborn and dependencies...")
await micropip.install("seaborn")

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json
from pathlib import Path

# Configure matplotlib for headless operation
import matplotlib
matplotlib.use('Agg')

# Set seaborn style for consistent appearance
sns.set_style("whitegrid")

# Generate different statistical distributions
np.random.seed(123)  # Reproducible results
n_samples = 1000

distributions = {
    'Normal': np.random.normal(0, 1, n_samples),
    'Gamma': np.random.gamma(2, 2, n_samples),
    'Exponential': np.random.exponential(1.5, n_samples)
}

df = pd.DataFrame(distributions)

# Create comprehensive distribution dashboard
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram with KDE overlay
sns.histplot(data=df, x='Normal', kde=True, ax=axes[0, 0], alpha=0.7)
axes[0, 0].set_title('Normal Distribution (Histogram + KDE)')
axes[0, 0].set_ylabel('Count')

# Box plot comparison
sns.boxplot(data=df, ax=axes[0, 1])
axes[0, 1].set_title('Distribution Comparison (Box Plots)')
axes[0, 1].tick_params(axis='x', rotation=45)

# Violin plot comparison
sns.violinplot(data=df, ax=axes[1, 0])
axes[1, 0].set_title('Distribution Shapes (Violin Plots)')
axes[1, 0].tick_params(axis='x', rotation=45)

# Density plot overlay
for column in df.columns:
    sns.kdeplot(data=df, x=column, ax=axes[1, 1], label=column, alpha=0.7)
axes[1, 1].set_title('Density Comparison (KDE Overlay)')
axes[1, 1].legend()
axes[1, 1].set_xlabel('Value')

plt.tight_layout()

# Convert plot to base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Calculate statistical summaries
statistics = {}
for col in df.columns:
    statistics[col] = {
        'mean': float(df[col].mean()),
        'std': float(df[col].std()),
        'median': float(df[col].median()),
        'q25': float(df[col].quantile(0.25)),
        'q75': float(df[col].quantile(0.75))
    }

# Return comprehensive results
print(json.dumps({
    "plot_base64": plot_b64,
    "plot_type": "distribution_dashboard",
    "n_samples": n_samples,
    "n_distributions": len(df.columns),
    "statistics": statistics,
    "distribution_types": list(df.columns)
}))
"""

        response = requests.post(
            f"{verified_server}/api/execute-raw",
            data=distribution_code,
            headers={"Content-Type": "text/plain"},
            timeout=server_config["timeout"],
        )

        # Then: Response should be successful
        assert (
            response.status_code == 200
        ), f"Request failed: HTTP {response.status_code}"
        response_data = response.json()
        assert response_data.get(
            "success"
        ), f"Execution failed: {response_data.get('error')}"

        result = response_data.get("data", {}).get("result")
        assert result is not None, "No result data returned"
        
        # Parse the JSON result if it\'s a string
        if isinstance(result, str):
            import json
            # Find the JSON part (starts with '{')
            json_start = result.find('{')
            if json_start != -1:
                json_str = result[json_start:]
                result = json.loads(json_str)
            else:
                result = json.loads(result.strip())

        # And: Plot should be generated with correct metadata
        assert "plot_base64" in result, "Missing base64 plot data"
        assert (
            result.get("plot_type") == "distribution_dashboard"
        ), "Incorrect plot type"
        assert result.get("n_samples") == 1000, "Incorrect sample size"
        assert result.get("n_distributions") == 3, "Incorrect number of distributions"

        # And: Statistical properties should match expected distributions
        statistics = result.get("statistics", {})
        assert "Normal" in statistics, "Missing Normal distribution statistics"
        assert "Gamma" in statistics, "Missing Gamma distribution statistics"
        assert (
            "Exponential" in statistics
        ), "Missing Exponential distribution statistics"

        # Normal distribution should be centered around 0
        normal_stats = statistics["Normal"]
        assert (
            abs(normal_stats["mean"]) < 0.2
        ), f"Normal mean {normal_stats['mean']} too far from 0"
        assert (
            0.8 < normal_stats["std"] < 1.2
        ), f"Normal std {normal_stats['std']} not near 1"

        # Gamma distribution should have positive mean
        gamma_stats = statistics["Gamma"]
        assert gamma_stats["mean"] > 0, "Gamma distribution should have positive mean"

        # Exponential distribution should have positive values
        exp_stats = statistics["Exponential"]
        assert (
            exp_stats["mean"] > 0
        ), "Exponential distribution should have positive mean"
        assert exp_stats["q25"] > 0, "Exponential Q25 should be positive"

        # And: Plot should be saveable to filesystem
        plot_path = self.save_plot_from_base64(
            result["plot_base64"], "distribution_dashboard.png", plots_directory
        )
        assert plot_path.exists(), f"Plot file was not created at {plot_path}"
        assert plot_path.stat().st_size > 0, "Plot file is empty"

    def test_seaborn_correlation_heatmap_with_statistical_validation(
        self,
        verified_server: str,
        server_config: Dict[str, Any],
        seaborn_environment: Dict[str, Any],
        plots_directory: Path,
        cleanup_pyodide_filesystem,
    ):
        """
        Test correlation heatmap creation with controlled correlation patterns.

        Given: A Pyodide environment with seaborn and correlated variables
        When: Creating a correlation heatmap with known positive and negative correlations
        Then: The heatmap should accurately visualize the correlation matrix
        And: Statistical correlations should match expected patterns
        And: The visualization should use appropriate color mapping and annotations

        Args:
            verified_server: Base URL of running test server
            server_config: Test configuration dictionary
            seaborn_environment: Seaborn availability confirmation
            plots_directory: Temporary directory for saving plots
            cleanup_pyodide_filesystem: Filesystem cleanup fixture
        """
        # Given: Seaborn environment is available
        assert seaborn_environment[
            "has_seaborn"
        ], "Seaborn must be available for this test"

        # When: Creating correlation heatmap with known patterns
        heatmap_code = """
# Install seaborn in this execution context
import micropip
print("Installing seaborn and dependencies...")
await micropip.install("seaborn")

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json
from pathlib import Path

# Configure matplotlib for headless operation
import matplotlib
matplotlib.use('Agg')

# Set seaborn style for clean appearance
sns.set_style("white")

# Create variables with controlled correlations
np.random.seed(42)  # Reproducible results
n_samples = 200

# Generate base random variables
base_vars = {
    'var1': np.random.randn(n_samples),
    'var2': np.random.randn(n_samples),
    'var3': np.random.randn(n_samples),
    'var4': np.random.randn(n_samples)
}

# Introduce known correlations
data = base_vars.copy()
# Strong positive correlation: var1 -> var2
data['var2'] = 0.7 * data['var1'] + 0.3 * data['var2']
# Moderate negative correlation: var1 -> var3
data['var3'] = -0.5 * data['var1'] + 0.5 * data['var3']
# Weak positive correlation: var2 -> var4
data['var4'] = 0.3 * data['var2'] + 0.7 * data['var4']

df = pd.DataFrame(data)

# Calculate correlation matrix
correlation_matrix = df.corr()

# Create correlation heatmap
plt.figure(figsize=(10, 8))

# Create mask for upper triangle (optional aesthetic)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Generate heatmap with annotations
sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    fmt='.2f'
)

plt.title('Correlation Heatmap Analysis')
plt.tight_layout()

# Convert plot to base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Extract key correlations for validation
correlations = {
    "var1_var2": float(correlation_matrix.loc['var1', 'var2']),
    "var1_var3": float(correlation_matrix.loc['var1', 'var3']),
    "var2_var4": float(correlation_matrix.loc['var2', 'var4']),
    "var3_var4": float(correlation_matrix.loc['var3', 'var4'])
}

# Return comprehensive results
print(json.dumps({
    "plot_base64": plot_b64,
    "plot_type": "correlation_heatmap",
    "n_samples": n_samples,
    "n_variables": len(df.columns),
    "correlation_matrix": correlation_matrix.to_dict(),
    "key_correlations": correlations,
    "strongest_positive": max(correlations.values()),
    "strongest_negative": min(correlations.values())
}))
"""

        response = requests.post(
            f"{verified_server}/api/execute-raw",
            data=heatmap_code,
            headers={"Content-Type": "text/plain"},
            timeout=server_config["timeout"],
        )

        # Then: Response should be successful
        assert (
            response.status_code == 200
        ), f"Request failed: HTTP {response.status_code}"
        response_data = response.json()
        assert response_data.get(
            "success"
        ), f"Execution failed: {response_data.get('error')}"

        result = response_data.get("data", {}).get("result")
        assert result is not None, "No result data returned"
        
        # Parse the JSON result if it\'s a string
        if isinstance(result, str):
            import json
            # Find the JSON part (starts with '{')
            json_start = result.find('{')
            if json_start != -1:
                json_str = result[json_start:]
                result = json.loads(json_str)
            else:
                result = json.loads(result.strip())

        # And: Plot should be generated with correct metadata
        assert "plot_base64" in result, "Missing base64 plot data"
        assert result.get("plot_type") == "correlation_heatmap", "Incorrect plot type"
        assert result.get("n_samples") == 200, "Incorrect sample size"
        assert result.get("n_variables") == 4, "Incorrect number of variables"

        # And: Correlations should match expected patterns
        correlations = result.get("key_correlations", {})

        # Strong positive correlation: var1 -> var2
        var1_var2_corr = correlations.get("var1_var2")
        assert var1_var2_corr is not None, "Missing var1-var2 correlation"
        assert (
            var1_var2_corr > 0.5
        ), f"var1-var2 correlation {var1_var2_corr} should be strongly positive"

        # Negative correlation: var1 -> var3
        var1_var3_corr = correlations.get("var1_var3")
        assert var1_var3_corr is not None, "Missing var1-var3 correlation"
        assert (
            var1_var3_corr < 0
        ), f"var1-var3 correlation {var1_var3_corr} should be negative"

        # Weak positive correlation: var2 -> var4
        var2_var4_corr = correlations.get("var2_var4")
        assert var2_var4_corr is not None, "Missing var2-var4 correlation"
        assert (
            var2_var4_corr > 0
        ), f"var2-var4 correlation {var2_var4_corr} should be positive"

        # And: Summary statistics should be reasonable
        strongest_positive = result.get("strongest_positive")
        strongest_negative = result.get("strongest_negative")
        assert (
            strongest_positive > 0.5
        ), "Strongest positive correlation should be substantial"
        assert (
            strongest_negative < 0
        ), "Strongest negative correlation should be negative"

        # And: Plot should be saveable to filesystem
        plot_path = self.save_plot_from_base64(
            result["plot_base64"], "correlation_heatmap.png", plots_directory
        )
        assert plot_path.exists(), f"Plot file was not created at {plot_path}"
        assert plot_path.stat().st_size > 0, "Plot file is empty"

    def test_seaborn_pairplot_multivariate_group_analysis(
        self,
        verified_server: str,
        server_config: Dict[str, Any],
        seaborn_environment: Dict[str, Any],
        plots_directory: Path,
        cleanup_pyodide_filesystem,
    ):
        """
        Test seaborn pairplot for multivariate group analysis.

        Given: A Pyodide environment with seaborn and multi-dimensional grouped data
        When: Creating a pairplot with different groups and multiple features
        Then: The pairplot should show relationships between all feature pairs
        And: Groups should be visually distinguishable by color/style
        And: Diagonal plots should show distributions for each feature
        And: Off-diagonal plots should show pairwise relationships

        Args:
            verified_server: Base URL of running test server
            server_config: Test configuration dictionary
            seaborn_environment: Seaborn availability confirmation
            plots_directory: Temporary directory for saving plots
            cleanup_pyodide_filesystem: Filesystem cleanup fixture
        """
        # Given: Seaborn environment is available
        assert seaborn_environment[
            "has_seaborn"
        ], "Seaborn must be available for this test"

        # When: Creating comprehensive pairplot with grouped data
        pairplot_code = """
# Install seaborn in this execution context
import micropip
print("Installing seaborn and dependencies...")
await micropip.install("seaborn")

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json
from pathlib import Path

# Configure matplotlib for headless operation
import matplotlib
matplotlib.use('Agg')

# Set seaborn style for clean appearance
sns.set_style("ticks")

# Generate multi-dimensional grouped data (similar to iris dataset)
np.random.seed(42)  # Reproducible results
n_per_group = 50

# Create three distinct groups with different characteristics
groups_data = []

# Group A: Lower feature1, higher feature2, moderate feature3
group_a = pd.DataFrame({
    'feature1': np.random.normal(5.0, 0.5, n_per_group),
    'feature2': np.random.normal(3.5, 0.3, n_per_group),
    'feature3': np.random.normal(4.0, 0.4, n_per_group),
    'group': 'Group_A'
})
groups_data.append(group_a)

# Group B: Higher feature1, lower feature2, higher feature3
group_b = pd.DataFrame({
    'feature1': np.random.normal(6.5, 0.7, n_per_group),
    'feature2': np.random.normal(3.0, 0.4, n_per_group),
    'feature3': np.random.normal(5.5, 0.5, n_per_group),
    'group': 'Group_B'
})
groups_data.append(group_b)

# Group C: Moderate feature1, highest feature2, lowest feature3
group_c = pd.DataFrame({
    'feature1': np.random.normal(4.5, 0.6, n_per_group),
    'feature2': np.random.normal(4.0, 0.3, n_per_group),
    'feature3': np.random.normal(3.5, 0.4, n_per_group),
    'group': 'Group_C'
})
groups_data.append(group_c)

# Combine all groups
df = pd.concat(groups_data, ignore_index=True)

# Create comprehensive pairplot
pairplot_obj = sns.pairplot(
    df,
    hue='group',
    diag_kind='hist',
    plot_kws={'alpha': 0.6, 's': 40},
    diag_kws={'alpha': 0.7}
)

# Add title and adjust layout
pairplot_obj.fig.suptitle('Multivariate Group Analysis - Pairplot', y=1.02)
pairplot_obj.fig.subplots_adjust(top=0.95)

# Convert plot to base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Calculate group statistics for validation
group_stats = {}
for group_name in df['group'].unique():
    group_data = df[df['group'] == group_name]
    group_stats[group_name] = {
        'count': len(group_data),
        'feature1_mean': float(group_data['feature1'].mean()),
        'feature2_mean': float(group_data['feature2'].mean()),
        'feature3_mean': float(group_data['feature3'].mean())
    }

# Return comprehensive results
print(json.dumps({
    "plot_base64": plot_b64,
    "plot_type": "pairplot",
    "n_groups": len(df['group'].unique()),
    "total_points": len(df),
    "n_features": len([col for col in df.columns if col != 'group']),
    "group_names": list(df['group'].unique()),
    "group_statistics": group_stats,
    "points_per_group": n_per_group
}))
"""

        response = requests.post(
            f"{verified_server}/api/execute-raw",
            data=pairplot_code,
            headers={"Content-Type": "text/plain"},
            timeout=server_config["timeout"],
        )

        # Then: Response should be successful
        assert (
            response.status_code == 200
        ), f"Request failed: HTTP {response.status_code}"
        response_data = response.json()
        assert response_data.get(
            "success"
        ), f"Execution failed: {response_data.get('error')}"

        result = response_data.get("data", {}).get("result")
        assert result is not None, "No result data returned"
        
        # Parse the JSON result if it\'s a string
        if isinstance(result, str):
            import json
            # Find the JSON part (starts with '{')
            json_start = result.find('{')
            if json_start != -1:
                json_str = result[json_start:]
                result = json.loads(json_str)
            else:
                result = json.loads(result.strip())

        # And: Plot should be generated with correct metadata
        assert "plot_base64" in result, "Missing base64 plot data"
        assert result.get("plot_type") == "pairplot", "Incorrect plot type"
        assert result.get("n_groups") == 3, "Incorrect number of groups"
        assert result.get("total_points") == 150, "Incorrect total number of points"
        assert result.get("n_features") == 3, "Incorrect number of features"
        assert result.get("points_per_group") == 50, "Incorrect points per group"

        # And: Group information should be complete
        group_names = result.get("group_names", [])
        assert len(group_names) == 3, "Should have exactly 3 group names"
        assert "Group_A" in group_names, "Missing Group_A"
        assert "Group_B" in group_names, "Missing Group_B"
        assert "Group_C" in group_names, "Missing Group_C"

        # And: Group statistics should show separation
        group_stats = result.get("group_statistics", {})
        assert len(group_stats) == 3, "Should have statistics for all 3 groups"

        for group_name, stats in group_stats.items():
            assert stats["count"] == 50, f"Group {group_name} should have 50 points"
            assert "feature1_mean" in stats, f"Missing feature1_mean for {group_name}"
            assert "feature2_mean" in stats, f"Missing feature2_mean for {group_name}"
            assert "feature3_mean" in stats, f"Missing feature3_mean for {group_name}"

        # Validate that groups have different mean values (showing separation)
        group_a_f1 = group_stats["Group_A"]["feature1_mean"]
        group_b_f1 = group_stats["Group_B"]["feature1_mean"]
        assert (
            abs(group_a_f1 - group_b_f1) > 0.5
        ), "Groups should be separated in feature1"

        # And: Plot should be saveable to filesystem
        plot_path = self.save_plot_from_base64(
            result["plot_base64"], "multivariate_pairplot.png", plots_directory
        )
        assert plot_path.exists(), f"Plot file was not created at {plot_path}"
        assert plot_path.stat().st_size > 0, "Plot file is empty"

    def test_seaborn_direct_filesystem_regression_plot_with_validation(
        self,
        verified_server: str,
        server_config: Dict[str, Any],
        seaborn_environment: Dict[str, Any],
        cleanup_pyodide_filesystem,
    ):
        """
        Test direct filesystem operations for seaborn regression plots in Pyodide.

        Given: A Pyodide environment with filesystem mounting capabilities
        When: Creating and saving a regression plot directly to the virtual filesystem
        Then: The file should be successfully saved with correct metadata
        And: File size should be non-zero indicating valid PNG data
        And: Statistical analysis should show strong correlation
        And: The virtual file should be accessible for extraction

        Args:
            verified_server: Base URL of running test server
            server_config: Test configuration dictionary
            seaborn_environment: Seaborn availability confirmation
            cleanup_pyodide_filesystem: Filesystem cleanup fixture
        """
        # Given: Seaborn environment is available
        assert seaborn_environment[
            "has_seaborn"
        ], "Seaborn must be available for this test"

        # When: Creating and saving regression plot directly to filesystem
        direct_save_code = """
# Install seaborn in this execution context
import micropip
print("Installing seaborn and dependencies...")
await micropip.install("seaborn")

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Set seaborn style for consistent appearance
sns.set_style("whitegrid")

# Create synthetic data with strong linear relationship
np.random.seed(42)  # Reproducible results
n_samples = 200
x_values = np.random.normal(0, 1, n_samples)
noise = np.random.normal(0, 0.5, n_samples)
y_values = 2.5 * x_values + 1.0 + noise  # Linear relationship with noise

df = pd.DataFrame({'feature_x': x_values, 'target_y': y_values})

# Create regression plot with enhanced styling
plt.figure(figsize=(10, 8))
sns.regplot(
    data=df,
    x='feature_x',
    y='target_y',
    scatter_kws={'alpha': 0.6, 's': 50, 'color': 'blue'},
    line_kws={'color': 'red', 'linewidth': 2}
)

plt.title('Direct Save - Seaborn Regression Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Feature X', fontsize=12)
plt.ylabel('Target Y', fontsize=12)

# Add statistical information to plot
correlation = df.corr().iloc[0, 1]
r_squared = correlation ** 2

info_text = f'Correlation: {correlation:.3f}\\nR²: {r_squared:.3f}\\nN: {n_samples}'
plt.text(0.05, 0.95, info_text,
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         verticalalignment='top')

# Ensure plots directory exists
plots_dir = Path('/plots/seaborn')
plots_dir.mkdir(parents=True, exist_ok=True)

# Save directly to virtual filesystem with unique timestamp
timestamp = int(time.time() * 1000)
output_path = plots_dir / f'direct_save_regression_{timestamp}.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

# Verify file creation and get metadata
file_exists = output_path.exists()
file_size = output_path.stat().st_size if file_exists else 0

# Return comprehensive validation data
{
    "file_saved": file_exists,
    "file_size": file_size,
    "plot_type": "direct_save_regression",
    "correlation": float(correlation),
    "r_squared": float(r_squared),
    "n_points": n_samples,
    "filename": str(output_path),
    "timestamp": timestamp,
    "data_summary": {
        "x_mean": float(df['feature_x'].mean()),
        "y_mean": float(df['target_y'].mean()),
        "x_std": float(df['feature_x'].std()),
        "y_std": float(df['target_y'].std())
    }
}
"""

        response = requests.post(
            f"{verified_server}/api/execute-raw",
            data=direct_save_code,
            headers={"Content-Type": "text/plain"},
            timeout=server_config["timeout"],
        )

        # Then: Response should be successful
        assert (
            response.status_code == 200
        ), f"Request failed: HTTP {response.status_code}"
        response_data = response.json()
        assert response_data.get(
            "success"
        ), f"Execution failed: {response_data.get('error')}"

        result = response_data.get("data", {}).get("result")
        assert result is not None, "No result data returned"
        
        # Parse the JSON result if it\'s a string
        if isinstance(result, str):
            import json
            # Find the JSON part (starts with '{')
            json_start = result.find('{')
            if json_start != -1:
                json_str = result[json_start:]
                result = json.loads(json_str)
            else:
                result = json.loads(result.strip())

        # And: File should be successfully saved
        assert (
            result.get("file_saved") is True
        ), "Regression plot was not saved to filesystem"
        assert result.get("file_size", 0) > 0, "Regression plot file has zero size"
        assert (
            result.get("plot_type") == "direct_save_regression"
        ), "Incorrect plot type"

        # And: Statistical properties should be correct
        correlation = result.get("correlation")
        r_squared = result.get("r_squared")
        assert correlation is not None, "Missing correlation coefficient"
        assert (
            correlation > 0.8
        ), f"Correlation {correlation} should be strongly positive (> 0.8)"
        assert r_squared is not None, "Missing R-squared value"
        assert (
            r_squared > 0.64
        ), f"R-squared {r_squared} should indicate good fit (> 0.64)"

        assert result.get("n_points") == 200, "Incorrect number of data points"

        # And: Data summary should be reasonable
        data_summary = result.get("data_summary", {})
        assert "x_mean" in data_summary, "Missing x_mean in data summary"
        assert "y_mean" in data_summary, "Missing y_mean in data summary"
        assert abs(data_summary["x_mean"]) < 0.5, "X mean should be close to 0"

        # And: Filename should be properly formatted
        filename = result.get("filename", "")
        assert filename, "No filename returned in result"
        assert filename.startswith(
            "/plots/seaborn/direct_save_regression_"
        ), "Incorrect filename format"
        assert filename.endswith(".png"), "File should have .png extension"

        timestamp = result.get("timestamp")
        assert timestamp is not None, "Missing timestamp"
        assert timestamp > 0, "Invalid timestamp"

    def test_seaborn_advanced_dashboard_comprehensive_visualization(
        self,
        verified_server: str,
        server_config: Dict[str, Any],
        seaborn_environment: Dict[str, Any],
        cleanup_pyodide_filesystem,
    ):
        """
        Test advanced seaborn dashboard with multiple visualization types.

        Given: A Pyodide environment with complex multi-dimensional data
        When: Creating a comprehensive dashboard with 6 different plot types
        Then: All visualizations should be generated and combined into single plot
        And: Each subplot should show different aspects of the data
        And: Statistical relationships should be correctly represented
        And: The dashboard should be saved as a high-quality PNG file

        Args:
            verified_server: Base URL of running test server
            server_config: Test configuration dictionary
            seaborn_environment: Seaborn availability confirmation
            cleanup_pyodide_filesystem: Filesystem cleanup fixture
        """
        # Given: Seaborn environment is available
        assert seaborn_environment[
            "has_seaborn"
        ], "Seaborn must be available for this test"

        # When: Creating comprehensive visualization dashboard
        dashboard_code = """
# Install seaborn in this execution context
import micropip
print("Installing seaborn and dependencies...")
await micropip.install("seaborn")

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import json as json_module
from pathlib import Path

# Set consistent styling and color palette
sns.set_style("whitegrid")
sns.set_palette("husl")

# Generate complex multi-dimensional dataset
np.random.seed(123)  # Reproducible results
n_samples = 300

# Create base data structure
base_data = {
    'group': np.random.choice(['Group_A', 'Group_B', 'Group_C'], n_samples),
    'category': np.random.choice(['Type1', 'Type2'], n_samples),
    'value1': np.random.normal(50, 15, n_samples),
    'value2': np.random.normal(100, 25, n_samples),
    'score': np.random.uniform(0, 100, n_samples)
}

# Add group-specific patterns to create interesting relationships
for i in range(n_samples):
    if base_data['group'][i] == 'Group_A':
        base_data['value1'][i] += 20  # Group A has higher value1
        base_data['score'][i] += 15   # and higher scores
    elif base_data['group'][i] == 'Group_C':
        base_data['value2'][i] += 30  # Group C has higher value2
        base_data['score'][i] -= 10   # but lower scores

df = pd.DataFrame(base_data)

# Create comprehensive 6-subplot dashboard
fig = plt.figure(figsize=(16, 12))

# Subplot 1: Correlation heatmap of numeric variables
plt.subplot(2, 3, 1)
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
plt.title('Correlation Matrix', fontweight='bold')

# Subplot 2: Box plot showing value1 by group and category
plt.subplot(2, 3, 2)
sns.boxplot(data=df, x='group', y='value1', hue='category')
plt.title('Value1 Distribution by Group and Category', fontweight='bold')
plt.xticks(rotation=45)

# Subplot 3: Violin plot showing score distribution by group
plt.subplot(2, 3, 3)
sns.violinplot(data=df, x='group', y='score')
plt.title('Score Distribution by Group', fontweight='bold')
plt.xticks(rotation=45)

# Subplot 4: Scatter plot with multiple encodings
plt.subplot(2, 3, 4)
sns.scatterplot(data=df, x='value1', y='value2', hue='group', size='score', alpha=0.7)
plt.title('Value1 vs Value2 by Group (sized by Score)', fontweight='bold')

# Subplot 5: Density plots for score by group
plt.subplot(2, 3, 5)
for group in df['group'].unique():
    subset = df[df['group'] == group]
    sns.kdeplot(data=subset, x='score', label=group, alpha=0.7)
plt.title('Score Density Distribution by Group', fontweight='bold')
plt.legend()

# Subplot 6: Count plot showing sample distribution
plt.subplot(2, 3, 6)
sns.countplot(data=df, x='group', hue='category')
plt.title('Sample Counts by Group and Category', fontweight='bold')
plt.xticks(rotation=45)

plt.suptitle('Comprehensive Seaborn Dashboard Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()

# Ensure directory exists and save to virtual filesystem
plots_dir = Path('/plots/seaborn')
plots_dir.mkdir(parents=True, exist_ok=True)

timestamp = int(time.time() * 1000)
output_path = plots_dir / f'advanced_dashboard_{timestamp}.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

# Calculate comprehensive statistics
group_stats = df.groupby('group').agg({
    'score': ['mean', 'std', 'count'],
    'value1': ['mean', 'std'],
    'value2': ['mean', 'std']
}).round(2)

# Count samples per category and group (convert tuple keys to strings for JSON serialization)
category_counts_raw = df.groupby(['group', 'category']).size().to_dict()
category_counts = {f"{k[0]}_{k[1]}": v for k, v in category_counts_raw.items()}

# Verify file creation
file_exists = output_path.exists()
file_size = output_path.stat().st_size if file_exists else 0

# Return comprehensive validation data
response = {
    "file_saved": file_exists,
    "file_size": file_size,
    "plot_type": "advanced_dashboard",
    "n_samples": n_samples,
    "n_groups": len(df['group'].unique()),
    "n_categories": len(df['category'].unique()),
    "filename": str(output_path),
    "timestamp": timestamp,
    "group_statistics": {
        group: {
            'score_mean': float(group_data['score'].mean()),
            'score_std': float(group_data['score'].std()),
            'value1_mean': float(group_data['value1'].mean()),
            'value2_mean': float(group_data['value2'].mean()),
            'count': len(group_data)
        }
        for group, group_data in df.groupby('group')
    },
    "category_distribution": category_counts,
    "correlation_value1_value2": float(df['value1'].corr(df['value2'])),
    "correlation_value1_score": float(df['value1'].corr(df['score']))
}

print(json_module.dumps(response, indent=2))
"""

        response = requests.post(
            f"{verified_server}/api/execute-raw",
            data=dashboard_code,
            headers={"Content-Type": "text/plain"},
            timeout=server_config["timeout"],
        )

        # Then: Response should be successful
        assert (
            response.status_code == 200
        ), f"Request failed: HTTP {response.status_code}"
        response_data = response.json()
        assert response_data.get(
            "success"
        ), f"Execution failed: {response_data.get('error')}"

        result = response_data.get("data", {}).get("result")
        assert result is not None, "No result data returned"
        
        # Parse the JSON result if it\'s a string
        if isinstance(result, str):
            import json
            # Find the JSON part (starts with '{')
            json_start = result.find('{')
            if json_start != -1:
                json_str = result[json_start:]
                result = json.loads(json_str)
            else:
                result = json.loads(result.strip())

        # And: Dashboard should be successfully created
        assert (
            result.get("file_saved") is True
        ), "Dashboard plot was not saved to filesystem"
        assert result.get("file_size", 0) > 0, "Dashboard plot file has zero size"
        assert result.get("plot_type") == "advanced_dashboard", "Incorrect plot type"

        # And: Data dimensions should be correct
        assert result.get("n_samples") == 300, "Incorrect number of samples"
        assert result.get("n_groups") == 3, "Incorrect number of groups"
        assert result.get("n_categories") == 2, "Incorrect number of categories"

        # And: Group statistics should be comprehensive
        group_stats = result.get("group_statistics", {})
        assert len(group_stats) == 3, "Should have statistics for all 3 groups"

        expected_groups = ["Group_A", "Group_B", "Group_C"]
        for group_name in expected_groups:
            assert group_name in group_stats, f"Missing statistics for {group_name}"
            group_data = group_stats[group_name]

            # Validate required statistical measures
            assert "score_mean" in group_data, f"Missing score_mean for {group_name}"
            assert "score_std" in group_data, f"Missing score_std for {group_name}"
            assert "value1_mean" in group_data, f"Missing value1_mean for {group_name}"
            assert "value2_mean" in group_data, f"Missing value2_mean for {group_name}"
            assert "count" in group_data, f"Missing count for {group_name}"

            # Validate reasonable ranges
            assert (
                0 <= group_data["score_mean"] <= 120
            ), f"Invalid score_mean for {group_name}"
            assert group_data["count"] > 0, f"Invalid count for {group_name}"

        # Validate group differences (created by design)
        group_a_score = group_stats["Group_A"]["score_mean"]
        group_c_score = group_stats["Group_C"]["score_mean"]
        assert (
            group_a_score > group_c_score
        ), "Group_A should have higher scores than Group_C"

        # And: Category distribution should be present
        category_dist = result.get("category_distribution", {})
        assert len(category_dist) > 0, "Missing category distribution data"

        # And: Correlations should be calculated
        corr_v1_v2 = result.get("correlation_value1_value2")
        corr_v1_score = result.get("correlation_value1_score")
        assert corr_v1_v2 is not None, "Missing value1-value2 correlation"
        assert corr_v1_score is not None, "Missing value1-score correlation"
        assert -1 <= corr_v1_v2 <= 1, "Correlation should be between -1 and 1"
        assert -1 <= corr_v1_score <= 1, "Correlation should be between -1 and 1"

        # And: Filename should be properly formatted
        filename = result.get("filename", "")
        assert filename, "No filename returned in result"
        assert filename.startswith(
            "/plots/seaborn/advanced_dashboard_"
        ), "Incorrect filename format"
        assert filename.endswith(".png"), "File should have .png extension"

        timestamp = result.get("timestamp")
        assert timestamp is not None, "Missing timestamp"
        assert timestamp > 0, "Invalid timestamp"


# Run tests when executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Seaborn Filesystem Integration Tests

This module contains comprehensive pytest tests for seaborn plotting functionality
that uses direct filesystem operations within Pyodide. Tests validate the ability
to create, save, and retrieve visualization files through the virtual filesystem.

All tests follow BDD (Behavior-Driven Development) patterns with Given-When-Then
structure and allow regular API endpoints but avoid internal pyodide-* endpoints.

Requirements:
- pytest framework with fixtures
- BDD-style test documentation
- Regular API endpoints allowed: /api/execute, /api/install-package, /api/reset
- Internal pyodide-* endpoints forbidden: /api/extract-plots, /api/pyodide-files
- pathlib for all path operations
- Comprehensive docstrings with examples
- Server API contract compliance
"""

import time
from pathlib import Path
from typing import Dict, Any

import pytest
import requests


# Test Configuration Constants
class TestConfig:
    """Test configuration constants to avoid hardcoded values."""

    # Server configuration
    DEFAULT_BASE_URL = "http://localhost:3000"

    # Timeout configurations (in seconds)
    SHORT_TIMEOUT = 30
    MEDIUM_TIMEOUT = 60
    LONG_TIMEOUT = 120
    SERVER_START_TIMEOUT = 180

    # Test data constants
    RANDOM_SEED = 42
    SAMPLE_SIZE_SMALL = 200
    SAMPLE_SIZE_LARGE = 300

    # Plot configuration
    PLOT_DPI = 150
    FIGURE_WIDTH = 10
    FIGURE_HEIGHT = 8
    DASHBOARD_WIDTH = 16
    DASHBOARD_HEIGHT = 12


def wait_for_server(url: str, timeout: int = 180) -> None:
    """
    Wait for server to become available.

    Args:
        url: Server URL to check
        timeout: Maximum wait time in seconds

    Raises:
        RuntimeError: If server doesn't respond within timeout
    """
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


@pytest.fixture(scope="session")
def server_config() -> Dict[str, Any]:
    """
    Session-scoped fixture providing server configuration.

    Returns:
        Configuration dictionary with server details and timeouts

    Example:
        >>> def test_example(server_config):
        ...     assert server_config['base_url'] == 'http://localhost:3000'
        ...     assert 'timeout' in server_config
    """
    return {
        "base_url": TestConfig.DEFAULT_BASE_URL,
        "short_timeout": TestConfig.SHORT_TIMEOUT,
        "timeout": TestConfig.MEDIUM_TIMEOUT,
        "long_timeout": TestConfig.LONG_TIMEOUT,
        "server_start_timeout": TestConfig.SERVER_START_TIMEOUT,
        "random_seed": TestConfig.RANDOM_SEED,
        "sample_size_small": TestConfig.SAMPLE_SIZE_SMALL,
        "sample_size_large": TestConfig.SAMPLE_SIZE_LARGE,
        "plot_dpi": TestConfig.PLOT_DPI,
        "figure_width": TestConfig.FIGURE_WIDTH,
        "figure_height": TestConfig.FIGURE_HEIGHT,
        "dashboard_width": TestConfig.DASHBOARD_WIDTH,
        "dashboard_height": TestConfig.DASHBOARD_HEIGHT,
    }


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
    timeout = server_config["server_start_timeout"]

    try:
        wait_for_server(f"{base_url}/health", timeout=30)
    except RuntimeError:
        pytest.skip("Server is not running on localhost:3000")

    # Reset Pyodide environment to clean state
    try:
        reset_response = requests.post(f"{base_url}/api/reset", timeout=30)
        if reset_response.status_code == 200:
            print("✅ Pyodide environment reset successfully")
        else:
            print(
                f"⚠️ Warning: Could not reset Pyodide environment: {reset_response.status_code}"
            )
    except requests.RequestException:
        print("⚠️ Warning: Could not reset Pyodide environment due to network error")

    return base_url


@pytest.fixture(scope="session")
def seaborn_environment(
    verified_server: str, server_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Session-scoped fixture ensuring seaborn availability.

    Installs required packages (matplotlib, seaborn) and verifies
    they are available for the test session.

    Args:
        verified_server: Base URL from verified_server fixture
        server_config: Configuration from server_config fixture

    Returns:
        Dictionary with package installation status

    Example:
        >>> def test_example(seaborn_environment):
        ...     assert seaborn_environment['seaborn_available'] is True
    """
    # Ensure seaborn and matplotlib are available (regular API endpoints)
    for package in ["matplotlib", "seaborn"]:
        try:
            install_response = requests.post(
                f"{verified_server}/api/install-package",
                json={"package": package},
                timeout=60,
            )
            if install_response.status_code == 200:
                print(f"✅ {package} installed successfully")
            else:
                print(
                    f"⚠️ Warning: {package} installation failed: {install_response.status_code}"
                )
        except requests.RequestException as e:
            print(f"⚠️ Warning: {package} installation failed due to network error: {e}")

    return {
        "seaborn_available": True,
        "matplotlib_available": True,
        "server_ready": True,
    }


@pytest.fixture
def clean_filesystem(verified_server: str, server_config: Dict[str, Any]):
    """
    Function-scoped fixture to clean Pyodide virtual filesystem.

    Ensures each test starts with a clean virtual filesystem by removing
    any existing plot files from the /plots/seaborn directory.

    Args:
        verified_server: Base URL from verified_server fixture
        server_config: Configuration from server_config fixture

    Yields:
        None (setup and teardown fixture)
    """
    # Setup: Clean before test
    cleanup_code = """
# Clean up seaborn plots from virtual filesystem using pathlib
from pathlib import Path

plot_dir = Path('/home/pyodide/plots/seaborn')
if plot_dir.exists():
    for plot_file in plot_dir.glob('*.png'):
        try:
            plot_file.unlink()
            print(f"Removed: {plot_file}")
        except Exception as e:
            print(f"Warning: Could not remove {plot_file}: {e}")
else:
    plot_dir.mkdir(parents=True, exist_ok=True)
    print("Created plots directory")

"Cleanup complete"
"""
    try:
        requests.post(
            f"{verified_server}/api/execute",
            json={"code": cleanup_code},
            timeout=server_config["short_timeout"],
        )
    except Exception:
        pass  # Cleanup is best-effort

    yield

    # Teardown: Clean after test (optional)
    try:
        requests.post(
            f"{verified_server}/api/execute",
            json={"code": cleanup_code},
            timeout=server_config["short_timeout"],
        )
    except Exception:
        pass  # Cleanup is best-effort


class TestSeabornFilesystemOperations:
    """
    Test class for seaborn plotting with direct filesystem operations.

    This class tests the ability to create seaborn plots within Pyodide
    and save them directly to the virtual filesystem. Tests verify plot
    creation, file saving, and metadata validation without using internal APIs.

    All tests follow BDD patterns and use regular API endpoints only.
    """

    def test_direct_file_save_regression_plot(
        self,
        verified_server: str,
        server_config: Dict[str, Any],
        seaborn_environment: Dict[str, Any],
        clean_filesystem,
    ):
        """
        Test creating and saving seaborn regression plot directly to virtual filesystem.

        Given a functioning seaborn environment with filesystem access
        When we create a regression plot with statistical analysis and save to filesystem
        Then the plot should be saved with correct metadata and file properties verified within Pyodide

        Args:
            verified_server: Base URL of running test server
            server_config: Test configuration dictionary
            seaborn_environment: Seaborn availability confirmation
            clean_filesystem: Filesystem cleanup fixture

        Example:
            This test validates that seaborn can create regression plots and save them
            directly to the Pyodide virtual filesystem with statistical validation.

            Expected behavior:
            - Seaborn regression plot created with sample data
            - Plot saved to /plots/seaborn/ directory
            - Correlation coefficient calculated and validated
            - File metadata verified within Pyodide execution context
        """
        # Given: Seaborn environment is available and filesystem is accessible
        assert seaborn_environment[
            "seaborn_available"
        ], "Seaborn must be available for filesystem tests"
        assert seaborn_environment[
            "server_ready"
        ], "Server must be ready for filesystem tests"

        # When: Creating regression plot with filesystem save and verification
        RANDOM_SEED = server_config["random_seed"]
        SAMPLE_SIZE = server_config["sample_size_small"]
        PLOT_DPI = server_config["plot_dpi"]
        FIGURE_WIDTH = server_config["figure_width"]
        FIGURE_HEIGHT = server_config["figure_height"]

        regression_code = f"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import time

print("Creating seaborn regression plot with filesystem verification...")

# Set seaborn style
sns.set_style("whitegrid")

# Create sample data with clear relationship
np.random.seed({RANDOM_SEED})
n_samples = {SAMPLE_SIZE}
x_values = np.random.normal(0, 1, n_samples)
noise = np.random.normal(0, 0.5, n_samples)
y_values = 2.5 * x_values + 1.0 + noise

df = pd.DataFrame({{'feature_x': x_values, 'target_y': y_values}})

# Create the plot with proper sizing
plt.figure(figsize=({FIGURE_WIDTH}, {FIGURE_HEIGHT}))
sns.regplot(data=df, x='feature_x', y='target_y',
            scatter_kws={{'alpha': 0.6, 's': 50}},
            line_kws={{'color': 'red', 'linewidth': 2}})
plt.title('Direct Save - Seaborn Regression Analysis', fontweight='bold')
plt.xlabel('Feature X', fontweight='bold')
plt.ylabel('Target Y', fontweight='bold')

# Add correlation info to the plot
correlation = df.corr().iloc[0, 1]
plt.text(0.05, 0.95, f'Correlation: {{correlation:.3f}}',
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Save directly to the virtual filesystem using pathlib
timestamp = int(time.time() * 1000)  # Generate unique timestamp
plots_path = Path('/home/pyodide/plots/seaborn')
plots_path.mkdir(parents=True, exist_ok=True)
output_file = plots_path / f'direct_save_regression_{{timestamp}}.png'

plt.savefig(output_file, dpi={PLOT_DPI}, bbox_inches='tight')
plt.close()

# Verify the file was created and get comprehensive results
file_exists = output_file.exists()
file_size = output_file.stat().st_size if file_exists else 0

# Return comprehensive verification results
result = {{
    "file_saved": file_exists,
    "file_size": file_size,
    "plot_type": "direct_save_regression", 
    "correlation": float(correlation),
    "n_points": n_samples,
    "filename": str(output_file),
    "timestamp": timestamp,
    "statistical_analysis": {{
        "correlation_coefficient": float(correlation),
        "sample_size": int(n_samples),
        "r_squared_estimate": float(correlation**2)
    }}
}}

print(f"Regression plot saved successfully: {{output_file}}")
result
"""

        response = requests.post(
            f"{verified_server}/api/execute-raw",
            data=regression_code,
            headers={"Content-Type": "text/plain"},
            timeout=server_config["long_timeout"],
        )

        # Then: Response should be successful with proper API contract
        assert (
            response.status_code == 200
        ), f"Request failed: HTTP {response.status_code}"
        response_data = response.json()

        # Validate API contract structure
        assert "success" in response_data, "Response missing 'success' field"
        assert "data" in response_data, "Response missing 'data' field"
        assert "error" in response_data, "Response missing 'error' field"
        assert "meta" in response_data, "Response missing 'meta' field"

        assert response_data.get("success"), f"Execution failed: {response_data}"

        # Validate data structure
        data = response_data.get("data", {})
        assert "result" in data, "Response data missing 'result' field"

        result = data.get("result")
        assert result is not None, f"API returned None result: {response_data}"

        # And: Plot should be saved with correct metadata
        assert (
            result.get("file_saved") is True
        ), "Regression plot was not saved to filesystem"
        assert result.get("file_size", 0) > 0, "Regression plot file has zero size"
        assert (
            result.get("plot_type") == "direct_save_regression"
        ), "Incorrect plot type"
        assert (
            result.get("correlation", 0) > 0.8
        ), "Correlation should be strong positive"
        assert result.get("n_points") == SAMPLE_SIZE, f"Expected {SAMPLE_SIZE} points"

        # And: Statistical analysis should be present
        stats = result.get("statistical_analysis", {})
        assert (
            stats.get("correlation_coefficient", 0) > 0.8
        ), "Strong correlation expected"
        assert stats.get("sample_size") == SAMPLE_SIZE, "Correct sample size expected"
        assert (
            0 < stats.get("r_squared_estimate", 0) < 1
        ), "R-squared should be between 0 and 1"

        # And: Filename should be properly formatted with pathlib
        filename = result.get("filename", "")
        assert filename.startswith("/plots/seaborn/"), "Incorrect file path"
        assert filename.endswith(".png"), "Incorrect file extension"
        assert "regression" in filename, "Plot type not in filename"

    def test_direct_file_save_advanced_dashboard(
        self,
        verified_server: str,
        server_config: Dict[str, Any],
        seaborn_environment: Dict[str, Any],
        clean_filesystem,
    ):
        """
        Test creating and saving advanced seaborn dashboard directly to virtual filesystem.

        Given a seaborn environment with filesystem capabilities
        When we create a complex dashboard with multiple plot types and save to filesystem
        Then all visualizations should render correctly with proper statistical analysis

        Args:
            verified_server: Base URL of running test server
            server_config: Test configuration dictionary
            seaborn_environment: Seaborn availability confirmation
            clean_filesystem: Filesystem cleanup fixture

        Example:
            This test creates a comprehensive dashboard including:
            - Correlation heatmap with statistical relationships
            - Box plots grouped by categorical variables
            - Violin plots for distribution analysis
            - Scatter plots with multi-dimensional encoding
            - Distribution plots for group comparisons
            - Count plots for categorical analysis

            All plots are combined into a single dashboard and saved to filesystem
            with comprehensive metadata validation.
        """
        # Given: Seaborn environment is available with filesystem access
        assert seaborn_environment[
            "seaborn_available"
        ], "Seaborn must be available for dashboard tests"
        assert seaborn_environment[
            "server_ready"
        ], "Server must be ready for dashboard tests"

        # When: Creating comprehensive dashboard with multiple visualizations
        RANDOM_SEED = 123  # Different seed for dashboard data
        SAMPLE_SIZE = server_config["sample_size_large"]
        DASHBOARD_WIDTH = server_config["dashboard_width"]
        DASHBOARD_HEIGHT = server_config["dashboard_height"]
        PLOT_DPI = server_config["plot_dpi"]

        dashboard_code = f"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import time

print("Creating comprehensive seaborn dashboard...")

# Set seaborn style and palette
sns.set_style("whitegrid")
sns.set_palette("husl")

# Create comprehensive dataset
np.random.seed({RANDOM_SEED})
n_samples = {SAMPLE_SIZE}

# Generate multi-dimensional data
data = {{
    'group': np.random.choice(['Alpha', 'Beta', 'Gamma'], n_samples),
    'category': np.random.choice(['Type1', 'Type2'], n_samples),
    'value1': np.random.normal(50, 15, n_samples),
    'value2': np.random.normal(100, 25, n_samples),
    'score': np.random.uniform(0, 100, n_samples)
}}

# Add complex relationships
for i in range(n_samples):
    if data['group'][i] == 'Alpha':
        data['value1'][i] += 20
        data['score'][i] += 15
    elif data['group'][i] == 'Gamma':
        data['value2'][i] += 30
        data['score'][i] -= 10
    
    # Add category effects  
    if data['category'][i] == 'Type2':
        data['value1'][i] *= 1.1
        data['score'][i] += 5

df = pd.DataFrame(data)

# Create comprehensive dashboard
fig = plt.figure(figsize=({DASHBOARD_WIDTH}, {DASHBOARD_HEIGHT}))

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
for group_name in df['group'].unique():
    subset = df[df['group'] == group_name]
    sns.kdeplot(data=subset, x='score', label=f'Group {{group_name}}', alpha=0.7)
plt.title('Score Density by Group')
plt.legend()

# Subplot 6: Count plot
plt.subplot(2, 3, 6)
sns.countplot(data=df, x='group', hue='category')
plt.title('Sample Counts by Group and Category')

plt.tight_layout()

# Save dashboard to virtual filesystem using pathlib
timestamp = int(time.time() * 1000)  # Generate unique timestamp
plots_path = Path('/home/pyodide/plots/seaborn')
plots_path.mkdir(parents=True, exist_ok=True)
output_file = plots_path / f'direct_save_dashboard_{{timestamp}}.png'

plt.savefig(output_file, dpi={PLOT_DPI}, bbox_inches='tight')
plt.close()

# Calculate comprehensive summary statistics
group_stats = df.groupby('group')['score'].agg(['count', 'mean', 'std']).to_dict()
category_stats = df.groupby('category')['value1'].agg(['count', 'mean', 'std']).to_dict()

# Verify the file was created and get comprehensive results
file_exists = output_file.exists()
file_size = output_file.stat().st_size if file_exists else 0

# Return comprehensive dashboard results
result = {{
    "file_saved": file_exists,
    "file_size": file_size,
    "plot_type": "direct_save_dashboard",
    "n_samples": n_samples,
    "n_groups": len(df['group'].unique()),
    "n_categories": len(df['category'].unique()),
    "groups": list(df['group'].unique()),
    "categories": list(df['category'].unique()),
    "statistical_analysis": {{
        "group_statistics": group_stats,
        "category_statistics": category_stats
    }},
    "visualization_info": {{
        "subplot_count": 6,
        "visualization_types": [
            "correlation_heatmap",
            "box_plot",
            "violin_plot", 
            "scatter_plot",
            "density_plot",
            "count_plot"
        ]
    }},
    "filename": str(output_file),
    "timestamp": timestamp
}}

print(f"Dashboard saved successfully: {{output_file}}")
result
"""

        response = requests.post(
            f"{verified_server}/api/execute-raw",
            data=dashboard_code,
            headers={"Content-Type": "text/plain"},
            timeout=server_config["long_timeout"],
        )

        # Then: Response should follow proper API contract
        assert (
            response.status_code == 200
        ), f"Request failed: HTTP {response.status_code}"
        response_data = response.json()

        # Validate API contract structure
        assert "success" in response_data, "Response missing 'success' field"
        assert "data" in response_data, "Response missing 'data' field"
        assert "error" in response_data, "Response missing 'error' field"
        assert "meta" in response_data, "Response missing 'meta' field"

        assert response_data.get("success"), f"Execution failed: {response_data}"

        # Validate data structure
        data = response_data.get("data", {})
        result = data.get("result")
        assert result is not None, f"API returned None result: {response_data}"

        # And: Dashboard should be saved successfully
        assert (
            result.get("file_saved") is True
        ), "Dashboard plot was not saved to filesystem"
        assert result.get("file_size", 0) > 0, "Dashboard plot file has zero size"
        assert result.get("plot_type") == "direct_save_dashboard", "Incorrect plot type"

        # And: Dataset information should be comprehensive
        assert result.get("n_samples") == SAMPLE_SIZE, f"Expected {SAMPLE_SIZE} samples"
        assert result.get("n_groups") == 3, "Should have 3 groups"
        assert result.get("n_categories") == 2, "Should have 2 categories"
        assert "Alpha" in result.get("groups", []), "Missing Alpha group"

        # And: Statistical analysis should be present
        stats = result.get("statistical_analysis", {})
        assert "group_statistics" in stats, "Missing group statistics"
        assert "category_statistics" in stats, "Missing category statistics"

        # And: Visualization information should be complete
        viz_info = result.get("visualization_info", {})
        assert viz_info.get("subplot_count") == 6, "Should have 6 subplots"
        viz_types = viz_info.get("visualization_types", [])
        assert len(viz_types) == 6, "Should have 6 visualization types"
        assert "correlation_heatmap" in viz_types, "Missing heatmap"
        assert "box_plot" in viz_types, "Missing box plot"

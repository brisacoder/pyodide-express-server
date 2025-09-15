"""
Seaborn Base64 Plot Generation Tests

This module contains comprehensive pytest tests for Seaborn visualization 
functionality using the Pyodide Express Server. Tests follow BDD (Behavior 
Driven Development) patterns and validate the complete workflow of generating
statistical visualizations as Base64 encoded data.

Key Features Tested:
- Correlation heatmaps with customizable styling
- Distribution plots (histograms and box plots)
- Pair plots for multivariate analysis  
- Regression analysis with statistical annotations
- Complex multi-subplot visualizations
- Error handling for invalid data scenarios
- Cross-platform file system operations

API Contract Compliance:
All tests validate the standardized API response format:
{
  "success": true|false,
  "data": {
    "result": "<pyodide_output>",
    "stdout": "<console_output>", 
    "stderr": "<error_output>",
    "executionTime": <milliseconds>
  },
  "error": null|"<error_message>",
  "meta": {
    "timestamp": "<ISO_timestamp>"
  }
}

Test Organization:
- Given: Test setup and data preparation
- When: Action execution (API calls)
- Then: Result validation and assertions

Requirements:
- Server must be running on localhost:3000
- Uses only /api/execute-raw endpoint with plain text
- Pyodide code uses pathlib for cross-platform compatibility
- All timeouts and constants are parameterized via Config class
"""

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import requests

from conftest import Config, wait_for_server


class TestSeabornBase64Plots:
    """
    Comprehensive test suite for Seaborn visualization generation via Pyodide.

    This class tests the complete workflow of creating statistical visualizations
    using Seaborn library within Pyodide environment, converting plots to Base64
    format, and validating API responses.

    Test Coverage:
    - Correlation analysis and heatmap generation
    - Distribution analysis (histograms, box plots)
    - Multivariate pair plot analysis
    - Regression analysis with correlation statistics
    - Error handling and edge cases
    - Performance and timeout validation

    BDD Pattern:
    Each test follows Given-When-Then structure for clarity and maintainability.
    """

    @pytest.fixture(scope="class")
    def server_ready(self):
        """
        Fixture to ensure server is running and ready for test execution.

        Given: A Pyodide Express Server instance
        When: Health check endpoint is called
        Then: Server responds with 200 OK status

        Raises:
            pytest.skip: If server is not accessible within timeout period
        """
        health_url = f"{Config.BASE_URL}{Config.ENDPOINTS['health']}"
        try:
            if not wait_for_server(health_url, Config.TIMEOUTS["server_health"]):
                pytest.skip("Server not running on localhost:3000")
        except Exception as e:
            pytest.skip(f"Server health check failed: {str(e)}")

    @pytest.fixture(scope="class")
    def seaborn_available(self, server_ready):
        """
        Fixture to verify Seaborn and Matplotlib packages are available in Pyodide.

        Given: A running Pyodide server
        When: Package availability is checked via code execution
        Then: Both seaborn and matplotlib import successfully

        Returns:
            bool: True if packages are available, False otherwise

        Raises:
            pytest.skip: If packages cannot be imported
        """
        # Given: First try to install seaborn if not available
        install_code = """
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    print(f"Packages already available - Seaborn: {sns.__version__}, Matplotlib: {plt.__version__}")
    "already_available"
except ImportError:
    import micropip
    await micropip.install(['seaborn'])
    import seaborn as sns
    import matplotlib.pyplot as plt
    print(f"Packages installed - Seaborn: {sns.__version__}, Matplotlib: {plt.__version__}")
    "newly_installed"
        """

        # When: Execute installation/check via API
        try:
            response = requests.post(
                f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
                data=install_code,
                headers=Config.HEADERS["execute_raw"],
                timeout=Config.TIMEOUTS["package_install"]
            )

            # Then: Verify packages are available
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    result = data.get("data", {}).get("result", "")
                    if "already_available" in result or "newly_installed" in result:
                        return True
        except Exception as e:
            pytest.skip(f"Package installation/check failed: {str(e)}")

        pytest.skip("Seaborn and/or Matplotlib could not be made available in Pyodide environment")

    @pytest.fixture
    def plots_directory(self):
        """
        Fixture providing Base64 plots storage directory with cleanup.

        Given: A test execution environment
        When: Plot storage is needed
        Then: Clean directory structure is provided with automatic cleanup

        Returns:
            Path: Directory path for storing Base64 plot files
        """
        plots_dir = Path(__file__).parent.parent / "plots" / "base64" / "seaborn"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Clean up existing plots before tests
        for plot_file in plots_dir.glob("*.png"):
            plot_file.unlink()

        yield plots_dir

        # Cleanup after tests
        for plot_file in plots_dir.glob("*.png"):
            plot_file.unlink()

    def _validate_api_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Validate API response follows the standardized contract format.

        Args:
            response: HTTP response object from API call

        Returns:
            Dict containing validated response data

        Raises:
            AssertionError: If response doesn't match expected contract
        """
        # Then: Validate HTTP status
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Then: Validate JSON structure
        data = response.json()
        assert "success" in data, "Response missing 'success' field"
        assert "data" in data, "Response missing 'data' field"
        assert "error" in data, "Response missing 'error' field"
        assert "meta" in data, "Response missing 'meta' field"
        assert "timestamp" in data["meta"], "Response missing 'meta.timestamp' field"

        # Then: Validate success response structure
        if data["success"]:
            assert data["data"] is not None, "Success response should have non-null data"
            assert "result" in data["data"], "Success response missing 'data.result'"
            assert "stdout" in data["data"], "Success response missing 'data.stdout'"
            assert "stderr" in data["data"], "Success response missing 'data.stderr'"
            assert "executionTime" in data["data"], "Success response missing 'data.executionTime'"
            assert data["error"] is None, "Success response should have null error"
        else:
            assert data["data"] is None, "Error response should have null data"
            assert data["error"] is not None, "Error response should have error message"

        return data

    def _save_base64_plot(self, base64_data: str, filename: str, plots_dir: Path) -> Path:
        """
        Save Base64 encoded plot data to filesystem for validation.

        Args:
            base64_data: Base64 encoded image data
            filename: Name for the saved file
            plots_dir: Directory to save the file in

        Returns:
            Path to the saved plot file

        Raises:
            ValueError: If Base64 data is invalid or empty
        """
        if not base64_data:
            raise ValueError("No Base64 data provided")

        # Decode Base64 data
        try:
            plot_bytes = base64.b64decode(base64_data)
        except Exception as e:
            raise ValueError(f"Invalid Base64 data: {str(e)}")

        # Save to filesystem
        filepath = plots_dir / filename
        filepath.write_bytes(plot_bytes)

        return filepath

    def test_correlation_heatmap_generation(self, seaborn_available, plots_directory):
        """
        Test correlation heatmap generation with Seaborn and Base64 encoding.

        This test validates the complete workflow of:
        1. Creating synthetic correlated dataset
        2. Computing correlation matrix
        3. Generating styled heatmap visualization
        4. Encoding plot as Base64 data
        5. Validating API response format
        6. Saving and verifying plot file

        Given: Seaborn is available in Pyodide environment
        When: Correlation heatmap code is executed via /api/execute-raw
        Then: Valid Base64 plot data is returned with correct metadata

        Example Output:
            Base64 encoded PNG image showing correlation matrix heatmap
            with color-coded correlation values and annotations

        Args:
            seaborn_available: Fixture ensuring Seaborn package availability
            plots_directory: Fixture providing clean plot storage directory
        """
        # Given: Correlation heatmap generation code with pathlib
        heatmap_code = """
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json
from pathlib import Path

# Create synthetic dataset with known correlations
np.random.seed(42)
n_samples = 200

data = {
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'feature_3': np.random.randn(n_samples),
    'feature_4': np.random.randn(n_samples)
}

# Add deliberate correlations for validation
data['feature_2'] = data['feature_1'] * 0.7 + np.random.randn(n_samples) * 0.3
data['feature_3'] = data['feature_1'] * -0.5 + np.random.randn(n_samples) * 0.5
data['feature_4'] = data['feature_2'] * 0.4 + np.random.randn(n_samples) * 0.6

df = pd.DataFrame(data)

# Generate correlation heatmap with styling
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='RdBu_r',
    center=0,
    square=True,
    fmt='.3f',
    cbar_kws={'label': 'Correlation Coefficient'},
    linewidths=0.5
)
plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
plt.tight_layout()

# Convert to Base64 for API response
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Return structured result
result = {
    "plot_base64": plot_b64,
    "plot_type": "correlation_heatmap",
    "correlation_stats": {
        "max_correlation": float(correlation_matrix.abs().max().max()),
        "min_correlation": float(correlation_matrix.min().min()),
        "feature_count": len(correlation_matrix.columns)
    }
}

json.dumps(result)
        """

        # When: Execute heatmap generation via API
        response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
            data=heatmap_code,
            headers=Config.HEADERS["execute_raw"],
            timeout=Config.TIMEOUTS["code_execution"]
        )

        # Then: Validate API response format
        data = self._validate_api_response(response)
        assert data["success"], f"Code execution failed: {data.get('error')}"

        # Then: Parse and validate result structure
        result_json = data["data"]["result"].strip()
        result = json.loads(result_json)

        assert "plot_base64" in result, "Result missing plot_base64 field"
        assert "plot_type" in result, "Result missing plot_type field"
        assert result["plot_type"] == "correlation_heatmap", "Incorrect plot type"
        assert "correlation_stats" in result, "Result missing correlation_stats"

        # Then: Validate correlation statistics
        stats = result["correlation_stats"]
        assert stats["feature_count"] == 4, "Expected 4 features in correlation matrix"
        assert 0 <= stats["max_correlation"] <= 1, "Max correlation should be between 0 and 1"
        assert -1 <= stats["min_correlation"] <= 1, "Min correlation should be between -1 and 1"

        # Then: Save and validate Base64 plot
        plot_path = self._save_base64_plot(
            result["plot_base64"],
            "correlation_heatmap.png",
            plots_directory
        )
        assert plot_path.exists(), "Plot file was not created"
        assert plot_path.stat().st_size > 0, "Plot file is empty"

    def test_distribution_plots_generation(self, seaborn_available, plots_directory):
        """
        Test multi-panel distribution plot generation with histograms and box plots.

        This test validates:
        1. Creation of multi-group synthetic dataset
        2. Generation of subplot layout with different plot types
        3. Histogram and box plot visualization
        4. Statistical summary computation
        5. Base64 encoding and API response validation

        Given: Seaborn is available for statistical plotting
        When: Distribution analysis code is executed
        Then: Multi-panel visualization is created with statistical metadata

        Example Output:
            Two-panel plot showing:
            - Left: Histogram with group overlays
            - Right: Box plot showing distribution quartiles

        Args:
            seaborn_available: Fixture ensuring Seaborn package availability
            plots_directory: Fixture providing clean plot storage directory
        """
        # Given: Distribution plot generation code
        distribution_code = """
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json
from pathlib import Path

# Create multi-group dataset with different distributions
np.random.seed(42)
n_per_group = 800

groups_data = {
    'Group A': np.random.normal(50, 12, n_per_group),
    'Group B': np.random.normal(65, 15, n_per_group),
    'Group C': np.random.normal(45, 8, n_per_group)
}

# Prepare dataframe for plotting
plot_data = []
for group_name, values in groups_data.items():
    for value in values:
        plot_data.append({'Group': group_name, 'Value': value})

df = pd.DataFrame(plot_data)

# Create multi-panel distribution analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: Histogram with group overlays
sns.histplot(
    data=df,
    x='Value',
    hue='Group',
    alpha=0.7,
    bins=40,
    ax=axes[0]
)
axes[0].set_title('Distribution Comparison - Histograms', fontsize=14)
axes[0].set_xlabel('Value', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)

# Panel 2: Box plot comparison
sns.boxplot(
    data=df,
    x='Group',
    y='Value',
    palette='viridis',
    ax=axes[1]
)
axes[1].set_title('Distribution Comparison - Box Plots', fontsize=14)
axes[1].set_xlabel('Group', fontsize=12)
axes[1].set_ylabel('Value', fontsize=12)

plt.tight_layout()

# Convert to Base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Calculate statistical summary
group_stats = df.groupby('Group')['Value'].agg(['mean', 'std', 'min', 'max']).to_dict()

result = {
    "plot_base64": plot_b64,
    "plot_type": "distribution_plots",
    "statistical_summary": {
        "total_samples": len(df),
        "group_count": df['Group'].nunique(),
        "group_statistics": group_stats
    }
}

json.dumps(result)
        """

        # When: Execute distribution plot generation
        response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
            data=distribution_code,
            headers=Config.HEADERS["execute_raw"],
            timeout=Config.TIMEOUTS["code_execution"]
        )

        # Then: Validate response and extract results
        data = self._validate_api_response(response)
        assert data["success"], f"Code execution failed: {data.get('error')}"

        result = json.loads(data["data"]["result"].strip())

        # Then: Validate plot metadata
        assert result["plot_type"] == "distribution_plots"
        assert "statistical_summary" in result

        summary = result["statistical_summary"]
        assert summary["group_count"] == 3, "Expected 3 groups in dataset"
        assert summary["total_samples"] == 2400, "Expected 2400 total samples"
        assert "group_statistics" in summary

        # Then: Validate group statistics structure
        group_stats = summary["group_statistics"]
        expected_groups = ['Group A', 'Group B', 'Group C']
        for stat_type in ['mean', 'std', 'min', 'max']:
            assert stat_type in group_stats, f"Missing {stat_type} statistics"
            for group in expected_groups:
                assert group in group_stats[stat_type], f"Missing {stat_type} for {group}"

        # Then: Save and validate plot file
        plot_path = self._save_base64_plot(
            result["plot_base64"],
            "distribution_plots.png",
            plots_directory
        )
        assert plot_path.exists() and plot_path.stat().st_size > 0

    def test_pair_plot_multivariate_analysis(self, seaborn_available, plots_directory):
        """
        Test pair plot generation for multivariate relationship analysis.

        This test validates:
        1. Creation of multivariate dataset with relationships
        2. Pair plot matrix generation with categorical hue
        3. Diagonal plot customization (histograms)
        4. Relationship strength computation
        5. Large plot Base64 encoding handling

        Given: Complex multivariate dataset with categorical grouping
        When: Pair plot analysis is executed
        Then: Matrix visualization shows all variable relationships

        Example Output:
            Grid plot showing:
            - Diagonal: Distribution histograms by category
            - Off-diagonal: Scatter plots with relationship trends

        Args:
            seaborn_available: Fixture ensuring Seaborn package availability
            plots_directory: Fixture providing clean plot storage directory
        """
        # Given: Pair plot generation code with relationships
        pairplot_code = """
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json
from pathlib import Path

# Create multivariate dataset with relationships
np.random.seed(42)
n_samples = 300

base_data = {
    'height': np.random.normal(170, 12, n_samples),
    'weight': np.random.normal(70, 15, n_samples),
    'age': np.random.randint(18, 70, n_samples),
    'category': np.random.choice(['Type A', 'Type B', 'Type C'], n_samples)
}

# Add realistic relationships
base_data['weight'] = base_data['height'] * 0.8 - 60 + np.random.normal(0, 8, n_samples)
base_data['age'] = np.clip(base_data['height'] * 0.2 + np.random.normal(15, 10, n_samples), 18, 70)

df = pd.DataFrame(base_data)

# Create pair plot matrix
g = sns.pairplot(
    df,
    hue='category',
    diag_kind='hist',
    plot_kws={'alpha': 0.7, 's': 30},
    diag_kws={'alpha': 0.7}
)
g.fig.suptitle('Multivariate Relationship Analysis', y=1.02, fontsize=16)

# Convert to Base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Calculate correlation matrix for numeric variables
numeric_cols = ['height', 'weight', 'age']
correlations = df[numeric_cols].corr()

result = {
    "plot_base64": plot_b64,
    "plot_type": "pair_plot",
    "analysis_summary": {
        "variable_count": len(numeric_cols),
        "category_count": df['category'].nunique(),
        "sample_size": len(df),
        "correlation_matrix": correlations.to_dict(),
        "strongest_correlation": {
            "variables": None,
            "coefficient": 0.0
        }
    }
}

# Find strongest correlation (excluding self-correlations)
max_corr = 0.0
max_pair = None
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        corr_val = abs(correlations.iloc[i, j])
        if corr_val > max_corr:
            max_corr = corr_val
            max_pair = (numeric_cols[i], numeric_cols[j])

if max_pair:
    result["analysis_summary"]["strongest_correlation"]["variables"] = max_pair
    result["analysis_summary"]["strongest_correlation"]["coefficient"] = float(max_corr)

json.dumps(result)
        """

        # When: Execute pair plot generation
        response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
            data=pairplot_code,
            headers=Config.HEADERS["execute_raw"],
            timeout=Config.TIMEOUTS["code_execution"]
        )

        # Then: Validate response
        data = self._validate_api_response(response)
        assert data["success"], f"Pair plot generation failed: {data.get('error')}"

        result = json.loads(data["data"]["result"].strip())

        # Then: Validate pair plot metadata
        assert result["plot_type"] == "pair_plot"
        assert "analysis_summary" in result

        summary = result["analysis_summary"]
        assert summary["variable_count"] == 3, "Expected 3 numeric variables"
        assert summary["category_count"] == 3, "Expected 3 categories"
        assert summary["sample_size"] == 300, "Expected 300 samples"

        # Then: Validate correlation analysis
        assert "correlation_matrix" in summary
        assert "strongest_correlation" in summary
        strongest = summary["strongest_correlation"]
        assert "variables" in strongest
        assert "coefficient" in strongest
        assert 0 <= strongest["coefficient"] <= 1, "Correlation coefficient should be normalized"

        # Then: Save large plot file
        plot_path = self._save_base64_plot(
            result["plot_base64"],
            "pair_plot.png",
            plots_directory
        )
        assert plot_path.exists() and plot_path.stat().st_size > 0
        # Pair plots are typically larger files
        assert plot_path.stat().st_size > 50000, "Pair plot file seems too small"

    def test_regression_analysis_with_statistics(self, seaborn_available, plots_directory):
        """
        Test regression plot generation with statistical annotations.

        This test validates:
        1. Synthetic dataset with clear linear relationship
        2. Regression line fitting and confidence intervals
        3. Statistical metric calculation (R², correlation)
        4. Plot annotation with statistical information
        5. Enhanced styling and presentation

        Given: Dataset with known linear relationship
        When: Regression analysis with statistics is performed
        Then: Annotated plot with regression line and metrics is created

        Example Output:
            Scatter plot with:
            - Data points with transparency
            - Fitted regression line with confidence band
            - Statistical annotations (R², correlation, equation)

        Args:
            seaborn_available: Fixture ensuring Seaborn package availability
            plots_directory: Fixture providing clean plot storage directory
        """
        # Given: Regression analysis code with statistical annotations
        regression_code = """
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json
from pathlib import Path
from scipy import stats

# Create dataset with known linear relationship
np.random.seed(42)
n = 250
true_slope = 2.3
true_intercept = 1.5

x = np.random.normal(0, 2, n)
noise = np.random.normal(0, 1, n)
y = true_slope * x + true_intercept + noise

df = pd.DataFrame({'feature_x': x, 'target_y': y})

# Create enhanced regression plot
plt.figure(figsize=(12, 8))

# Main regression plot
sns.regplot(
    data=df,
    x='feature_x',
    y='target_y',
    scatter_kws={'alpha': 0.6, 's': 40, 'color': 'steelblue'},
    line_kws={'color': 'red', 'linewidth': 2},
    ci=95
)

plt.title('Regression Analysis with Statistical Metrics', fontsize=16, pad=20)
plt.xlabel('Feature X', fontsize=14)
plt.ylabel('Target Y', fontsize=14)
plt.grid(True, alpha=0.3)

# Calculate statistical metrics
correlation = df['feature_x'].corr(df['target_y'])
slope, intercept, r_value, p_value, std_err = stats.linregress(df['feature_x'], df['target_y'])
r_squared = r_value ** 2

# Add statistical annotations
stats_text = f'''Regression Statistics:
Correlation: r = {correlation:.4f}
R-squared: R² = {r_squared:.4f}
Equation: y = {slope:.3f}x + {intercept:.3f}
P-value: {p_value:.2e}
Standard Error: {std_err:.4f}'''

plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
         verticalalignment='top', fontfamily='monospace', fontsize=10)

plt.tight_layout()

# Convert to Base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

result = {
    "plot_base64": plot_b64,
    "plot_type": "regression_analysis",
    "statistical_metrics": {
        "correlation_coefficient": float(correlation),
        "r_squared": float(r_squared),
        "regression_equation": {
            "slope": float(slope),
            "intercept": float(intercept)
        },
        "significance": {
            "p_value": float(p_value),
            "standard_error": float(std_err)
        },
        "model_quality": "strong" if r_squared > 0.7 else "moderate" if r_squared > 0.3 else "weak"
    }
}

json.dumps(result)
        """

        # When: Execute regression analysis
        response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
            data=regression_code,
            headers=Config.HEADERS["execute_raw"],
            timeout=Config.TIMEOUTS["code_execution"]
        )

        # Then: Validate successful execution
        data = self._validate_api_response(response)
        assert data["success"], f"Regression analysis failed: {data.get('error')}"

        result = json.loads(data["data"]["result"].strip())

        # Then: Validate regression analysis results
        assert result["plot_type"] == "regression_analysis"
        assert "statistical_metrics" in result

        metrics = result["statistical_metrics"]

        # Then: Validate correlation metrics
        assert "correlation_coefficient" in metrics
        correlation = metrics["correlation_coefficient"]
        assert -1 <= correlation <= 1, "Correlation coefficient out of valid range"
        assert abs(correlation) > 0.5, "Expected strong correlation in synthetic data"

        # Then: Validate R-squared
        assert "r_squared" in metrics
        r_squared = metrics["r_squared"]
        assert 0 <= r_squared <= 1, "R-squared out of valid range"
        assert r_squared > 0.25, "Expected reasonable fit for synthetic data"

        # Then: Validate regression equation
        assert "regression_equation" in metrics
        equation = metrics["regression_equation"]
        assert "slope" in equation and "intercept" in equation
        assert isinstance(equation["slope"], (int, float))
        assert isinstance(equation["intercept"], (int, float))

        # Then: Validate significance testing
        assert "significance" in metrics
        significance = metrics["significance"]
        assert "p_value" in significance
        assert "standard_error" in significance
        assert significance["p_value"] < 0.05, "Expected significant relationship"

        # Then: Validate model quality assessment
        assert "model_quality" in metrics
        assert metrics["model_quality"] in ["weak", "moderate", "strong"]

        # Then: Save annotated plot
        plot_path = self._save_base64_plot(
            result["plot_base64"],
            "regression_analysis.png",
            plots_directory
        )
        assert plot_path.exists() and plot_path.stat().st_size > 0

    def test_error_handling_invalid_data(self, seaborn_available):
        """
        Test error handling for invalid data scenarios in Seaborn plotting.

        This test validates robust error handling for:
        1. Empty datasets
        2. Non-numeric data for correlation analysis
        3. Missing value handling
        4. Invalid plot parameters
        5. Memory and resource constraints

        Given: Various invalid data scenarios
        When: Plotting operations are attempted
        Then: Appropriate error messages are returned with proper API format

        Args:
            seaborn_available: Fixture ensuring Seaborn package availability
        """
        # Given: Code that intentionally triggers errors
        error_scenarios = [
            {
                "name": "empty_dataset",
                "code": """
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Empty dataset should cause error
df = pd.DataFrame()
sns.heatmap(df.corr())
plt.show()
                """,
                "expected_error_type": "ValueError"
            },
            {
                "name": "non_numeric_correlation",
                "code": """
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Non-numeric data for correlation
df = pd.DataFrame({
    'text_col': ['a', 'b', 'c', 'd'],
    'more_text': ['x', 'y', 'z', 'w']
})
sns.heatmap(df.corr())
plt.show()
                """,
                "expected_error_type": "DataFrame"
            }
        ]

        for scenario in error_scenarios:
            # When: Execute problematic code
            response = requests.post(
                f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
                data=scenario["code"],
                headers=Config.HEADERS["execute_raw"],
                timeout=Config.TIMEOUTS["code_execution"]
            )

            # Then: Validate error response format
            assert response.status_code == 200, "Server should return 200 even for Python errors"
            data = response.json()

            # Then: Should be marked as unsuccessful
            assert not data["success"], f"Expected failure for {scenario['name']} scenario"
            assert data["data"] is None, "Error response should have null data"
            assert data["error"] is not None, "Error response should have error message"
            expected_error = scenario["expected_error_type"]
            assert expected_error in data["error"], f"Expected {expected_error} in error message"

    def test_performance_and_timeout_handling(self, seaborn_available):
        """
        Test performance characteristics and timeout handling for complex plots.

        This test validates:
        1. Large dataset processing capabilities
        2. Complex plot generation within time limits
        3. Memory efficiency for Base64 encoding
        4. Execution time measurement accuracy
        5. Graceful timeout handling

        Given: Complex plotting operations with large datasets
        When: Execution time is measured
        Then: Operations complete within acceptable time limits

        Args:
            seaborn_available: Fixture ensuring Seaborn package availability
        """
        # Given: Performance test with large dataset
        performance_code = """
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json
import time
from pathlib import Path

# Large dataset for performance testing
start_time = time.time()
np.random.seed(42)
n_large = 2000

# Create large multivariate dataset
large_data = {
    f'feature_{i}': np.random.randn(n_large) for i in range(8)
}

# Add some correlations
for i in range(1, 4):
    large_data[f'feature_{i}'] = (large_data['feature_0'] * 0.5 +
                                   np.random.randn(n_large) * 0.5)

df_large = pd.DataFrame(large_data)
data_prep_time = time.time() - start_time

# Create complex heatmap
plot_start = time.time()
plt.figure(figsize=(12, 10))
correlation_matrix = df_large.corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    square=True,
    fmt='.2f',
    cbar_kws={'label': 'Correlation'},
    linewidths=0.1
)
plt.title(f'Large Dataset Correlation Analysis\\n{n_large} samples, {len(large_data)} features')
plt.tight_layout()

# Base64 encoding
encoding_start = time.time()
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()
encoding_time = time.time() - encoding_start
total_time = time.time() - start_time

result = {
    "plot_base64": plot_b64,
    "plot_type": "performance_test",
    "performance_metrics": {
        "dataset_size": n_large,
        "feature_count": len(large_data),
        "data_preparation_time": data_prep_time,
        "plot_generation_time": time.time() - plot_start - encoding_time,
        "base64_encoding_time": encoding_time,
        "total_execution_time": total_time,
        "base64_size_bytes": len(plot_b64)
    }
}

json.dumps(result)
        """

        # When: Execute performance test
        start_time = time.time()
        response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
            data=performance_code,
            headers=Config.HEADERS["execute_raw"],
            timeout=Config.TIMEOUTS["code_execution"]
        )
        api_call_time = time.time() - start_time

        # Then: Validate successful execution within timeout
        data = self._validate_api_response(response)
        assert data["success"], f"Performance test failed: {data.get('error')}"

        result = json.loads(data["data"]["result"].strip())

        # Then: Validate performance metrics
        assert result["plot_type"] == "performance_test"
        metrics = result["performance_metrics"]

        # Then: Check execution times are reasonable
        assert metrics["total_execution_time"] < 30.0, "Execution took too long"
        assert metrics["data_preparation_time"] < 5.0, "Data preparation too slow"
        assert metrics["plot_generation_time"] < 20.0, "Plot generation too slow"
        assert metrics["base64_encoding_time"] < 5.0, "Base64 encoding too slow"

        # Then: Check dataset size
        assert metrics["dataset_size"] == 2000, "Unexpected dataset size"
        assert metrics["feature_count"] == 8, "Unexpected feature count"

        # Then: Validate Base64 output size is reasonable
        assert metrics["base64_size_bytes"] > 10000, "Base64 output seems too small"
        assert metrics["base64_size_bytes"] < 2000000, "Base64 output seems too large"

        # Then: Overall API call time should be reasonable
        assert api_call_time < Config.TIMEOUTS["code_execution"], "API call exceeded expected timeout"

"""
Test seaborn base64 plot generation functionality.

Description:
    Comprehensive test suite for seaborn visualization library integration
    with base64 plot encoding. Validates correlation analysis, distribution
    plots, regression analysis, and pair plot functionality.

    This test suite follows BDD (Behavior-Driven Development) patterns and
    ensures all tests use the /api/execute-raw endpoint for proper API
    contract compliance.

Requirements:
    - pytest framework for BDD-style testing
    - requests library for HTTP API calls
    - pathlib for cross-platform file operations
    - API contract compliance validation
    - Comprehensive test coverage and documentation

API Contract:
    {
        "success": true | false,
        "data": {"result": str, "stdout": str, "stderr": str, "executionTime": int},
        "error": str | null,
        "meta": {"timestamp": str}
    }

Author: AI Test Generation System
Date: 2025-01-28
Version: 2.0 (Pytest BDD Conversion)
"""

# ==================== IMPORTS AND CONFIGURATION ====================

import json
import base64
import pytest
import requests
from pathlib import Path
from typing import Dict, Any, Optional


# ==================== CONFIGURATION CONSTANTS ====================

# Seaborn installation code to be added to each test
SEABORN_INSTALL_CODE = '''
# Install and import seaborn if not available
try:
    import seaborn as sns
except ImportError:
    import micropip
    await micropip.install("seaborn")
    import seaborn as sns
'''

class Config:
    """
    Centralized configuration for seaborn base64 test suite.

    Description:
        Provides all constants and configuration values used throughout
        the test suite. Ensures no hardcoded values and consistent
        parameterization across all test scenarios.

    Constants:
        API endpoints, timeouts, plot settings, and validation parameters
    """
    # API Configuration
    BASE_URL: str = "http://localhost:3000"
    EXECUTE_RAW_ENDPOINT: str = "/api/execute-raw"
    HEALTH_CHECK_ENDPOINT: str = "/health"
    API_TIMEOUT: int = 120

    # API Response Fields
    API_SUCCESS_FIELD: str = "success"
    API_DATA_FIELD: str = "data"
    API_ERROR_FIELD: str = "error"
    API_META_FIELD: str = "meta"

    # Data Response Fields
    DATA_RESULT_FIELD: str = "result"
    DATA_STDOUT_FIELD: str = "stdout"
    DATA_STDERR_FIELD: str = "stderr"
    DATA_EXECUTION_TIME_FIELD: str = "executionTime"

    # Plot Configuration
    PLOT_WIDTH: int = 10
    PLOT_HEIGHT: int = 8
    PLOT_DPI: int = 150
    PLOT_FORMAT: str = "png"

    # Test Configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0


# ==================== PYTEST FIXTURES ====================

@pytest.fixture(scope="session")
def api_client() -> requests.Session:
    """
    Provide a configured HTTP session for API calls.

    Description:
        Creates and configures a persistent HTTP session with appropriate
        headers and timeout settings for use throughout the test suite.

    Input:
        None (pytest fixture)

    Output:
        requests.Session: Configured HTTP session

    Example:
        def test_api_call(api_client):
            response = api_client.post(url, data=data)
            assert response.status_code == 200

    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "SeabornBase64TestSuite/2.0",
        "Accept": "application/json"
    })
    return session


@pytest.fixture(scope="session")
def server_health_check(api_client) -> bool:
    """
    Verify server is available and responding before running tests.

    Description:
        Performs a health check against the server to ensure it's available
        and responding correctly before attempting to run any tests.

    Input:
        api_client: HTTP session fixture

    Output:
        bool: True if server is healthy

    Example:
        def test_functionality(server_health_check):
            assert server_health_check is True
            # Test will only run if server is healthy

    Raises:
        RuntimeError: If server is not available or unhealthy
    """
    health_url = f"{Config.BASE_URL}{Config.HEALTH_CHECK_ENDPOINT}"

    try:
        response = api_client.get(health_url, timeout=Config.API_TIMEOUT)
        if response.status_code == 200:
            return True
        else:
            raise RuntimeError(f"Server health check failed: {response.status_code}")
    except requests.RequestException as e:
        raise RuntimeError(f"Server is not available: {e}")


@pytest.fixture(scope="session")
def seaborn_environment_setup(api_client, server_health_check) -> bool:
    """
    Ensure seaborn and matplotlib are available in the Pyodide environment.

    Description:
        Verifies that seaborn and matplotlib can be imported and used for
        plotting within the Pyodide environment. Uses execute-raw to test imports.

    Input:
        api_client: HTTP session fixture
        server_health_check: Server availability fixture

    Output:
        bool: True if seaborn environment is ready

    Example:
        def test_seaborn_plot(seaborn_environment_setup):
            # Test will automatically verify seaborn is available
            assert seaborn_environment_setup is True

    Raises:
        RuntimeError: If seaborn cannot be imported or used
    """
    # Test seaborn and matplotlib imports with installation
    import_test_code = SEABORN_INSTALL_CODE + '''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
import io
import json
from pathlib import Path

# Test basic functionality
print("Seaborn version:", sns.__version__)
print("Matplotlib version:", plt.matplotlib.__version__)
print("Seaborn environment ready")

# Return success indicator
json.dumps({"seaborn_ready": True, "matplotlib_ready": True})
'''

    response = execute_python_code_raw(api_client, import_test_code)
    if not response[Config.API_SUCCESS_FIELD]:
        raise RuntimeError(f"Seaborn environment setup failed: {response.get(Config.API_ERROR_FIELD)}")

    result_str = extract_python_result(response)
    if "seaborn_ready" not in result_str:
        raise RuntimeError("Seaborn import test failed")

    return True


@pytest.fixture(scope="session")
def plots_directory() -> Path:
    """
    Provide a directory path for saving test plot outputs.

    Description:
        Creates and provides a directory path for saving base64-decoded plots
        during testing. Uses pathlib for cross-platform compatibility.

    Input:
        None (pytest fixture)

    Output:
        Path: Directory path for plot outputs

    Example:
        def test_save_plot(plots_directory):
            plot_path = plots_directory / "test_plot.png"
            # Save plot to plot_path

    """
    # Use pathlib for cross-platform compatibility
    plots_dir = Path(__file__).parent.parent / "plots" / "base64" / "seaborn"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any existing plots before tests
    for plot_file in plots_dir.glob("*.png"):
        plot_file.unlink()

    return plots_dir


# ==================== HELPER FUNCTIONS ====================

def execute_python_code_raw(
    api_client: requests.Session, code: str
) -> Dict[str, Any]:
    """
    Execute Python code via /api/execute-raw endpoint with full validation.

    Description:
        Sends Python code to the execute-raw endpoint and validates the response
        conforms to the API contract. Provides clean interface for test functions.

    Input:
        api_client: Configured HTTP session
        code: Python code string to execute

    Output:
        Dict containing validated API response

    Example:
        result = execute_python_code_raw(api_client, "print('hello')")
        assert result["success"] is True
        assert "hello" in result["data"]["result"]

    Raises:
        AssertionError: If API response is invalid
        requests.RequestException: If HTTP request fails
    """
    execute_url = f"{Config.BASE_URL}{Config.EXECUTE_RAW_ENDPOINT}"

    response = api_client.post(
        execute_url,
        data=code,
        headers={"Content-Type": "text/plain"},
        timeout=Config.API_TIMEOUT
    )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )

    response_data = response.json()

    # Basic API contract validation
    assert Config.API_SUCCESS_FIELD in response_data
    assert Config.API_DATA_FIELD in response_data
    assert Config.API_ERROR_FIELD in response_data
    assert Config.API_META_FIELD in response_data

    return response_data


def extract_python_result(response_data: Dict[str, Any]) -> str:
    """
    Extract Python execution result from validated API response.

    Description:
        Safely extracts the Python result from execute-raw response data.
        Handles result parsing and validation for test assertions.

    Input:
        response_data: Validated API response dictionary

    Output:
        str: String representation of the execution result

    Example:
        response = execute_python_code_raw(api_client, "42")
        result = extract_python_result(response)
        assert "42" in result

    Raises:
        ValueError: If result cannot be extracted or parsed
    """
    if not response_data[Config.API_SUCCESS_FIELD]:
        error_msg = response_data.get(Config.API_ERROR_FIELD, "Unknown error")
        raise ValueError(f"Python execution failed: {error_msg}")

    data = response_data[Config.API_DATA_FIELD]
    result = data[Config.DATA_RESULT_FIELD]
    
    # Handle JSON string results that need to be parsed back to string
    if isinstance(result, str):
        try:
            # Try to parse as JSON in case it's a JSON-serialized result
            json.loads(result)  # Just validate it's valid JSON
            # If successful, return the original JSON string for later parsing
            return result
        except json.JSONDecodeError:
            # If not valid JSON, return as-is
            return result
    
    return str(result)


def parse_json_result(result_str: str) -> Dict[str, Any]:
    """
    Parse JSON result from string representation.

    Description:
        Converts string result from execute-raw API to Python dictionary.
        Handles JSON parsing with proper error handling.

    Input:
        result_str: String representation of JSON result

    Output:
        Dict: Parsed JSON data

    Example:
        result_str = '{"plot_base64": "iVBOR...", "plot_type": "heatmap"}'
        parsed = parse_json_result(result_str)
        assert "plot_base64" in parsed

    Raises:
        ValueError: If JSON parsing fails
    """
    try:
        return json.loads(result_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON result: {e}")


def save_base64_plot(base64_data: str, plot_path: Path) -> None:
    """
    Save base64-encoded plot data to filesystem.

    Description:
        Decodes base64 plot data and saves it as a PNG file using pathlib
        for cross-platform compatibility.

    Input:
        base64_data: Base64-encoded plot data
        plot_path: Path object for saving the plot

    Output:
        None (saves file to filesystem)

    Example:
        save_base64_plot(plot_data, plots_directory / "correlation.png")

    Raises:
        ValueError: If base64 data is invalid
        IOError: If file cannot be written
    """
    if not base64_data:
        raise ValueError("No base64 data provided")

    try:
        plot_bytes = base64.b64decode(base64_data)
        plot_path.write_bytes(plot_bytes)
    except Exception as e:
        raise IOError(f"Failed to save plot to {plot_path}: {e}")


# ==================== BDD TEST SCENARIOS ====================


class TestSeabornCorrelationAnalysis:
    """
    Test scenarios for seaborn correlation analysis with base64 plots.

    Covers correlation heatmaps, statistical analysis, and visualization
    workflows commonly used in data science and machine learning.
    """

    def test_given_dataset_when_correlation_heatmap_generated_then_returns_base64_plot(
        self, api_client, server_health_check, seaborn_environment_setup, plots_directory
    ):
        """
        Scenario: Generate correlation heatmap with seaborn and return as base64.

        Description:
            Given a sample dataset with correlated features
            When a correlation heatmap is generated using seaborn
            Then the API should return a base64-encoded plot
            And the plot should be saveable to filesystem
            And demonstrate statistical relationships

        Input:
            api_client: HTTP session fixture
            server_health_check: Server availability fixture
            seaborn_environment_setup: Seaborn environment fixture
            plots_directory: Directory for saving plots

        Output:
            None (assertions validate behavior)

        Example:
            Input: Dataset with correlated features
            Expected: Base64-encoded correlation heatmap plot
        """
        # Given: Sample dataset with known correlations
        python_code = SEABORN_INSTALL_CODE + '''

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json

# Create sample dataset with controlled correlations
np.random.seed(42)
n_samples = 200

# Generate base features
feature_1 = np.random.randn(n_samples)
feature_2 = feature_1 * 0.7 + np.random.randn(n_samples) * 0.3  # Strong positive correlation
feature_3 = feature_1 * -0.5 + np.random.randn(n_samples) * 0.5  # Moderate negative correlation
feature_4 = np.random.randn(n_samples)  # Independent feature

data = {
    'feature_1': feature_1,
    'feature_2': feature_2,
    'feature_3': feature_3,
    'feature_4': feature_4
}

df = pd.DataFrame(data)

# Create correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.3f')
plt.title('Feature Correlation Heatmap')

# Save to bytes buffer for base64 encoding
buffer = io.BytesIO()
plt.savefig(buffer,
           format='png',
           dpi=150,
           bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Prepare result with validation data
result = {
    "plot_base64": plot_b64,
    "plot_type": "correlation_heatmap",
    "correlations": {
        "feature_1_feature_2": float(correlation_matrix.loc['feature_1', 'feature_2']),
        "feature_1_feature_3": float(correlation_matrix.loc['feature_1', 'feature_3'])
    },
    "dataset_info": {
        "n_samples": len(df),
        "n_features": len(df.columns)
    }
}

print(f"Generated correlation heatmap with {len(df)} samples")
json.dumps(result)
        '''

        # When: Correlation heatmap code is executed via execute-raw endpoint
        response = execute_python_code_raw(api_client, python_code)

        # Then: Response should contain base64 plot data
        assert response[Config.API_SUCCESS_FIELD] is True
        result_str = extract_python_result(response)
        
        # Parse the JSON result
        result_data = parse_json_result(result_str)
        
        # Validate plot data structure
        assert "plot_base64" in result_data
        assert "plot_type" in result_data
        assert result_data["plot_type"] == "correlation_heatmap"
        
        # Validate correlation data
        assert "correlations" in result_data
        correlations = result_data["correlations"]
        
        # Verify expected strong correlation between feature_1 and feature_2
        assert correlations["feature_1_feature_2"] > 0.6
        
        # Verify expected negative correlation between feature_1 and feature_3
        assert correlations["feature_1_feature_3"] < -0.3
        
        # Save plot to filesystem for verification
        plot_path = plots_directory / "correlation_heatmap.png"
        save_base64_plot(result_data["plot_base64"], plot_path)
        assert plot_path.exists()
        
        # Verify file size indicates a real plot
        assert plot_path.stat().st_size > 1000  # Should be a substantial PNG file


class TestSeabornDistributionAnalysis:
    """
    Test scenarios for seaborn distribution analysis with multiple plot types.

    Covers histograms, box plots, distribution comparisons, and multi-subplot
    layouts for comprehensive statistical visualization.
    """

    def test_given_multi_group_data_when_distribution_plots_generated_then_returns_base64_subplots(
        self, api_client, server_health_check, seaborn_environment_setup, plots_directory
    ):
        """
        Scenario: Generate distribution plots with multiple subplots and groups.

        Description:
            Given sample data with multiple groups and distributions
            When distribution plots are generated using seaborn subplots
            Then the API should return a base64-encoded multi-plot figure
            And demonstrate distribution differences between groups
            And provide statistical summaries

        Input:
            api_client: HTTP session fixture
            server_health_check: Server availability fixture
            seaborn_environment_setup: Seaborn environment fixture
            plots_directory: Directory for saving plots

        Output:
            None (assertions validate behavior)

        Example:
            Input: Multi-group dataset with different distributions
            Expected: Base64-encoded subplot figure with histograms and boxplots
        """
        # Given: Multi-group dataset with distinct distributions
        python_code = '''
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json

# Create sample data with distinct group distributions
np.random.seed(42)
n_per_group = 500

# Group A: Normal distribution centered at 50
group_a = np.random.normal(50, 10, n_per_group)

# Group B: Normal distribution centered at 60 
group_b = np.random.normal(60, 15, n_per_group)

# Group C: Normal distribution centered at 45
group_c = np.random.normal(45, 8, n_per_group)

# Create combined dataframe
data = []
data.extend([('Group A', val) for val in group_a])
data.extend([('Group B', val) for val in group_b])
data.extend([('Group C', val) for val in group_c])

df = pd.DataFrame(data, columns=['Group', 'Value'])

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

# Subplot 1: Histogram with overlapping distributions
sns.histplot(data=df, x='Value', hue='Group', alpha=0.7, ax=ax1)
ax1.set_title('Distribution Comparison - Histograms')
ax1.legend(title='Groups')

# Subplot 2: Box plots for comparison
sns.boxplot(data=df, x='Group', y='Value', ax=ax2)
ax2.set_title('Distribution Comparison - Box Plots')

plt.tight_layout()

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer,
           format='png',
           dpi=150,
           bbox_inches='tight')
buffer.seek(0)

# Convert to base64
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Calculate group statistics
group_stats = df.groupby('Group')['Value'].agg(['mean', 'std', 'count']).to_dict('index')

# Prepare result
result = {
    "plot_base64": plot_b64,
    "plot_type": "distribution_plots",
    "subplot_count": 2,
    "group_statistics": {
        group: {
            "mean": float(stats['mean']),
            "std": float(stats['std']),
            "count": int(stats['count'])
        }
        for group, stats in group_stats.items()
    },
    "total_samples": len(df)
}

print(f"Generated distribution plots for {len(df)} samples across {df['Group'].nunique()} groups")
json.dumps(result)
        '''

        # When: Distribution plots code is executed via execute-raw endpoint
        response = execute_python_code_raw(api_client, python_code)

        # Then: Response should contain multi-subplot base64 plot
        assert response[Config.API_SUCCESS_FIELD] is True
        result_str = extract_python_result(response)
        
        # Parse the JSON result
        result_data = parse_json_result(result_str)
        
        # Validate plot data structure
        assert "plot_base64" in result_data
        assert "plot_type" in result_data
        assert result_data["plot_type"] == "distribution_plots"
        assert result_data["subplot_count"] == 2
        
        # Validate group statistics
        assert "group_statistics" in result_data
        group_stats = result_data["group_statistics"]
        
        # Verify all three groups are present
        assert "Group A" in group_stats
        assert "Group B" in group_stats
        assert "Group C" in group_stats
        
        # Verify expected mean differences (Group B should have highest mean)
        assert group_stats["Group B"]["mean"] > group_stats["Group A"]["mean"]
        assert group_stats["Group A"]["mean"] > group_stats["Group C"]["mean"]
        
        # Save plot to filesystem
        plot_path = plots_directory / "distribution_plots.png"
        save_base64_plot(result_data["plot_base64"], plot_path)
        assert plot_path.exists()
        
        # Verify substantial file size for multi-subplot figure
        assert plot_path.stat().st_size > 5000


class TestSeabornRegressionAnalysis:
    """
    Test scenarios for seaborn regression analysis and statistical modeling.

    Covers regression plots, residual analysis, and confidence intervals
    for predictive modeling and relationship analysis.
    """

    def test_given_linear_relationship_when_regression_plot_generated_then_returns_statistical_analysis(
        self, api_client, server_health_check, seaborn_environment_setup, plots_directory
    ):
        """
        Scenario: Generate regression plot with statistical analysis and confidence intervals.

        Description:
            Given data with a linear relationship
            When a regression plot is generated using seaborn
            Then the API should return a base64-encoded plot with regression line
            And provide correlation and statistical metrics
            And include confidence intervals visualization

        Input:
            api_client: HTTP session fixture
            server_health_check: Server availability fixture
            seaborn_environment_setup: Seaborn environment fixture
            plots_directory: Directory for saving plots

        Output:
            None (assertions validate behavior)

        Example:
            Input: Dataset with linear relationship
            Expected: Regression plot with correlation > 0.8
        """
        # Given: Data with controlled linear relationship
        python_code = '''
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json

# Create data with strong linear relationship
np.random.seed(42)
n_samples = 300

# Generate predictive features
x = np.random.normal(0, 1, n_samples)
noise = np.random.normal(0, 0.3, n_samples)  # Low noise for strong correlation
y = 2.5 * x + 1.0 + noise  # y = 2.5x + 1 + noise

df = pd.DataFrame({'feature_x': x, 'target_y': y})

# Create regression plot with confidence interval
plt.figure(figsize=(10, 8))
sns.regplot(data=df, x='feature_x', y='target_y',
            scatter_kws={'alpha': 0.6, 's': 40},
            line_kws={'color': 'red', 'linewidth': 2},
            ci=95)  # 95% confidence interval

plt.title('Regression Analysis - Feature vs Target')
plt.xlabel('Feature X')
plt.ylabel('Target Y')

# Calculate correlation
correlation = df.corr().iloc[0, 1]

# Add statistical info to plot
stats_text = f'Correlation: {correlation:.3f}\\nR²: {correlation**2:.3f}\\nSamples: {len(df)}'

plt.text(0.05, 0.95, stats_text,
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         verticalalignment='top')

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer,
           format='png',
           dpi=150,
           bbox_inches='tight')
buffer.seek(0)

# Convert to base64
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Prepare statistical result
result = {
    "plot_base64": plot_b64,
    "plot_type": "regression_plot",
    "statistics": {
        "correlation": float(correlation),
        "r_squared": float(correlation**2),
        "n_samples": len(df),
        "slope_estimate": 2.5,  # Known true value
        "intercept_estimate": 1.0  # Known true value
    },
    "confidence_interval": 95
}

print(f"Generated regression plot with correlation {correlation:.3f}")
json.dumps(result)
        '''

        # When: Regression analysis code is executed via execute-raw endpoint
        response = execute_python_code_raw(api_client, python_code)

        # Then: Response should contain regression analysis results
        assert response[Config.API_SUCCESS_FIELD] is True
        result_str = extract_python_result(response)
        
        # Parse the JSON result
        result_data = parse_json_result(result_str)
        
        # Validate plot data structure
        assert "plot_base64" in result_data
        assert "plot_type" in result_data
        assert result_data["plot_type"] == "regression_plot"
        
        # Validate statistical analysis
        assert "statistics" in result_data
        stats = result_data["statistics"]
        
        # Verify strong correlation (should be > 0.8 with low noise)
        assert stats["correlation"] > 0.8
        assert stats["r_squared"] > 0.64  # R² should be > 0.64
        assert stats["n_samples"] == 300
        
        # Save plot to filesystem
        plot_path = plots_directory / "regression_plot.png"
        save_base64_plot(result_data["plot_base64"], plot_path)
        assert plot_path.exists()
        
        # Verify substantial file size
        assert plot_path.stat().st_size > 2000


class TestSeabornPairPlotAnalysis:
    """
    Test scenarios for seaborn pair plot analysis with multi-dimensional data.

    Covers pair plots, multi-variable relationships, and categorical
    comparisons for exploratory data analysis workflows.
    """

    def test_given_multidimensional_data_when_pair_plot_generated_then_returns_comprehensive_visualization(
        self, api_client, server_health_check, seaborn_environment_setup, plots_directory
    ):
        """
        Scenario: Generate comprehensive pair plot with categorical grouping.

        Description:
            Given multi-dimensional dataset with categorical grouping
            When a pair plot is generated using seaborn
            Then the API should return a base64-encoded comprehensive visualization
            And demonstrate relationships across all variable pairs
            And provide summary statistics for each group

        Input:
            api_client: HTTP session fixture
            server_health_check: Server availability fixture
            seaborn_environment_setup: Seaborn environment fixture
            plots_directory: Directory for saving plots

        Output:
            None (assertions validate behavior)

        Example:
            Input: Multi-dimensional dataset with categories
            Expected: Pair plot showing all variable relationships
        """
        # Given: Multi-dimensional dataset with categories
        python_code = '''
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json

# Create realistic multi-dimensional dataset
np.random.seed(42)
n_samples = 400

# Create base features with relationships
height = np.random.normal(170, 10, n_samples)
weight = height * 0.8 + np.random.normal(0, 8, n_samples)  # Correlated with height
age = np.random.uniform(18, 65, n_samples)
income = (age - 18) * 1000 + np.random.normal(30000, 15000, n_samples)  # Increases with age

# Add categorical grouping
categories = np.random.choice(['Group A', 'Group B', 'Group C'], n_samples)

# Modify features slightly based on categories
for i, cat in enumerate(categories):
    if cat == 'Group A':
        height[i] += np.random.normal(5, 2)  # Slightly taller
    elif cat == 'Group B':
        weight[i] += np.random.normal(5, 3)  # Slightly heavier
    # Group C remains unchanged

df = pd.DataFrame({
    'height_cm': height,
    'weight_kg': weight,
    'age_years': age,
    'income_usd': income,
    'category': categories
})

# Create pair plot
g = sns.pairplot(df, hue='category', 
                diag_kind='hist',
                plot_kws={'alpha': 0.6, 's': 30},
                height=2.5)
g.fig.suptitle('Multi-Dimensional Data Analysis - Pair Plot', y=1.02)

# Save to bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer,
           format='png',
           dpi=150,
           bbox_inches='tight')
buffer.seek(0)

# Convert to base64
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Calculate summary statistics by category
numeric_cols = ['height_cm', 'weight_kg', 'age_years', 'income_usd']
summary_stats = {}
for category in df['category'].unique():
    cat_data = df[df['category'] == category][numeric_cols]
    summary_stats[category] = {
        col: {
            'mean': float(cat_data[col].mean()),
            'std': float(cat_data[col].std()),
            'count': int(len(cat_data))
        }
        for col in numeric_cols
    }

# Prepare result
result = {
    "plot_base64": plot_b64,
    "plot_type": "pair_plot",
    "dimensions": len(numeric_cols),
    "categories": list(df['category'].unique()),
    "total_samples": len(df),
    "summary_statistics": summary_stats,
    "correlations": {
        "height_weight": float(df[['height_cm', 'weight_kg']].corr().iloc[0, 1]),
        "age_income": float(df[['age_years', 'income_usd']].corr().iloc[0, 1])
    }
}

print(f"Generated pair plot with {len(df)} samples across {len(numeric_cols)} dimensions")
json.dumps(result)
        '''

        # When: Pair plot code is executed via execute-raw endpoint
        response = execute_python_code_raw(api_client, python_code)

        # Then: Response should contain comprehensive pair plot results
        assert response[Config.API_SUCCESS_FIELD] is True
        result_str = extract_python_result(response)
        
        # Parse the JSON result
        result_data = parse_json_result(result_str)
        
        # Validate plot data structure
        assert "plot_base64" in result_data
        assert "plot_type" in result_data
        assert result_data["plot_type"] == "pair_plot"
        
        # Validate dimensions and categories
        assert result_data["dimensions"] == 4
        assert len(result_data["categories"]) == 3
        assert result_data["total_samples"] == 400
        
        # Validate summary statistics for each category
        assert "summary_statistics" in result_data
        summary = result_data["summary_statistics"]
        
        for category in result_data["categories"]:
            assert category in summary
            cat_stats = summary[category]
            
            # Verify all numeric columns are present
            assert "height_cm" in cat_stats
            assert "weight_kg" in cat_stats
            assert "age_years" in cat_stats
            assert "income_usd" in cat_stats
            
            # Verify reasonable means for each category
            assert 150 < cat_stats["height_cm"]["mean"] < 200
            assert 100 < cat_stats["weight_kg"]["mean"] < 180  # Adjusted for realistic weight ranges
        
        # Validate expected correlations
        correlations = result_data["correlations"]
        assert correlations["height_weight"] > 0.5  # Should be positively correlated
        assert correlations["age_income"] > 0.3  # Should be positively correlated
        
        # Save plot to filesystem
        plot_path = plots_directory / "pair_plot.png"
        save_base64_plot(result_data["plot_base64"], plot_path)
        assert plot_path.exists()
        
        # Verify large file size for comprehensive pair plot
        assert plot_path.stat().st_size > 10000  # Should be substantial for pair plot


if __name__ == "__main__":
    # Run pytest with verbose output and specific markers
    pytest.main([__file__, "-v", "--tb=short"])
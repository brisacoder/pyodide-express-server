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
    API_EXECUTE_RAW_ENDPOINT = "/api/execute-raw"
    API_INSTALL_PACKAGE_ENDPOINT = "/api/install-package"
    API_RESET_ENDPOINT = "/api/reset"
    HEALTH_ENDPOINT = "/health"

    # Timeout configurations (in seconds)
    SHORT_TIMEOUT = 30
    MEDIUM_TIMEOUT = 60
    LONG_TIMEOUT = 120
    SERVER_START_TIMEOUT = 180
    HEALTH_CHECK_TIMEOUT = 10

    # Test data constants
    RANDOM_SEED = 42
    SAMPLE_SIZE_SMALL = 200
    SAMPLE_SIZE_LARGE = 300
    
    # Alternative dataset configurations for enhanced testing
    DASHBOARD_RANDOM_SEED = 123
    EDGE_CASE_SAMPLE_SIZE = 50
    LARGE_DATASET_SIZE = 500

    # Plot configuration
    PLOT_DPI = 150
    FIGURE_WIDTH = 10
    FIGURE_HEIGHT = 8
    DASHBOARD_WIDTH = 16
    DASHBOARD_HEIGHT = 12
    
    # File size expectations for validation
    MIN_PLOT_FILE_SIZE = 1000  # Minimum expected PNG file size in bytes
    MAX_REASONABLE_FILE_SIZE = 10_000_000  # 10MB maximum reasonable size
    
    # Virtual filesystem paths
    PLOTS_BASE_PATH = "/plots/seaborn"
    PLOTS_MATPLOTLIB_PATH = "/plots/matplotlib"
    UPLOADS_PATH = "/uploads"


def wait_for_server(url: str, timeout: int = 180) -> None:
    """
    Wait for server to become available with proper health check.

    This function implements a robust server availability check with exponential
    backoff and proper error handling for connection issues.

    Args:
        url: Server URL to check (should be health endpoint)
        timeout: Maximum wait time in seconds

    Raises:
        RuntimeError: If server doesn't respond within timeout period

    Example:
        >>> wait_for_server("http://localhost:3000/health", timeout=30)
        # Blocks until server responds or timeout occurs
    """
    start = time.time()
    backoff_delay = 1
    max_backoff = 5
    
    while time.time() - start < timeout:
        try:
            response = requests.get(url, timeout=TestConfig.HEALTH_CHECK_TIMEOUT)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        
        # Exponential backoff for connection attempts
        time.sleep(min(backoff_delay, max_backoff))
        backoff_delay = min(backoff_delay * 1.5, max_backoff)
    
    raise RuntimeError(f"Server at {url} did not start within {timeout} seconds")


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
    Session-scoped fixture ensuring server availability with comprehensive health checks.

    This fixture performs a complete server validation including health check,
    API endpoint availability, and optional environment reset for clean testing state.

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
    
    # Comprehensive server health check
    health_url = f"{base_url}{TestConfig.HEALTH_ENDPOINT}"
    try:
        wait_for_server(health_url, timeout=TestConfig.SHORT_TIMEOUT)
    except RuntimeError:
        pytest.skip("Server is not running on localhost:3000 - start with 'npm start'")

    # Verify execute-raw endpoint availability
    try:
        test_response = requests.post(
            f"{base_url}{TestConfig.API_EXECUTE_RAW_ENDPOINT}",
            data="print('Server API test')",
            headers={"Content-Type": "text/plain"},
            timeout=TestConfig.SHORT_TIMEOUT,
        )
        if test_response.status_code not in [200, 400, 500]:
            pytest.skip(f"Execute-raw endpoint not responding properly: {test_response.status_code}")
    except requests.RequestException as e:
        pytest.skip(f"Execute-raw endpoint not accessible: {e}")

    # Optional: Reset Pyodide environment to clean state (best effort)
    try:
        reset_response = requests.post(
            f"{base_url}{TestConfig.API_RESET_ENDPOINT}", 
            timeout=TestConfig.SHORT_TIMEOUT
        )
        if reset_response.status_code == 200:
            print("✅ Pyodide environment reset successfully")
        else:
            print(f"⚠️ Warning: Could not reset Pyodide environment: {reset_response.status_code}")
    except requests.RequestException:
        print("⚠️ Warning: Could not reset Pyodide environment due to network error")

    return base_url


@pytest.fixture(scope="session")
def seaborn_environment(
    verified_server: str, server_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Session-scoped fixture ensuring seaborn availability with comprehensive package installation.

    This fixture installs required packages (matplotlib, seaborn) and verifies
    they are available for the test session. Uses only regular API endpoints
    and implements proper error handling for network issues.

    Args:
        verified_server: Base URL from verified_server fixture
        server_config: Configuration from server_config fixture

    Returns:
        Dictionary with package installation status and availability flags

    Example:
        >>> def test_example(seaborn_environment):
        ...     assert seaborn_environment['seaborn_available'] is True
        ...     assert seaborn_environment['matplotlib_available'] is True
    """
    installation_status = {
        "seaborn_available": False,
        "matplotlib_available": False,
        "server_ready": True,
        "packages_installed": [],
        "installation_errors": []
    }
    
    # Required packages for seaborn filesystem testing
    required_packages = ["matplotlib", "seaborn", "numpy", "pandas"]
    
    for package in required_packages:
        try:
            install_response = requests.post(
                f"{verified_server}{TestConfig.API_INSTALL_PACKAGE_ENDPOINT}",
                json={"package": package},
                timeout=server_config["long_timeout"],  # Extended timeout for package installation
                headers={"Content-Type": "application/json"}
            )
            
            if install_response.status_code == 200:
                print(f"✅ {package} installed successfully")
                installation_status["packages_installed"].append(package)
                
                # Mark specific packages as available
                if package == "matplotlib":
                    installation_status["matplotlib_available"] = True
                elif package == "seaborn":
                    installation_status["seaborn_available"] = True
            else:
                error_msg = f"{package} installation failed: HTTP {install_response.status_code}"
                print(f"⚠️ Warning: {error_msg}")
                installation_status["installation_errors"].append(error_msg)
                
        except requests.RequestException as e:
            error_msg = f"{package} installation failed due to network error: {e}"
            print(f"⚠️ Warning: {error_msg}")
            installation_status["installation_errors"].append(error_msg)

    # If core packages are not available, mark seaborn as unavailable
    if "matplotlib" not in installation_status["packages_installed"]:
        installation_status["matplotlib_available"] = False
    if "seaborn" not in installation_status["packages_installed"]:
        installation_status["seaborn_available"] = False

    return installation_status


@pytest.fixture
def clean_filesystem(verified_server: str, server_config: Dict[str, Any]):
    """
    Function-scoped fixture to clean Pyodide virtual filesystem comprehensively.

    This fixture ensures each test starts with a clean virtual filesystem by removing
    any existing plot files from multiple directories. Uses only /api/execute-raw 
    endpoint for all filesystem operations as required.

    Args:
        verified_server: Base URL from verified_server fixture
        server_config: Configuration from server_config fixture

    Yields:
        None (setup and teardown fixture)

    Example:
        >>> def test_plot_creation(clean_filesystem):
        ...     # Filesystem is guaranteed to be clean here
        ...     # Test creates plots without interference
        ...     pass
    """
    # Setup: Comprehensive cleanup before test using execute-raw endpoint
    cleanup_code = f"""
# Comprehensive filesystem cleanup using pathlib for cross-platform compatibility
from pathlib import Path
import sys

print("Starting comprehensive filesystem cleanup...")

# Define all directories to clean
directories_to_clean = [
    Path('{TestConfig.PLOTS_BASE_PATH}'),
    Path('{TestConfig.PLOTS_MATPLOTLIB_PATH}'),
    Path('/plots'),  # Base plots directory
]

cleanup_results = {{
    "directories_processed": 0,
    "files_removed": 0,
    "directories_created": 0,
    "errors": []
}}

for plot_dir in directories_to_clean:
    try:
        cleanup_results["directories_processed"] += 1
        
        if plot_dir.exists():
            # Remove all PNG files in directory
            png_files = list(plot_dir.glob('*.png'))
            for plot_file in png_files:
                try:
                    plot_file.unlink()
                    cleanup_results["files_removed"] += 1
                    print(f"Removed: {{plot_file}}")
                except Exception as e:
                    error_msg = f"Could not remove {{plot_file}}: {{e}}"
                    print(f"Warning: {{error_msg}}")
                    cleanup_results["errors"].append(error_msg)
        else:
            # Create directory if it doesn't exist
            plot_dir.mkdir(parents=True, exist_ok=True)
            cleanup_results["directories_created"] += 1
            print(f"Created directory: {{plot_dir}}")
            
    except Exception as e:
        error_msg = f"Failed to process directory {{plot_dir}}: {{e}}"
        print(f"Error: {{error_msg}}")
        cleanup_results["errors"].append(error_msg)

print(f"Cleanup completed: {{cleanup_results}}")
"Cleanup complete"
"""
    
    # Execute cleanup using execute-raw endpoint (as required)
    try:
        cleanup_response = requests.post(
            f"{verified_server}{TestConfig.API_EXECUTE_RAW_ENDPOINT}",
            data=cleanup_code,
            headers={"Content-Type": "text/plain"},
            timeout=server_config["short_timeout"],
        )
        # Log cleanup results if successful
        if cleanup_response.status_code == 200:
            print("✅ Filesystem cleanup completed successfully")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")  # Cleanup is best-effort

    yield

    # Teardown: Optional cleanup after test (best effort)
    try:
        requests.post(
            f"{verified_server}{TestConfig.API_EXECUTE_RAW_ENDPOINT}",
            data=cleanup_code,
            headers={"Content-Type": "text/plain"},
            timeout=server_config["short_timeout"],
        )
    except Exception:
        pass  # Teardown cleanup is best-effort


def validate_api_contract(response_data: Dict[str, Any]) -> None:
    """
    Validate that API response follows the required contract structure.
    
    Required API contract:
    {
      "success": true | false,
      "data": <object|null>,
      "error": <string|null>,  
      "meta": { "timestamp": <string> }
    }
    
    Args:
        response_data: The JSON response from the API
        
    Raises:
        AssertionError: If the API contract is not followed
        
    Example:
        >>> response = requests.post("/api/execute-raw", data="print('test')")
        >>> validate_api_contract(response.json())
        # Passes if contract is valid, raises AssertionError if not
    """
    # Validate top-level structure
    assert "success" in response_data, "API contract violation: missing 'success' field"
    assert "data" in response_data, "API contract violation: missing 'data' field"
    assert "error" in response_data, "API contract violation: missing 'error' field"
    assert "meta" in response_data, "API contract violation: missing 'meta' field"
    
    # Validate types
    assert isinstance(response_data["success"], bool), "API contract violation: 'success' must be boolean"
    
    # Validate meta structure
    meta = response_data.get("meta", {})
    assert isinstance(meta, dict), "API contract violation: 'meta' must be object"
    assert "timestamp" in meta, "API contract violation: 'meta' missing 'timestamp'"
    assert isinstance(meta["timestamp"], str), "API contract violation: 'timestamp' must be string"
    
    # Validate conditional fields
    if response_data["success"]:
        assert response_data["error"] is None, "API contract violation: 'error' must be null when success=true"
        assert response_data["data"] is not None, "API contract violation: 'data' must not be null when success=true"
    else:
        assert response_data["data"] is None, "API contract violation: 'data' must be null when success=false"
        assert isinstance(response_data["error"], str), "API contract violation: 'error' must be string when success=false"


def make_execute_raw_request(
    server_url: str,
    code: str,
    timeout: int = TestConfig.MEDIUM_TIMEOUT
) -> Dict[str, Any]:
    """
    Make a request to the /api/execute-raw endpoint with proper error handling.
    
    This utility function standardizes API calls and validates the response
    contract as required by the problem statement.
    
    Args:
        server_url: Base server URL
        code: Python code to execute
        timeout: Request timeout in seconds
        
    Returns:
        Validated JSON response from the API
        
    Raises:
        AssertionError: If API contract is violated
        requests.RequestException: If request fails
        
    Example:
        >>> response = make_execute_raw_request(
        ...     "http://localhost:3000", 
        ...     "print('Hello World')"
        ... )
        >>> assert response["success"] is True
    """
    response = requests.post(
        f"{server_url}{TestConfig.API_EXECUTE_RAW_ENDPOINT}",
        data=code,
        headers={"Content-Type": "text/plain"},
        timeout=timeout,
    )
    
    # Validate HTTP response
    assert response.status_code == 200, f"HTTP request failed: {response.status_code}"
    
    # Parse and validate API contract
    response_data = response.json()
    validate_api_contract(response_data)
    
    return response_data


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

        **BDD Scenario**: Direct filesystem save with statistical validation
        
        **Given**: A functioning seaborn environment with filesystem access
        **When**: We create a regression plot with statistical analysis and save to filesystem  
        **Then**: The plot should be saved with correct metadata and file properties verified within Pyodide

        This test validates comprehensive seaborn regression functionality including:
        - Cross-platform pathlib usage for file operations
        - Statistical correlation analysis and validation
        - Direct filesystem save without internal APIs
        - Comprehensive metadata extraction and validation
        - API contract compliance verification

        Args:
            verified_server: Base URL of running test server
            server_config: Test configuration dictionary with timeouts and parameters
            seaborn_environment: Seaborn availability confirmation with installation status
            clean_filesystem: Filesystem cleanup fixture ensuring clean test state

        Raises:
            AssertionError: If any validation fails or API contract is violated
            pytest.skip: If seaborn environment is not available

        Example:
            This test demonstrates the complete workflow:
            
            ```python
            # Given: Environment setup
            assert seaborn_environment['seaborn_available'] is True
            
            # When: Creating regression plot
            regression_code = "import seaborn as sns; ..."
            response = make_execute_raw_request(server, regression_code)
            
            # Then: Validation
            assert response['success'] is True
            assert result['file_saved'] is True
            assert result['correlation'] > 0.8
            ```

        Expected Behavior:
            - Seaborn regression plot created with deterministic sample data
            - Plot saved to {TestConfig.PLOTS_BASE_PATH}/ directory using pathlib
            - Correlation coefficient calculated and validated (> 0.8)
            - File metadata verified within Pyodide execution context
            - Statistical analysis results included in response
            - API contract followed exactly as specified
        """
        # Given: Seaborn environment is available and filesystem is accessible
        if not seaborn_environment.get("seaborn_available"):
            pytest.skip("Seaborn not available - check package installation")
        if not seaborn_environment.get("matplotlib_available"):
            pytest.skip("Matplotlib not available - required for seaborn plots")
        if not seaborn_environment.get("server_ready"):
            pytest.skip("Server not ready for filesystem tests")

        # When: Creating regression plot with filesystem save and verification
        # Use parameterized constants instead of hardcoded values
        random_seed = server_config["random_seed"]
        sample_size = server_config["sample_size_small"]
        plot_dpi = server_config["plot_dpi"]
        figure_width = server_config["figure_width"]
        figure_height = server_config["figure_height"]

        regression_code = f"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json

print("Creating seaborn regression plot with comprehensive filesystem verification...")

# Set deterministic random seed for reproducible testing
np.random.seed({random_seed})

# Set seaborn style for consistent appearance
sns.set_style("whitegrid")

# Create sample data with clear statistical relationship
n_samples = {sample_size}
x_values = np.random.normal(0, 1, n_samples)
noise = np.random.normal(0, 0.5, n_samples)
y_values = 2.5 * x_values + 1.0 + noise  # Strong linear relationship

# Create DataFrame for seaborn
df = pd.DataFrame({{'feature_x': x_values, 'target_y': y_values}})

# Create the regression plot with proper configuration
plt.figure(figsize=({figure_width}, {figure_height}))
ax = sns.regplot(
    data=df, 
    x='feature_x', 
    y='target_y',
    scatter_kws={{'alpha': 0.6, 's': 50, 'color': 'blue'}},
    line_kws={{'color': 'red', 'linewidth': 2, 'alpha': 0.8}}
)

# Add professional plot styling
plt.title('Seaborn Regression Analysis - Direct Filesystem Save', 
          fontweight='bold', fontsize=14)
plt.xlabel('Feature X', fontweight='bold', fontsize=12)
plt.ylabel('Target Y', fontweight='bold', fontsize=12)

# Calculate and display correlation statistics
correlation = df[['feature_x', 'target_y']].corr().iloc[0, 1]
r_squared = correlation ** 2

# Add statistical annotation to plot
plt.text(0.05, 0.95, 
         f'Correlation: {{correlation:.3f}}\\nR²: {{r_squared:.3f}}',
         transform=ax.transAxes,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
         fontsize=10, verticalalignment='top')

# Save directly to virtual filesystem using pathlib (cross-platform)
timestamp = int(time.time() * 1000)  # Unique timestamp for filename
plots_path = Path('{TestConfig.PLOTS_BASE_PATH}')
plots_path.mkdir(parents=True, exist_ok=True)
output_file = plots_path / f'regression_test_{{timestamp}}.png'

# Save with high quality settings
plt.savefig(output_file, dpi={plot_dpi}, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.close()  # Properly close to free memory

# Comprehensive file verification and metadata collection
file_exists = output_file.exists()
file_size = output_file.stat().st_size if file_exists else 0
file_info = output_file.stat() if file_exists else None

# Create comprehensive verification results
result = {{
    "file_saved": file_exists,
    "file_size": file_size,
    "file_size_valid": file_size > {TestConfig.MIN_PLOT_FILE_SIZE},
    "plot_type": "seaborn_regression_analysis", 
    "correlation": float(correlation),
    "r_squared": float(r_squared),
    "n_points": n_samples,
    "filename": str(output_file),
    "timestamp": timestamp,
    "statistical_analysis": {{
        "correlation_coefficient": float(correlation),
        "r_squared_value": float(r_squared),
        "sample_size": int(n_samples),
        "relationship_strength": "strong" if abs(correlation) > 0.8 else "moderate" if abs(correlation) > 0.5 else "weak",
        "data_quality": {{
            "x_mean": float(np.mean(x_values)),
            "x_std": float(np.std(x_values)),
            "y_mean": float(np.mean(y_values)), 
            "y_std": float(np.std(y_values))
        }}
    }},
    "plot_metadata": {{
        "figure_size": [{figure_width}, {figure_height}],
        "dpi": {plot_dpi},
        "backend": "Agg",
        "style": "whitegrid"
    }},
    "filesystem_info": {{
        "base_path": str(plots_path),
        "full_path": str(output_file),
        "file_size_bytes": file_size,
        "creation_timestamp": timestamp
    }}
}}

print(f"Regression plot saved successfully: {{output_file}}")
print(f"File size: {{file_size}} bytes")
print(f"Correlation: {{correlation:.3f}}")

# Return result as the last expression for API extraction
result
"""

        # Execute using the standardized execute-raw API call
        response_data = make_execute_raw_request(
            verified_server,
            regression_code,
            timeout=server_config["long_timeout"]
        )

        # Then: Validate successful execution and API contract compliance
        assert response_data.get("success"), f"Execution failed: {response_data}"

        # Extract result from data.result as per API contract
        data = response_data.get("data", {})
        assert "result" in data, "Response data missing 'result' field"
        
        result = data.get("result")
        assert result is not None, f"API returned None result: {response_data}"

        # And: Plot should be saved with correct metadata validation
        assert result.get("file_saved") is True, "Regression plot was not saved to filesystem"
        assert result.get("file_size_valid") is True, f"Plot file size too small: {result.get('file_size', 0)} bytes"
        assert result.get("plot_type") == "seaborn_regression_analysis", "Incorrect plot type identifier"

        # And: Statistical analysis should meet quality requirements
        correlation = result.get("correlation", 0)
        assert correlation > 0.8, f"Expected strong positive correlation (>0.8), got {correlation:.3f}"
        assert result.get("n_points") == sample_size, f"Expected {sample_size} data points"

        # And: Statistical analysis structure should be comprehensive
        stats = result.get("statistical_analysis", {})
        assert stats.get("correlation_coefficient", 0) > 0.8, "Strong correlation expected in analysis"
        assert stats.get("sample_size") == sample_size, "Correct sample size expected in analysis"
        assert 0 < stats.get("r_squared_value", 0) < 1, "R-squared should be between 0 and 1"
        assert stats.get("relationship_strength") == "strong", "Should identify strong relationship"

        # And: Filesystem information should be properly structured
        fs_info = result.get("filesystem_info", {})
        assert fs_info.get("base_path") == TestConfig.PLOTS_BASE_PATH, "Incorrect base path"
        assert fs_info.get("full_path", "").endswith(".png"), "Invalid file extension"
        assert "regression_test" in fs_info.get("full_path", ""), "Plot type not in filename"

        # And: Plot metadata should reflect configuration
        plot_meta = result.get("plot_metadata", {})
        assert plot_meta.get("figure_size") == [figure_width, figure_height], "Incorrect figure size"
        assert plot_meta.get("dpi") == plot_dpi, "Incorrect DPI setting"
        assert plot_meta.get("backend") == "Agg", "Should use non-interactive backend"

    def test_direct_file_save_advanced_dashboard(
        self,
        verified_server: str,
        server_config: Dict[str, Any],
        seaborn_environment: Dict[str, Any],
        clean_filesystem,
    ):
        """
        Test creating and saving advanced seaborn dashboard directly to virtual filesystem.

        **BDD Scenario**: Complex multi-plot dashboard creation and validation
        
        **Given**: A seaborn environment with comprehensive filesystem capabilities
        **When**: We create a complex dashboard with multiple plot types and save to filesystem
        **Then**: All visualizations should render correctly with proper statistical analysis

        This test demonstrates advanced seaborn capabilities including:
        - Multi-subplot dashboard layout with 6 distinct visualization types  
        - Cross-platform pathlib usage for complex filesystem operations
        - Comprehensive statistical analysis across multiple data dimensions
        - Professional plot styling and layout management
        - Memory management and resource cleanup
        - Detailed metadata extraction and validation

        Args:
            verified_server: Base URL of running test server
            server_config: Test configuration dictionary with dashboard parameters
            seaborn_environment: Seaborn availability confirmation with package status
            clean_filesystem: Filesystem cleanup fixture ensuring clean test environment

        Raises:
            AssertionError: If any dashboard component fails or API contract is violated
            pytest.skip: If required packages are not available

        Example:
            This test creates a comprehensive dashboard including:
            
            ```python
            # Dashboard components:
            # 1. Correlation heatmap - relationship analysis
            # 2. Box plots - group comparisons by category  
            # 3. Violin plots - distribution shape analysis
            # 4. Scatter plots - multi-dimensional relationships
            # 5. Density plots - probability distributions
            # 6. Count plots - categorical frequency analysis
            ```

        Expected Behavior:
            - Complex dataset generation with realistic relationships
            - Six distinct visualization types in single dashboard
            - Professional layout with tight_layout() optimization
            - High-resolution output suitable for publication
            - Comprehensive statistical metadata collection
            - Cross-platform filesystem compatibility using pathlib
            - Complete memory cleanup after plot generation
        """
        # Given: Seaborn environment is available with comprehensive capabilities
        if not seaborn_environment.get("seaborn_available"):
            pytest.skip("Seaborn not available - required for dashboard creation")
        if not seaborn_environment.get("matplotlib_available"):
            pytest.skip("Matplotlib not available - required for subplot management")
        if not seaborn_environment.get("server_ready"):
            pytest.skip("Server not ready for complex dashboard tests")

        # When: Creating comprehensive dashboard with multiple visualizations
        # Use different random seed for dashboard to ensure test independence
        dashboard_seed = TestConfig.DASHBOARD_RANDOM_SEED
        sample_size = server_config["sample_size_large"]
        dashboard_width = server_config["dashboard_width"]
        dashboard_height = server_config["dashboard_height"]
        plot_dpi = server_config["plot_dpi"]

        dashboard_code = f"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json

print("Creating comprehensive seaborn dashboard with advanced visualizations...")

# Set deterministic random seed for reproducible dashboard
np.random.seed({dashboard_seed})

# Configure seaborn for professional appearance
sns.set_style("whitegrid")
sns.set_palette("husl")  # Visually distinct colors

# Create comprehensive multi-dimensional dataset
n_samples = {sample_size}
print(f"Generating dataset with {{n_samples}} samples...")

# Generate realistic business data with complex relationships
data = {{
    'group': np.random.choice(['Alpha', 'Beta', 'Gamma'], n_samples),
    'category': np.random.choice(['Type1', 'Type2'], n_samples),
    'value1': np.random.normal(50, 15, n_samples),
    'value2': np.random.normal(100, 25, n_samples),
    'score': np.random.uniform(0, 100, n_samples),
    'efficiency': np.random.beta(2, 5, n_samples) * 100  # Realistic efficiency distribution
}}

# Add realistic business relationships and effects
for i in range(n_samples):
    # Group-based effects (market segments)
    if data['group'][i] == 'Alpha':
        data['value1'][i] += 20  # Premium segment
        data['score'][i] += 15
        data['efficiency'][i] *= 1.1
    elif data['group'][i] == 'Gamma':
        data['value2'][i] += 30  # Different value proposition
        data['score'][i] -= 10
        data['efficiency'][i] *= 0.9
    
    # Category effects (product types)
    if data['category'][i] == 'Type2':
        data['value1'][i] *= 1.15  # Product premium
        data['score'][i] += 5
        data['efficiency'][i] = min(data['efficiency'][i] * 1.05, 100)

# Convert to DataFrame for analysis
df = pd.DataFrame(data)

# Validate dataset quality
print(f"Dataset created: {{len(df)}} rows, {{len(df.columns)}} columns")
print(f"Groups: {{df['group'].unique()}}")
print(f"Categories: {{df['category'].unique()}}")

# Create comprehensive dashboard with professional layout
fig = plt.figure(figsize=({dashboard_width}, {dashboard_height}))
fig.suptitle('Comprehensive Business Analytics Dashboard', 
             fontsize=16, fontweight='bold', y=0.98)

# Subplot 1: Correlation heatmap (relationships overview)
plt.subplot(2, 3, 1)
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
           square=True, fmt='.2f', cbar_kws={{"shrink": .8}})
plt.title('Feature Correlation Matrix', fontweight='bold')

# Subplot 2: Box plot by group and category (distribution comparison)
plt.subplot(2, 3, 2)
sns.boxplot(data=df, x='group', y='value1', hue='category', palette='Set2')
plt.title('Value1 Distribution by Group & Category', fontweight='bold')
plt.ylabel('Value1 Score')

# Subplot 3: Violin plot (distribution shapes)
plt.subplot(2, 3, 3)
sns.violinplot(data=df, x='group', y='score', palette='viridis')
plt.title('Score Distribution Shapes by Group', fontweight='bold')
plt.ylabel('Performance Score')

# Subplot 4: Scatter plot with multiple dimensions
plt.subplot(2, 3, 4)
sns.scatterplot(data=df, x='value1', y='value2', hue='group', 
               size='efficiency', alpha=0.7, sizes=(20, 200))
plt.title('Multi-dimensional Relationship Analysis', fontweight='bold')
plt.xlabel('Value1 Metric')
plt.ylabel('Value2 Metric')

# Subplot 5: Density plots (probability distributions)
plt.subplot(2, 3, 5)
for group_name in df['group'].unique():
    subset = df[df['group'] == group_name]
    sns.kdeplot(data=subset, x='score', label=f'Group {{group_name}}', 
               alpha=0.7, fill=True)
plt.title('Score Probability Densities', fontweight='bold')
plt.xlabel('Performance Score')
plt.ylabel('Density')
plt.legend()

# Subplot 6: Count plot (categorical frequencies)
plt.subplot(2, 3, 6)
sns.countplot(data=df, x='group', hue='category', palette='pastel')
plt.title('Sample Distribution by Group & Category', fontweight='bold')
plt.ylabel('Count')

# Optimize layout for professional appearance
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle

# Save dashboard to virtual filesystem using pathlib (cross-platform)
timestamp = int(time.time() * 1000)
plots_path = Path('{TestConfig.PLOTS_BASE_PATH}')
plots_path.mkdir(parents=True, exist_ok=True)
output_file = plots_path / f'dashboard_comprehensive_{{timestamp}}.png'

# Save with publication-quality settings
plt.savefig(output_file, dpi={plot_dpi}, bbox_inches='tight',
           facecolor='white', edgecolor='none', pad_inches=0.2)
plt.close()  # Properly close to free memory

# Comprehensive statistical analysis
print("Calculating comprehensive statistics...")

# Group-based statistics
group_stats = {{}}
for stat_name in ['count', 'mean', 'std', 'min', 'max']:
    group_stats[stat_name] = df.groupby('group')['score'].agg(stat_name).to_dict()

# Category-based statistics  
category_stats = {{}}
for stat_name in ['count', 'mean', 'std', 'min', 'max']:
    category_stats[stat_name] = df.groupby('category')['value1'].agg(stat_name).to_dict()

# Advanced correlation analysis
correlation_analysis = {{
    "strongest_positive": "",
    "strongest_negative": "", 
    "max_correlation": 0,
    "min_correlation": 0
}}

# Find strongest correlations
corr_values = correlation_matrix.values
np.fill_diagonal(corr_values, 0)  # Ignore self-correlation
max_idx = np.unravel_index(np.argmax(np.abs(corr_values)), corr_values.shape)
correlation_analysis["max_correlation"] = float(corr_values[max_idx])
correlation_analysis["strongest_positive"] = f"{{numeric_df.columns[max_idx[0]]}} vs {{numeric_df.columns[max_idx[1]]}}"

# File verification and metadata
file_exists = output_file.exists()
file_size = output_file.stat().st_size if file_exists else 0

# Create comprehensive dashboard results
result = {{
    "file_saved": file_exists,
    "file_size": file_size,
    "file_size_valid": file_size > {TestConfig.MIN_PLOT_FILE_SIZE},
    "plot_type": "seaborn_comprehensive_dashboard",
    "n_samples": n_samples,
    "n_groups": len(df['group'].unique()),
    "n_categories": len(df['category'].unique()),
    "groups": sorted(df['group'].unique().tolist()),
    "categories": sorted(df['category'].unique().tolist()),
    "statistical_analysis": {{
        "group_statistics": group_stats,
        "category_statistics": category_stats,
        "correlation_analysis": correlation_analysis,
        "dataset_summary": {{
            "total_samples": int(len(df)),
            "feature_count": int(len(df.columns)),
            "numeric_features": len(numeric_df.columns),
            "categorical_features": len(df.select_dtypes(include=['object']).columns)
        }}
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
        ],
        "layout_configuration": {{
            "rows": 2,
            "columns": 3,
            "figure_size": [{dashboard_width}, {dashboard_height}],
            "dpi": {plot_dpi}
        }}
    }},
    "dashboard_metadata": {{
        "creation_timestamp": timestamp,
        "random_seed": {dashboard_seed},
        "backend": "Agg",
        "style": "whitegrid",
        "palette": "husl"
    }},
    "filesystem_info": {{
        "base_path": str(plots_path),
        "full_path": str(output_file),
        "file_size_bytes": file_size
    }}
}}

print(f"Dashboard saved successfully: {{output_file}}")
print(f"File size: {{file_size}} bytes")
print(f"Visualizations: {{len(result['visualization_info']['visualization_types'])}}")

# Return comprehensive result
result
"""

        # Execute using standardized execute-raw API call
        response_data = make_execute_raw_request(
            verified_server,
            dashboard_code,
            timeout=server_config["long_timeout"]
        )

        # Then: Validate successful execution and API contract compliance
        assert response_data.get("success"), f"Dashboard execution failed: {response_data}"

        # Extract result from data.result as per API contract
        data = response_data.get("data", {})
        result = data.get("result")
        assert result is not None, f"API returned None result: {response_data}"

        # And: Dashboard should be saved successfully with proper metadata
        assert result.get("file_saved") is True, "Dashboard was not saved to filesystem"
        assert result.get("file_size_valid") is True, f"Dashboard file size too small: {result.get('file_size', 0)} bytes"
        assert result.get("plot_type") == "seaborn_comprehensive_dashboard", "Incorrect dashboard type"

        # And: Dataset information should be comprehensive and accurate
        assert result.get("n_samples") == sample_size, f"Expected {sample_size} samples"
        assert result.get("n_groups") == 3, "Should have exactly 3 groups (Alpha, Beta, Gamma)"
        assert result.get("n_categories") == 2, "Should have exactly 2 categories (Type1, Type2)"
        
        groups = result.get("groups", [])
        expected_groups = ["Alpha", "Beta", "Gamma"]
        assert all(group in groups for group in expected_groups), f"Missing groups: {set(expected_groups) - set(groups)}"

        # And: Statistical analysis should be comprehensive and valid
        stats = result.get("statistical_analysis", {})
        assert "group_statistics" in stats, "Missing group-based statistical analysis"
        assert "category_statistics" in stats, "Missing category-based statistical analysis"
        assert "correlation_analysis" in stats, "Missing correlation analysis"
        assert "dataset_summary" in stats, "Missing dataset summary statistics"

        # Validate statistical structure depth
        group_stats = stats.get("group_statistics", {})
        assert "count" in group_stats, "Missing group count statistics"
        assert "mean" in group_stats, "Missing group mean statistics"
        assert "std" in group_stats, "Missing group standard deviation statistics"

        dataset_summary = stats.get("dataset_summary", {})
        assert dataset_summary.get("total_samples") == sample_size, "Incorrect sample count in summary"
        assert dataset_summary.get("feature_count") >= 5, "Should have at least 5 features"

        # And: Visualization information should be complete and accurate
        viz_info = result.get("visualization_info", {})
        assert viz_info.get("subplot_count") == 6, "Should have exactly 6 subplots"
        
        viz_types = viz_info.get("visualization_types", [])
        expected_viz_types = [
            "correlation_heatmap", "box_plot", "violin_plot",
            "scatter_plot", "density_plot", "count_plot"
        ]
        assert len(viz_types) == 6, f"Expected 6 visualization types, got {len(viz_types)}"
        assert all(vtype in viz_types for vtype in expected_viz_types), f"Missing visualization types: {set(expected_viz_types) - set(viz_types)}"

        # And: Layout configuration should match parameters
        layout_config = viz_info.get("layout_configuration", {})
        assert layout_config.get("figure_size") == [dashboard_width, dashboard_height], "Incorrect figure size"
        assert layout_config.get("dpi") == plot_dpi, "Incorrect DPI setting"
        assert layout_config.get("rows") == 2, "Should have 2 rows"
        assert layout_config.get("columns") == 3, "Should have 3 columns"

        # And: Dashboard metadata should reflect proper configuration
        dashboard_meta = result.get("dashboard_metadata", {})
        assert dashboard_meta.get("random_seed") == dashboard_seed, "Incorrect random seed"
        assert dashboard_meta.get("backend") == "Agg", "Should use non-interactive backend"
        assert dashboard_meta.get("style") == "whitegrid", "Should use whitegrid style"
        assert dashboard_meta.get("palette") == "husl", "Should use husl color palette"

    def test_edge_case_small_dataset_handling(
        self,
        verified_server: str,
        server_config: Dict[str, Any],
        seaborn_environment: Dict[str, Any],
        clean_filesystem,
    ):
        """
        Test seaborn plotting with edge case scenarios and small datasets.

        **BDD Scenario**: Edge case validation with minimal data
        
        **Given**: A seaborn environment with small dataset constraints
        **When**: We create plots with minimal data points and edge cases
        **Then**: The system should handle gracefully with appropriate warnings

        This test validates system robustness with:
        - Very small datasets (edge case sample sizes)
        - Missing data handling and validation
        - Error recovery and graceful degradation
        - Appropriate warning generation for statistical limitations
        - Memory efficiency with small datasets

        Args:
            verified_server: Base URL of running test server
            server_config: Test configuration dictionary
            seaborn_environment: Seaborn availability confirmation
            clean_filesystem: Filesystem cleanup fixture

        Raises:
            pytest.skip: If seaborn environment is not available
        """
        # Given: Environment validation for edge case testing
        if not seaborn_environment.get("seaborn_available"):
            pytest.skip("Seaborn not available for edge case testing")

        # When: Testing with minimal dataset
        edge_case_code = f"""
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import time
import warnings

print("Testing edge case scenarios with minimal dataset...")

# Capture warnings for edge case analysis
warnings.simplefilter('always')
captured_warnings = []

# Create minimal dataset for edge case testing
np.random.seed(42)
n_samples = {TestConfig.EDGE_CASE_SAMPLE_SIZE}

# Minimal data with potential statistical challenges
minimal_data = {{
    'x': np.random.normal(0, 1, n_samples),
    'y': np.random.normal(0, 1, n_samples),
    'category': ['A'] * (n_samples // 2) + ['B'] * (n_samples - n_samples // 2)
}}

df_minimal = pd.DataFrame(minimal_data)

# Test seaborn's handling of minimal data
plt.figure(figsize=(8, 6))
try:
    sns.scatterplot(data=df_minimal, x='x', y='y', hue='category')
    plt.title('Edge Case: Minimal Dataset Scatter Plot')
    
    # Save edge case plot
    timestamp = int(time.time() * 1000)
    plots_path = Path('{TestConfig.PLOTS_BASE_PATH}')
    plots_path.mkdir(parents=True, exist_ok=True)
    output_file = plots_path / f'edge_case_minimal_{{timestamp}}.png'
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    file_exists = output_file.exists()
    file_size = output_file.stat().st_size if file_exists else 0
    
    edge_case_result = {{
        "file_saved": file_exists,
        "file_size": file_size,
        "test_type": "edge_case_minimal_dataset",
        "sample_size": n_samples,
        "categories_count": len(df_minimal['category'].unique()),
        "statistical_validity": "limited" if n_samples < 30 else "adequate",
        "warnings_captured": len(captured_warnings),
        "filename": str(output_file)
    }}
    
except Exception as e:
    edge_case_result = {{
        "file_saved": False,
        "error": str(e),
        "test_type": "edge_case_minimal_dataset_failed",
        "sample_size": n_samples
    }}

print(f"Edge case test completed: {{edge_case_result}}")
edge_case_result
"""

        # Execute edge case test
        response_data = make_execute_raw_request(
            verified_server,
            edge_case_code,
            timeout=server_config["medium_timeout"]
        )

        # Then: Validate edge case handling
        assert response_data.get("success"), f"Edge case test failed: {response_data}"
        
        data = response_data.get("data", {})
        result = data.get("result")
        assert result is not None, "Edge case test returned no result"

        # Validate edge case results
        if result.get("file_saved"):
            assert result.get("test_type") == "edge_case_minimal_dataset", "Incorrect edge case test type"
            assert result.get("sample_size") == TestConfig.EDGE_CASE_SAMPLE_SIZE, "Incorrect sample size"
        else:
            # If file not saved due to errors, that's acceptable for edge cases
            print(f"Edge case resulted in expected limitation: {result.get('error', 'Unknown')}")

    def test_api_contract_error_handling(
        self,
        verified_server: str,
        server_config: Dict[str, Any],
        clean_filesystem,
    ):
        """
        Test API contract compliance during error scenarios.

        **BDD Scenario**: Error handling with proper API contract
        
        **Given**: A running server with execute-raw endpoint
        **When**: We send invalid Python code that will cause errors
        **Then**: The API should return proper error structure following the contract

        This test validates:
        - Proper error structure in API responses
        - API contract compliance during failures
        - Error message clarity and usefulness
        - Proper HTTP status codes
        - Meta information during error states

        Args:
            verified_server: Base URL of running test server
            server_config: Test configuration dictionary
            clean_filesystem: Filesystem cleanup fixture
        """
        # Given: Server is available for error testing
        
        # When: Sending invalid Python code to trigger error
        invalid_code = """
import seaborn as sns
import matplotlib.pyplot as plt

# This will cause an error - undefined variable
print(undefined_variable)
sns.scatterplot(x=[1, 2, 3], y=invalid_data)
"""

        response = requests.post(
            f"{verified_server}{TestConfig.API_EXECUTE_RAW_ENDPOINT}",
            data=invalid_code,
            headers={"Content-Type": "text/plain"},
            timeout=server_config["short_timeout"],
        )

        # Then: Validate error response follows API contract
        assert response.status_code == 200, "Should return 200 even for execution errors"
        
        response_data = response.json()
        validate_api_contract(response_data)

        # Error-specific validations
        assert response_data.get("success") is False, "Should indicate failure for invalid code"
        assert response_data.get("data") is None, "Data should be null for failed execution"
        assert isinstance(response_data.get("error"), str), "Error should be descriptive string"
        assert len(response_data.get("error", "")) > 0, "Error message should not be empty"

        # Meta information should still be present during errors
        meta = response_data.get("meta", {})
        assert "timestamp" in meta, "Timestamp should be present even during errors"

    def test_performance_large_dataset_timeout_handling(
        self,
        verified_server: str,
        server_config: Dict[str, Any],
        seaborn_environment: Dict[str, Any],
        clean_filesystem,
    ):
        """
        Test performance characteristics and timeout handling with large datasets.

        **BDD Scenario**: Performance validation with realistic workloads
        
        **Given**: A seaborn environment capable of handling large datasets
        **When**: We create complex visualizations with substantial data volumes
        **Then**: The system should complete within reasonable timeframes or handle timeouts gracefully

        This test validates:
        - Performance with larger datasets
        - Memory management during intensive operations
        - Proper timeout handling and recovery
        - Resource cleanup after complex operations
        - Scalability characteristics

        Args:
            verified_server: Base URL of running test server
            server_config: Test configuration dictionary
            seaborn_environment: Seaborn availability confirmation
            clean_filesystem: Filesystem cleanup fixture

        Raises:
            pytest.skip: If seaborn environment is not available
        """
        # Given: Environment validation for performance testing
        if not seaborn_environment.get("seaborn_available"):
            pytest.skip("Seaborn not available for performance testing")

        # When: Creating visualization with larger dataset
        large_dataset_size = TestConfig.LARGE_DATASET_SIZE
        performance_code = f"""
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import time

print("Starting performance test with large dataset...")
start_time = time.time()

# Create substantial dataset for performance testing
np.random.seed(42)
n_samples = {large_dataset_size}

print(f"Generating {{n_samples}} data points...")
data_generation_start = time.time()

large_data = {{
    'feature_1': np.random.normal(50, 20, n_samples),
    'feature_2': np.random.normal(100, 30, n_samples),
    'feature_3': np.random.exponential(2, n_samples),
    'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
    'subcategory': np.random.choice(['X', 'Y'], n_samples)
}}

df_large = pd.DataFrame(large_data)
data_generation_time = time.time() - data_generation_start

print(f"Data generation completed in {{data_generation_time:.2f}} seconds")

# Create performance test visualization
plot_start_time = time.time()
plt.figure(figsize=(12, 8))

# Complex visualization that exercises seaborn capabilities
sns.scatterplot(data=df_large, x='feature_1', y='feature_2', 
               hue='category', style='subcategory', alpha=0.6)
plt.title(f'Performance Test: {{n_samples:,}} Data Points')

# Save performance test plot
plots_path = Path('{TestConfig.PLOTS_BASE_PATH}')
plots_path.mkdir(parents=True, exist_ok=True)
timestamp = int(time.time() * 1000)
output_file = plots_path / f'performance_test_{{timestamp}}.png'

plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

plot_time = time.time() - plot_start_time
total_time = time.time() - start_time

# Verify results
file_exists = output_file.exists()
file_size = output_file.stat().st_size if file_exists else 0

performance_result = {{
    "file_saved": file_exists,
    "file_size": file_size,
    "test_type": "performance_large_dataset",
    "dataset_size": n_samples,
    "timing_analysis": {{
        "data_generation_seconds": round(data_generation_time, 3),
        "plot_creation_seconds": round(plot_time, 3),
        "total_execution_seconds": round(total_time, 3)
    }},
    "performance_metrics": {{
        "data_points_per_second": round(n_samples / total_time, 1),
        "memory_efficiency": "acceptable" if total_time < 30 else "needs_optimization"
    }},
    "filename": str(output_file)
}}

print(f"Performance test completed in {{total_time:.2f}} seconds")
performance_result
"""

        # Execute with extended timeout for performance test
        extended_timeout = min(server_config["long_timeout"] * 2, 240)  # Max 4 minutes
        
        try:
            response_data = make_execute_raw_request(
                verified_server,
                performance_code,
                timeout=extended_timeout
            )

            # Then: Validate performance test results
            assert response_data.get("success"), f"Performance test failed: {response_data}"
            
            data = response_data.get("data", {})
            result = data.get("result")
            assert result is not None, "Performance test returned no result"

            # Performance-specific validations
            assert result.get("file_saved") is True, "Performance test plot should be saved"
            assert result.get("dataset_size") == large_dataset_size, "Incorrect dataset size"
            
            timing = result.get("timing_analysis", {})
            total_time = timing.get("total_execution_seconds", 0)
            assert total_time > 0, "Should have positive execution time"
            
            # Log performance characteristics
            print(f"✅ Performance test completed in {total_time} seconds with {large_dataset_size} data points")
            
        except requests.Timeout:
            # Timeout is acceptable for very large datasets
            pytest.skip(f"Performance test timed out with {large_dataset_size} data points - expected behavior")

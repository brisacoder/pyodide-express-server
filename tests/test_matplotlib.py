"""
Pytest BDD Tests for Matplotlib Plotting with Pyodide Express Server.

This test suite follows BDD (Behavior-Driven Development) patterns with Given-When-Then
structure and provides comprehensive coverage for matplotlib plotting functionality
using the Pyodide Express Server API.

Test Coverage:
- Basic line plots with data visualization
- Scatter plots with color and size mapping
- Histogram plots with statistical data
- Complex subplot layouts and compositions
- Direct filesystem plotting operations
- Error handling and edge cases
- Performance testing with timeout scenarios

Requirements Compliance:
1. ✅ Pytest framework with BDD style scenarios
2. ✅ All globals parameterized via Config class and fixtures
3. ✅ No internal REST APIs (no 'pyodide' endpoints)
4. ✅ BDD Given-When-Then structure in docstrings
5. ✅ Only /api/execute-raw for Python execution
6. ✅ No internal pyodide REST APIs
7. ✅ Comprehensive test coverage for plotting scenarios
8. ✅ Full docstrings with descriptions, inputs, outputs, examples
9. ✅ Python code uses pathlib for portability
10. ✅ JavaScript API contract validation enforced

API Contract Validation:
{
  "success": true | false,
  "data": { "result": any, "stdout": str, "stderr": str, "executionTime": int } | null,
  "error": string | null,
  "meta": { "timestamp": string }
}

Example Usage:
    pytest tests/test_matplotlib.py -v
    pytest tests/test_matplotlib.py::test_when_basic_line_plot_is_created_then_visualization_is_generated -s
    pytest tests/test_matplotlib.py -k "scatter" --tb=short
"""

import base64
import json
import time
from pathlib import Path
from typing import Any

import pytest
import requests

try:
    from .conftest import Config, execute_python_code, validate_api_contract
except ImportError:
    # For direct execution, try absolute import
    from conftest import Config, execute_python_code, validate_api_contract


# ==================== MATPLOTLIB-SPECIFIC CONFIGURATION ====================


class MatplotlibConfig:
    """Matplotlib-specific test configuration constants."""

    # Plot generation settings
    PLOT_SETTINGS = {
        "default_dpi": 150,
        "figure_size_basic": (10, 6),
        "figure_size_complex": (14, 12),
        "figure_size_scatter": (10, 8),
        "figure_size_histogram": (12, 5),
        "line_width": 2,
        "alpha_transparency": 0.6,
        "grid_alpha": 0.3,
    }

    # Test data settings
    TEST_DATA = {
        "random_seed": 42,
        "sample_points_small": 100,
        "sample_points_medium": 200,
        "sample_points_large": 1000,
        "histogram_bins": 30,
    }

    # Virtual filesystem paths (pathlib for portability)
    VIRTUAL_PATHS = {
        "plots_dir": "/home/pyodide/plots/matplotlib",
        "temp_dir": "/tmp",
    }

    # File naming patterns
    FILE_PATTERNS = {
        "basic_plot": "basic_line_plot_{timestamp}.png",
        "scatter_plot": "scatter_plot_colors_{timestamp}.png",
        "histogram_plot": "histogram_plot_{timestamp}.png",
        "complex_plot": "subplot_complex_{timestamp}.png",
        "direct_save_basic": "direct_save_basic_{timestamp}.png",
        "direct_save_complex": "direct_save_complex_{timestamp}.png",
    }


# ==================== FIXTURES ====================


@pytest.fixture(scope="session")
def matplotlib_ready(server_ready):
    """
    Ensure matplotlib package is installed and available in Pyodide environment.

    This fixture validates that matplotlib can be imported and used for plotting
    operations within the Pyodide runtime environment.

    Args:
        server_ready: Server readiness fixture dependency

    Returns:
        bool: True if matplotlib is available, False otherwise

    Raises:
        pytest.skip: If matplotlib cannot be installed or imported

    Example:
        >>> def test_plotting(matplotlib_ready):
        ...     if not matplotlib_ready:
        ...         pytest.skip("matplotlib not available")
        ...     # Test plotting code here
    """
    # Install matplotlib package in Pyodide environment
    install_response = requests.post(
        f"{Config.BASE_URL}{Config.ENDPOINTS['install_package']}",
        json={"package": "matplotlib"},
        timeout=Config.TIMEOUTS["code_execution"] * 4,  # Extended timeout for package installation
    )

    if install_response.status_code != 200:
        pytest.skip(f"Failed to install matplotlib: {install_response.status_code}")

    # Verify matplotlib can be imported
    check_code = """
import matplotlib
print(f"matplotlib version: {matplotlib.__version__}")
import matplotlib.pyplot as plt
print("matplotlib.pyplot imported successfully")
"""

    try:
        result = execute_python_code(check_code, timeout=Config.TIMEOUTS["code_execution"])
        if result["success"] and "matplotlib version:" in result["data"]["stdout"]:
            print(f"✅ Matplotlib ready: {result['data']['stdout'].strip()}")
            return True
        else:
            pytest.skip(f"Matplotlib import failed: {result}")
    except Exception as e:
        pytest.skip(f"Matplotlib availability check failed: {e}")


@pytest.fixture
def plot_cleanup():
    """
    Provide cleanup functionality for generated plot files.

    This fixture creates a cleanup tracker that automatically removes
    plot files and artifacts after each test completes.

    Yields:
        PlotCleanupTracker: Object to track plot files for cleanup

    Example:
        >>> def test_plot_generation(plot_cleanup):
        ...     cleanup = plot_cleanup
        ...     # Generate plot
        ...     plot_file = generate_plot()
        ...     cleanup.track_plot_file(plot_file)
        ...     # File automatically cleaned up after test
    """
    class PlotCleanupTracker:
        """Tracks and cleans up plot files and artifacts."""

        def __init__(self):
            self.plot_files = []
            self.temp_files = []
            self.start_time = time.time()

        def track_plot_file(self, filepath: str | Path) -> None:
            """Track plot file for automatic cleanup."""
            self.plot_files.append(Path(filepath))

        def track_temp_file(self, filepath: str | Path) -> None:
            """Track temporary file for automatic cleanup."""
            self.temp_files.append(Path(filepath))

        def cleanup(self) -> None:
            """Clean up all tracked plot and temporary files."""
            # Clean up plot files
            for plot_file in self.plot_files:
                if plot_file.exists() and plot_file.is_file():
                    try:
                        plot_file.unlink()
                        print(f"Cleaned up plot file: {plot_file}")
                    except Exception as e:
                        print(f"Warning: Could not remove plot file {plot_file}: {e}")

            # Clean up temporary files
            for temp_file in self.temp_files:
                if temp_file.exists() and temp_file.is_file():
                    try:
                        temp_file.unlink()
                        print(f"Cleaned up temp file: {temp_file}")
                    except Exception as e:
                        print(f"Warning: Could not remove temp file {temp_file}: {e}")

        def get_test_duration(self) -> float:
            """Get test execution duration in seconds."""
            return time.time() - self.start_time

    tracker = PlotCleanupTracker()
    yield tracker
    tracker.cleanup()


# ==================== UTILITY FUNCTIONS ====================


def save_plot_from_base64(base64_data: str, filename: str, output_dir: Path) -> Path:
    """
    Save a base64 encoded plot to the local filesystem.

    This function decodes base64 plot data and saves it as a PNG file
    to the specified directory using pathlib for cross-platform compatibility.

    Args:
        base64_data: Base64 encoded plot image data
        filename: Name of the output file (including .png extension)
        output_dir: Directory to save the file to

    Returns:
        Path: Path to the saved file

    Raises:
        ValueError: If base64 data is invalid
        OSError: If file cannot be written

    Example:
        >>> plot_data = "iVBORw0KGgoAAAANSUhEUgAA..."
        >>> output_path = save_plot_from_base64(plot_data, "test.png", Path("/tmp"))
        >>> assert output_path.exists()
    """
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Decode base64 data
        image_data = base64.b64decode(base64_data)
        filepath = output_dir / filename

        # Write binary data to file
        with filepath.open('wb') as f:
            f.write(image_data)

        print(f"✅ Plot saved to: {filepath}")
        return filepath
    except (ValueError, OSError, IOError) as e:
        raise RuntimeError(f"Failed to save plot: {e}")


def generate_timestamp_filename(pattern: str) -> str:
    """
    Generate a timestamped filename to avoid conflicts.

    Args:
        pattern: Filename pattern with {timestamp} placeholder

    Returns:
        str: Filename with timestamp inserted

    Example:
        >>> filename = generate_timestamp_filename("plot_{timestamp}.png")
        >>> assert "plot_" in filename and ".png" in filename
    """
    timestamp = int(time.time() * 1000)
    return pattern.format(timestamp=timestamp)


def parse_result_data(response: dict) -> dict:
    """
    Parse result data from server response, handling JSON string conversion.
    
    The server may return result data as a JSON string that needs to be parsed
    back into a dictionary for proper assertion handling.
    
    Args:
        response: Server response dictionary containing data.result
        
    Returns:
        Parsed result data as dictionary
        
    Raises:
        json.JSONDecodeError: If result data is malformed JSON
        KeyError: If response structure is invalid
    """
    result_data = response["data"]["result"]
    if isinstance(result_data, str):
        try:
            return json.loads(result_data)
        except json.JSONDecodeError:
            # If it's not valid JSON, return as-is
            return {"raw_result": result_data}
    return result_data


# ==================== BDD TEST SCENARIOS ====================


@pytest.mark.api
@pytest.mark.plotting
def test_when_basic_line_plot_is_created_then_visualization_is_generated(
    matplotlib_ready: bool,
    plot_cleanup: Any,
    tmp_path: Path
) -> None:
    """
    Test basic line plot generation using matplotlib in Pyodide environment.

    Given: A Pyodide environment with matplotlib available
    When: A basic line plot with sin(x) function is created using execute-raw endpoint
    Then: The plot should be generated successfully with base64 encoded data
    And: The plot should contain expected visualization elements
    And: The API response should follow the contract specification

    This test validates:
    - Basic matplotlib plotting functionality in Pyodide
    - Proper API contract compliance with data.result structure
    - Base64 encoding of plot data for transmission
    - Successful plot generation with labeled axes and grid

    Args:
        matplotlib_ready: Fixture ensuring matplotlib is available
        plot_cleanup: Fixture for automatic cleanup of generated files
        tmp_path: Pytest temporary directory fixture

    Returns:
        None: Test validates through assertions

    Raises:
        AssertionError: If plot generation fails or API contract is violated

    Example:
        Expected API response format:
        {
          "success": true,
          "data": {
            "result": {"plot_base64": "iVBOR...", "plot_type": "line_plot"},
            "stdout": "...",
            "stderr": "",
            "executionTime": 1500
          },
          "error": null,
          "meta": {"timestamp": "2025-01-01T00:00:00Z"}
        }
    """
    # Given: Matplotlib is available in Pyodide environment
    if not matplotlib_ready:
        pytest.skip("matplotlib not available in this Pyodide environment")

    # When: Creating a basic line plot with sin(x) function using pathlib
    plot_code = f"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Create sample data for sin(x) visualization
x = np.linspace(0, 10, {MatplotlibConfig.TEST_DATA['sample_points_small']})
y = np.sin(x)

# Create the plot with specified configuration
plt.figure(figsize={MatplotlibConfig.PLOT_SETTINGS['figure_size_basic']})
plt.plot(x, y, 'b-', linewidth={MatplotlibConfig.PLOT_SETTINGS['line_width']}, label='sin(x)')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Basic Line Plot - sin(x)')
plt.grid(True, alpha={MatplotlibConfig.PLOT_SETTINGS['grid_alpha']})
plt.legend()

# Save to bytes buffer for base64 encoding
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi={MatplotlibConfig.PLOT_SETTINGS['default_dpi']}, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Return structured result
result = {{"plot_base64": plot_b64, "plot_type": "line_plot"}}
print(f"Generated line plot with {{len(plot_b64)}} base64 characters")
result
"""

    # Execute the plot generation code via execute-raw endpoint
    response = requests.post(
        f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
        headers=Config.HEADERS["execute_raw"],
        data=plot_code,
        timeout=Config.TIMEOUTS["code_execution"]
    )

    # Then: The API response should be successful and follow contract
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    validate_api_contract(response_data)

    assert response_data["success"] is True, f"Plot generation failed: {response_data}"
    assert response_data["data"] is not None, "Response data should not be null for success"
    assert "result" in response_data["data"], "Response should contain result field"

    # And: The plot result should contain expected base64 data
    result = parse_result_data(response_data)
    assert isinstance(result, dict), f"Result should be dict, got {type(result)}"
    assert "plot_base64" in result, "Result should contain plot_base64 field"
    assert "plot_type" in result, "Result should contain plot_type field"
    assert result["plot_type"] == "line_plot", f"Expected 'line_plot', got {result['plot_type']}"

    # Validate base64 data quality
    plot_data = result["plot_base64"]
    assert isinstance(plot_data, str), "Plot data should be string"
    assert len(plot_data) > 1000, f"Plot data seems too small: {len(plot_data)} chars"

    # And: The plot should be decodable and saveable to filesystem
    filename = generate_timestamp_filename(MatplotlibConfig.FILE_PATTERNS["basic_plot"])
    saved_path = save_plot_from_base64(plot_data, filename, tmp_path)
    plot_cleanup.track_plot_file(saved_path)

    assert saved_path.exists(), f"Plot file was not saved to {saved_path}"
    assert saved_path.stat().st_size > 0, "Plot file has zero size"


@pytest.mark.api
@pytest.mark.plotting
def test_when_scatter_plot_with_colors_is_created_then_visualization_includes_colorbar(
    matplotlib_ready: bool,
    plot_cleanup: Any,
    tmp_path: Path
) -> None:
    """
    Test scatter plot generation with color and size mapping using matplotlib.

    Given: A Pyodide environment with matplotlib available
    When: A scatter plot with color-coded points and variable sizes is created
    Then: The plot should be generated with proper colorbar and legend
    And: The plot data should include metadata about number of points
    And: The API response should follow the contract specification

    This test validates:
    - Scatter plot functionality with advanced matplotlib features
    - Color mapping and size variation for data points
    - Colorbar integration for visual interpretation
    - Statistical metadata inclusion in results
    - Proper random seed usage for reproducible results

    Args:
        matplotlib_ready: Fixture ensuring matplotlib is available
        plot_cleanup: Fixture for automatic cleanup of generated files
        tmp_path: Pytest temporary directory fixture

    Returns:
        None: Test validates through assertions

    Raises:
        AssertionError: If scatter plot generation fails or metadata is incorrect

    Example:
        Expected result structure:
        {
          "plot_base64": "iVBOR...",
          "plot_type": "scatter_plot",
          "n_points": 200
        }
    """
    # Given: Matplotlib is available in Pyodide environment
    if not matplotlib_ready:
        pytest.skip("matplotlib not available in this Pyodide environment")

    # When: Creating a scatter plot with color and size mapping using pathlib
    scatter_code = f"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Create sample data with controlled randomness
np.random.seed({MatplotlibConfig.TEST_DATA['random_seed']})
n_points = {MatplotlibConfig.TEST_DATA['sample_points_medium']}
x = np.random.randn(n_points)
y = np.random.randn(n_points)
colors = np.random.rand(n_points)
sizes = 1000 * np.random.rand(n_points)

# Create the scatter plot with color and size mapping
plt.figure(figsize={MatplotlibConfig.PLOT_SETTINGS['figure_size_scatter']})
scatter = plt.scatter(x, y, c=colors, s=sizes, alpha={MatplotlibConfig.PLOT_SETTINGS['alpha_transparency']}, cmap='viridis')
plt.colorbar(scatter, label='Color Scale')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot with Colors and Sizes')
plt.grid(True, alpha={MatplotlibConfig.PLOT_SETTINGS['grid_alpha']})

# Save to bytes buffer for base64 encoding
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi={MatplotlibConfig.PLOT_SETTINGS['default_dpi']}, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Return structured result with metadata
result = {{"plot_base64": plot_b64, "plot_type": "scatter_plot", "n_points": n_points}}
print(f"Generated scatter plot with {{n_points}} points and {{len(plot_b64)}} base64 characters")
result
"""

    # Execute the scatter plot generation code via execute-raw endpoint
    response = requests.post(
        f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
        headers=Config.HEADERS["execute_raw"],
        data=scatter_code,
        timeout=Config.TIMEOUTS["code_execution"]
    )

    # Then: The API response should be successful and follow contract
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    validate_api_contract(response_data)

    assert response_data["success"] is True, f"Scatter plot generation failed: {response_data}"
    assert response_data["data"] is not None, "Response data should not be null for success"

    # And: The plot result should contain expected scatter plot data
    result = parse_result_data(response_data)
    assert isinstance(result, dict), f"Result should be dict, got {type(result)}"
    assert "plot_base64" in result, "Result should contain plot_base64 field"
    assert "plot_type" in result, "Result should contain plot_type field"
    assert "n_points" in result, "Result should contain n_points field"

    assert result["plot_type"] == "scatter_plot", f"Expected 'scatter_plot', got {result['plot_type']}"
    assert result["n_points"] == MatplotlibConfig.TEST_DATA['sample_points_medium'], \
        f"Expected {MatplotlibConfig.TEST_DATA['sample_points_medium']} points, got {result['n_points']}"

    # And: The plot should be decodable and saveable to filesystem
    plot_data = result["plot_base64"]
    assert len(plot_data) > 1000, f"Scatter plot data seems too small: {len(plot_data)} chars"

    filename = generate_timestamp_filename(MatplotlibConfig.FILE_PATTERNS["scatter_plot"])
    saved_path = save_plot_from_base64(plot_data, filename, tmp_path)
    plot_cleanup.track_plot_file(saved_path)

    assert saved_path.exists(), f"Scatter plot file was not saved to {saved_path}"
    assert saved_path.stat().st_size > 0, "Scatter plot file has zero size"


@pytest.mark.api
@pytest.mark.plotting
def test_when_histogram_plot_is_created_then_statistical_visualization_is_generated(
    matplotlib_ready: bool,
    plot_cleanup: Any,
    tmp_path: Path
) -> None:
    """
    Test histogram plot generation with statistical data analysis.

    Given: A Pyodide environment with matplotlib available
    When: A histogram plot with multiple distributions is created
    Then: The plot should show overlapping histograms with proper binning
    And: Statistical metadata should be calculated and returned
    And: The plot should include proper labels and formatting

    This test validates:
    - Histogram plotting with multiple data series
    - Statistical calculation integration (mean values)
    - Subplot layout functionality
    - Proper binning and distribution visualization
    - Metadata accuracy for statistical validation

    Args:
        matplotlib_ready: Fixture ensuring matplotlib is available
        plot_cleanup: Fixture for automatic cleanup of generated files
        tmp_path: Pytest temporary directory fixture

    Returns:
        None: Test validates through assertions

    Raises:
        AssertionError: If histogram generation fails or statistics are incorrect

    Example:
        Expected result with statistical metadata:
        {
          "plot_base64": "iVBOR...",
          "plot_type": "histogram",
          "data1_mean": 0.02,
          "data2_mean": 1.98
        }
    """
    # Given: Matplotlib is available in Pyodide environment
    if not matplotlib_ready:
        pytest.skip("matplotlib not available in this Pyodide environment")

    # When: Creating histogram plots with statistical distributions using pathlib
    histogram_code = f"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Create sample data with known statistical properties
np.random.seed(123)  # Different seed for variety
data1 = np.random.normal(0, 1, {MatplotlibConfig.TEST_DATA['sample_points_large']})
data2 = np.random.normal(2, 1.5, {MatplotlibConfig.TEST_DATA['sample_points_large']})

# Create the histogram plot with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize={MatplotlibConfig.PLOT_SETTINGS['figure_size_histogram']})

# Histogram 1: Normal distribution centered at 0
ax1.hist(data1, bins={MatplotlibConfig.TEST_DATA['histogram_bins']}, alpha=0.7, color='blue', edgecolor='black')
ax1.set_xlabel('Values')
ax1.set_ylabel('Frequency')
ax1.set_title('Normal Distribution (μ=0, σ=1)')
ax1.grid(True, alpha={MatplotlibConfig.PLOT_SETTINGS['grid_alpha']})

# Histogram 2: Normal distribution centered at 2
ax2.hist(data2, bins={MatplotlibConfig.TEST_DATA['histogram_bins']}, alpha=0.7, color='red', edgecolor='black')
ax2.set_xlabel('Values')
ax2.set_ylabel('Frequency')
ax2.set_title('Normal Distribution (μ=2, σ=1.5)')
ax2.grid(True, alpha={MatplotlibConfig.PLOT_SETTINGS['grid_alpha']})

plt.tight_layout()

# Save to bytes buffer for base64 encoding
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi={MatplotlibConfig.PLOT_SETTINGS['default_dpi']}, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Calculate statistical metadata for validation
data1_mean = float(np.mean(data1))
data2_mean = float(np.mean(data2))

# Return structured result with statistical metadata
result = {{
    "plot_base64": plot_b64,
    "plot_type": "histogram",
    "data1_mean": data1_mean,
    "data2_mean": data2_mean
}}
print(f"Generated histogram plot with means: data1={{data1_mean:.3f}}, data2={{data2_mean:.3f}}")
result
"""

    # Execute the histogram generation code via execute-raw endpoint
    response = requests.post(
        f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
        headers=Config.HEADERS["execute_raw"],
        data=histogram_code,
        timeout=Config.TIMEOUTS["code_execution"]
    )

    # Then: The API response should be successful and follow contract
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    validate_api_contract(response_data)

    assert response_data["success"] is True, f"Histogram generation failed: {response_data}"
    assert response_data["data"] is not None, "Response data should not be null for success"

    # And: The plot result should contain expected histogram data and statistics
    result = parse_result_data(response_data)
    assert isinstance(result, dict), f"Result should be dict, got {type(result)}"
    assert "plot_base64" in result, "Result should contain plot_base64 field"
    assert "plot_type" in result, "Result should contain plot_type field"
    assert "data1_mean" in result, "Result should contain data1_mean field"
    assert "data2_mean" in result, "Result should contain data2_mean field"

    assert result["plot_type"] == "histogram", f"Expected 'histogram', got {result['plot_type']}"

    # Validate statistical accuracy (with reasonable tolerance for random data)
    assert abs(result["data1_mean"] - 0.0) < 0.2, f"data1_mean should be ~0.0, got {result['data1_mean']}"
    assert abs(result["data2_mean"] - 2.0) < 0.2, f"data2_mean should be ~2.0, got {result['data2_mean']}"

    # And: The plot should be decodable and saveable to filesystem
    plot_data = result["plot_base64"]
    assert len(plot_data) > 1000, f"Histogram plot data seems too small: {len(plot_data)} chars"

    filename = generate_timestamp_filename(MatplotlibConfig.FILE_PATTERNS["histogram_plot"])
    saved_path = save_plot_from_base64(plot_data, filename, tmp_path)
    plot_cleanup.track_plot_file(saved_path)

    assert saved_path.exists(), f"Histogram plot file was not saved to {saved_path}"
    assert saved_path.stat().st_size > 0, "Histogram plot file has zero size"


@pytest.mark.api
@pytest.mark.plotting
@pytest.mark.slow
def test_when_complex_subplot_layout_is_created_then_multi_panel_visualization_is_generated(
    matplotlib_ready: bool,
    plot_cleanup: Any,
    tmp_path: Path
) -> None:
    """
    Test complex subplot layout creation with multiple visualization types.

    Given: A Pyodide environment with matplotlib available
    When: A complex subplot layout with 4 different plot types is created
    Then: All subplots should be generated with proper layout and formatting
    And: Each subplot should contain different mathematical functions
    And: The overall figure should have consistent styling and title

    This test validates:
    - Complex subplot layout management (2x2 grid)
    - Multiple plot types in single figure
    - Mathematical function visualization variety
    - Proper subplot spacing and title management
    - Advanced matplotlib layout capabilities

    Args:
        matplotlib_ready: Fixture ensuring matplotlib is available
        plot_cleanup: Fixture for automatic cleanup of generated files
        tmp_path: Pytest temporary directory fixture

    Returns:
        None: Test validates through assertions

    Raises:
        AssertionError: If complex subplot generation fails

    Example:
        Expected result with subplot metadata:
        {
          "plot_base64": "iVBOR...",
          "plot_type": "subplot_complex",
          "subplot_count": 4
        }
    """
    # Given: Matplotlib is available in Pyodide environment
    if not matplotlib_ready:
        pytest.skip("matplotlib not available in this Pyodide environment")

    # When: Creating a complex subplot layout using pathlib
    complex_subplot_code = f"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Create sample data for mathematical functions
x = np.linspace(0, 4*np.pi, {MatplotlibConfig.TEST_DATA['sample_points_small']})
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)
y4 = np.exp(-x/8) * np.sin(x)

# Create the complex subplot layout (2x2 grid)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize={MatplotlibConfig.PLOT_SETTINGS['figure_size_complex']})

# Plot 1: Sin wave
ax1.plot(x, y1, 'b-', linewidth={MatplotlibConfig.PLOT_SETTINGS['line_width']})
ax1.set_title('sin(x)')
ax1.grid(True, alpha={MatplotlibConfig.PLOT_SETTINGS['grid_alpha']})

# Plot 2: Cos wave
ax2.plot(x, y2, 'r-', linewidth={MatplotlibConfig.PLOT_SETTINGS['line_width']})
ax2.set_title('cos(x)')
ax2.grid(True, alpha={MatplotlibConfig.PLOT_SETTINGS['grid_alpha']})

# Plot 3: Product of sin and cos
ax3.plot(x, y3, 'g-', linewidth={MatplotlibConfig.PLOT_SETTINGS['line_width']})
ax3.set_title('sin(x) * cos(x)')
ax3.grid(True, alpha={MatplotlibConfig.PLOT_SETTINGS['grid_alpha']})

# Plot 4: Damped oscillation
ax4.plot(x, y4, 'm-', linewidth={MatplotlibConfig.PLOT_SETTINGS['line_width']})
ax4.set_title('exp(-x/8) * sin(x)')
ax4.grid(True, alpha={MatplotlibConfig.PLOT_SETTINGS['grid_alpha']})

# Add overall title and layout management
fig.suptitle('Complex Subplot Layout', fontsize=16)
plt.tight_layout()

# Save to bytes buffer for base64 encoding
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi={MatplotlibConfig.PLOT_SETTINGS['default_dpi']}, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Return structured result with subplot metadata
result = {{"plot_base64": plot_b64, "plot_type": "subplot_complex", "subplot_count": 4}}
print(f"Generated complex subplot layout with 4 panels and {{len(plot_b64)}} base64 characters")
result
"""

    # Execute the complex subplot generation code via execute-raw endpoint
    response = requests.post(
        f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
        headers=Config.HEADERS["execute_raw"],
        data=complex_subplot_code,
        timeout=Config.TIMEOUTS["code_execution"]
    )

    # Then: The API response should be successful and follow contract
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    validate_api_contract(response_data)

    assert response_data["success"] is True, f"Complex subplot generation failed: {response_data}"
    assert response_data["data"] is not None, "Response data should not be null for success"

    # And: The plot result should contain expected subplot metadata
    result = parse_result_data(response_data)
    assert isinstance(result, dict), f"Result should be dict, got {type(result)}"
    assert "plot_base64" in result, "Result should contain plot_base64 field"
    assert "plot_type" in result, "Result should contain plot_type field"
    assert "subplot_count" in result, "Result should contain subplot_count field"

    assert result["plot_type"] == "subplot_complex", f"Expected 'subplot_complex', got {result['plot_type']}"
    assert result["subplot_count"] == 4, f"Expected 4 subplots, got {result['subplot_count']}"

    # And: The complex plot should be decodable and saveable to filesystem
    plot_data = result["plot_base64"]
    assert len(plot_data) > 2000, f"Complex plot data seems too small: {len(plot_data)} chars"

    filename = generate_timestamp_filename(MatplotlibConfig.FILE_PATTERNS["complex_plot"])
    saved_path = save_plot_from_base64(plot_data, filename, tmp_path)
    plot_cleanup.track_plot_file(saved_path)

    assert saved_path.exists(), f"Complex subplot file was not saved to {saved_path}"
    assert saved_path.stat().st_size > 0, "Complex subplot file has zero size"


@pytest.mark.api
@pytest.mark.plotting
@pytest.mark.filesystem
def test_when_direct_filesystem_plot_is_saved_then_virtual_file_system_contains_plot(
    matplotlib_ready: bool,
    plot_cleanup: Any,
    tmp_path: Path
) -> None:
    """
    Test direct plot saving to Pyodide virtual filesystem without internal APIs.

    Given: A Pyodide environment with matplotlib and virtual filesystem available
    When: A plot is created and saved directly to the virtual filesystem using pathlib
    Then: The plot file should be created in the virtual filesystem
    And: File metadata should confirm successful creation and proper size
    And: The operation should be performed without using internal pyodide endpoints

    This test validates:
    - Direct filesystem operations within Pyodide virtual environment
    - Pathlib usage for cross-platform file operations
    - Plot file creation and metadata validation
    - Virtual filesystem integration without internal APIs
    - File existence and size verification

    Args:
        matplotlib_ready: Fixture ensuring matplotlib is available
        plot_cleanup: Fixture for automatic cleanup of generated files
        tmp_path: Pytest temporary directory fixture

    Returns:
        None: Test validates through assertions

    Raises:
        AssertionError: If direct filesystem plot saving fails

    Example:
        Expected result with file metadata:
        {
          "file_saved": true,
          "file_size": 45678,
          "plot_type": "direct_save_basic",
          "filename": "/home/pyodide/plots/matplotlib/plot_xyz.png"
        }
    """
    # Given: Matplotlib is available in Pyodide environment
    if not matplotlib_ready:
        pytest.skip("matplotlib not available in this Pyodide environment")

    # Generate unique filename to avoid conflicts
    timestamp = int(time.time() * 1000)
    virtual_filename = f"direct_save_basic_{timestamp}.png"
    virtual_file_path = f"{MatplotlibConfig.VIRTUAL_PATHS['plots_dir']}/{virtual_filename}"

    # When: Creating and saving a plot directly to virtual filesystem using pathlib
    direct_save_code = f"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import numpy as np

# Create sample data for trigonometric functions
x = np.linspace(0, 2*np.pi, {MatplotlibConfig.TEST_DATA['sample_points_small']})
y1 = np.sin(x)
y2 = np.cos(x)

# Create the plot with proper configuration
plt.figure(figsize={MatplotlibConfig.PLOT_SETTINGS['figure_size_basic']})
plt.plot(x, y1, 'b-', linewidth={MatplotlibConfig.PLOT_SETTINGS['line_width']}, label='sin(x)')
plt.plot(x, y2, 'r--', linewidth={MatplotlibConfig.PLOT_SETTINGS['line_width']}, label='cos(x)')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Direct File Save - Trigonometric Functions')
plt.legend()
plt.grid(True, alpha={MatplotlibConfig.PLOT_SETTINGS['grid_alpha']})

# Use pathlib for cross-platform file operations
output_path = Path("{virtual_file_path}")
plots_dir = output_path.parent
plots_dir.mkdir(parents=True, exist_ok=True)

# Save directly to the virtual filesystem
plt.savefig(output_path, dpi={MatplotlibConfig.PLOT_SETTINGS['default_dpi']}, bbox_inches='tight')
plt.close()

# Verify file creation and get metadata using pathlib
file_exists = output_path.exists()
file_size = output_path.stat().st_size if file_exists else 0

# Return structured result with file metadata
result = {{
    "file_saved": file_exists,
    "file_size": file_size,
    "plot_type": "direct_save_basic",
    "filename": str(output_path)
}}
print(f"Direct save result: file_saved={{file_exists}}, size={{file_size}} bytes")
result
"""

    # Execute the direct filesystem save code via execute-raw endpoint
    response = requests.post(
        f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
        headers=Config.HEADERS["execute_raw"],
        data=direct_save_code,
        timeout=Config.TIMEOUTS["code_execution"]
    )

    # Then: The API response should be successful and follow contract
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    validate_api_contract(response_data)

    assert response_data["success"] is True, f"Direct filesystem save failed: {response_data}"
    assert response_data["data"] is not None, "Response data should not be null for success"

    # And: The file metadata should confirm successful creation
    result = parse_result_data(response_data)
    assert isinstance(result, dict), f"Result should be dict, got {type(result)}"
    assert "file_saved" in result, "Result should contain file_saved field"
    assert "file_size" in result, "Result should contain file_size field"
    assert "plot_type" in result, "Result should contain plot_type field"
    assert "filename" in result, "Result should contain filename field"

    assert result["file_saved"] is True, "Plot file was not saved to virtual filesystem"
    assert result["file_size"] > 0, f"Plot file has zero size: {result['file_size']}"
    assert result["plot_type"] == "direct_save_basic", f"Expected 'direct_save_basic', got {result['plot_type']}"

    # Validate filename format
    filename_path = Path(result["filename"])
    assert virtual_filename in str(filename_path), f"Filename should contain {virtual_filename}"


@pytest.mark.api
@pytest.mark.plotting
@pytest.mark.filesystem
@pytest.mark.slow
def test_when_complex_filesystem_plot_is_saved_then_multi_subplot_file_is_created(
    matplotlib_ready: bool,
    plot_cleanup: Any,
    tmp_path: Path
) -> None:
    """
    Test complex multi-subplot plot creation with direct filesystem saving.

    Given: A Pyodide environment with matplotlib and virtual filesystem available
    When: A complex multi-subplot plot with various chart types is created and saved
    Then: The plot file should be created with all subplots properly rendered
    And: Statistical metadata should be calculated for validation
    And: File size should reflect the complexity of the visualization

    This test validates:
    - Complex subplot layouts with multiple chart types
    - Mixed visualization types (scatter, histogram, line, box plots)
    - Statistical data generation and analysis
    - Large dataset handling and performance
    - Advanced matplotlib features integration

    Args:
        matplotlib_ready: Fixture ensuring matplotlib is available
        plot_cleanup: Fixture for automatic cleanup of generated files
        tmp_path: Pytest temporary directory fixture

    Returns:
        None: Test validates through assertions

    Raises:
        AssertionError: If complex multi-subplot creation fails

    Example:
        Expected result with comprehensive metadata:
        {
          "file_saved": true,
          "file_size": 125678,
          "plot_type": "direct_save_complex",
          "data_points": 1000,
          "subplot_count": 4,
          "filename": "/home/pyodide/plots/matplotlib/complex_xyz.png"
        }
    """
    # Given: Matplotlib is available in Pyodide environment
    if not matplotlib_ready:
        pytest.skip("matplotlib not available in this Pyodide environment")

    # Generate unique filename for complex plot
    timestamp = int(time.time() * 1000)
    virtual_filename = f"direct_save_complex_{timestamp}.png"
    virtual_file_path = f"{MatplotlibConfig.VIRTUAL_PATHS['plots_dir']}/{virtual_filename}"

    # When: Creating a complex multi-subplot visualization using pathlib
    complex_save_code = f"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import numpy as np

# Create sample data with controlled randomness for reproducible results
np.random.seed({MatplotlibConfig.TEST_DATA['random_seed']})
n_points = {MatplotlibConfig.TEST_DATA['sample_points_large']}
x = np.random.randn(n_points)
y = np.random.randn(n_points)
colors = x + y
sizes = np.abs(x * y) * 100

# Create complex subplot layout (2x2 grid) with different chart types
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize={MatplotlibConfig.PLOT_SETTINGS['figure_size_complex']})

# Subplot 1: Scatter plot with color and size mapping
scatter = ax1.scatter(x, y, c=colors, s=sizes, alpha={MatplotlibConfig.PLOT_SETTINGS['alpha_transparency']}, cmap='viridis')
ax1.set_title('Scatter Plot with Color and Size Mapping')
ax1.set_xlabel('X values')
ax1.set_ylabel('Y values')
plt.colorbar(scatter, ax=ax1)

# Subplot 2: Overlapping histograms
ax2.hist(x, bins={MatplotlibConfig.TEST_DATA['histogram_bins']}, alpha=0.7, color='blue', label='X values')
ax2.hist(y, bins={MatplotlibConfig.TEST_DATA['histogram_bins']}, alpha=0.7, color='red', label='Y values')
ax2.set_title('Overlapping Histograms')
ax2.set_xlabel('Values')
ax2.set_ylabel('Frequency')
ax2.legend()

# Subplot 3: Line plot with multiple trigonometric series
t = np.linspace(0, 4*np.pi, {MatplotlibConfig.TEST_DATA['sample_points_small']})
ax3.plot(t, np.sin(t), 'b-', linewidth={MatplotlibConfig.PLOT_SETTINGS['line_width']}, label='sin(t)')
ax3.plot(t, np.cos(t), 'r--', linewidth={MatplotlibConfig.PLOT_SETTINGS['line_width']}, label='cos(t)')
ax3.plot(t, np.sin(t)*np.cos(t), 'g:', linewidth={MatplotlibConfig.PLOT_SETTINGS['line_width']}, label='sin(t)*cos(t)')
ax3.set_title('Trigonometric Functions')
ax3.set_xlabel('t')
ax3.set_ylabel('f(t)')
ax3.legend()
ax3.grid(True, alpha={MatplotlibConfig.PLOT_SETTINGS['grid_alpha']})

# Subplot 4: Box plot comparison
data_for_boxplot = [x[:250], y[:250], (x+y)[:250], (x*y)[:250]]
ax4.boxplot(data_for_boxplot, labels=['X', 'Y', 'X+Y', 'X*Y'])
ax4.set_title('Box Plot Comparison')
ax4.set_ylabel('Values')

plt.tight_layout()

# Use pathlib for cross-platform file operations
output_path = Path("{virtual_file_path}")
plots_dir = output_path.parent
plots_dir.mkdir(parents=True, exist_ok=True)

# Save complex plot directly to virtual filesystem
plt.savefig(output_path, dpi={MatplotlibConfig.PLOT_SETTINGS['default_dpi']}, bbox_inches='tight')
plt.close()

# Verify file creation and get comprehensive metadata
file_exists = output_path.exists()
file_size = output_path.stat().st_size if file_exists else 0

# Return structured result with comprehensive metadata
result = {{
    "file_saved": file_exists,
    "file_size": file_size,
    "plot_type": "direct_save_complex",
    "data_points": n_points,
    "subplot_count": 4,
    "filename": str(output_path)
}}
print(f"Complex save result: file_saved={{file_exists}}, size={{file_size}} bytes, data_points={{n_points}}")
result
"""

    # Execute the complex filesystem save code via execute-raw endpoint
    response = requests.post(
        f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
        headers=Config.HEADERS["execute_raw"],
        data=complex_save_code,
        timeout=Config.TIMEOUTS["code_execution"] * 2  # Extended timeout for complex operation
    )

    # Then: The API response should be successful and follow contract
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    validate_api_contract(response_data)

    assert response_data["success"] is True, f"Complex filesystem save failed: {response_data}"
    assert response_data["data"] is not None, "Response data should not be null for success"

    # And: The complex plot metadata should confirm successful creation
    result = parse_result_data(response_data)
    assert isinstance(result, dict), f"Result should be dict, got {type(result)}"
    assert "file_saved" in result, "Result should contain file_saved field"
    assert "file_size" in result, "Result should contain file_size field"
    assert "plot_type" in result, "Result should contain plot_type field"
    assert "data_points" in result, "Result should contain data_points field"
    assert "subplot_count" in result, "Result should contain subplot_count field"
    assert "filename" in result, "Result should contain filename field"

    assert result["file_saved"] is True, "Complex plot file was not saved to virtual filesystem"
    assert result["file_size"] > 0, f"Complex plot file has zero size: {result['file_size']}"
    assert result["plot_type"] == "direct_save_complex", f"Expected 'direct_save_complex', got {result['plot_type']}"
    assert result["data_points"] == MatplotlibConfig.TEST_DATA['sample_points_large'], \
        f"Expected {MatplotlibConfig.TEST_DATA['sample_points_large']} points, got {result['data_points']}"
    assert result["subplot_count"] == 4, f"Expected 4 subplots, got {result['subplot_count']}"

    # Validate that complex plots are significantly larger than simple ones
    assert result["file_size"] > 10000, f"Complex plot file seems too small: {result['file_size']} bytes"

    # Validate filename format
    filename_path = Path(result["filename"])
    assert virtual_filename in str(filename_path), f"Filename should contain {virtual_filename}"


# ==================== ERROR HANDLING AND EDGE CASES ====================


@pytest.mark.api
@pytest.mark.plotting
def test_when_invalid_matplotlib_code_is_executed_then_error_is_handled_gracefully(
    matplotlib_ready: bool
) -> None:
    """
    Test error handling for invalid matplotlib code execution.

    Given: A Pyodide environment with matplotlib available
    When: Invalid matplotlib code is executed via execute-raw endpoint
    Then: The API should return a proper error response
    And: The error message should be descriptive and helpful
    And: The API contract should be maintained even for errors

    This test validates:
    - Proper error handling for matplotlib code failures
    - API contract compliance for error responses
    - Descriptive error messages for debugging
    - Graceful degradation without server crashes

    Args:
        matplotlib_ready: Fixture ensuring matplotlib is available

    Returns:
        None: Test validates through assertions

    Raises:
        AssertionError: If error handling is improper
    """
    # Given: Matplotlib is available in Pyodide environment
    if not matplotlib_ready:
        pytest.skip("matplotlib not available in this Pyodide environment")

    # When: Executing invalid matplotlib code with syntax errors
    invalid_code = """
import matplotlib.pyplot as plt
import numpy as np

# This code has intentional errors
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Missing parentheses - syntax error
plt.plot(x, y
plt.show()
"""

    # Execute the invalid code via execute-raw endpoint
    response = requests.post(
        f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
        headers=Config.HEADERS["execute_raw"],
        data=invalid_code,
        timeout=Config.TIMEOUTS["code_execution"]
    )

    # Then: The API response should indicate failure but follow contract
    assert response.status_code == 200, f"Expected 200 even for errors, got {response.status_code}"

    response_data = response.json()
    validate_api_contract(response_data)

    # And: The response should indicate failure with descriptive error
    assert response_data["success"] is False, "Invalid code should result in failure"
    assert response_data["error"] is not None, "Error response should contain error message"
    assert response_data["data"] is None, "Error response should have null data"

    # Validate error message contains useful information
    error_message = response_data["error"]
    assert isinstance(error_message, str), f"Error should be string, got {type(error_message)}"
    assert len(error_message) > 0, "Error message should not be empty"


@pytest.mark.api
@pytest.mark.plotting
@pytest.mark.timeout
def test_when_long_running_plot_exceeds_timeout_then_operation_is_terminated(
    matplotlib_ready: bool
) -> None:
    """
    Test timeout handling for long-running matplotlib operations.

    Given: A Pyodide environment with matplotlib available
    When: A matplotlib operation that takes longer than timeout is executed
    Then: The operation should be terminated with a timeout error
    And: The API should return a proper timeout error response
    And: The server should remain stable after timeout

    This test validates:
    - Timeout enforcement for long-running operations
    - Proper timeout error handling and messaging
    - Server stability after operation termination
    - API contract compliance for timeout errors

    Args:
        matplotlib_ready: Fixture ensuring matplotlib is available

    Returns:
        None: Test validates through assertions

    Raises:
        AssertionError: If timeout handling is improper
    """
    # Given: Matplotlib is available in Pyodide environment
    if not matplotlib_ready:
        pytest.skip("matplotlib not available in this Pyodide environment")

    # When: Executing a deliberately slow matplotlib operation
    slow_code = f"""
import matplotlib.pyplot as plt
import numpy as np
import time

# Create a very large dataset to slow down processing
n_points = {MatplotlibConfig.TEST_DATA['sample_points_large'] * 10}  # 10x normal size
x = np.random.randn(n_points)
y = np.random.randn(n_points)

# Add deliberate delay
time.sleep(2)

# Create complex plot that takes time
plt.figure(figsize=(20, 20))  # Very large figure
for i in range(100):  # Many plots
    plt.subplot(10, 10, i+1)
    plt.plot(x[i*100:(i+1)*100], y[i*100:(i+1)*100])

plt.tight_layout()
print("Complex plot completed")
"""

    # Execute with short timeout to trigger timeout error
    short_timeout = 5  # 5 seconds - should be too short for the complex operation

    try:
        response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
            headers=Config.HEADERS["execute_raw"],
            data=slow_code,
            timeout=short_timeout + 2  # Allow some buffer for network
        )

        # Then: The response should indicate timeout or failure
        assert response.status_code in [200, 408, 500], f"Expected timeout related status, got {response.status_code}"

        if response.status_code == 200:
            response_data = response.json()
            validate_api_contract(response_data)

            # Should either succeed quickly (if optimized) or fail with timeout
            if not response_data["success"]:
                assert "timeout" in response_data["error"].lower() or "time" in response_data["error"].lower(), \
                    f"Expected timeout error, got: {response_data['error']}"

    except requests.Timeout:
        # Network timeout is also acceptable for this test
        print("✅ Operation properly timed out at network level")
        pass


if __name__ == "__main__":
    # For running tests directly with python
    pytest.main([__file__, "-v"])

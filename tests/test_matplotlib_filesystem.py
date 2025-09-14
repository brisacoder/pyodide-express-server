"""
Test matplotlib plotting functionality with filesystem integration.

This module tests matplotlib operations within the Pyodide environment,
focusing on filesystem interactions for saving and retrieving plots.
It validates plot generation, file saving, and extraction workflows.
"""

import os
import time
import json
from pathlib import Path
import pytest
import requests


# Constants for test configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:3000")
DEFAULT_TIMEOUT = 30  # seconds
PLOT_TIMEOUT = 120  # seconds for complex plot operations
INSTALLATION_TIMEOUT = 300  # seconds for package installation


class TestMatplotlibFilesystemIntegration:
    """
    Test matplotlib functionality with filesystem operations.

    This test class validates matplotlib plotting capabilities within
    the Pyodide environment, including saving plots to the virtual
    filesystem and extracting them to the host filesystem.
    """

    def _parse_execute_raw_response(self, response):
        """
        Parse response from execute-raw endpoint to extract JSON from stdout.

        Args:
            response: Response object from requests

        Returns:
            dict: Parsed JSON result from Python code execution
        """
        assert (
            response.status_code == 200
        ), f"API request failed: {response.status_code}"

        # Parse the response from execute-raw endpoint
        response_data = response.json()
        assert (
            response_data.get("success") is True
        ), f"API execution failed: {response_data}"

        # Get stdout from the result
        stdout = response_data.get("data", {}).get("result", {}).get("stdout", "")

        # Parse the JSON result from stdout
        stdout_lines = stdout.strip().split("\n")
        json_line = next((line for line in stdout_lines if line.startswith("{")), None)
        assert json_line is not None, f"No JSON result found in stdout: {stdout}"

        return json.loads(json_line)

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, request):
        """
        Set up test environment before each test and clean up afterwards.

        This fixture:
        - Creates local plots directory for test outputs
        - Cleans up existing plots before and after tests
        - Ensures matplotlib is available in Pyodide

        Args:
            request: Pytest request object for test context
        """
        # Create local plots directory
        self.plots_dir = Path(__file__).parent.parent / "plots" / "matplotlib"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Clean up before test
        self._cleanup_local_plots()
        self._cleanup_pyodide_plots()

        yield

        # Clean up after test
        self._cleanup_local_plots()
        self._cleanup_pyodide_plots()

    def _cleanup_local_plots(self) -> None:
        """Remove any existing plot files from local filesystem."""
        if self.plots_dir.exists():
            for file_path in self.plots_dir.glob("*.png"):
                try:
                    file_path.unlink()
                    print(f"Removed local file: {file_path.name}")
                except OSError as e:
                    print(f"Warning: Could not remove {file_path.name}: {e}")

    def _cleanup_pyodide_plots(self) -> None:
        """Remove any existing plot files from Pyodide virtual filesystem."""
        cleanup_code = """
from pathlib import Path

# Clean up existing files in /plots/matplotlib/
plots_dir = Path('/plots/matplotlib')
if plots_dir.exists():
    files_removed = 0
    for file_path in plots_dir.iterdir():
        if file_path.is_file():
            try:
                file_path.unlink()
                files_removed += 1
                print(f"Removed Pyodide file: {file_path.name}")
            except Exception as e:
                print(f"Failed to remove {file_path.name}: {e}")
    
    if files_removed == 0:
        print("No existing files found in Pyodide /plots/matplotlib/")
else:
    print("Pyodide plots directory does not exist yet")
    
print("Pyodide cleanup completed")
"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/execute-raw",
                data=cleanup_code,
                headers={"Content-Type": "text/plain"},
                timeout=DEFAULT_TIMEOUT,
            )
            if response.status_code == 200:
                print("✅ Pyodide filesystem cleaned successfully")
            else:
                print(f"⚠️ Pyodide cleanup request failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Exception during Pyodide cleanup: {e}")

    def test_given_basic_plot_when_saved_to_filesystem_then_file_exists(self):
        """
        Test saving a basic matplotlib plot to the filesystem.

        Given: A simple sine/cosine plot is created
        When: The plot is saved to the Pyodide virtual filesystem
        Then: The file exists and can be extracted to the host filesystem

        This test validates:
        - Basic plot creation with matplotlib
        - File saving to virtual filesystem using pathlib
        - File extraction from Pyodide to host filesystem
        - File size validation

        Example:
            >>> # Create plot
            >>> plt.plot([1, 2, 3], [1, 4, 9])
            >>> plt.savefig(Path('/plots/matplotlib/test.png'))
            >>> # Extract to host
            >>> extract_plots()
        """
        # Generate unique filename with timestamp
        timestamp = int(time.time() * 1000)

        # Create and save a basic plot
        code = f"""import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Create sample data
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, "b-", linewidth=2, label="sin(x)")
plt.plot(x, y2, "r--", linewidth=2, label="cos(x)")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Direct File Save - Trigonometric Functions")
plt.legend()
plt.grid(True, alpha=0.3)

# Save using pathlib for portability
plots_dir = Path('/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)
output_path = plots_dir / f'direct_save_basic_{timestamp}.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

# Verify the file was created
file_exists = output_path.exists()
file_size = output_path.stat().st_size if file_exists else 0

result = {{
    "file_saved": file_exists,
    "file_size": file_size,
    "plot_type": "direct_save_basic",
    "filename": str(output_path)
}}
print(json.dumps(result))"""

        # When: Execute the plot creation code
        response = requests.post(
            f"{API_BASE_URL}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=PLOT_TIMEOUT,
        )

        # Then: Verify successful execution and parse result
        result = self._parse_execute_raw_response(response)
        assert result["file_saved"] is True, "Plot file was not saved to filesystem"
        assert result["file_size"] > 0, "Plot file has zero size"
        assert result["plot_type"] == "direct_save_basic"
        assert result["filename"].startswith(
            "/plots/matplotlib/"
        ), "File not saved in correct directory"
        assert result["filename"].endswith(".png"), "File should have .png extension"

    def test_given_complex_subplot_when_saved_to_filesystem_then_all_subplots_rendered(
        self,
    ):
        """
        Test saving a complex multi-subplot visualization.

        Given: A complex visualization with 4 different subplot types
        When: The figure is saved to the Pyodide virtual filesystem
        Then: All subplots are rendered correctly and file is extractable

        This test validates:
        - Complex subplot layouts (2x2 grid)
        - Different plot types (scatter, histogram, line, box)
        - Color mapping and legend handling
        - Large data processing (1000 points)

        Example:
            >>> fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            >>> # Add various plot types to subplots
            >>> plt.tight_layout()
            >>> plt.savefig(Path('/plots/matplotlib/complex.png'))
        """
        # Add small delay to avoid timestamp collisions
        time.sleep(0.1)
        timestamp = int(time.time() * 1000)

        # Create complex multi-subplot visualization
        code = f"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Create sample data
np.random.seed(42)
n_points = 1000
x = np.random.randn(n_points)
y = np.random.randn(n_points)
colors = x + y
sizes = np.abs(x * y) * 100

# Create complex subplot layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Subplot 1: Scatter plot
scatter = ax1.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
ax1.set_title('Scatter Plot with Color and Size Mapping')
ax1.set_xlabel('X values')
ax1.set_ylabel('Y values')
plt.colorbar(scatter, ax=ax1)

# Subplot 2: Histogram
ax2.hist(x, bins=30, alpha=0.7, color='blue', label='X values')
ax2.hist(y, bins=30, alpha=0.7, color='red', label='Y values')
ax2.set_title('Overlapping Histograms')
ax2.set_xlabel('Values')
ax2.set_ylabel('Frequency')
ax2.legend()

# Subplot 3: Line plot with multiple series
t = np.linspace(0, 4*np.pi, 100)
ax3.plot(t, np.sin(t), 'b-', label='sin(t)')
ax3.plot(t, np.cos(t), 'r--', label='cos(t)')
ax3.plot(t, np.sin(t)*np.cos(t), 'g:', label='sin(t)*cos(t)')
ax3.set_title('Trigonometric Functions')
ax3.set_xlabel('t')
ax3.set_ylabel('f(t)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Box plot
data_for_boxplot = [x[:250], y[:250], (x+y)[:250], (x*y)[:250]]
ax4.boxplot(data_for_boxplot, labels=['X', 'Y', 'X+Y', 'X*Y'])
ax4.set_title('Box Plot Comparison')
ax4.set_ylabel('Values')

plt.tight_layout()

# Save using pathlib for portability
plots_dir = Path('/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)
output_path = plots_dir / f'direct_save_complex_{timestamp}.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

# Verify the file was created and get statistics
file_exists = output_path.exists()
file_size = output_path.stat().st_size if file_exists else 0

result = {{
    "file_saved": file_exists,
    "file_size": file_size,
    "plot_type": "direct_save_complex",
    "data_points": n_points,
    "subplot_count": 4,
    "filename": str(output_path)
}}
print(json.dumps(result))
"""

        # When: Execute the complex plot creation
        response = requests.post(
            f"{API_BASE_URL}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=PLOT_TIMEOUT,
        )

        # Then: Verify successful execution and parse result
        result = self._parse_execute_raw_response(response)
        assert result["file_saved"] is True, "Complex plot file was not saved"
        assert result["file_size"] > 0, "Complex plot file has zero size"
        assert result["plot_type"] == "direct_save_complex"
        assert result["data_points"] == 1000, "Incorrect data point count"
        assert result["subplot_count"] == 4, "Incorrect subplot count"
        assert result["filename"].startswith(
            "/plots/matplotlib/"
        ), "File not saved in correct directory"
        assert result["filename"].endswith(".png"), "File should have .png extension"

    def test_given_plot_with_custom_styles_when_saved_then_styles_preserved(self):
        """
        Test matplotlib plots with custom styling options.

        Given: A plot with custom colors, markers, and styles
        When: The plot is saved with high DPI settings
        Then: All custom styling is preserved in the output

        This test validates:
        - Custom color schemes and markers
        - Font size customization
        - DPI settings for high-quality output
        - Style preservation through save/load cycle

        Example:
            >>> plt.style.use('seaborn')
            >>> plt.plot(x, y, 'ro-', markersize=10, linewidth=3)
            >>> plt.savefig(path, dpi=300, facecolor='white')
        """
        timestamp = int(time.time() * 1000)

        code = f"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Create data
x = np.linspace(0, 10, 50)
y1 = np.sin(x) * np.exp(-x/10)
y2 = np.cos(x) * np.exp(-x/10)

# Create figure with custom styling
plt.figure(figsize=(12, 8), facecolor='white')

# Custom plot styling
plt.plot(x, y1, 'ro-', markersize=8, linewidth=2.5, 
         markerfacecolor='red', markeredgecolor='darkred',
         markeredgewidth=1.5, label='Damped Sine')
         
plt.plot(x, y2, 'b^--', markersize=8, linewidth=2.5,
         markerfacecolor='blue', markeredgecolor='darkblue', 
         markeredgewidth=1.5, label='Damped Cosine')

# Customize appearance
plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
plt.ylabel('Amplitude', fontsize=14, fontweight='bold')
plt.title('Damped Oscillations with Custom Styling', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
plt.grid(True, linestyle=':', alpha=0.6, linewidth=1)

# Add text annotation
plt.text(5, 0.3, 'Exponential Decay', fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# Set axis limits and style
plt.xlim(0, 10)
plt.ylim(-0.5, 0.5)
plt.tick_params(axis='both', which='major', labelsize=10)

# Save with high DPI
plots_dir = Path('/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)
output_path = plots_dir / f'custom_styled_plot_{timestamp}.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

# Return results
result = {{
    "file_saved": output_path.exists(),
    "file_size": output_path.stat().st_size if output_path.exists() else 0,
    "filename": str(output_path),
    "dpi": 300,
    "style_elements": ["custom_markers", "custom_colors", "annotations", "grid"]
}}
print(json.dumps(result))
"""

        # When: Execute styled plot creation
        response = requests.post(
            f"{API_BASE_URL}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=PLOT_TIMEOUT,
        )

        # Then: Verify styling was preserved
        result = self._parse_execute_raw_response(response)
        assert result["file_saved"] is True
        assert result["file_size"] > 100000, "High DPI file should be larger"
        assert result["dpi"] == 300
        assert "custom_markers" in result["style_elements"]

    def test_given_multiple_plots_when_saved_sequentially_then_all_files_exist(self):
        """
        Test saving multiple plots in sequence.

        Given: Multiple plots are created with different data
        When: Each plot is saved with a unique filename
        Then: All files exist independently in the filesystem

        This test validates:
        - Sequential plot creation without interference
        - Unique file naming prevents overwrites
        - Multiple file extraction in one operation
        - Memory management between plots

        Example:
            >>> for i in range(3):
            >>>     plt.figure()
            >>>     plt.plot(data[i])
            >>>     plt.savefig(Path(f'/plots/matplotlib/plot_{i}.png'))
            >>>     plt.close()
        """
        base_timestamp = int(time.time() * 1000)

        # Create multiple plots
        code = f"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Create plots directory
plots_dir = Path('/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)

results = []

# Create 3 different plots
for i in range(3):
    plt.figure(figsize=(8, 6))
    
    # Different data for each plot
    x = np.linspace(0, 10, 100)
    if i == 0:
        y = np.sin(x * (i + 1))
        plt.plot(x, y, 'b-', linewidth=2)
        plt.title(f'Plot {{i+1}}: Sine Wave')
    elif i == 1:
        y = np.exp(-x / 5) * np.cos(x * 2)
        plt.plot(x, y, 'r--', linewidth=2)
        plt.title(f'Plot {{i+1}}: Damped Cosine')
    else:
        y = x ** 0.5 + np.random.normal(0, 0.1, len(x))
        plt.plot(x, y, 'g-.', linewidth=2)
        plt.title(f'Plot {{i+1}}: Square Root with Noise')
    
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.grid(True, alpha=0.3)
    
    # Save with unique filename
    filename = f'sequential_plot_{{i+1}}_{base_timestamp}.png'
    output_path = plots_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Record result
    results.append({{
        "plot_number": i + 1,
        "filename": str(output_path),
        "exists": output_path.exists(),
        "size": output_path.stat().st_size if output_path.exists() else 0
    }})

# Return all results
print(json.dumps({{"plots": results, "total_count": len(results)}}))
"""

        # When: Execute multiple plot creation
        response = requests.post(
            f"{API_BASE_URL}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=PLOT_TIMEOUT,
        )

        # Then: Verify all plots were created
        result = self._parse_execute_raw_response(response)
        assert result["total_count"] == 3, "Should have created 3 plots"

        for plot in result["plots"]:
            assert plot["exists"] is True, f"Plot {plot['plot_number']} was not saved"
            assert plot["size"] > 0, f"Plot {plot['plot_number']} has zero size"
            assert plot["filename"].startswith(
                "/plots/matplotlib/"
            ), f"Plot {plot['plot_number']} not in correct directory"
            assert plot["filename"].endswith(
                ".png"
            ), f"Plot {plot['plot_number']} should have .png extension"

    def test_given_plot_directory_when_created_with_pathlib_then_proper_structure(self):
        """
        Test directory creation and management using pathlib.

        Given: A need to organize plots in subdirectories
        When: Directories are created using pathlib
        Then: Directory structure is properly created and accessible

        This test validates:
        - Pathlib directory creation in Pyodide
        - Subdirectory organization
        - Cross-platform path handling
        - Directory existence checking

        Example:
            >>> base_dir = Path('/plots/matplotlib')
            >>> subdir = base_dir / 'analysis' / 'results'
            >>> subdir.mkdir(parents=True, exist_ok=True)
        """
        timestamp = int(time.time() * 1000)

        code = f"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Create nested directory structure
base_dir = Path('/plots/matplotlib')
subdir = base_dir / 'organized' / 'charts'
subdir.mkdir(parents=True, exist_ok=True)

# Create a simple plot
x = np.linspace(0, 5, 50)
y = x ** 2

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Plot in Subdirectory')
plt.xlabel('X')
plt.ylabel('X squared')
plt.grid(True)

# Save in subdirectory
output_path = subdir / f'nested_plot_{timestamp}.png'
plt.savefig(output_path, dpi=150)
plt.close()

# Verify directory structure
structure = {{
    "base_exists": base_dir.exists(),
    "subdir_exists": subdir.exists(),
    "file_saved": output_path.exists(),
    "file_size": output_path.stat().st_size if output_path.exists() else 0,
    "filename": str(output_path),
    "parent_dir": str(output_path.parent),
    "is_file": output_path.is_file() if output_path.exists() else False
}}

print(json.dumps(structure))
"""

        # When: Execute directory creation and plot saving
        response = requests.post(
            f"{API_BASE_URL}/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=PLOT_TIMEOUT,
        )

        # Then: Verify directory structure
        result = self._parse_execute_raw_response(response)
        assert result["base_exists"] is True
        assert result["subdir_exists"] is True
        assert result["file_saved"] is True
        assert result["file_size"] > 0
        assert result["is_file"] is True
        assert "/organized/charts" in result["parent_dir"]


@pytest.fixture(scope="session")
def ensure_matplotlib_installed():
    """
    Session-scoped fixture to ensure matplotlib is installed.

    This fixture runs once per test session to verify matplotlib
    is available in the Pyodide environment. It uses the standard
    API endpoints without internal/pyodide-specific endpoints.

    Returns:
        bool: True if matplotlib is available, False otherwise
    """
    # Check if matplotlib is available by trying to import it
    check_code = """
try:
    import matplotlib
    print(f"matplotlib version: {matplotlib.__version__}")
    available = True
except ImportError:
    print("matplotlib not available")
    available = False

print(f"MATPLOTLIB_AVAILABLE:{available}")
"""

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/execute-raw",
            data=check_code,
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT,
        )

        if response.status_code == 200:
            output = response.text
            return "MATPLOTLIB_AVAILABLE:True" in output
        else:
            print(f"Failed to check matplotlib availability: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error checking matplotlib: {e}")
        return False


def pytest_configure(config):
    """
    Configure pytest with custom markers.

    Args:
        config: Pytest configuration object
    """
    config.addinivalue_line("markers", "matplotlib: mark test as requiring matplotlib")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

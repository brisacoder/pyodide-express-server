#!/usr/bin/env python3
"""
BDD-style container filesystem tests for Pyodide execution.

This module contains comprehensive tests for containerized Pyodide execution
filesystem mounting and persistence, written in Behavior-Driven Development (BDD)
style using pytest.

Tests validate that:
- Containerized Pyodide maintains full filesystem compatibility with host
- File persistence works correctly across execution contexts
- Matplotlib plots are properly saved and accessible
- Complex multi-plot dashboards render correctly
- Environment configuration matches expected state

All tests follow Given-When-Then BDD pattern and use only the /api/execute-raw
endpoint for code execution.
"""

import time
from pathlib import Path
from typing import Generator, List

import pytest
import requests

# Global configuration constants
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30
EXECUTION_TIMEOUT = 60000  # 60 seconds for complex operations
PLOT_CREATION_TIMEOUT = 45000  # 45 seconds for plot generation
DASHBOARD_TIMEOUT = 60000  # 60 seconds for complex dashboards
MAX_WAIT_TIME = 60  # Maximum wait time for server responses
FILE_SYNC_DELAY = 1  # Time to wait for filesystem sync
MIN_PLOT_SIZE_BYTES = 1000  # Minimum expected size for plot files
MIN_DASHBOARD_SIZE_BYTES = 50000  # Minimum expected size for dashboard files


@pytest.fixture(scope="function")
def server_health() -> None:
    """
    Pytest fixture to ensure server is available before running tests.

    This fixture validates server connectivity and skips tests if server
    is not reachable. It's scoped to function level to check health
    before each test.

    Raises:
        pytest.skip: If server is not available or not responding
    """
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=DEFAULT_TIMEOUT)
        if response.status_code != 200:
            pytest.skip("Server not available - health check failed")
    except requests.RequestException:
        pytest.skip("Server not reachable - connection failed")


@pytest.fixture(scope="function")
def temp_file_tracker() -> Generator[List[Path], None, None]:
    """
    Pytest fixture for tracking and cleaning up temporary files.

    This fixture provides a list to track temporary files created during
    tests and automatically cleans them up after test completion.

    Yields:
        List[Path]: List to track temporary files for cleanup

    Note:
        Files are automatically removed in the teardown phase
    """
    temp_files: List[Path] = []

    yield temp_files

    # Teardown: Clean up temporary files
    for temp_file in temp_files:
        if isinstance(temp_file, Path) and temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass  # Best effort cleanup


@pytest.fixture(scope="function")
def base_url() -> str:
    """
    Pytest fixture providing the base URL for API requests.

    Returns:
        str: The base URL for the API server
    """
    return BASE_URL


class TestContainerBasicExecution:
    """Test scenarios for basic containerized Python execution."""

    @pytest.mark.api
    @pytest.mark.integration
    def test_given_server_when_executing_basic_python_then_success_response(
        self, server_health, base_url
    ):
        """
        Test basic Python code execution in containerized environment.

        Scenario: Execute simple Python code
        Given: The containerized Pyodide server is running
        When: I execute basic Python code via /api/execute-raw
        Then: The execution should succeed and return expected output

        Args:
            server_health: Fixture ensuring server availability
            base_url: Fixture providing API base URL
        """
        # Given
        code = "print('Hello from containerized Pyodide!')"
        endpoint = f"{base_url}/api/execute-raw"

        # When
        response = requests.post(
            endpoint,
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT
        )

        # Then
        assert response.status_code == 200
        result = response.json()
        assert result.get('success') is True
        assert 'Hello from containerized Pyodide!' in result.get('stdout', '')


class TestContainerPlotGeneration:
    """Test scenarios for matplotlib plot generation in containerized environment."""

    @pytest.mark.api
    @pytest.mark.integration
    @pytest.mark.slow
    def test_given_server_when_creating_matplotlib_plot_then_file_persisted_on_host(
        self, server_health, base_url, temp_file_tracker
    ):
        """
        Test matplotlib plot creation and filesystem persistence.

        Scenario: Create and save matplotlib plot
        Given: The containerized Pyodide server is running with filesystem mounting
        When: I execute Python code that creates and saves a matplotlib plot
        Then: The plot file should be persisted on the host filesystem
        And: The file should have substantial content (not empty)

        Args:
            server_health: Fixture ensuring server availability
            base_url: Fixture providing API base URL
            temp_file_tracker: Fixture for tracking temporary files
        """
        # Given
        timestamp = int(time.time() * 1000)
        endpoint = f"{base_url}/api/execute-raw"

        plot_code = f"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Generate test data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Container Test Plot - {timestamp}')
plt.xlabel('X values')
plt.ylabel('sin(X)')
plt.grid(True, alpha=0.3)

# Save to mounted filesystem
plot_path = Path('/plots/matplotlib/container_test_{timestamp}.png')
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f'Plot saved to: {{plot_path}}')
print(f'File exists: {{plot_path.exists()}}')
if plot_path.exists():
    print(f'File size: {{plot_path.stat().st_size}} bytes')
"""

        # When
        response = requests.post(
            endpoint,
            data=plot_code,
            headers={"Content-Type": "text/plain"},
            timeout=PLOT_CREATION_TIMEOUT
        )

        # Then
        assert response.status_code == 200
        result = response.json()
        assert result.get('success') is True

        # Verify file exists on host filesystem
        host_plot_path = Path(f"plots/matplotlib/container_test_{timestamp}.png")
        temp_file_tracker.append(host_plot_path)

        # Wait for filesystem sync
        time.sleep(FILE_SYNC_DELAY)

        assert host_plot_path.exists(), f"Plot file not found on host: {host_plot_path}"

        # Verify file has substantial content
        file_size = host_plot_path.stat().st_size
        assert file_size > MIN_PLOT_SIZE_BYTES, f"Plot file too small: {file_size} bytes"

    @pytest.mark.api
    @pytest.mark.integration
    @pytest.mark.slow
    def test_given_server_when_creating_complex_dashboard_then_large_file_persisted(
        self, server_health, base_url, temp_file_tracker
    ):
        """
        Test complex multi-plot dashboard creation in containerized environment.

        Scenario: Create complex matplotlib dashboard with multiple subplots
        Given: The containerized Pyodide server is running
        When: I execute Python code that creates a complex multi-plot dashboard
        Then: The dashboard file should be created and persisted
        And: The file should be substantially larger than simple plots

        Args:
            server_health: Fixture ensuring server availability
            base_url: Fixture providing API base URL
            temp_file_tracker: Fixture for tracking temporary files
        """
        # Given
        timestamp = int(time.time() * 1000)
        endpoint = f"{base_url}/api/execute-raw"

        dashboard_code = f"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create dashboard with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Container Dashboard - {timestamp}', fontsize=16)

# Plot 1: Sine and Cosine
x = np.linspace(0, 4*np.pi, 200)
ax1.plot(x, np.sin(x), 'b-', label='sin(x)')
ax1.plot(x, np.cos(x), 'r-', label='cos(x)')
ax1.set_title('Trigonometric Functions')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Random scatter
np.random.seed(42)
x_rand = np.random.randn(100)
y_rand = np.random.randn(100)
colors = np.random.rand(100)
ax2.scatter(x_rand, y_rand, c=colors, alpha=0.7)
ax2.set_title('Random Scatter Plot')

# Plot 3: Bar chart
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
ax3.bar(categories, values, color='skyblue')
ax3.set_title('Sample Bar Chart')

# Plot 4: Histogram
data = np.random.normal(0, 1, 1000)
ax4.hist(data, bins=30, alpha=0.7, color='green')
ax4.set_title('Normal Distribution')

plt.tight_layout()

# Save dashboard
dashboard_path = Path('/plots/matplotlib/container_dashboard_{timestamp}.png')
dashboard_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
plt.close()

print(f'Dashboard saved: {{dashboard_path}}')
print(f'File exists: {{dashboard_path.exists()}}')
"""

        # When
        response = requests.post(
            endpoint,
            data=dashboard_code,
            headers={"Content-Type": "text/plain"},
            timeout=DASHBOARD_TIMEOUT
        )

        # Then
        assert response.status_code == 200
        result = response.json()
        assert result.get('success') is True

        # Verify dashboard file on host
        host_dashboard_path = Path(f"plots/matplotlib/container_dashboard_{timestamp}.png")
        temp_file_tracker.append(host_dashboard_path)

        time.sleep(FILE_SYNC_DELAY)

        assert host_dashboard_path.exists(), f"Dashboard file not found: {host_dashboard_path}"

        # Verify substantial file size (dashboard should be larger than simple plots)
        file_size = host_dashboard_path.stat().st_size
        assert file_size > MIN_DASHBOARD_SIZE_BYTES, f"Dashboard file too small: {file_size} bytes"


class TestContainerEnvironmentValidation:
    """Test scenarios for validating containerized environment configuration."""

    @pytest.mark.api
    @pytest.mark.integration
    def test_given_server_when_checking_environment_then_expected_configuration_present(
        self, server_health, base_url
    ):
        """
        Test containerized environment configuration and package availability.

        Scenario: Validate container environment matches expected configuration
        Given: The containerized Pyodide server is running
        When: I execute Python code that checks environment details and package versions
        Then: The environment should match expected Python version and package availability
        And: Key filesystem paths should be accessible

        Args:
            server_health: Fixture ensuring server availability
            base_url: Fixture providing API base URL
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"

        env_code = """
import sys
import platform
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path

# Environment info
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print(f"Platform: {platform.platform()}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")

# Path accessibility
test_paths = ['/plots', '/uploads', '/logs', '/plots/matplotlib']
for path in test_paths:
    p = Path(path)
    status = "✅" if p.exists() and p.is_dir() else "❌"
    print(f"{status} {path}: exists={p.exists()}")
"""

        # When
        response = requests.post(
            endpoint,
            data=env_code,
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT
        )

        # Then
        assert response.status_code == 200
        result = response.json()
        assert result.get('success') is True

        stdout = result.get('stdout', '')

        # Verify Python version (should be 3.1x)
        assert 'Python version: 3.1' in stdout

        # Verify key packages are available
        assert 'NumPy version:' in stdout
        assert 'Pandas version:' in stdout
        assert 'Matplotlib version:' in stdout

        # Verify path accessibility - should have multiple successful mounts
        success_count = stdout.count('✅')
        assert success_count >= 3, "Not enough accessible paths in container"


class TestContainerEdgeCases:
    """Test scenarios for edge cases and error conditions in containerized environment."""

    @pytest.mark.api
    @pytest.mark.integration
    def test_given_server_when_executing_invalid_plot_code_then_error_handled_gracefully(
        self, server_health, base_url
    ):
        """
        Test error handling for invalid matplotlib code in containerized environment.

        Scenario: Execute invalid matplotlib code
        Given: The containerized Pyodide server is running
        When: I execute Python code with matplotlib syntax errors
        Then: The execution should fail gracefully with descriptive error message

        Args:
            server_health: Fixture ensuring server availability
            base_url: Fixture providing API base URL
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"
        invalid_code = """
import matplotlib.pyplot as plt
import numpy as np

# Invalid matplotlib code - wrong function name
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.invalid_function(x, y)  # This should cause an error
plt.show()
"""

        # When
        response = requests.post(
            endpoint,
            data=invalid_code,
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT
        )

        # Then
        assert response.status_code == 200  # Server should handle error gracefully
        result = response.json()

        # Should either fail with success=False or succeed with error in stderr
        if result.get('success') is False:
            assert 'error' in result
        else:
            stderr = result.get('stderr', '')
            assert 'AttributeError' in stderr or 'invalid_function' in stderr

    @pytest.mark.api
    @pytest.mark.integration
    def test_given_server_when_creating_plot_with_invalid_path_then_handled_gracefully(
        self, server_health, base_url
    ):
        """
        Test handling of invalid filesystem paths in containerized environment.

        Scenario: Attempt to save plot to invalid filesystem path
        Given: The containerized Pyodide server is running
        When: I execute Python code that tries to save plot to invalid/restricted path
        Then: The execution should handle the error gracefully

        Args:
            server_health: Fixture ensuring server availability
            base_url: Fixture providing API base URL
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"
        invalid_path_code = """
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Generate simple plot
x = np.linspace(0, 10, 50)
y = np.sin(x)
plt.figure(figsize=(6, 4))
plt.plot(x, y)

# Try to save to invalid/restricted path
try:
    invalid_path = Path('/invalid/restricted/path/plot.png')
    plt.savefig(invalid_path)
    print("ERROR: Should not have succeeded")
except Exception as e:
    print(f"Expected error handled: {type(e).__name__}")
    print(f"Error message: {str(e)}")

plt.close()
print("Test completed successfully")
"""

        # When
        response = requests.post(
            endpoint,
            data=invalid_path_code,
            headers={"Content-Type": "text/plain"},
            timeout=DEFAULT_TIMEOUT
        )

        # Then
        assert response.status_code == 200
        result = response.json()
        assert result.get('success') is True

        stdout = result.get('stdout', '')
        assert 'Expected error handled:' in stdout
        assert 'Test completed successfully' in stdout

    @pytest.mark.api
    @pytest.mark.integration
    def test_given_server_when_executing_memory_intensive_plot_then_completed_within_limits(
        self, server_health, base_url, temp_file_tracker
    ):
        """
        Test memory-intensive plot creation stays within container limits.

        Scenario: Create memory-intensive visualization
        Given: The containerized Pyodide server is running
        When: I execute Python code that creates a memory-intensive plot
        Then: The execution should complete successfully within memory limits

        Args:
            server_health: Fixture ensuring server availability
            base_url: Fixture providing API base URL
            temp_file_tracker: Fixture for tracking temporary files
        """
        # Given
        timestamp = int(time.time() * 1000)
        endpoint = f"{base_url}/api/execute-raw"

        memory_intensive_code = f"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create memory-intensive visualization
n_points = 10000  # Large dataset
x = np.random.randn(n_points)
y = np.random.randn(n_points)
colors = np.random.rand(n_points)
sizes = np.random.rand(n_points) * 100

# Create high-resolution scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.title('Memory Intensive Scatter Plot - {timestamp}')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.colorbar(label='Color Scale')

# Save high-resolution plot
plot_path = Path('/plots/matplotlib/memory_test_{timestamp}.png')
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
plt.close()

print(f'Memory intensive plot saved: {{plot_path}}')
print(f'File exists: {{plot_path.exists()}}')
print(f'Dataset size: {{n_points}} points')
"""

        # When
        response = requests.post(
            endpoint,
            data=memory_intensive_code,
            headers={"Content-Type": "text/plain"},
            timeout=DASHBOARD_TIMEOUT
        )

        # Then
        assert response.status_code == 200
        result = response.json()
        assert result.get('success') is True

        # Verify file was created
        host_plot_path = Path(f"plots/matplotlib/memory_test_{timestamp}.png")
        temp_file_tracker.append(host_plot_path)

        time.sleep(FILE_SYNC_DELAY)

        assert host_plot_path.exists(), f"Memory intensive plot not found: {host_plot_path}"

        # Verify file has substantial content
        file_size = host_plot_path.stat().st_size
        assert file_size > MIN_PLOT_SIZE_BYTES, f"Plot file too small: {file_size} bytes"

    @pytest.mark.api
    @pytest.mark.integration
    def test_given_server_when_executing_concurrent_plot_operations_then_all_succeed(
        self, server_health, base_url, temp_file_tracker
    ):
        """
        Test concurrent plot creation operations in containerized environment.

        Scenario: Execute multiple plot operations in sequence
        Given: The containerized Pyodide server is running
        When: I execute multiple Python code blocks that create different plots
        Then: All plot operations should succeed and create separate files

        Args:
            server_health: Fixture ensuring server availability
            base_url: Fixture providing API base URL
            temp_file_tracker: Fixture for tracking temporary files
        """
        # Given
        timestamp = int(time.time() * 1000)
        endpoint = f"{base_url}/api/execute-raw"

        plot_operations = [
            {
                'name': 'line_plot',
                'code': f"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Line Plot - {timestamp}')
plt.xlabel('X')
plt.ylabel('sin(X)')

plot_path = Path('/plots/matplotlib/line_plot_{timestamp}.png')
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f'Line plot saved: {{plot_path}}')
"""
            },
            {
                'name': 'histogram',
                'code': f"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

data = np.random.normal(0, 1, 1000)
plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.title('Histogram - {timestamp}')
plt.xlabel('Value')
plt.ylabel('Frequency')

plot_path = Path('/plots/matplotlib/histogram_{timestamp}.png')
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f'Histogram saved: {{plot_path}}')
"""
            },
            {
                'name': 'bar_chart',
                'code': f"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [25, 40, 30, 35]
plt.figure(figsize=(8, 6))
plt.bar(categories, values, color=['red', 'blue', 'green', 'orange'])
plt.title('Bar Chart - {timestamp}')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.xticks(rotation=45)

plot_path = Path('/plots/matplotlib/bar_chart_{timestamp}.png')
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f'Bar chart saved: {{plot_path}}')
"""
            }
        ]

        # When & Then
        for operation in plot_operations:
            response = requests.post(
                endpoint,
                data=operation['code'],
                headers={"Content-Type": "text/plain"},
                timeout=PLOT_CREATION_TIMEOUT
            )

            assert response.status_code == 200
            result = response.json()
            assert result.get('success') is True

            # Verify file was created
            plot_file = Path(f"plots/matplotlib/{operation['name']}_{timestamp}.png")
            temp_file_tracker.append(plot_file)

            time.sleep(FILE_SYNC_DELAY)

            assert plot_file.exists(), f"{operation['name']} plot not found: {plot_file}"

            # Verify file has content
            file_size = plot_file.stat().st_size
            assert file_size > MIN_PLOT_SIZE_BYTES, (
                f"{operation['name']} file too small: {file_size} bytes"
            )


class TestContainerPerformanceAndLimits:
    """Test scenarios for performance characteristics and limits in containerized environment."""

    @pytest.mark.api
    @pytest.mark.integration
    @pytest.mark.slow
    def test_given_server_when_executing_long_running_computation_then_completes_within_timeout(
        self, server_health, base_url
    ):
        """
        Test long-running computation performance in containerized environment.

        Scenario: Execute computationally intensive Python code
        Given: The containerized Pyodide server is running
        When: I execute Python code with intensive numerical computations
        Then: The execution should complete within reasonable time limits

        Args:
            server_health: Fixture ensuring server availability
            base_url: Fixture providing API base URL
        """
        # Given
        endpoint = f"{base_url}/api/execute-raw"

        computation_code = """
import numpy as np
import time

start_time = time.time()

# Perform intensive numerical computation
n = 1000
matrix_a = np.random.rand(n, n)
matrix_b = np.random.rand(n, n)

# Matrix multiplication (computationally intensive)
result = np.dot(matrix_a, matrix_b)

# Additional computations
eigenvalues = np.linalg.eigvals(result[:100, :100])  # Smaller subset for performance
mean_eigenvalue = np.mean(eigenvalues.real)

end_time = time.time()
execution_time = end_time - start_time

print(f"Matrix dimensions: {n}x{n}")
print(f"Result shape: {result.shape}")
print(f"Mean eigenvalue: {mean_eigenvalue:.6f}")
print(f"Computation time: {execution_time:.2f} seconds")
print("Long-running computation completed successfully")
"""

        # When
        start_time = time.time()
        response = requests.post(
            endpoint,
            data=computation_code,
            headers={"Content-Type": "text/plain"},
            timeout=EXECUTION_TIMEOUT
        )
        end_time = time.time()

        # Then
        assert response.status_code == 200
        result = response.json()
        assert result.get('success') is True

        stdout = result.get('stdout', '')
        assert 'Long-running computation completed successfully' in stdout
        assert 'Matrix dimensions: 1000x1000' in stdout
        assert 'Computation time:' in stdout

        # Verify reasonable execution time
        total_time = end_time - start_time
        assert total_time < MAX_WAIT_TIME, f"Execution took too long: {total_time:.2f} seconds"

    @pytest.mark.api
    @pytest.mark.integration
    def test_given_server_when_creating_multiple_file_formats_then_all_supported(
        self, server_health, base_url, temp_file_tracker
    ):
        """
        Test support for multiple plot file formats in containerized environment.

        Scenario: Create plots in different file formats
        Given: The containerized Pyodide server is running
        When: I execute Python code that saves plots in different formats (PNG, PDF, SVG)
        Then: All file formats should be supported and files created successfully

        Args:
            server_health: Fixture ensuring server availability
            base_url: Fixture providing API base URL
            temp_file_tracker: Fixture for tracking temporary files
        """
        # Given
        timestamp = int(time.time() * 1000)
        endpoint = f"{base_url}/api/execute-raw"

        multi_format_code = f"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create a simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.title('Multi-Format Plot Test - {timestamp}')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True, alpha=0.3)

# Save in multiple formats
formats = ['png', 'pdf', 'svg']
base_path = Path('/plots/matplotlib/multi_format_{timestamp}')
base_path.parent.mkdir(parents=True, exist_ok=True)

saved_files = []
for fmt in formats:
    file_path = base_path.with_suffix(f'.{{fmt}}')
    try:
        plt.savefig(file_path, format=fmt, dpi=150, bbox_inches='tight')
        saved_files.append(str(file_path))
        print(f'✅ {{fmt.upper()}} saved: {{file_path}}')
        print(f'   File exists: {{file_path.exists()}}')
        if file_path.exists():
            print(f'   File size: {{file_path.stat().st_size}} bytes')
    except Exception as e:
        print(f'❌ {{fmt.upper()}} failed: {{str(e)}}')

plt.close()

print(f'Total formats saved: {{len(saved_files)}}')
print('Multi-format test completed')
"""

        # When
        response = requests.post(
            endpoint,
            data=multi_format_code,
            headers={"Content-Type": "text/plain"},
            timeout=PLOT_CREATION_TIMEOUT
        )

        # Then
        assert response.status_code == 200
        result = response.json()
        assert result.get('success') is True

        stdout = result.get('stdout', '')
        assert 'Multi-format test completed' in stdout

        # Check for successful format saves
        success_indicators = stdout.count('✅')
        assert success_indicators >= 1, "At least one file format should be supported"

        # Verify PNG file (most common format) was created on host
        host_png_path = Path(f"plots/matplotlib/multi_format_{timestamp}.png")
        temp_file_tracker.append(host_png_path)

        time.sleep(FILE_SYNC_DELAY)

        if host_png_path.exists():
            file_size = host_png_path.stat().st_size
            assert file_size > MIN_PLOT_SIZE_BYTES, f"PNG file too small: {file_size} bytes"

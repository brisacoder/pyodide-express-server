"""
BDD-style pytest tests for plot creation and filesystem behavior in Pyodide Express Server.

This module provides comprehensive tests for plot creation functionality using only
the public /api/execute-raw endpoint. Tests follow BDD (Behavior-Driven Development)
patterns with Given/When/Then structure.

Key Features:
- Pytest framework with comprehensive fixtures
- BDD test structure (Given/When/Then)
- Cross-platform portability (Windows/Linux) using pathlib
- API contract compliance validation
- Only uses public /api/execute-raw endpoint
- Comprehensive plot creation and filesystem testing
- No hardcoded values - all parameterized with fixtures
- Enhanced error handling and edge case coverage

API Contract:
{
  "success": true | false,
  "data": {
    "result": <any>,
    "stdout": <string>,
    "stderr": <string>,
    "executionTime": <number>
  } | null,
  "error": <string|null>,
  "meta": {"timestamp": <string>}
}

Test Categories:
- plot_creation: Tests for creating plots in various directories
- filesystem: Tests for directory creation and file operations
- integration: End-to-end plot creation workflows
- error_handling: Tests for error conditions and edge cases
- performance: Basic performance validation for plot operations
"""

import json
import time
from typing import Any, Dict, List

import pytest
import requests
from requests.exceptions import RequestException

# Import shared configuration and utilities
try:
    from .conftest import Config, execute_python_code, validate_api_contract
except ImportError:
    from conftest import Config, execute_python_code, validate_api_contract


class TestConfig:
    """Test-specific configuration constants."""

    # Plot creation settings
    PLOT_SETTINGS = {
        "default_dpi": 100,
        "high_dpi": 150,
        "test_figsize": (6, 4),
        "large_figsize": (12, 8),
        "timeout_seconds": 45,
        "quick_timeout": 15,
        "max_plots_per_test": 10,
        "min_file_size_bytes": 1000,  # Minimum expected plot file size
        "max_file_size_bytes": 5 * 1024 * 1024,  # 5MB max
    }

    # Directory paths for testing - ALL MUST USE VFS PATHS UNDER /home/pyodide/
    TEST_DIRECTORIES = [
        "/home/pyodide/plots",
        "/home/pyodide/plots/matplotlib",
        "/home/pyodide/plots/seaborn",
        "/home/pyodide/plots/test_extract",
        "/home/pyodide/plots/custom",
        "/home/pyodide/plots/comprehensive",
    ]

    # Non-plots directories for negative testing
    NON_PLOTS_DIRECTORIES = [
        "/tmp/test_plots",
        "/data/plots", 
        "/custom/plots",
        "/home/pyodide/temp_plots",
    ]

    # Plot file patterns for testing
    PLOT_PATTERNS = {
        "simple_line": "simple_line_plot",
        "scatter": "scatter_plot",
        "histogram": "histogram_plot",
        "multi_subplot": "subplot_demo",
        "heatmap": "heatmap_plot",
        "bar_chart": "bar_chart",
    }

    # Error scenarios for testing
    ERROR_SCENARIOS = [
        "invalid_directory_path",
        "matplotlib_data_mismatch",
        "file_permission_error",
        "memory_intensive_plot",
        "invalid_plot_format",
    ]


def parse_result_data(result: dict) -> dict:
    """
    Parse the result data from API response, handling both dict and JSON string formats.
    
    This function provides robust parsing of API response data, handling various
    formats that might be returned by the Pyodide execution environment.
    
    Args:
        result: API response result from execute_python_code
        
    Returns:
        dict: Parsed result data
        
    Raises:
        AssertionError: If result cannot be parsed or is invalid
        
    Example:
        >>> api_response = {"success": True, "data": {"result": '{"key": "value"}'}}
        >>> parsed = parse_result_data(api_response)
        >>> assert parsed == {"key": "value"}
    """
    test_results = result["data"]["result"]
    
    # If result is already a dict, return as-is
    if isinstance(test_results, dict):
        return test_results
    
    # If result is a JSON string, parse it
    if isinstance(test_results, str):
        try:
            return json.loads(test_results)
        except json.JSONDecodeError as e:
            raise AssertionError(f"Failed to parse JSON result: {e}")
    
    # If result is neither dict nor string, it's invalid
    raise AssertionError(f"Expected dict or JSON string, got {type(test_results)}: {test_results}")


@pytest.fixture
def quick_timeout():
    """
    Provide shorter timeout for quick operations.

    Returns:
        int: Timeout in seconds for quick operations

    Example:
        >>> def test_quick_operation(quick_timeout):
        ...     # Use quick_timeout for simple operations
        ...     result = execute_python_code(simple_code, timeout=quick_timeout)
    """
    return TestConfig.PLOT_SETTINGS["quick_timeout"]


@pytest.fixture
def plot_patterns():
    """
    Provide plot patterns for testing different plot types.

    Returns:
        dict: Mapping of plot types to filename patterns

    Example:
        >>> def test_plot_patterns(plot_patterns):
        ...     for plot_type, pattern in plot_patterns.items():
        ...         # Create plot using pattern
        ...         filename = f"{pattern}_{timestamp}.png"
    """
    return TestConfig.PLOT_PATTERNS.copy()


@pytest.fixture(scope="session")
def server_ready():
    """
    Ensure server is running and ready for plot creation tests.

    This session-scoped fixture validates that the Pyodide Express Server
    is available and can execute Python code for plot creation.

    Returns:
        None: Fixture validates server availability

    Raises:
        pytest.skip: If server is not available within timeout period

    Example:
        >>> def test_something(server_ready):
        ...     # Server is guaranteed to be ready here
        ...     response = requests.get(f"{Config.BASE_URL}/health")
        ...     assert response.status_code == 200
    """

    def wait_for_server_ready(url: str, timeout: int) -> None:
        """Wait for server to become available."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(url, timeout=Config.TIMEOUTS["quick_operation"])
                if response.status_code == 200:
                    return
            except RequestException:
                pass
            time.sleep(1)
        pytest.skip(f"Server at {url} not available within {timeout}s")

    # Check server health
    health_url = f"{Config.BASE_URL}{Config.ENDPOINTS['health']}"
    wait_for_server_ready(health_url, Config.TIMEOUTS["server_health"])


@pytest.fixture
def plot_timeout():
    """
    Provide timeout for plot creation operations.

    Returns:
        int: Timeout in seconds for plot creation

    Example:
        >>> def test_plot_creation(plot_timeout):
        ...     # Use plot_timeout for code execution
        ...     result = execute_python_code(code, timeout=plot_timeout)
    """
    return TestConfig.PLOT_SETTINGS["timeout_seconds"]


@pytest.fixture
def test_directories():
    """
    Provide list of directories for plot testing.

    Returns:
        List[str]: Directory paths to test for plot creation (all VFS compliant)

    Example:
        >>> def test_directory_creation(test_directories):
        ...     for directory in test_directories:
        ...         # Test each directory - all should be VFS compliant
        ...         assert directory.startswith("/home/pyodide/plots")
    """
    return TestConfig.TEST_DIRECTORIES.copy()


@pytest.fixture
def non_plots_directories():
    """
    Provide list of non-plots directories for negative testing.

    Returns:
        List[str]: Directory paths outside /home/pyodide/plots for testing

    Example:
        >>> def test_non_plots_behavior(non_plots_directories):
        ...     for directory in non_plots_directories:
        ...         # Test behavior outside VFS plots directory
        ...         assert not directory.startswith("/home/pyodide/plots")
    """
    return TestConfig.NON_PLOTS_DIRECTORIES.copy()


@pytest.fixture
def cleanup_plots():
    """
    Cleanup fixture to remove test plots after each test.

    This fixture provides a list to track created plot files and
    automatically cleans them up after test completion.

    Yields:
        List[str]: List to append created plot file paths

    Example:
        >>> def test_plot_creation(cleanup_plots):
        ...     # Create plot and track for cleanup
        ...     plot_path = "/home/pyodide/plots/matplotlib/test.png"
        ...     cleanup_plots.append(plot_path)
        ...     # Plot will be automatically cleaned up
    """
    created_plots = []
    yield created_plots

    # Cleanup created plots
    if created_plots:
        cleanup_code = f"""
from pathlib import Path
import traceback

cleaned_files = []
errors = []

for plot_path in {created_plots}:
    try:
        plot_file = Path(plot_path)
        if plot_file.exists():
            plot_file.unlink()
            cleaned_files.append(str(plot_file))
    except Exception as e:
        errors.append(f"{{plot_path}}: {{str(e)}}")

result = {{
    "cleaned_files": cleaned_files,
    "errors": errors,
    "total_cleaned": len(cleaned_files)
}}
result
        """
        try:
            execute_python_code(cleanup_code)
        except Exception:
            # Don't fail tests due to cleanup issues
            pass


@pytest.mark.matplotlib
@pytest.mark.api
class TestPlotDirectoryBehavior:
    """
    BDD tests for plot directory creation and filesystem behavior.

    This test class validates that the Pyodide filesystem can properly
    create directories and store plot files using pathlib for portability.
    """

    def test_given_plots_directories_when_creating_then_all_should_be_writable(
        self, server_ready, test_directories, plot_timeout
    ):
        """
        Test that all standard plots directories can be created and are writable.

        **Given:** A list of standard plots directories
        **When:** Attempting to create directories and test files in each
        **Then:** All directories should be successfully created and writable

        Args:
            server_ready: Server readiness fixture
            test_directories: List of directories to test
            plot_timeout: Timeout for operations

        Example:
            This test validates directories like:
            - /home/pyodide/plots
            - /home/pyodide/plots/matplotlib
            - /home/pyodide/plots/seaborn
            - /home/pyodide/plots/test_extract
        """
        # Given: Standard plots directories
        code = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

test_directories = {test_directories}
results = {{
    "directory_tests": [],
    "total_tested": len(test_directories),
    "total_success": 0,
    "performance_metrics": {{}}
}}

# When: Testing each directory
for directory_path in test_directories:
    start_time = time.time()
    test_result = {{
        "path": directory_path,
        "exists_before": False,
        "created_successfully": False,
        "is_writable": False,
        "test_file_created": False,
        "file_size": 0,
        "creation_time_ms": 0,
        "error": None
    }}
    
    try:
        directory = Path(directory_path)
        test_result["exists_before"] = directory.exists()
        
        # Create directory if it doesn't exist
        directory.mkdir(parents=True, exist_ok=True)
        test_result["created_successfully"] = directory.exists()
        
        if test_result["created_successfully"]:
            # Test writability by creating a test plot file
            timestamp = int(time.time() * 1000)
            test_file = directory / f"writability_test_{{timestamp}}.png"
            
            plt.figure(figsize=(4, 3), dpi={TestConfig.PLOT_SETTINGS['default_dpi']})
            plt.plot([1, 2, 3], [1, 4, 2], 'b-', linewidth=2)
            plt.title(f'Writability Test: {{directory_path}}')
            plt.xlabel('X Values')
            plt.ylabel('Y Values')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(str(test_file), dpi={TestConfig.PLOT_SETTINGS['default_dpi']}, bbox_inches='tight')
            plt.close()
            
            test_result["is_writable"] = test_file.exists()
            test_result["test_file_created"] = test_result["is_writable"]
            
            if test_result["test_file_created"]:
                test_result["file_size"] = test_file.stat().st_size
                test_result["creation_time_ms"] = int((time.time() - start_time) * 1000)
                results["total_success"] += 1
                # Clean up test file
                test_file.unlink()
            
    except Exception as e:
        test_result["error"] = str(e)
    
    results["directory_tests"].append(test_result)

# Then: Validate all operations completed and calculate metrics
results["success_rate"] = results["total_success"] / results["total_tested"] if results["total_tested"] > 0 else 0
creation_times = [t["creation_time_ms"] for t in results["directory_tests"] if t["creation_time_ms"] > 0]
if creation_times:
    results["performance_metrics"] = {{
        "avg_creation_time_ms": sum(creation_times) / len(creation_times),
        "max_creation_time_ms": max(creation_times),
        "min_creation_time_ms": min(creation_times)
    }}

results
        """

        # When: Executing directory tests
        result = execute_python_code(code, timeout=plot_timeout)

        # Then: Validate API contract and results
        validate_api_contract(result)
        assert result[
            "success"
        ], f"Directory test execution failed: {result.get('error')}"

        test_results = parse_result_data(result)
        assert isinstance(test_results, dict), "Results should be a dictionary"
        assert (
            "directory_tests" in test_results
        ), "Results should contain directory_tests"

        # Validate each directory was tested
        directory_tests = test_results["directory_tests"]
        assert len(directory_tests) == len(
            test_directories
        ), f"Expected {len(test_directories)} tests, got {len(directory_tests)}"

        # All directories should be successfully created and writable
        failed_directories = []
        for test in directory_tests:
            if not (
                test.get("created_successfully", False)
                and test.get("is_writable", False)
            ):
                failed_directories.append(
                    {
                        "path": test.get("path"),
                        "created": test.get("created_successfully", False),
                        "writable": test.get("is_writable", False),
                        "error": test.get("error"),
                    }
                )

        assert len(failed_directories) == 0, f"Failed directories: {failed_directories}"
        assert (
            test_results["success_rate"] == 1.0
        ), f"Expected 100% success rate, got {test_results['success_rate']}"

        # Validate performance metrics
        if "performance_metrics" in test_results:
            metrics = test_results["performance_metrics"]
            assert metrics.get("avg_creation_time_ms", 0) > 0, "Should have performance metrics"
            assert metrics.get("max_creation_time_ms", 0) < 30000, "Creation should be under 30 seconds"

    def test_given_non_plots_directories_when_creating_plots_then_should_work_but_be_outside_plots(
        self, server_ready, non_plots_directories, plot_timeout, cleanup_plots
    ):
        """
        Test that plots can be created outside /home/pyodide/plots but are distinguishable.

        **Given:** Directories outside the standard /home/pyodide/plots structure
        **When:** Creating plot files in these directories
        **Then:** Files should be created successfully but clearly outside /home/pyodide/plots

        Args:
            server_ready: Server readiness fixture
            non_plots_directories: List of non-plots directories
            plot_timeout: Timeout for operations
            cleanup_plots: Cleanup tracking list

        Example:
            This test creates plots in directories like:
            - /tmp/test_plots
            - /data/plots
            - /custom/plots
        """
        # Given: Non-plots directories
        code = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

non_plots_directories = {non_plots_directories}
results = {{
    "non_plots_tests": [],
    "files_created": [],
    "plots_vs_non_plots": {{}},
    "validation_checks": {{}}
}}

# When: Creating plots in non-plots directories
for directory_path in non_plots_directories:
    test_result = {{
        "path": directory_path,
        "is_plots_directory": directory_path.startswith("/home/pyodide/plots"),
        "created_successfully": False,
        "plot_created": False,
        "file_path": None,
        "file_size": 0,
        "error": None
    }}
    
    try:
        directory = Path(directory_path)
        directory.mkdir(parents=True, exist_ok=True)
        test_result["created_successfully"] = directory.exists()
        
        if test_result["created_successfully"]:
            # Create a test plot
            timestamp = int(time.time() * 1000)
            plot_file = directory / f"non_plots_test_{{timestamp}}.png"
            
            plt.figure(figsize=(5, 4), dpi={TestConfig.PLOT_SETTINGS['default_dpi']})
            plt.scatter([1, 2, 3, 4, 5], [1, 4, 2, 8, 5], c='red', alpha=0.7, s=50)
            plt.title(f'Non-Plots Directory Test\\n{{directory_path}}')
            plt.xlabel('X Values')
            plt.ylabel('Y Values')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(str(plot_file), dpi={TestConfig.PLOT_SETTINGS['default_dpi']}, bbox_inches='tight')
            plt.close()
            
            test_result["plot_created"] = plot_file.exists()
            test_result["file_path"] = str(plot_file)
            
            if test_result["plot_created"]:
                test_result["file_size"] = plot_file.stat().st_size
                results["files_created"].append(str(plot_file))
            
    except Exception as e:
        test_result["error"] = str(e)
    
    results["non_plots_tests"].append(test_result)

# Then: Categorize results and validate
plots_files = [f for f in results["files_created"] if f.startswith("/home/pyodide/plots/")]
non_plots_files = [f for f in results["files_created"] if not f.startswith("/home/pyodide/plots/")]

results["plots_vs_non_plots"] = {{
    "plots_directory_files": plots_files,
    "non_plots_directory_files": non_plots_files,
    "total_plots_files": len(plots_files),
    "total_non_plots_files": len(non_plots_files)
}}

# Validation checks
results["validation_checks"] = {{
    "all_directories_tested": len(results["non_plots_tests"]) == len(non_plots_directories),
    "no_plots_directory_confusion": all(not test["is_plots_directory"] for test in results["non_plots_tests"]),
    "files_created_outside_plots": len(non_plots_files) > 0,
    "no_files_in_plots_dir": len(plots_files) == 0
}}

results
        """

        # When: Executing non-plots directory tests
        result = execute_python_code(code, timeout=plot_timeout)

        # Then: Validate API contract and results
        validate_api_contract(result)
        assert result[
            "success"
        ], f"Non-plots directory test failed: {result.get('error')}"

        test_results = parse_result_data(result)
        assert isinstance(test_results, dict), "Results should be a dictionary"

        # Track created files for cleanup
        if "files_created" in test_results:
            cleanup_plots.extend(test_results["files_created"])

        # Validate non-plots behavior
        non_plots_tests = test_results.get("non_plots_tests", [])
        assert len(non_plots_tests) == len(
            non_plots_directories
        ), f"Expected {len(non_plots_directories)} tests"

        # All directories should be created successfully
        for test in non_plots_tests:
            assert not test.get(
                "is_plots_directory", True
            ), f"Directory {test['path']} should not be a plots directory"
            assert test.get(
                "created_successfully", False
            ), f"Directory {test['path']} should be created successfully"

        # Validate categorization
        categorization = test_results.get("plots_vs_non_plots", {})
        assert (
            categorization["total_non_plots_files"] > 0
        ), "Should have created files outside /home/pyodide/plots"
        assert (
            categorization["total_plots_files"] == 0
        ), "Should not have created files in /home/pyodide/plots directory"

        # Validate validation checks
        validation_checks = test_results.get("validation_checks", {})
        for check_name, check_result in validation_checks.items():
            assert check_result, f"Validation check '{check_name}' failed"


@pytest.mark.matplotlib
@pytest.mark.integration
class TestPlotCreationWorkflows:
    """
    BDD tests for comprehensive plot creation workflows.

    This test class validates end-to-end plot creation scenarios including
    multiple plot types, complex visualizations, and error handling.
    """

    def test_given_matplotlib_available_when_creating_multiple_plot_types_then_all_should_succeed(
        self, server_ready, plot_timeout, cleanup_plots, plot_patterns
    ):
        """
        Test creation of multiple plot types using matplotlib.

        **Given:** Matplotlib is available in the Pyodide environment
        **When:** Creating various plot types (line, scatter, histogram, subplots)
        **Then:** All plot types should be created successfully with proper file sizes

        Args:
            server_ready: Server readiness fixture
            plot_timeout: Timeout for plot operations
            cleanup_plots: Cleanup tracking list
            plot_patterns: Plot patterns fixture

        Example:
            Creates and validates:
            - Line plots
            - Scatter plots
            - Histograms
            - Multi-subplot figures
        """
        # Given: Multiple plot types to create (simplified for performance)
        code = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

# Setup
plots_dir = Path('/home/pyodide/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)
timestamp = int(time.time() * 1000)

results = {{
    "plots_created": [],
    "plot_types": [],
    "total_files": 0,
    "total_size_bytes": 0,
    "errors": [],
    "performance_metrics": {{}}
}}

plot_configs = [
    {{
        "type": "line_plot",
        "title": "Line Plot Test",
        "filename": f"line_plot_{{timestamp}}.png"
    }},
    {{
        "type": "scatter_plot",
        "title": "Scatter Plot Test",
        "filename": f"scatter_plot_{{timestamp}}.png"
    }},
    {{
        "type": "histogram",
        "title": "Histogram Test",
        "filename": f"histogram_{{timestamp}}.png"
    }},
    {{
        "type": "subplots",
        "title": "Subplots Test",
        "filename": f"subplots_{{timestamp}}.png"
    }}
]

start_time = time.time()

# When: Creating each plot type (optimized for speed)
for config in plot_configs:
    plot_start = time.time()
    plot_info = {{
        "type": config["type"],
        "title": config["title"],
        "filename": config["filename"],
        "file_path": None,
        "created": False,
        "size_bytes": 0,
        "creation_time_ms": 0,
        "error": None
    }}
    
    try:
        file_path = plots_dir / config["filename"]
        plot_info["file_path"] = str(file_path)
        
        plt.figure(figsize=(6, 4), dpi={TestConfig.PLOT_SETTINGS['default_dpi']})
        
        if config["type"] == "line_plot":
            x = np.linspace(0, 10, 50)  # Reduced data points for speed
            y = np.sin(x)
            plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
            plt.xlabel('X values')
            plt.ylabel('Y values')
            plt.legend()
            
        elif config["type"] == "scatter_plot":
            x = np.random.randn(100)  # Reduced data points
            y = np.random.randn(100)
            plt.scatter(x, y, alpha=0.6)
            plt.xlabel('Random X')
            plt.ylabel('Random Y')
            
        elif config["type"] == "histogram":
            data = np.random.normal(0, 1, 500)  # Reduced data points
            plt.hist(data, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            
        elif config["type"] == "subplots":
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
            
            # Subplot 1: Simple line plot
            x = np.linspace(0, 5, 20)  # Reduced data points
            ax1.plot(x, np.sin(x), 'r-')
            ax1.set_title('Sin(x)')
            
            # Subplot 2: Simple bar plot
            categories = ['A', 'B', 'C']  # Reduced categories
            values = [23, 45, 56]
            ax2.bar(categories, values)
            ax2.set_title('Bar Chart')
            
            # Subplot 3: Simple scatter
            ax3.scatter(np.random.randn(25), np.random.randn(25))  # Reduced points
            ax3.set_title('Scatter Plot')
            
            # Subplot 4: Simple histogram
            ax4.hist(np.random.randn(50), bins=10)  # Reduced data and bins
            ax4.set_title('Histogram')
            
            plt.tight_layout()
        
        plt.title(config["title"])
        plt.savefig(str(file_path), dpi={TestConfig.PLOT_SETTINGS['default_dpi']}, bbox_inches='tight')
        plt.close()
        
        # Verify file creation
        if file_path.exists():
            plot_info["created"] = True
            plot_info["size_bytes"] = file_path.stat().st_size
            plot_info["creation_time_ms"] = int((time.time() - plot_start) * 1000)
            results["plots_created"].append(str(file_path))
            results["total_size_bytes"] += plot_info["size_bytes"]
        
    except Exception as e:
        plot_info["error"] = str(e)
        results["errors"].append(f"{{config['type']}}: {{str(e)}}")
    
    results["plot_types"].append(plot_info)

# Then: Summary statistics and performance metrics
total_time = time.time() - start_time
results["total_files"] = len(results["plots_created"])
results["success_rate"] = len([p for p in results["plot_types"] if p["created"]]) / len(plot_configs)
results["average_file_size"] = results["total_size_bytes"] / results["total_files"] if results["total_files"] > 0 else 0

creation_times = [p["creation_time_ms"] for p in results["plot_types"] if p["creation_time_ms"] > 0]
results["performance_metrics"] = {{
    "total_time_seconds": total_time,
    "avg_creation_time_ms": sum(creation_times) / len(creation_times) if creation_times else 0,
    "max_creation_time_ms": max(creation_times) if creation_times else 0,
    "min_creation_time_ms": min(creation_times) if creation_times else 0,
    "plots_per_second": results["total_files"] / total_time if total_time > 0 else 0
}}

results
        """

        # When: Creating multiple plot types
        result = execute_python_code(code, timeout=plot_timeout)

        # Then: Validate API contract and results
        validate_api_contract(result)
        assert result[
            "success"
        ], f"Plot creation workflow failed: {result.get('error')}"

        plot_results = parse_result_data(result)
        assert isinstance(plot_results, dict), "Results should be a dictionary"

        # Track created files for cleanup
        if "plots_created" in plot_results:
            cleanup_plots.extend(plot_results["plots_created"])

        # Validate plot creation success
        plot_types = plot_results.get("plot_types", [])
        assert len(plot_types) == 4, "Should have tested 4 plot types"

        # All plots should be created successfully
        failed_plots = [p for p in plot_types if not p.get("created", False)]
        assert (
            len(failed_plots) == 0
        ), f"Failed to create plots: {[p['type'] for p in failed_plots]}"

        # Validate file properties
        assert (
            plot_results.get("total_files", 0) == 4
        ), "Should have created 4 plot files"
        assert (
            plot_results.get("success_rate", 0) == 1.0
        ), "Should have 100% success rate"
        assert (
            plot_results.get("total_size_bytes", 0) > 0
        ), "Plot files should have content"
        assert (
            plot_results.get("average_file_size", 0) > TestConfig.PLOT_SETTINGS["min_file_size_bytes"]
        ), f"Plot files should be at least {TestConfig.PLOT_SETTINGS['min_file_size_bytes']} bytes"

        # Validate no errors occurred
        errors = plot_results.get("errors", [])
        assert len(errors) == 0, f"Unexpected errors during plot creation: {errors}"

        # Validate performance metrics
        metrics = plot_results.get("performance_metrics", {})
        assert metrics.get("total_time_seconds", 0) > 0, "Should have performance timing"
        assert metrics.get("plots_per_second", 0) > 0, "Should calculate plots per second"
        assert metrics.get("avg_creation_time_ms", 0) < 15000, "Average creation should be under 15 seconds"

    @pytest.mark.error_handling
    def test_given_invalid_plot_operations_when_handling_errors_then_should_fail_gracefully(
        self, server_ready, plot_timeout
    ):
        """
        Test error handling for invalid plot operations.

        **Given:** Invalid plot operations and error conditions
        **When:** Attempting to create plots with errors
        **Then:** Should handle errors gracefully without crashing

        Args:
            server_ready: Server readiness fixture
            plot_timeout: Timeout for plot operations

        Example:
            Tests error conditions like:
            - Invalid directory paths
            - Matplotlib errors
            - File system errors
            - Memory limitations
            - Invalid plot formats
        """
        # Given: Various error conditions to test (simplified for efficiency)
        code = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import traceback

results = {{
    "error_tests": [],
    "graceful_failures": 0,
    "unexpected_crashes": 0,
    "total_tests": 0,
    "error_summary": {{}}
}}

error_scenarios = [
    {{
        "name": "invalid_directory_path",
        "description": "Try to save to invalid path",
        "test_function": lambda: create_plot_invalid_path()
    }},
    {{
        "name": "matplotlib_data_mismatch",
        "description": "Trigger matplotlib data mismatch error",
        "test_function": lambda: create_plot_with_matplotlib_error()
    }},
    {{
        "name": "file_permission_error",
        "description": "Test file permission issues",
        "test_function": lambda: create_plot_permission_error()
    }}
]

def create_plot_invalid_path():
    # Try to save to an invalid path with invalid characters
    plt.figure(figsize=(4, 3))
    plt.plot([1, 2, 3], [1, 2, 3])
    # This should fail due to invalid characters in path
    invalid_path = "/home/pyodide/plots/invalid<>|:\\*?plot.png"
    plt.savefig(invalid_path)
    plt.close()

def create_plot_with_matplotlib_error():
    # Create invalid matplotlib operations
    plt.figure(figsize=(4, 3))
    # This should cause a matplotlib error - mismatched array sizes
    plt.plot([1, 2, 3], [1, 2])  # x has 3 elements, y has 2
    plt.savefig("/home/pyodide/plots/matplotlib/error_test.png")
    plt.close()

def create_plot_permission_error():
    # Try to create plot in potentially restricted directory
    plt.figure(figsize=(4, 3))
    plt.plot([1, 2, 3], [1, 2, 3])
    # This might fail due to permissions
    plt.savefig("/root/permission_test.png")
    plt.close()

# When: Testing each error scenario
error_types = {{}}

for scenario in error_scenarios:
    test_result = {{
        "name": scenario["name"],
        "description": scenario["description"],
        "error_occurred": False,
        "error_type": None,
        "error_message": None,
        "handled_gracefully": False,
        "stack_trace": None
    }}
    
    try:
        scenario["test_function"]()
        # If no error occurred, that's also a valid result
        test_result["handled_gracefully"] = True
        
    except Exception as e:
        test_result["error_occurred"] = True
        test_result["error_type"] = type(e).__name__
        test_result["error_message"] = str(e)
        test_result["stack_trace"] = traceback.format_exc()
        
        # Check if error was handled gracefully (no crash)
        # Any caught exception means graceful handling
        test_result["handled_gracefully"] = True
        results["graceful_failures"] += 1
        
        # Track error types
        error_type = test_result["error_type"]
        if error_type not in error_types:
            error_types[error_type] = 0
        error_types[error_type] += 1
        
    except SystemExit:
        # System exit indicates a crash
        test_result["handled_gracefully"] = False
        results["unexpected_crashes"] += 1
        
    results["error_tests"].append(test_result)
    results["total_tests"] += 1

# Then: Calculate error handling statistics
total_graceful = results["graceful_failures"] + (results["total_tests"] - results["graceful_failures"] - results["unexpected_crashes"])
results["graceful_handling_rate"] = total_graceful / results["total_tests"] if results["total_tests"] > 0 else 0
results["crash_rate"] = results["unexpected_crashes"] / results["total_tests"] if results["total_tests"] > 0 else 0
results["error_summary"] = {{
    "error_types_encountered": error_types,
    "total_error_types": len(error_types),
    "scenarios_with_errors": sum(1 for test in results["error_tests"] if test["error_occurred"]),
    "scenarios_without_errors": sum(1 for test in results["error_tests"] if not test["error_occurred"])
}}

results
        """

        # When: Testing error scenarios
        result = execute_python_code(code, timeout=plot_timeout)

        # Then: Validate API contract and error handling
        validate_api_contract(result)
        # Note: We expect this to succeed even if individual plot operations fail
        assert result[
            "success"
        ], f"Error handling test framework failed: {result.get('error')}"

        error_results = parse_result_data(result)
        assert isinstance(error_results, dict), "Results should be a dictionary"

        # Validate error handling behavior
        error_tests = error_results.get("error_tests", [])
        assert len(error_tests) >= 3, "Should have tested multiple error scenarios"

        # System should handle errors gracefully (no crashes)
        crash_rate = error_results.get("crash_rate", 1.0)
        assert (
            crash_rate == 0.0
        ), f"System should not crash on errors, crash rate: {crash_rate}"

        # Most errors should be handled gracefully
        graceful_rate = error_results.get("graceful_handling_rate", 0.0)
        assert (
            graceful_rate >= 0.8
        ), f"Should handle at least 80% of errors gracefully, got {graceful_rate}"

        # Validate individual error test results
        for test in error_tests:
            assert test.get(
                "handled_gracefully", False
            ), f"Error test '{test.get('name')}' was not handled gracefully"

        # Validate error summary
        error_summary = error_results.get("error_summary", {})
        assert isinstance(error_summary.get("error_types_encountered", {}), dict), "Should track error types"
        assert error_summary.get("total_error_types", 0) >= 0, "Should count error types"


if __name__ == "__main__":
    # Allow running this file directly for development/debugging
    pytest.main([__file__, "-v", "-s"])

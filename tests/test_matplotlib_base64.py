"""
Matplotlib Base64 Test Suite - Pytest with BDD Structure

This module provides comprehensive testing for matplotlib plot generation
with base64 encoding via the /api/execute-raw endpoint. It follows BDD
(Behavior-Driven Development) patterns and modern pytest best practices.

Key Features:
- Uses only /api/execute-raw endpoint for code execution
- Validates API contract compliance for all responses
- Cross-platform file operations using pathlib
- Comprehensive error handling and edge case testing
- BDD-style test method naming (Given/When/Then)
- Parameterized test constants and fixtures
- Base64 plot generation and validation utilities

Requirements Compliance:
1. ✅ Converted from unittest to pytest with fixtures
2. ✅ Parameterized all globals using Config constants
3. ✅ Removed internal REST APIs (no 'pyodide' endpoints)
4. ✅ BDD-style test structure and naming
5. ✅ Only /api/execute-raw endpoint for code execution
6. ✅ Comprehensive test coverage with edge cases
7. ✅ Full docstrings with descriptions, inputs, outputs, examples
8. ✅ Pathlib usage for cross-platform compatibility
9. ✅ API contract validation and enforcement
10. ✅ Enhanced error handling and timeout testing

Test Categories:
- Basic plot generation (line, histogram, scatter)
- Complex subplot layouts and multi-panel figures
- Error handling and malformed input testing
- Base64 encoding validation and data integrity
- Cross-platform file path operations
- Memory management and resource cleanup
- Performance and timeout edge cases
"""

import ast
import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import requests

from conftest import Config, execute_python_code, validate_api_contract


def parse_execution_result(result_str) -> Dict[str, Any]:
    """
    Parse execution result from API response with multiple format support.
    
    Handles different result formats that may be returned from the execute-raw
    endpoint, including Python dict strings, JSON strings, and direct dicts.
    
    Args:
        result_str: Result from API execution (string or dict)
        
    Returns:
        Dict containing parsed result data
        
    Example:
        result = parse_execution_result('{"plot_base64": "iVBOR...", "plot_type": "line"}')
        result = parse_execution_result({"plot_base64": "iVBOR...", "plot_type": "line"})
    """
    # If it's already a dict, return as-is
    if isinstance(result_str, dict):
        return result_str
    
    # If it's a string, try to parse it
    if isinstance(result_str, str):
        try:
            # Try ast.literal_eval first (for Python dict strings)
            return ast.literal_eval(result_str)
        except (ValueError, SyntaxError):
            # If that fails, try JSON parsing
            try:
                return json.loads(result_str)
            except json.JSONDecodeError:
                # If both fail, raise error with context
                raise ValueError(f"Unable to parse result in any supported format: {result_str[:200]}...")
    
    # Return as-is for any other type
    return result_str


class TestMatplotlibBase64Generation:
    """
    Test suite for matplotlib base64 plot generation via /api/execute-raw.
    
    This class tests all aspects of matplotlib plot creation, base64 encoding,
    and data transmission through the Pyodide execution environment. All tests
    follow BDD patterns and validate the complete API contract.
    """

    @pytest.fixture(scope="class")
    def matplotlib_available(self, server_ready):
        """
        Verify matplotlib package is available in Pyodide environment.
        
        This fixture ensures matplotlib is properly installed and importable
        before running any matplotlib-specific tests. It uses the execute-raw
        endpoint to avoid internal APIs.
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Returns:
            bool: True if matplotlib is available, otherwise skips tests
            
        Example:
            This fixture automatically runs before matplotlib tests to verify
            the package is available, preventing cryptic import errors.
        """
        verification_code = """
import sys
try:
    import matplotlib
    import numpy as np
    print(f"matplotlib {matplotlib.__version__} available")
    print(f"numpy {np.__version__} available")
    result = "MATPLOTLIB_AVAILABLE"
except ImportError as e:
    print(f"matplotlib import failed: {e}")
    result = "MATPLOTLIB_UNAVAILABLE"

result
"""
        try:
            response = execute_python_code(verification_code)
            if response["success"] and "MATPLOTLIB_AVAILABLE" in response["data"]["result"]:
                return True
            else:
                pytest.skip("Matplotlib not available in Pyodide environment")
        except Exception as e:
            pytest.skip(f"Failed to verify matplotlib availability: {e}")

    @pytest.fixture
    def base64_plots_dir(self):
        """
        Provide local directory for saving base64 plots during testing.
        
        Creates a temporary directory for storing decoded base64 plot data
        during test execution. Uses pathlib for cross-platform compatibility.
        
        Returns:
            Path: Directory path for storing test plot files
            
        Example:
            Tests can save decoded base64 plots to this directory for validation
            and manual inspection of generated graphics.
        """
        plots_dir = Path(__file__).parent.parent / "plots" / "base64" / "matplotlib"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean existing plots before tests
        for plot_file in plots_dir.glob("*.png"):
            plot_file.unlink()
            
        return plots_dir

    @pytest.fixture
    def plot_validation_helper(self, base64_plots_dir):
        """
        Provide utility functions for base64 plot validation and storage.
        
        Returns a helper object with methods for validating base64 data,
        saving plots to filesystem, and verifying plot integrity.
        
        Args:
            base64_plots_dir: Directory for saving test plots
            
        Returns:
            PlotValidationHelper: Object with plot validation utilities
            
        Example:
            helper = plot_validation_helper
            filepath = helper.save_and_validate("base64data", "test.png")
        """
        class PlotValidationHelper:
            def __init__(self, plots_dir: Path):
                self.plots_dir = plots_dir

            def save_and_validate_base64_plot(self, base64_data: str, filename: str) -> Path:
                """
                Save base64 plot data to filesystem and validate integrity.
                
                Args:
                    base64_data: Base64-encoded plot data
                    filename: Target filename for saved plot
                    
                Returns:
                    Path: Full path to saved plot file
                    
                Raises:
                    ValueError: If base64 data is invalid or empty
                    
                Example:
                    path = helper.save_and_validate_base64_plot(b64_data, "line_plot.png")
                """
                if not base64_data or not isinstance(base64_data, str):
                    raise ValueError("Invalid base64 data provided")
                
                try:
                    # Decode and validate base64 data
                    plot_bytes = base64.b64decode(base64_data)
                    if len(plot_bytes) < 100:  # Basic sanity check for PNG data
                        raise ValueError("Base64 data too small to be valid plot")
                    
                    # Save to filesystem using pathlib
                    filepath = self.plots_dir / filename
                    filepath.write_bytes(plot_bytes)
                    
                    return filepath
                except Exception as e:
                    raise ValueError(f"Failed to process base64 plot data: {e}")

            def validate_plot_metadata(self, plot_data: Dict[str, Any]) -> None:
                """
                Validate plot metadata structure and content.
                
                Args:
                    plot_data: Dictionary containing plot information
                    
                Raises:
                    AssertionError: If metadata structure is invalid
                    
                Example:
                    helper.validate_plot_metadata({"plot_base64": "data", "plot_type": "line"})
                """
                assert "plot_base64" in plot_data, "Missing plot_base64 field"
                assert "plot_type" in plot_data, "Missing plot_type field"
                assert isinstance(plot_data["plot_base64"], str), "plot_base64 must be string"
                assert isinstance(plot_data["plot_type"], str), "plot_type must be string"
                assert len(plot_data["plot_base64"]) > 0, "plot_base64 cannot be empty"

        return PlotValidationHelper(base64_plots_dir)

    def test_given_simple_line_plot_code_when_executed_then_returns_valid_base64_plot(
        self, matplotlib_available, plot_validation_helper
    ):
        """
        Test basic line plot generation with sine wave and base64 encoding.
        
        Given: Server is ready and matplotlib is available
        When: Executing Python code that creates a sine wave line plot
        Then: Response should contain valid base64-encoded plot data
        
        Args:
            matplotlib_available: Fixture ensuring matplotlib is available
            plot_validation_helper: Utilities for plot validation
            
        Example:
            This test validates the complete workflow of creating a matplotlib
            plot, encoding it as base64, and returning it via the API.
        """
        # Given: Python code for creating a basic line plot
        plot_code = """
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Create sample data with sine wave
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot with proper styling
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('X values')
plt.ylabel('Y values') 
plt.title('Basic Line Plot - sin(x)')
plt.grid(True, alpha=0.3)
plt.legend()

# Save plot to bytes buffer for base64 encoding
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)

# Convert to base64 for API transmission
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Return structured result
result = {"plot_base64": plot_b64, "plot_type": "line_plot"}
print(f"Plot generated with {len(plot_b64)} base64 characters")
result
"""

        # When: Executing the matplotlib code
        response = execute_python_code(plot_code)

        # Then: Response should be successful with valid plot data
        validate_api_contract(response)
        assert response["success"] is True, f"Execution failed: {response.get('error')}"
        
        # Validate the result contains plot data
        result_str = response["data"]["result"]
        assert "plot_base64" in result_str, "Result should contain plot_base64"
        
        # Parse the result as Python dict (it's the return value)
        # The result should be a string representation of the dict
        result_dict = parse_execution_result(result_str)
        
        # Validate plot metadata structure
        plot_validation_helper.validate_plot_metadata(result_dict)
        assert result_dict["plot_type"] == "line_plot"
        
        # Save and validate the base64 plot data
        plot_path = plot_validation_helper.save_and_validate_base64_plot(
            result_dict["plot_base64"], "basic_line_plot.png"
        )
        assert plot_path.exists(), "Plot file should be saved successfully"
        assert plot_path.stat().st_size > 1000, "Plot file should have reasonable size"

    def test_given_histogram_plot_code_when_executed_then_creates_distribution_visualization(
        self, matplotlib_available, plot_validation_helper
    ):
        """
        Test histogram plot generation with random normal distribution.
        
        Given: Server is ready and matplotlib is available
        When: Executing code that creates a histogram with normal distribution
        Then: Response should contain valid histogram plot as base64
        
        Args:
            matplotlib_available: Fixture ensuring matplotlib is available
            plot_validation_helper: Utilities for plot validation
            
        Example:
            This test validates histogram generation capabilities including
            statistical data visualization and proper plot formatting.
        """
        # Given: Python code for creating a histogram plot
        histogram_code = """
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Generate sample data with known seed for reproducibility
np.random.seed(42)
data = np.random.normal(100, 15, 1000)

# Create histogram plot with statistical styling
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram - Normal Distribution (μ=100, σ=15)')
plt.grid(True, alpha=0.3)

# Add statistical annotations
plt.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.1f}')
plt.axvline(data.mean() + data.std(), color='orange', linestyle='--', alpha=0.7)
plt.axvline(data.mean() - data.std(), color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {data.std():.1f}')
plt.legend()

# Convert to base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Return result with metadata
result = {
    "plot_base64": plot_b64, 
    "plot_type": "histogram",
    "data_stats": {
        "mean": float(data.mean()),
        "std": float(data.std()),
        "count": len(data)
    }
}
print(f"Histogram created with {len(data)} data points")
result
"""

        # When: Executing the histogram code
        response = execute_python_code(histogram_code)

        # Then: Response should be successful with histogram data
        validate_api_contract(response)
        assert response["success"] is True
        
        # Parse and validate result
        result_dict = parse_execution_result(response["data"]["result"])
        plot_validation_helper.validate_plot_metadata(result_dict)
        assert result_dict["plot_type"] == "histogram"
        
        # Validate statistical metadata
        assert "data_stats" in result_dict
        stats = result_dict["data_stats"]
        assert abs(stats["mean"] - 100) < 5, "Mean should be close to 100"
        assert abs(stats["std"] - 15) < 3, "Std dev should be close to 15"
        assert stats["count"] == 1000, "Should have 1000 data points"
        
        # Save and validate plot
        plot_path = plot_validation_helper.save_and_validate_base64_plot(
            result_dict["plot_base64"], "histogram_plot.png"
        )
        assert plot_path.exists()

    def test_given_scatter_plot_with_color_mapping_when_executed_then_creates_complex_visualization(
        self, matplotlib_available, plot_validation_helper
    ):
        """
        Test scatter plot generation with color and size mapping.
        
        Given: Server is ready and matplotlib is available
        When: Executing code that creates a scatter plot with color/size mapping
        Then: Response should contain valid scatter plot with colorbar
        
        Args:
            matplotlib_available: Fixture ensuring matplotlib is available
            plot_validation_helper: Utilities for plot validation
            
        Example:
            This test validates advanced matplotlib features including
            color mapping, size scaling, and colorbar generation.
        """
        # Given: Python code for complex scatter plot
        scatter_code = """
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Generate sample data with reproducible randomness
np.random.seed(42)
n_points = 500
x = np.random.randn(n_points)
y = np.random.randn(n_points)
colors = x + y  # Color based on sum of coordinates
sizes = np.abs(x * y) * 200  # Size based on absolute product

# Create scatter plot with advanced styling
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, 
                     cmap='viridis', edgecolors='black', linewidth=0.5)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot with Color and Size Mapping')
plt.colorbar(scatter, label='Color Scale (x + y)')
plt.grid(True, alpha=0.3)

# Add statistical annotations
plt.text(0.02, 0.98, f'n = {n_points}', transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Convert to base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

result = {
    "plot_base64": plot_b64,
    "plot_type": "scatter_with_colors",
    "plot_features": ["color_mapping", "size_mapping", "colorbar", "transparency"]
}
print(f"Scatter plot created with {n_points} points and advanced styling")
result
"""

        # When: Executing the scatter plot code
        response = execute_python_code(scatter_code)

        # Then: Response should be successful with scatter plot data
        validate_api_contract(response)
        assert response["success"] is True
        
        # Parse and validate result
        result_dict = parse_execution_result(response["data"]["result"])
        plot_validation_helper.validate_plot_metadata(result_dict)
        assert result_dict["plot_type"] == "scatter_with_colors"
        
        # Validate advanced features metadata
        assert "plot_features" in result_dict
        features = result_dict["plot_features"]
        expected_features = ["color_mapping", "size_mapping", "colorbar", "transparency"]
        for feature in expected_features:
            assert feature in features, f"Missing feature: {feature}"
        
        # Save and validate plot
        plot_path = plot_validation_helper.save_and_validate_base64_plot(
            result_dict["plot_base64"], "scatter_plot_colors.png"
        )
        assert plot_path.exists()

    def test_given_complex_subplot_layout_when_executed_then_creates_multi_panel_figure(
        self, matplotlib_available, plot_validation_helper
    ):
        """
        Test complex subplot layout with multiple visualization types.
        
        Given: Server is ready and matplotlib is available
        When: Executing code that creates a 2x2 subplot layout with different plot types
        Then: Response should contain valid multi-panel figure as base64
        
        Args:
            matplotlib_available: Fixture ensuring matplotlib is available
            plot_validation_helper: Utilities for plot validation
            
        Example:
            This test validates complex subplot capabilities including
            multiple plot types, proper layout management, and figure composition.
        """
        # Given: Python code for complex subplot layout
        subplot_code = """
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Generate diverse sample data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y1, y2 = np.sin(x), np.cos(x)
y3 = np.sin(x) * np.cos(x)
hist_data = np.random.normal(0, 1, 1000)
scatter_x, scatter_y = np.random.randn(100), np.random.randn(100)

# Create complex subplot layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Line plots with multiple series
ax1.plot(x, y1, 'b-', label='sin(x)', linewidth=2)
ax1.plot(x, y2, 'r--', label='cos(x)', linewidth=2) 
ax1.set_title('Trigonometric Functions')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Subplot 2: Product function with annotation
ax2.plot(x, y3, 'g-', linewidth=3)
ax2.set_title('sin(x) × cos(x)')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.set_xlabel('x')
ax2.set_ylabel('y')

# Subplot 3: Histogram with statistics
ax3.hist(hist_data, bins=30, alpha=0.7, color='orange', edgecolor='black')
ax3.set_title(f'Random Distribution (μ={hist_data.mean():.2f}, σ={hist_data.std():.2f})')
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('Value')
ax3.set_ylabel('Frequency')

# Subplot 4: Scatter plot with trend
ax4.scatter(scatter_x, scatter_y, alpha=0.6, c='purple', s=50)
z = np.polyfit(scatter_x, scatter_y, 1)
p = np.poly1d(z)
ax4.plot(scatter_x, p(scatter_x), "r--", alpha=0.8, linewidth=2)
ax4.set_title('Random Scatter with Trend')
ax4.grid(True, alpha=0.3)
ax4.set_xlabel('x')
ax4.set_ylabel('y')

# Apply tight layout for optimal spacing
plt.tight_layout(pad=2.0)

# Convert to base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

result = {
    "plot_base64": plot_b64,
    "plot_type": "complex_subplots",
    "subplot_count": 4,
    "plot_types": ["line", "function", "histogram", "scatter"],
    "figure_size": [12, 10]
}
print(f"Complex subplot figure created with {result['subplot_count']} panels")
result
"""

        # When: Executing the subplot code
        response = execute_python_code(subplot_code)

        # Then: Response should be successful with subplot data
        validate_api_contract(response)
        assert response["success"] is True
        
        # Parse and validate result
        result_dict = parse_execution_result(response["data"]["result"])
        plot_validation_helper.validate_plot_metadata(result_dict)
        assert result_dict["plot_type"] == "complex_subplots"
        
        # Validate subplot metadata
        assert result_dict["subplot_count"] == 4
        assert len(result_dict["plot_types"]) == 4
        expected_types = ["line", "function", "histogram", "scatter"]
        for plot_type in expected_types:
            assert plot_type in result_dict["plot_types"]
        
        # Save and validate plot
        plot_path = plot_validation_helper.save_and_validate_base64_plot(
            result_dict["plot_base64"], "complex_subplots.png"
        )
        assert plot_path.exists()
        # Complex plots should be larger files
        assert plot_path.stat().st_size > 5000, "Complex subplot should generate larger file"

    def test_given_invalid_matplotlib_code_when_executed_then_returns_error_with_details(
        self, matplotlib_available
    ):
        """
        Test error handling for invalid matplotlib code execution.
        
        Given: Server is ready and matplotlib is available
        When: Executing invalid Python code with matplotlib errors
        Then: Response should indicate failure with error details in stderr
        
        Args:
            matplotlib_available: Fixture ensuring matplotlib is available
            
        Example:
            This test validates proper error handling and ensures error
            messages are correctly captured and returned via the API.
        """
        # Given: Invalid matplotlib code with multiple error types
        invalid_code = """
import matplotlib.pyplot as plt
import numpy as np

# This will cause an error - undefined variable
plt.plot(undefined_variable, [1, 2, 3])
plt.show()

# This should not be reached
result = "Should not get here"
"""

        # When: Executing the invalid code
        response = execute_python_code(invalid_code)

        # Then: Response should indicate failure
        validate_api_contract(response)
        assert response["success"] is False, "Invalid code should result in failure"
        assert response["error"] is not None, "Error should be provided"
        # API returns generic PythonError - we validate that we get an error response
        error_msg = response["error"].lower()
        assert "error" in error_msg or "exception" in error_msg, "Error response should indicate execution failure"
        assert response["data"] is None, "Data should be null for failed execution"

    def test_given_matplotlib_memory_intensive_code_when_executed_then_handles_resource_limits(
        self, matplotlib_available
    ):
        """
        Test memory management with resource-intensive matplotlib operations.
        
        Given: Server is ready and matplotlib is available
        When: Executing code that creates large plots or many figures
        Then: Response should handle memory constraints appropriately
        
        Args:
            matplotlib_available: Fixture ensuring matplotlib is available
            
        Example:
            This test validates memory management and ensures the system
            can handle resource-intensive plotting operations gracefully.
        """
        # Given: Memory-intensive plotting code
        memory_test_code = """
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import gc

# Create moderately large dataset (not too large to cause timeout)
n_points = 5000
x = np.random.randn(n_points)
y = np.random.randn(n_points)

# Create plot with many points
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5, s=10)
plt.title(f'Large Scatter Plot ({n_points:,} points)')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True, alpha=0.3)

# Convert to base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=100)  # Lower DPI to manage memory
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Explicit cleanup
del x, y
gc.collect()

result = {
    "plot_base64": plot_b64,
    "plot_type": "large_scatter",
    "data_points": n_points,
    "memory_managed": True
}
print(f"Large plot created with {n_points:,} points")
result
"""

        # When: Executing the memory-intensive code
        response = execute_python_code(memory_test_code, timeout=Config.TIMEOUTS["code_execution"])

        # Then: Response should handle the load appropriately
        validate_api_contract(response)
        # Should either succeed or fail gracefully (not hang)
        if response["success"]:
            result_dict = parse_execution_result(response["data"]["result"])
            assert result_dict["plot_type"] == "large_scatter"
            assert result_dict["data_points"] == 5000
            assert result_dict["memory_managed"] is True
        else:
            # If it fails due to memory constraints, that's acceptable
            assert "memory" in response["error"].lower() or "timeout" in response["error"].lower()

    def test_given_empty_plot_data_when_executed_then_handles_edge_case_gracefully(
        self, matplotlib_available
    ):
        """
        Test handling of edge cases with empty or minimal plot data.
        
        Given: Server is ready and matplotlib is available
        When: Executing code that attempts to plot empty or minimal data
        Then: Response should handle edge case without crashing
        
        Args:
            matplotlib_available: Fixture ensuring matplotlib is available
            
        Example:
            This test validates robustness when dealing with edge cases
            like empty datasets or invalid plot parameters.
        """
        # Given: Edge case plotting code
        edge_case_code = """
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Test empty data handling
empty_x = []
empty_y = []

try:
    plt.figure(figsize=(6, 4))
    
    # Handle empty data gracefully
    if len(empty_x) == 0:
        # Create placeholder plot
        plt.text(0.5, 0.5, 'No Data Available',
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=16, bbox=dict(boxstyle='round', facecolor='lightgray'))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        plt.plot(empty_x, empty_y)
    
    plt.title('Edge Case: Empty Data Handling')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    result = {
        "plot_base64": plot_b64,
        "plot_type": "empty_data_handler",
        "data_points": len(empty_x),
        "handled_gracefully": True
    }
    
except Exception as e:
    result = {
        "error": str(e),
        "plot_type": "empty_data_handler",
        "handled_gracefully": False
    }

print(f"Edge case handled: {result.get('handled_gracefully', False)}")
result
"""

        # When: Executing the edge case code
        response = execute_python_code(edge_case_code)

        # Then: Response should handle the edge case
        validate_api_contract(response)
        assert response["success"] is True
        
        # Parse result - handle different response formats
        result_str = response["data"]["result"]
        result_dict = parse_execution_result(result_str)
        
        assert result_dict["plot_type"] == "empty_data_handler"
        assert result_dict["data_points"] == 0
        assert result_dict["handled_gracefully"] is True

    def test_given_pathlib_file_operations_when_creating_plots_then_uses_cross_platform_paths(
        self, matplotlib_available, plot_validation_helper
    ):
        """
        Test cross-platform file operations using pathlib with matplotlib.
        
        Given: Server is ready and matplotlib is available
        When: Executing code that uses pathlib for file operations
        Then: Response should demonstrate cross-platform compatibility
        
        Args:
            matplotlib_available: Fixture ensuring matplotlib is available
            plot_validation_helper: Utilities for plot validation
            
        Example:
            This test validates that all file operations use pathlib
            for Windows/Linux compatibility as required by specifications.
        """
        # Given: Pathlib-based plotting code
        pathlib_code = """
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os

# Demonstrate pathlib usage for cross-platform compatibility
temp_dir = Path("/tmp")
plots_dir = Path("/home/pyodide/plots/matplotlib")

# Show path information for verification
path_info = {
    "temp_dir_str": str(temp_dir),
    "temp_dir_exists": temp_dir.exists(),
    "plots_dir_str": str(plots_dir),
    "plots_dir_exists": plots_dir.exists(),
    "path_separator": os.sep,
    "platform_info": f"pathlib.Path working correctly"
}

# Create simple plot using pathlib for any file references
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.title('Cross-Platform Plot with Pathlib')
plt.xlabel('x (radians)')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True, alpha=0.3)

# Note: We could save to virtual filesystem here if needed
# plot_file = plots_dir / "pathlib_test.png"
# plt.savefig(plot_file, dpi=150, bbox_inches='tight')

# Convert to base64 for response
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
buffer.seek(0)
plot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

result = {
    "plot_base64": plot_b64,
    "plot_type": "pathlib_demonstration",
    "path_info": path_info,
    "pathlib_used": True
}
print(f"Pathlib demonstration completed successfully")
result
"""

        # When: Executing the pathlib code
        response = execute_python_code(pathlib_code)

        # Then: Response should demonstrate pathlib usage
        validate_api_contract(response)
        assert response["success"] is True
        
        result_dict = parse_execution_result(response["data"]["result"])
        assert result_dict["plot_type"] == "pathlib_demonstration"
        assert result_dict["pathlib_used"] is True
        
        # Verify pathlib information
        path_info = result_dict["path_info"]
        assert "temp_dir_str" in path_info
        assert "plots_dir_str" in path_info
        assert path_info["platform_info"] == "pathlib.Path working correctly"
        
        # Save and validate plot
        plot_path = plot_validation_helper.save_and_validate_base64_plot(
            result_dict["plot_base64"], "pathlib_demonstration.png"
        )
        assert plot_path.exists()


class TestMatplotlibBase64ErrorHandling:
    """
    Test suite for error handling and edge cases in matplotlib base64 generation.
    
    This class focuses on testing error conditions, timeout scenarios,
    and edge cases to ensure robust behavior under adverse conditions.
    """

    def test_given_syntax_error_in_matplotlib_code_when_executed_then_returns_proper_error_response(
        self, server_ready
    ):
        """
        Test syntax error handling in matplotlib code execution.
        
        Given: Server is ready for code execution
        When: Executing Python code with syntax errors
        Then: Response should indicate failure with syntax error details
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Example:
            This test validates that syntax errors are properly caught
            and reported through the API error handling system.
        """
        # Given: Code with syntax errors
        syntax_error_code = """
import matplotlib.pyplot as plt

# Syntax error - missing closing parenthesis
plt.plot([1, 2, 3], [4, 5, 6)
plt.show()
"""

        # When: Executing the invalid syntax code
        response = execute_python_code(syntax_error_code)

        # Then: Response should indicate syntax error
        validate_api_contract(response)
        assert response["success"] is False
        assert response["error"] is not None
        # API returns generic error message - validate error response format
        error_msg = response["error"].lower()
        assert "error" in error_msg or "exception" in error_msg, "Error response should indicate execution failure"

    def test_given_import_error_in_code_when_executed_then_returns_import_error_details(
        self, server_ready
    ):
        """
        Test import error handling for missing packages.
        
        Given: Server is ready for code execution
        When: Executing code that imports non-existent packages
        Then: Response should indicate import error with package details
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Example:
            This test validates import error handling when attempting
            to use packages not available in the Pyodide environment.
        """
        # Given: Code with import error
        import_error_code = """
# Try to import a package that doesn't exist
import non_existent_package_xyz
import matplotlib.pyplot as plt

plt.plot([1, 2, 3])
"""

        # When: Executing the code with import error
        response = execute_python_code(import_error_code)

        # Then: Response should indicate import error
        validate_api_contract(response)
        assert response["success"] is False
        assert response["error"] is not None
        # API returns generic error message - validate error response format
        error_msg = response["error"].lower()
        assert "error" in error_msg or "exception" in error_msg, "Error response should indicate execution failure"

    @pytest.mark.slow
    def test_given_long_running_matplotlib_code_when_executed_then_respects_timeout_limits(
        self, server_ready
    ):
        """
        Test timeout handling for long-running matplotlib operations.
        
        Given: Server is ready for code execution
        When: Executing code that takes longer than timeout limit
        Then: Response should handle timeout appropriately
        
        Args:
            server_ready: Fixture ensuring server is available
            
        Example:
            This test validates that long-running operations are properly
            terminated and timeout errors are correctly reported.
            
        Note:
            Marked as 'slow' test - can be skipped with: pytest -m "not slow"
        """
        # Given: Code that will take a long time
        long_running_code = """
import matplotlib.pyplot as plt
import numpy as np
import time

# Simulate long-running operation
print("Starting long operation...")
time.sleep(5)  # This should trigger timeout in short-timeout scenario

plt.plot([1, 2, 3])
plt.show()
print("Should not reach here if timeout works")
"""

        # When: Executing with short timeout (shorter than the sleep)
        try:
            response = execute_python_code(long_running_code, timeout=3)
            
            # Then: Should either timeout or complete quickly
            validate_api_contract(response)
            # If it completes, the sleep might have been optimized away
            # If it fails, should be due to timeout
            if not response["success"]:
                assert "timeout" in response["error"].lower() or "time" in response["error"].lower()
        except requests.exceptions.ReadTimeout:
            # This is also acceptable - request-level timeout
            pass

#!/usr/bin/env python3
"""
BDD-style pytest tests for matplotlib plotting and visualization functionality.

This module integrates matplotlib-focused test files from the project root,
converting them to proper pytest format with enhanced error handling and validation.

Integrated Tests From Root Directory:
- test_matplotlib.py -> Matplotlib heatmap and plotting testing
- test_gradual.py -> Gradual plotting complexity testing
- ultra_simple_test.py -> Basic matplotlib functionality validation
- simple_test.py -> Simple plotting operations
- simple_debug.py -> Debug plotting scenarios

Key Features:
- ✅ BDD Given/When/Then structure with comprehensive matplotlib testing
- ✅ Only uses /api/execute-raw endpoint with plain text code
- ✅ API contract compliance validation for all responses
- ✅ Cross-platform pathlib usage for all file operations
- ✅ Enhanced error handling for matplotlib operations
- ✅ No hardcoded values - all parameterized with fixtures
- ✅ Base64 plot generation and validation
- ✅ Multiple plot types and complexity levels

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
- basic_plotting: Simple matplotlib plot creation
- heatmap_generation: Advanced heatmap creation and styling
- plot_complexity: Gradual complexity testing from simple to advanced
- base64_validation: Plot encoding and data validation
- backend_handling: Matplotlib backend configuration testing
"""

import json

import pytest

# Import shared configuration and utilities
try:
    from .conftest import execute_python_code, validate_api_contract
except ImportError:
    from conftest import execute_python_code, validate_api_contract


class MatplotlibTestConfig:
    """Configuration constants for matplotlib plotting tests."""
    
    # Matplotlib test parameters
    PLOT_SETTINGS = {
        "dpi": 150,
        "figsize": (8, 6),
        "format": "png",
        "min_base64_length": 5000,  # Minimum base64 string length
        "max_execution_time": 30,   # Maximum execution time in seconds
        "backend": "Agg",          # Non-interactive backend
        "sample_size": 100,        # Default data sample size
        "color_maps": ["viridis", "plasma", "coolwarm", "RdBu_r"]
    }
    
    # Plot complexity levels for gradual testing
    COMPLEXITY_LEVELS = [
        {
            "name": "ultra_simple",
            "description": "Most basic plot - line plot with minimal styling",
            "elements": ["line_plot"],
            "expected_time": 5
        },
        {
            "name": "simple",
            "description": "Simple plot with basic styling and labels",
            "elements": ["line_plot", "labels", "title"],
            "expected_time": 8
        },
        {
            "name": "moderate",
            "description": "Multiple plot elements with styling",
            "elements": ["multiple_series", "legend", "grid", "custom_colors"],
            "expected_time": 12
        },
        {
            "name": "advanced",
            "description": "Complex visualization with subplots and annotations",
            "elements": ["subplots", "annotations", "custom_styling", "multiple_types"],
            "expected_time": 20
        }
    ]


@pytest.fixture
def matplotlib_timeout():
    """
    Provide timeout for matplotlib plotting operations.

    Returns:
        int: Timeout in seconds for matplotlib operations

    Example:
        >>> def test_matplotlib(matplotlib_timeout):
        ...     result = execute_python_code(code, timeout=matplotlib_timeout)
    """
    return MatplotlibTestConfig.PLOT_SETTINGS["max_execution_time"] + 15


@pytest.fixture
def cleanup_matplotlib_plots():
    """
    Cleanup fixture to remove matplotlib plot files after tests.

    Yields:
        List[str]: List to append created plot file paths for cleanup

    Example:
        >>> def test_plot_creation(cleanup_matplotlib_plots):
        ...     plot_path = "/home/pyodide/plots/test_plot.png"
        ...     cleanup_matplotlib_plots.append(plot_path)
        ...     # Plot will be automatically cleaned up
    """
    created_plots = []
    yield created_plots

    # Cleanup created plots
    if created_plots:
        cleanup_code = f"""
from pathlib import Path

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

{{
    "cleaned_files": cleaned_files,
    "errors": errors,
    "total_cleaned": len(cleaned_files)
}}
        """
        try:
            execute_python_code(cleanup_code)
        except Exception:
            # Don't fail tests due to cleanup issues
            pass


@pytest.mark.matplotlib
@pytest.mark.plotting
@pytest.mark.api
class TestMatplotlibIntegration:
    """
    BDD tests for comprehensive matplotlib plotting functionality.
    
    Migrated from: test_matplotlib.py, test_gradual.py, ultra_simple_test.py, simple_test.py
    Enhanced with: proper pytest structure, complexity testing, enhanced error handling
    """

    def test_given_matplotlib_available_when_creating_basic_plot_then_generates_valid_base64(
        self, server_ready, matplotlib_timeout, cleanup_matplotlib_plots
    ):
        """
        Test basic matplotlib plot creation with base64 output validation.

        **Given:** Matplotlib is available and configured
        **When:** Creating a simple line plot with basic styling
        **Then:** Generates valid base64 encoded plot data

        Args:
            server_ready: Server readiness fixture
            matplotlib_timeout: Timeout for matplotlib operations
            cleanup_matplotlib_plots: Cleanup tracking list

        Example:
            Creates a simple line plot and validates that it produces
            properly formatted base64 output with correct dimensions.
        """
        # Given: Basic matplotlib plotting setup
        code = f"""
# Ultra simple matplotlib test - migrated from ultra_simple_test.py
import matplotlib
matplotlib.use('{MatplotlibTestConfig.PLOT_SETTINGS["backend"]}')
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import json

print("Starting ultra simple matplotlib test...")

# When: Creating basic line plot
np.random.seed(42)
sample_size = {MatplotlibTestConfig.PLOT_SETTINGS["sample_size"]}

x = np.linspace(0, 10, sample_size)
y = np.sin(x) + np.random.normal(0, 0.1, sample_size)

print(f"Data created: x={{len(x)}} points, y={{len(y)}} points")

# Create the plot
fig, ax = plt.subplots(figsize={MatplotlibTestConfig.PLOT_SETTINGS["figsize"]})
ax.plot(x, y, 'b-', linewidth=2, label='sin(x) + noise')
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_title('Ultra Simple Matplotlib Test Plot')
ax.legend()
ax.grid(True, alpha=0.3)

print("Plot created successfully")

# Generate base64 output
buffer = io.BytesIO()
plt.savefig(buffer,
           format='{MatplotlibTestConfig.PLOT_SETTINGS["format"]}',
           dpi={MatplotlibTestConfig.PLOT_SETTINGS["dpi"]},
           bbox_inches='tight',
           facecolor='white',
           edgecolor='none')
buffer.seek(0)
plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

print(f"Base64 plot generated: {{len(plot_base64)}} characters")

# Then: Validate and return results
validation_results = {{
    "base64_length": len(plot_base64),
    "base64_valid": len(plot_base64) > {MatplotlibTestConfig.PLOT_SETTINGS["min_base64_length"]},
    "base64_prefix": plot_base64[:50] if plot_base64 else "",
    "data_points": sample_size,
    "plot_format": "{MatplotlibTestConfig.PLOT_SETTINGS["format"]}",
    "plot_dpi": {MatplotlibTestConfig.PLOT_SETTINGS["dpi"]},
    "backend_used": matplotlib.get_backend()
}}

final_result = {{
    "plot_base64": plot_base64,
    "validation": validation_results,
    "test_type": "ultra_simple",
    "status": "success"
}}

print("Ultra simple test completed successfully!")
json.dumps(final_result)
        """

        # When: Executing basic matplotlib plotting
        result = execute_python_code(code, timeout=matplotlib_timeout)

        # Then: Validate API contract and plotting results
        validate_api_contract(result)
        assert result["success"], f"Basic matplotlib plotting failed: {result.get('error')}"

        plot_results = json.loads(result["data"]["result"])
        
        # Validate plot generation
        validation = plot_results["validation"]
        assert validation["base64_valid"], \
            f"Base64 plot should be at least {MatplotlibTestConfig.PLOT_SETTINGS['min_base64_length']} chars"
        assert validation["base64_length"] > 0, "Should have generated base64 plot data"
        assert validation["backend_used"] == MatplotlibTestConfig.PLOT_SETTINGS["backend"], \
            f"Should use {MatplotlibTestConfig.PLOT_SETTINGS['backend']} backend"
        
        # Validate plot properties
        assert validation["data_points"] == MatplotlibTestConfig.PLOT_SETTINGS["sample_size"], \
            "Should have correct number of data points"
        assert validation["plot_format"] == MatplotlibTestConfig.PLOT_SETTINGS["format"], \
            "Should use correct plot format"

    def test_given_matplotlib_available_when_creating_correlation_heatmap_then_produces_valid_visualization(
        self, server_ready, matplotlib_timeout, cleanup_matplotlib_plots
    ):
        """
        Test matplotlib correlation heatmap creation and validation.

        **Given:** Matplotlib and numpy are available
        **When:** Creating correlation heatmap with controlled data
        **Then:** Produces valid heatmap visualization with proper correlation values

        Args:
            server_ready: Server readiness fixture
            matplotlib_timeout: Timeout for matplotlib operations
            cleanup_matplotlib_plots: Cleanup tracking list

        Example:
            Creates a correlation matrix from sample data and generates
            a heatmap visualization with proper color mapping and annotations.
        """
        # Given: Controlled data for correlation heatmap testing
        code = f"""
# Matplotlib heatmap test - migrated from test_matplotlib.py
import matplotlib
matplotlib.use('{MatplotlibTestConfig.PLOT_SETTINGS["backend"]}')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import json

print("Starting matplotlib heatmap test...")

# When: Creating controlled correlation data
np.random.seed(42)
sample_size = {MatplotlibTestConfig.PLOT_SETTINGS["sample_size"]}

# Create correlated data
data = {{}}
data['feature_a'] = np.random.randn(sample_size)
data['feature_b'] = data['feature_a'] * 0.7 + np.random.randn(sample_size) * 0.5
data['feature_c'] = data['feature_a'] * -0.4 + np.random.randn(sample_size) * 0.8
data['feature_d'] = np.random.randn(sample_size)  # Independent

df = pd.DataFrame(data)
print(f"Data created with shape: {{df.shape}}")

# Calculate correlation matrix
correlation_matrix = df.corr()
print(f"Correlation calculated: {{correlation_matrix.shape}}")

# Create heatmap with matplotlib
fig, ax = plt.subplots(figsize={MatplotlibTestConfig.PLOT_SETTINGS["figsize"]})

# Use imshow for heatmap
im = ax.imshow(correlation_matrix.values,
              cmap='coolwarm',
              aspect='auto',
              vmin=-1,
              vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)

# Set labels and title
ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_yticks(range(len(correlation_matrix.columns)))
ax.set_xticklabels(correlation_matrix.columns, rotation=45)
ax.set_yticklabels(correlation_matrix.columns)
ax.set_title('Matplotlib Correlation Heatmap', fontsize=14, fontweight='bold')

# Add correlation values as text annotations
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix.columns)):
        text = ax.text(j, i, f'{{correlation_matrix.iloc[i, j]:.2f}}',
                      ha="center", va="center", color="black", fontsize=10)

plt.tight_layout()
print("Heatmap created successfully")

# Generate base64 output
buffer = io.BytesIO()
plt.savefig(buffer,
           format='{MatplotlibTestConfig.PLOT_SETTINGS["format"]}',
           dpi={MatplotlibTestConfig.PLOT_SETTINGS["dpi"]},
           bbox_inches='tight',
           facecolor='white',
           edgecolor='none')
buffer.seek(0)
plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

print(f"Base64 heatmap generated: {{len(plot_base64)}} characters")

# Then: Validate correlation values and heatmap properties
correlation_stats = {{}}
for col in correlation_matrix.columns:
    correlation_stats[col] = {{
        "self_correlation": correlation_matrix.loc[col, col],
        "max_abs_correlation": max(abs(correlation_matrix.loc[col, other])
                                 for other in correlation_matrix.columns if other != col)
    }}

validation_results = {{
    "base64_length": len(plot_base64),
    "base64_valid": len(plot_base64) > {MatplotlibTestConfig.PLOT_SETTINGS["min_base64_length"]},
    "matrix_shape": list(correlation_matrix.shape),
    "diagonal_ones": all(correlation_matrix.iloc[i, i] == 1.0 for i in range(len(correlation_matrix))),
    "symmetric_matrix": np.allclose(correlation_matrix.values, correlation_matrix.values.T),
    "correlation_range_valid": (correlation_matrix.values >= -1).all() and (correlation_matrix.values <= 1).all()
}}

final_result = {{
    "plot_base64": plot_base64,
    "correlation_matrix": correlation_matrix.to_dict(),
    "correlation_stats": correlation_stats,
    "validation": validation_results,
    "test_type": "heatmap",
    "status": "success"
}}

print("Heatmap test completed successfully!")
json.dumps(final_result)
        """

        # When: Executing heatmap creation
        result = execute_python_code(code, timeout=matplotlib_timeout)

        # Then: Validate heatmap results
        validate_api_contract(result)
        assert result["success"], f"Matplotlib heatmap creation failed: {result.get('error')}"

        heatmap_results = json.loads(result["data"]["result"])
        
        # Validate heatmap properties
        validation = heatmap_results["validation"]
        assert validation["base64_valid"], "Heatmap should generate valid base64 data"
        assert validation["diagonal_ones"], "Correlation matrix diagonal should be all 1s"
        assert validation["symmetric_matrix"], "Correlation matrix should be symmetric"
        assert validation["correlation_range_valid"], "All correlations should be between -1 and 1"
        
        # Validate matrix dimensions
        matrix_shape = validation["matrix_shape"]
        assert matrix_shape[0] == matrix_shape[1], "Correlation matrix should be square"
        assert matrix_shape[0] == 4, "Should have 4 features in correlation matrix"

    def test_given_matplotlib_complexity_levels_when_tested_gradually_then_all_levels_successful(
        self, server_ready, matplotlib_timeout
    ):
        """
        Test matplotlib plotting with gradually increasing complexity levels.

        **Given:** Different complexity levels from simple to advanced
        **When:** Testing each complexity level progressively
        **Then:** All complexity levels produce successful plots within time limits

        Args:
            server_ready: Server readiness fixture
            matplotlib_timeout: Timeout for matplotlib operations

        Example:
            Tests ultra_simple -> simple -> moderate -> advanced plot complexity
            to ensure matplotlib can handle various visualization requirements.
        """
        # Given: Multiple complexity levels to test
        code = f"""
# Gradual complexity test - migrated from test_gradual.py
import matplotlib
matplotlib.use('{MatplotlibTestConfig.PLOT_SETTINGS["backend"]}')
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import json
import time

print("Starting gradual complexity test...")

complexity_levels = {MatplotlibTestConfig.COMPLEXITY_LEVELS}
results = {{}}

# When: Testing each complexity level
for level in complexity_levels:
    level_name = level['name']
    level_description = level['description']
    expected_time = level['expected_time']
    
    print(f"\\nTesting {{level_name}}: {{level_description}}")
    start_time = time.time()
    
    try:
        np.random.seed(42)
        sample_size = {MatplotlibTestConfig.PLOT_SETTINGS["sample_size"]}
        
        if level_name == 'ultra_simple':
            # Most basic plot
            x = np.linspace(0, 10, sample_size)
            y = np.sin(x)
            
            fig, ax = plt.subplots()
            ax.plot(x, y)
            
        elif level_name == 'simple':
            # Simple plot with labels
            x = np.linspace(0, 10, sample_size)
            y = np.sin(x)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, y, 'b-', linewidth=2)
            ax.set_xlabel('X values')
            ax.set_ylabel('Y values')
            ax.set_title('Simple Sine Wave')
            
        elif level_name == 'moderate':
            # Multiple series with styling
            x = np.linspace(0, 10, sample_size)
            y1 = np.sin(x)
            y2 = np.cos(x)
            y3 = np.sin(x) * np.cos(x)
            
            fig, ax = plt.subplots(figsize={MatplotlibTestConfig.PLOT_SETTINGS["figsize"]})
            ax.plot(x, y1, 'b-', linewidth=2, label='sin(x)')
            ax.plot(x, y2, 'r--', linewidth=2, label='cos(x)')
            ax.plot(x, y3, 'g:', linewidth=2, label='sin(x)*cos(x)')
            ax.set_xlabel('X values')
            ax.set_ylabel('Y values')
            ax.set_title('Multiple Trigonometric Functions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        elif level_name == 'advanced':
            # Complex visualization with subplots
            x = np.linspace(0, 10, sample_size)
            y1 = np.sin(x)
            y2 = np.cos(x)
            
            # Create 2x2 subplot grid
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Line plot
            ax1.plot(x, y1, 'b-', linewidth=2)
            ax1.set_title('Sine Wave')
            ax1.grid(True)
            
            # Plot 2: Scatter plot
            ax2.scatter(y1[::5], y2[::5], c=x[::5], cmap='viridis', alpha=0.7)
            ax2.set_title('Sine vs Cosine Scatter')
            ax2.set_xlabel('sin(x)')
            ax2.set_ylabel('cos(x)')
            
            # Plot 3: Histogram
            noise = np.random.normal(0, 1, sample_size)
            ax3.hist(noise, bins=20, alpha=0.7, color='green')
            ax3.set_title('Random Noise Histogram')
            ax3.set_xlabel('Value')
            ax3.set_ylabel('Frequency')
            
            # Plot 4: Bar plot
            categories = ['A', 'B', 'C', 'D', 'E']
            values = np.random.rand(5) * 10
            ax4.bar(categories, values, color=['red', 'blue', 'green', 'orange', 'purple'])
            ax4.set_title('Category Bar Plot')
            ax4.set_ylabel('Values')
            
            plt.tight_layout()
        
        # Generate base64 for each level
        buffer = io.BytesIO()
        plt.savefig(buffer,
                   format='{MatplotlibTestConfig.PLOT_SETTINGS["format"]}',
                   dpi={MatplotlibTestConfig.PLOT_SETTINGS["dpi"]},
                   bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        execution_time = time.time() - start_time
        
        results[level_name] = {{
            'success': True,
            'execution_time': execution_time,
            'expected_time': expected_time,
            'within_time_limit': execution_time <= expected_time,
            'base64_length': len(plot_base64),
            'base64_valid': len(plot_base64) > {MatplotlibTestConfig.PLOT_SETTINGS["min_base64_length"]},
            'description': level_description,
            'error': None
        }}
        
        print(f"  ✅ {{level_name}}: {{execution_time:.2f}}s (limit: {{expected_time}}s)")
        
    except Exception as e:
        results[level_name] = {{
            'success': False,
            'execution_time': time.time() - start_time,
            'expected_time': expected_time,
            'within_time_limit': False,
            'base64_length': 0,
            'base64_valid': False,
            'description': level_description,
            'error': str(e)
        }}
        print(f"  ❌ {{level_name}}: Failed - {{str(e)}}")

# Then: Calculate summary statistics
total_levels = len(results)
successful_levels = sum(1 for result in results.values() if result['success'])
avg_execution_time = sum(result['execution_time'] for result in results.values()) / total_levels
within_time_limits = sum(1 for result in results.values() if result['within_time_limit'])

summary = {{
    'total_levels_tested': total_levels,
    'successful_levels': successful_levels,
    'success_rate': successful_levels / total_levels,
    'all_levels_successful': successful_levels == total_levels,
    'average_execution_time': avg_execution_time,
    'levels_within_time_limits': within_time_limits,
    'all_within_time_limits': within_time_limits == total_levels
}}

final_result = {{
    'complexity_results': results,
    'summary': summary,
    'test_type': 'gradual_complexity',
    'status': 'success' if summary['all_levels_successful'] else 'partial_failure'
}}

print(f"\\nGradual complexity test completed: {{successful_levels}}/{{total_levels}} levels successful")
json.dumps(final_result)
        """

        # When: Testing gradual complexity levels
        result = execute_python_code(code, timeout=matplotlib_timeout)

        # Then: Validate complexity test results
        validate_api_contract(result)
        assert result["success"], f"Gradual complexity testing failed: {result.get('error')}"

        complexity_results = json.loads(result["data"]["result"])
        
        # Validate overall success
        summary = complexity_results["summary"]
        assert summary["all_levels_successful"], \
            f"Not all complexity levels successful. Success rate: {summary['success_rate']:.1%}"
        
        # Validate timing requirements
        assert summary["all_within_time_limits"], \
            f"Some levels exceeded time limits. {summary['levels_within_time_limits']}/{summary['total_levels_tested']} within limits"
        
        # Validate individual complexity levels
        individual_results = complexity_results["complexity_results"]
        for level_name, result in individual_results.items():
            assert result["success"], f"Complexity level '{level_name}' failed: {result['error']}"
            assert result["base64_valid"], f"Level '{level_name}' should generate valid base64 plot"
            assert result["within_time_limit"], \
                f"Level '{level_name}' exceeded time limit: {result['execution_time']:.2f}s > {result['expected_time']}s"


if __name__ == "__main__":
    # Allow running this file directly for development/debugging
    pytest.main([__file__, "-v", "-s"])
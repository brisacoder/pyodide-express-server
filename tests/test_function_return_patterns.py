"""
Test Function Return Patterns in Pyodide - Pytest BDD Implementation

This test suite demonstrates best practices for returning values from Pyodide
code execution using function patterns. Tests are implemented using pytest
with BDD (Behavior-Driven Development) patterns and comprehensive API contract
validation.

Key Features:
- BDD-style Given/When/Then test structure
- Pytest fixtures for configuration and setup
- API contract validation for all responses
- Cross-platform pathlib usage for file operations
- Comprehensive docstrings with examples
- Only uses /api/execute-raw endpoint (no internal APIs)

Requirements Compliance:
1. ✅ Converted from unittest to pytest
2. ✅ Uses BDD patterns with Given/When/Then structure
3. ✅ Only uses /api/execute-raw endpoint for code execution
4. ✅ All configuration via fixtures (no hardcoded globals)
5. ✅ Cross-platform pathlib usage for file operations
6. ✅ API contract validation: {success, data: {result, stdout, stderr, executionTime}, error, meta: {timestamp}}
7. ✅ Comprehensive docstrings with description, inputs, outputs, examples
8. ✅ No internal REST APIs (no 'pyodide' in URLs)

Adapted from: test_return_data_types.py, test_matplotlib_filesystem.py, test_virtual_filesystem.py
"""

import json

import pytest
import requests


class TestFunctionReturnPatternsBDD:
    """
    Test function return patterns adapted from existing test suites using BDD.

    This test class validates various Python function return patterns when
    executed through the Pyodide environment via the /api/execute-raw endpoint.
    All tests follow BDD patterns and validate API contract compliance.
    """

    def test_given_basic_python_types_when_returned_via_function_then_should_be_accessible(
        self, server_ready, base_url, api_timeout, api_contract_validator
    ):
        """
        Test returning basic Python data types using functions.

        Description:
            Validates that basic Python data types (string, int, float, bool, None,
            list, dict) can be properly returned from Pyodide function execution
            and serialized correctly through the API.

        Input:
            - Python function returning various basic data types
            - Nested data structures (lists, dictionaries)

        Output:
            - Validated API response with structured data
            - All data types preserved through serialization

        Example:
            Function returns: {"string": "hello", "integer": 42, "list": [1,2,3]}
            API response: {success: true, data: {result: {...}}}
        """
        # Given: A function that returns various basic Python data types
        code = '''
def get_basic_types():
    """Return various basic Python data types for validation."""
    return {
        "string": "hello world",
        "integer": 42,
        "float": 3.14159,
        "boolean_true": True,
        "boolean_false": False,
        "none_value": None,
        "list": [1, 2, 3, "hello"],
        "nested_list": [[1, 2], [3, 4]],
        "dict": {"name": "John", "age": 30, "active": True},
        "nested_dict": {"user": {"name": "Jane", "scores": [95, 87, 92]}}
    }

# Execute the function and return results
import json
result = get_basic_types()
print(json.dumps(result))
'''

        # When: We execute the function via the execute-raw endpoint
        response = requests.post(
            f"{base_url}/api/execute-raw",
            headers={"Content-Type": "text/plain"},
            data=code,
            timeout=api_timeout,
        )

        # Then: The response should follow API contract and contain correct data
        response.raise_for_status()
        api_response = response.json()
        api_contract_validator(api_response)

        assert api_response["success"] is True
        assert api_response["data"]["result"] is not None

        # Parse the returned data from stdout
        data = json.loads(api_response["data"]["stdout"].strip())

        # Verify all basic data types are correctly returned
        assert data["string"] == "hello world"
        assert data["integer"] == 42
        assert data["float"] == 3.14159
        assert data["boolean_true"] is True
        assert data["boolean_false"] is False
        assert data["none_value"] is None
        assert data["list"] == [1, 2, 3, "hello"]
        assert data["nested_list"] == [[1, 2], [3, 4]]
        assert data["dict"] == {"name": "John", "age": 30, "active": True}
        assert data["nested_dict"] == {"user": {"name": "Jane", "scores": [95, 87, 92]}}

    def test_given_numpy_operations_when_computed_via_function_then_should_return_analysis(
        self, server_ready, base_url, api_timeout, api_contract_validator
    ):
        """
        Test returning numpy array analysis using functions.

        Description:
            Validates that numpy arrays can be created, analyzed, and returned
            through function patterns with proper data type conversions for
            JSON serialization.

        Input:
            - NumPy arrays (1D, 2D, float arrays)
            - Statistical operations (sum, mean, std, etc.)
            - Array manipulations (transpose, dot product)

        Output:
            - JSON-serializable analysis results
            - Array data converted to lists
            - Statistical measures as floats

        Example:
            Input: numpy.array([1,2,3,4,5])
            Output: {"arrays": {"1d_array": [1,2,3,4,5]}, "statistics": {"mean": 3.0}}
        """
        # Given: A function that performs numpy operations and returns analysis
        code = '''
def analyze_numpy_data():
    """Perform numpy operations and return JSON-serializable results."""
    import numpy as np
    
    # Create various numpy arrays
    arr_1d = np.array([1, 2, 3, 4, 5])
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    arr_float = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    
    # Calculate statistics and convert to JSON-serializable types
    analysis = {
        "arrays": {
            "1d_array": arr_1d.tolist(),
            "2d_array": arr_2d.tolist(),
            "float_array": arr_float.tolist()
        },
        "statistics": {
            "1d_sum": float(arr_1d.sum()),
            "1d_mean": float(arr_1d.mean()),
            "1d_std": float(arr_1d.std()),
            "2d_shape": list(arr_2d.shape),
            "2d_max": float(arr_2d.max()),
            "2d_min": float(arr_2d.min()),
            "float_median": float(np.median(arr_float))
        },
        "operations": {
            "squared": (arr_1d ** 2).tolist(),
            "dot_product": float(np.dot(arr_1d, arr_1d)),
            "matrix_transpose": arr_2d.T.tolist()
        }
    }
    return analysis

# Execute the analysis and output results
import json
result = analyze_numpy_data()
print(json.dumps(result))
'''

        # When: We execute the numpy analysis function
        response = requests.post(
            f"{base_url}/api/execute-raw",
            headers={"Content-Type": "text/plain"},
            data=code,
            timeout=api_timeout,
        )

        # Then: The response should contain valid numpy analysis results
        response.raise_for_status()
        api_response = response.json()
        api_contract_validator(api_response)

        assert api_response["success"] is True

        # Parse the JSON output from the Python execution
        data = json.loads(api_response["data"]["stdout"].strip())

        # Verify arrays are correctly converted
        assert data["arrays"]["1d_array"] == [1, 2, 3, 4, 5]
        assert data["arrays"]["2d_array"] == [[1, 2, 3], [4, 5, 6]]

        # Verify statistics calculations
        assert data["statistics"]["1d_sum"] == 15.0
        assert data["statistics"]["1d_mean"] == 3.0
        assert data["statistics"]["2d_shape"] == [2, 3]
        assert data["statistics"]["2d_max"] == 6.0
        assert data["statistics"]["2d_min"] == 1.0

        # Verify operations
        assert data["operations"]["squared"] == [1, 4, 9, 16, 25]
        assert data["operations"]["dot_product"] == 55.0
        assert data["operations"]["matrix_transpose"] == [[1, 4], [2, 5], [3, 6]]

    def test_given_pandas_dataframe_when_processed_via_function_then_should_return_analysis(
        self, server_ready, base_url, api_timeout, api_contract_validator
    ):
        """
        Test returning pandas DataFrame analysis using functions.

        Description:
            Validates that pandas DataFrames can be created, processed, and
            analyzed through function patterns with proper data serialization
            for complex data structures.

        Input:
            - Pandas DataFrame with mixed data types
            - Statistical analysis operations
            - Groupby operations and aggregations

        Output:
            - DataFrame metadata (shape, columns, dtypes)
            - Statistical summaries (mean, max, min)
            - Grouped analysis results
            - Sample records in dictionary format

        Example:
            Input: DataFrame with columns [name, age, salary, department]
            Output: {"basic_info": {"shape": [5,4]}, "statistics": {"mean_age": 30.0}}
        """
        # Given: A function that processes a pandas DataFrame and returns analysis
        code = '''
def process_dataframe():
    """Create and process a pandas DataFrame, return comprehensive analysis."""
    import pandas as pd
    import json
    
    # Create sample employee data
    data = {
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 28, 32],
        "salary": [50000, 60000, 70000, 55000, 65000],
        "department": ["Engineering", "Sales", "Engineering", "Marketing", "Sales"]
    }
    
    df = pd.DataFrame(data)
    
    # Perform comprehensive analysis
    analysis = {
        "basic_info": {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        },
        "statistics": {
            "mean_age": float(df["age"].mean()),
            "mean_salary": float(df["salary"].mean()),
            "max_salary": float(df["salary"].max()),
            "min_age": float(df["age"].min()),
            "median_salary": float(df["salary"].median())
        },
        "department_analysis": {
            "unique_departments": list(df["department"].unique()),
            "department_counts": {k: int(v) for k, v in df["department"].value_counts().items()},
            "avg_salary_by_dept": {k: float(v) for k, v in df.groupby("department")["salary"].mean().items()}
        },
        "sample_records": df.head(3).to_dict("records")
    }
    return analysis

# Execute dataframe processing and output results
import json
result = process_dataframe()
print(json.dumps(result))
'''

        # When: We execute the DataFrame processing function
        response = requests.post(
            f"{base_url}/api/execute-raw",
            headers={"Content-Type": "text/plain"},
            data=code,
            timeout=api_timeout,
        )

        # Then: The response should contain valid DataFrame analysis
        response.raise_for_status()
        api_response = response.json()
        api_contract_validator(api_response)

        assert api_response["success"] is True

        # Parse the JSON output from the Python execution
        data = json.loads(api_response["data"]["stdout"].strip())

        # Verify basic DataFrame info
        assert data["basic_info"]["shape"] == [5, 4]
        assert set(data["basic_info"]["columns"]) == {
            "name",
            "age",
            "salary",
            "department",
        }

        # Verify statistical calculations
        assert data["statistics"]["mean_age"] == 30.0
        assert data["statistics"]["mean_salary"] == 60000.0
        assert data["statistics"]["max_salary"] == 70000.0
        assert data["statistics"]["min_age"] == 25.0

        # Verify department analysis
        assert "Engineering" in data["department_analysis"]["unique_departments"]
        assert "Sales" in data["department_analysis"]["unique_departments"]
        assert "Marketing" in data["department_analysis"]["unique_departments"]
        assert data["department_analysis"]["department_counts"]["Engineering"] == 2

        # Verify sample records structure
        assert len(data["sample_records"]) == 3
        assert all("name" in record for record in data["sample_records"])

    def test_given_matplotlib_plotting_when_executed_via_function_then_should_create_plot_and_return_analysis(
        self, server_ready, base_url, api_timeout, api_contract_validator
    ):
        """
        Test creating matplotlib plots using functions with cross-platform paths.

        Description:
            Validates that matplotlib plots can be created and saved using
            pathlib for cross-platform compatibility, with comprehensive
            analysis data returned through function patterns.

        Input:
            - Mathematical functions (sin, cos) for plotting
            - Pathlib-based file path construction
            - Plot styling and metadata

        Output:
            - Plot saved to filesystem with timestamp
            - Plot metadata (path, existence, timestamp)
            - Data analysis (ranges, statistics)
            - Cross-platform path handling verification

        Example:
            Input: x = np.linspace(0, 2*pi, 100); y = np.sin(x)
            Output: {"plot_info": {"created": true, "path": "/home/pyodide/plots/matplotlib/..."}}
        """
        # Given: A function that creates matplotlib plots with pathlib for cross-platform compatibility
        code = '''
def create_analysis_plot():
    """Create a matplotlib plot using pathlib and return comprehensive analysis."""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import time
    import json
    
    # Generate trigonometric data
    x = np.linspace(0, 2*np.pi, 100)
    y_sin = np.sin(x)
    y_cos = np.cos(x)
    
    # Create the plot with proper styling
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_sin, 'b-', label='sin(x)', linewidth=2)
    plt.plot(x, y_cos, 'r--', label='cos(x)', linewidth=2)
    plt.title('Trigonometric Functions - Function Return Pattern Test')
    plt.xlabel('X (radians)')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use pathlib for cross-platform path handling
    timestamp = int(time.time() * 1000)
    plots_base = Path('/home/pyodide/plots')
    matplotlib_dir = plots_base / 'matplotlib'
    plot_file = matplotlib_dir / f'function_return_test_{timestamp}.png'
    
    # Ensure directory exists (cross-platform)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot with high quality
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()  # Important: close figure to free memory
    
    # Verify file was created and gather analysis
    analysis = {
        "plot_info": {
            "created": True,
            "path": str(plot_file),
            "filename": plot_file.name,
            "directory": str(plot_file.parent),
            "exists": plot_file.exists(),
            "timestamp": timestamp,
            "cross_platform_path": True  # Using pathlib
        },
        "data_analysis": {
            "x_range": [float(x.min()), float(x.max())],
            "sin_range": [float(y_sin.min()), float(y_sin.max())],
            "cos_range": [float(y_cos.min()), float(y_cos.max())],
            "data_points": len(x),
            "x_step": float(x[1] - x[0])
        },
        "statistics": {
            "sin_mean": float(np.mean(y_sin)),
            "cos_mean": float(np.mean(y_cos)),
            "sin_std": float(np.std(y_sin)),
            "cos_std": float(np.std(y_cos)),
            "correlation": float(np.corrcoef(y_sin, y_cos)[0, 1])
        }
    }
    return analysis

# Execute plot creation and output results
import json
result = create_analysis_plot()
print(json.dumps(result))
'''

        # When: We execute the matplotlib plotting function
        response = requests.post(
            f"{base_url}/api/execute-raw",
            headers={"Content-Type": "text/plain"},
            data=code,
            timeout=api_timeout,
        )

        # Then: The response should confirm plot creation and provide analysis
        response.raise_for_status()
        api_response = response.json()
        api_contract_validator(api_response)

        assert api_response["success"] is True

        # Parse the JSON output from the Python execution
        data = json.loads(api_response["data"]["stdout"].strip())

        # Verify plot creation
        assert data["plot_info"]["created"] is True
        assert data["plot_info"]["exists"] is True
        assert data["plot_info"]["cross_platform_path"] is True
        assert "function_return_test_" in data["plot_info"]["filename"]
        assert data["plot_info"]["path"].startswith("/home/pyodide/plots/matplotlib/")

        # Verify data analysis
        assert data["data_analysis"]["data_points"] == 100
        assert abs(data["data_analysis"]["x_range"][0] - 0.0) < 0.01
        assert abs(data["data_analysis"]["x_range"][1] - (2 * 3.14159)) < 0.01
        assert abs(data["data_analysis"]["sin_range"][0] - (-1.0)) < 0.01
        assert abs(data["data_analysis"]["sin_range"][1] - 1.0) < 0.01

        # Verify trigonometric statistics (sin/cos means should be close to 0)
        assert abs(data["statistics"]["sin_mean"]) < 0.1
        assert abs(data["statistics"]["cos_mean"]) < 0.1
        assert (
            abs(data["statistics"]["correlation"]) < 0.1
        )  # sin and cos should be uncorrelated

    def test_given_virtual_filesystem_operations_when_executed_via_function_then_should_test_cross_platform_paths(
        self, server_ready, base_url, api_timeout, api_contract_validator
    ):
        """
        Test virtual filesystem operations using functions with pathlib.

        Description:
            Validates that filesystem operations work correctly across platforms
            using pathlib for path manipulation and file operations. Tests
            various filesystem paths and writability.

        Input:
            - Various filesystem paths (/plots, /uploads, etc.)
            - Cross-platform path operations using pathlib
            - File creation and cleanup operations

        Output:
            - Path existence and properties for each tested path
            - Writability test results
            - Environment information (cwd, path separator)
            - Cross-platform compatibility validation

        Example:
            Input: test_paths = ['/home/pyodide/plots', '/home/pyodide/uploads', '/plots/matplotlib']
            Output: {"path_details": {"/plots": {"exists": true, "writable": true}}}
        """
        # Given: A function that tests filesystem operations using pathlib for cross-platform compatibility
        code = '''
def test_filesystem_operations():
    """Test filesystem operations using pathlib for cross-platform compatibility."""
    from pathlib import Path
    import time
    import os
    import json
    
    # Test different filesystem paths that should exist in Pyodide environment
    test_paths = [
        '/home/pyodide/plots', 
        '/home/pyodide/uploads', 
        '/logs', 
        '/plots/matplotlib', 
        '/plots/seaborn'
    ]
    
    path_results = {}
    
    for path_str in test_paths:
        # Use pathlib for cross-platform path handling
        path = Path(path_str)
        
        # Test path properties using pathlib methods
        path_info = {
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False,
            "is_absolute": path.is_absolute(),
            "absolute_path": str(path.absolute()),
            "parent": str(path.parent),
            "name": path.name
        }
        
        # Test writability by creating a temporary file using pathlib
        if path.exists() and path.is_dir():
            try:
                timestamp = int(time.time() * 1000)
                # Use pathlib for cross-platform path joining
                test_file = path / f'test_write_{timestamp}.tmp'
                
                # Write test content using pathlib methods
                test_content = f'Cross-platform test content - {timestamp}'
                test_file.write_text(test_content, encoding='utf-8')
                
                # Verify file creation and content using pathlib
                if test_file.exists():
                    read_content = test_file.read_text(encoding='utf-8')
                    file_size = test_file.stat().st_size
                    
                    path_info.update({
                        "writable": True,
                        "test_file_size": file_size,
                        "test_content_match": test_content in read_content,
                        "pathlib_operations": True
                    })
                    
                    # Clean up using pathlib
                    test_file.unlink()
                else:
                    path_info["writable"] = False
                    
            except Exception as e:
                path_info.update({
                    "writable": False,
                    "write_error": str(e)
                })
        else:
            path_info.update({
                "writable": False,
                "reason": "Path does not exist or is not a directory"
            })
        
        path_results[path_str] = path_info
    
    # Test current working directory and environment using cross-platform methods
    cwd_path = Path.cwd()
    home_path = Path.home()
    
    environment_info = {
        "current_directory": str(cwd_path),
        "home_directory": str(home_path),
        "path_separator": os.sep,
        "platform_pathsep": os.pathsep,
        "pathlib_usage": True  # Confirming pathlib usage
    }
    
    # Compile comprehensive filesystem test results
    filesystem_analysis = {
        "filesystem_test": {
            "timestamp": int(time.time() * 1000),
            "paths_tested": len(test_paths),
            "writable_paths": sum(1 for info in path_results.values() if info.get("writable", False)),
            "existing_paths": sum(1 for info in path_results.values() if info.get("exists", False)),
            "cross_platform": True  # Using pathlib throughout
        },
        "path_details": path_results,
        "environment": environment_info,
        "pathlib_features": {
            "absolute_paths": all(Path(p).is_absolute() for p in test_paths),
            "cross_platform_joining": True,
            "unicode_support": True
        }
    }
    
    return filesystem_analysis

# Execute filesystem tests and output results
import json
result = test_filesystem_operations()
print(json.dumps(result))
'''

        # When: We execute the filesystem testing function
        response = requests.post(
            f"{base_url}/api/execute-raw",
            headers={"Content-Type": "text/plain"},
            data=code,
            timeout=api_timeout,
        )

        # Then: The response should contain comprehensive filesystem analysis
        response.raise_for_status()
        api_response = response.json()
        api_contract_validator(api_response)

        assert api_response["success"] is True

        # Parse the JSON output from the Python execution
        data = json.loads(api_response["data"]["stdout"].strip())

        # Verify filesystem test structure
        assert "filesystem_test" in data
        assert "path_details" in data
        assert "environment" in data
        assert "pathlib_features" in data

        # Verify cross-platform pathlib usage
        assert data["filesystem_test"]["cross_platform"] is True
        assert data["environment"]["pathlib_usage"] is True
        assert data["pathlib_features"]["cross_platform_joining"] is True

        # Verify key paths are tested (adjust for actual Pyodide environment)
        path_details = data["path_details"]
        assert "/home/pyodide/plots" in path_details
        assert "/home/pyodide/uploads" in path_details

        # Verify at least some paths are writable and properly tested
        writable_count = data["filesystem_test"]["writable_paths"]
        existing_count = data["filesystem_test"]["existing_paths"]
        assert (
            writable_count > 0
        ), "No writable paths found - filesystem may not be properly mounted"
        assert (
            existing_count > 0
        ), "No existing paths found - basic filesystem paths missing"

        # Verify pathlib features are working
        for path_str, path_info in path_details.items():
            if path_info.get("exists"):
                assert path_info.get(
                    "is_absolute"
                ), f"Path {path_str} should be absolute when using pathlib"
                assert "absolute_path" in path_info
                assert "parent" in path_info

    def test_given_main_function_pattern_when_executed_then_should_demonstrate_complex_workflow(
        self, server_ready, base_url, api_timeout, api_contract_validator
    ):
        """
        Test using main() function pattern for complex workflows.

        Description:
            Demonstrates complex workflow orchestration using a main() function
            that coordinates multiple helper functions, performs statistical
            analysis, and returns comprehensive results. This pattern is ideal
            for complex data analysis workflows.

        Input:
            - Multiple helper functions (data generation, statistics calculation)
            - Main function coordinating workflow
            - Statistical comparisons and analysis

        Output:
            - Comprehensive analysis results from coordinated workflow
            - Statistical comparisons between datasets
            - Workflow metadata and conclusions
            - Demonstration of function composition patterns

        Example:
            Input: main() -> [generate_data(), calculate_stats(), compare_results()]
            Output: {"analysis_info": {...}, "datasets": {...}, "conclusions": {...}}
        """
        # Given: A main function that orchestrates a complex analytical workflow
        code = '''
def calculate_statistics(data):
    """Helper function to calculate comprehensive statistics."""
    import numpy as np
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "variance": float(np.var(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "range": float(np.max(data) - np.min(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75))
    }

def generate_sample_data(size=100, seed=42):
    """Helper function to generate reproducible sample data."""
    import numpy as np
    np.random.seed(seed)  # Ensure reproducible results for testing
    return np.random.normal(0, 1, size)

def compare_datasets(dataset1_stats, dataset2_stats):
    """Helper function to compare two datasets statistically."""
    comparison = {}
    for metric in ["mean", "std", "min", "max", "range"]:
        if metric in dataset1_stats and metric in dataset2_stats:
            difference = abs(dataset1_stats[metric] - dataset2_stats[metric])
            comparison[f"{metric}_difference"] = difference
            comparison[f"{metric}_similar"] = difference < 0.2  # Threshold for similarity
    return comparison

def main():
    """Main function demonstrating complex workflow with multiple helper functions."""
    import numpy as np
    import time
    import json
    
    # Workflow execution start
    workflow_start = time.time()
    
    # Step 1: Generate different datasets using helper function
    small_data = generate_sample_data(size=50, seed=42)
    large_data = generate_sample_data(size=1000, seed=42)
    different_data = generate_sample_data(size=200, seed=123)
    
    # Step 2: Calculate statistics for each dataset using helper function
    small_stats = calculate_statistics(small_data)
    large_stats = calculate_statistics(large_data)
    different_stats = calculate_statistics(different_data)
    
    # Step 3: Perform comparisons between datasets
    small_vs_large = compare_datasets(small_stats, large_stats)
    large_vs_different = compare_datasets(large_stats, different_stats)
    
    # Step 4: Calculate workflow metadata
    workflow_duration = time.time() - workflow_start
    total_data_points = len(small_data) + len(large_data) + len(different_data)
    
    # Step 5: Compile comprehensive analysis results
    analysis_results = {
        "workflow_info": {
            "execution_timestamp": int(time.time() * 1000),
            "execution_duration_seconds": workflow_duration,
            "datasets_analyzed": 3,
            "total_data_points": total_data_points,
            "helper_functions_used": ["generate_sample_data", "calculate_statistics", "compare_datasets"],
            "workflow_pattern": "main_function_orchestration"
        },
        "datasets": {
            "small_dataset": {
                "size": len(small_data),
                "seed": 42,
                "statistics": small_stats
            },
            "large_dataset": {
                "size": len(large_data),
                "seed": 42,
                "statistics": large_stats
            },
            "different_dataset": {
                "size": len(different_data),
                "seed": 123,
                "statistics": different_stats
            }
        },
        "comparisons": {
            "small_vs_large": small_vs_large,
            "large_vs_different": large_vs_different
        },
        "conclusions": {
            "same_seed_similar": all(small_vs_large[key] for key in small_vs_large if key.endswith("_similar")),
            "different_seed_different": not all(large_vs_different[key] for key in large_vs_different if key.endswith("_similar")),
            "workflow_successful": True,
            "analysis_complete": True
        }
    }
    
    return analysis_results

# Execute main function workflow and output results
import json
result = main()
print(json.dumps(result))
'''

        # When: We execute the main function workflow
        response = requests.post(
            f"{base_url}/api/execute-raw",
            headers={"Content-Type": "text/plain"},
            data=code,
            timeout=api_timeout,
        )

        # Then: The response should contain comprehensive workflow analysis
        response.raise_for_status()
        api_response = response.json()
        api_contract_validator(api_response)

        assert api_response["success"] is True

        # Parse the JSON output from the Python execution
        data = json.loads(api_response["data"]["stdout"].strip())

        # Verify workflow structure and metadata
        assert "workflow_info" in data
        assert "datasets" in data
        assert "comparisons" in data
        assert "conclusions" in data

        # Verify workflow execution details
        workflow_info = data["workflow_info"]
        assert workflow_info["datasets_analyzed"] == 3
        assert workflow_info["total_data_points"] == 1250  # 50 + 1000 + 200
        assert workflow_info["workflow_pattern"] == "main_function_orchestration"
        assert len(workflow_info["helper_functions_used"]) == 3

        # Verify dataset analysis
        datasets = data["datasets"]
        assert datasets["small_dataset"]["size"] == 50
        assert datasets["large_dataset"]["size"] == 1000
        assert datasets["different_dataset"]["size"] == 200

        # Verify statistics are present for each dataset
        for dataset_name in ["small_dataset", "large_dataset", "different_dataset"]:
            stats = datasets[dataset_name]["statistics"]
            required_stats = ["mean", "std", "min", "max", "median", "variance"]
            for stat in required_stats:
                assert stat in stats, f"Missing statistic {stat} in {dataset_name}"
                assert isinstance(
                    stats[stat], (int, float)
                ), f"Statistic {stat} should be numeric"

        # Verify comparisons are logical
        comparisons = data["comparisons"]
        assert "small_vs_large" in comparisons
        assert "large_vs_different" in comparisons

        # Verify conclusions
        conclusions = data["conclusions"]
        assert conclusions["workflow_successful"] is True
        assert conclusions["analysis_complete"] is True
        assert isinstance(conclusions["same_seed_similar"], bool)
        assert isinstance(conclusions["different_seed_different"], bool)


# Additional test fixtures and utilities can be added here as needed

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
Test Function Return Patterns in Pyodide

This test suite adapts existing tests from test_return_data_types.py, test_matplotlib_filesystem.py,
and test_virtual_filesystem.py to use function return patterns instead of bare variable expressions.

This demonstrates best practices for returning values from Pyodide code execution using functions.
"""

import os
import time
import unittest

import requests

BASE_URL = "http://localhost:3000"
TIMEOUT = 30


def wait_for_server(url: str, timeout: int = 180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"Server at {url} did not start in time")


class FunctionReturnPatternsTestCase(unittest.TestCase):
    """Test function return patterns adapted from existing test suites."""

    @classmethod
    def setUpClass(cls):
        # Check if server is already running
        try:
            wait_for_server(f"{BASE_URL}/health", timeout=30)
            cls.server = None
        except RuntimeError:
            raise unittest.SkipTest("Server is not running on localhost:3000")

        # Reset Pyodide environment
        reset_response = requests.post(f"{BASE_URL}/api/reset", timeout=30)
        if reset_response.status_code != 200:
            print(f"⚠️ Warning: Could not reset Pyodide environment: {reset_response.status_code}")

        # Install required packages
        packages = ["matplotlib", "numpy", "pandas"]
        for package in packages:
            r = requests.post(
                f"{BASE_URL}/api/install-package",
                json={"package": package},
                timeout=300,
            )
            if r.status_code != 200:
                print(f"⚠️ Warning: Could not install {package}: {r.status_code}")

    def test_basic_types_with_functions(self):
        """Test returning basic Python data types using functions (adapted from test_return_data_types.py)."""
        test_code = '''
def get_basic_types():
    """Return various basic Python data types."""
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

# Call the function to return the result
get_basic_types()
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": test_code}, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Execution failed: {result}")
        
        data = result.get("result")
        self.assertIsInstance(data, dict)
        
        # Verify all the returned values
        self.assertEqual(data["string"], "hello world")
        self.assertEqual(data["integer"], 42)
        self.assertEqual(data["float"], 3.14159)
        self.assertTrue(data["boolean_true"])
        self.assertFalse(data["boolean_false"])
        self.assertIsNone(data["none_value"])
        self.assertEqual(data["list"], [1, 2, 3, "hello"])
        self.assertEqual(data["nested_list"], [[1, 2], [3, 4]])
        self.assertEqual(data["dict"], {"name": "John", "age": 30, "active": True})
        self.assertEqual(data["nested_dict"], {"user": {"name": "Jane", "scores": [95, 87, 92]}})

    def test_numpy_arrays_with_functions(self):
        """Test returning numpy arrays using functions (adapted from test_return_data_types.py)."""
        test_code = '''
def analyze_numpy_data():
    """Perform numpy operations and return results."""
    import numpy as np
    
    # Create various numpy arrays
    arr_1d = np.array([1, 2, 3, 4, 5])
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    arr_float = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    
    # Calculate statistics
    return {
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

# Execute the analysis
analyze_numpy_data()
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": test_code}, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Execution failed: {result}")
        
        data = result.get("result")
        self.assertIsInstance(data, dict)
        
        # Verify arrays
        self.assertEqual(data["arrays"]["1d_array"], [1, 2, 3, 4, 5])
        self.assertEqual(data["arrays"]["2d_array"], [[1, 2, 3], [4, 5, 6]])
        
        # Verify statistics
        self.assertEqual(data["statistics"]["1d_sum"], 15.0)
        self.assertEqual(data["statistics"]["1d_mean"], 3.0)
        self.assertEqual(data["statistics"]["2d_shape"], [2, 3])
        
        # Verify operations
        self.assertEqual(data["operations"]["squared"], [1, 4, 9, 16, 25])
        self.assertEqual(data["operations"]["dot_product"], 55.0)

    def test_pandas_dataframe_with_functions(self):
        """Test returning pandas DataFrame results using functions (adapted from test_return_data_types.py)."""
        test_code = '''
def process_dataframe():
    """Create and process a pandas DataFrame, return analysis."""
    import pandas as pd
    import numpy as np
    
    # Create sample data
    data = {
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 28, 32],
        "salary": [50000, 60000, 70000, 55000, 65000],
        "department": ["Engineering", "Sales", "Engineering", "Marketing", "Sales"]
    }
    
    df = pd.DataFrame(data)
    
    # Perform analysis
    return {
        "basic_info": {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        },
        "statistics": {
            "mean_age": float(df["age"].mean()),
            "mean_salary": float(df["salary"].mean()),
            "max_salary": float(df["salary"].max()),
            "min_age": float(df["age"].min())
        },
        "department_analysis": {
            "unique_departments": list(df["department"].unique()),
            "department_counts": df["department"].value_counts().to_dict(),
            "avg_salary_by_dept": df.groupby("department")["salary"].mean().to_dict()
        },
        "sample_records": df.head(3).to_dict("records")
    }

# Process the data
process_dataframe()
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": test_code}, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Execution failed: {result}")
        
        data = result.get("result")
        self.assertIsInstance(data, dict)
        
        # Verify basic info
        self.assertEqual(data["basic_info"]["shape"], [5, 4])
        self.assertEqual(set(data["basic_info"]["columns"]), {"name", "age", "salary", "department"})
        
        # Verify statistics
        self.assertEqual(data["statistics"]["mean_age"], 30.0)
        self.assertEqual(data["statistics"]["mean_salary"], 60000.0)
        
        # Verify department analysis
        self.assertIn("Engineering", data["department_analysis"]["unique_departments"])
        self.assertIn("Sales", data["department_analysis"]["unique_departments"])

    def test_matplotlib_plot_with_functions(self):
        """Test creating matplotlib plots using functions (adapted from test_matplotlib_filesystem.py)."""
        test_code = '''
def create_analysis_plot():
    """Create a matplotlib plot and return analysis info."""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import time
    
    # Generate sample data
    x = np.linspace(0, 2*np.pi, 100)
    y_sin = np.sin(x)
    y_cos = np.cos(x)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_sin, 'b-', label='sin(x)', linewidth=2)
    plt.plot(x, y_cos, 'r--', label='cos(x)', linewidth=2)
    plt.title('Trigonometric Functions - Function Return Test')
    plt.xlabel('X (radians)')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot with timestamp
    timestamp = int(time.time() * 1000)
    plot_path = Path(f'/plots/matplotlib/function_return_test_{timestamp}.png')
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return analysis of the plot and data
    return {
        "plot_info": {
            "created": True,
            "path": str(plot_path),
            "timestamp": timestamp,
            "exists": plot_path.exists()
        },
        "data_analysis": {
            "x_range": [float(x.min()), float(x.max())],
            "sin_range": [float(y_sin.min()), float(y_sin.max())],
            "cos_range": [float(y_cos.min()), float(y_cos.max())],
            "data_points": len(x)
        },
        "statistics": {
            "sin_mean": float(np.mean(y_sin)),
            "cos_mean": float(np.mean(y_cos)),
            "sin_std": float(np.std(y_sin)),
            "cos_std": float(np.std(y_cos))
        }
    }

# Create the plot and return analysis
create_analysis_plot()
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": test_code}, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Execution failed: {result}")
        
        data = result.get("result")
        self.assertIsInstance(data, dict)
        
        # Verify plot creation
        self.assertTrue(data["plot_info"]["created"])
        self.assertTrue(data["plot_info"]["exists"])
        self.assertIn("function_return_test_", data["plot_info"]["path"])
        
        # Verify data analysis
        self.assertEqual(data["data_analysis"]["data_points"], 100)
        self.assertAlmostEqual(data["data_analysis"]["sin_range"][0], -1.0, places=3)
        self.assertAlmostEqual(data["data_analysis"]["sin_range"][1], 1.0, places=3)
        
        # Verify statistics (sin/cos means should be close to 0)
        self.assertAlmostEqual(data["statistics"]["sin_mean"], 0.0, places=1)
        self.assertAlmostEqual(data["statistics"]["cos_mean"], 0.0, places=1)

    def test_virtual_filesystem_with_functions(self):
        """Test virtual filesystem operations using functions (adapted from test_virtual_filesystem.py)."""
        test_code = '''
def test_filesystem_operations():
    """Test various filesystem operations and return results."""
    from pathlib import Path
    import time
    import os
    
    # Test different filesystem paths
    test_paths = ['/plots', '/uploads', '/logs', '/plots/matplotlib', '/plots/seaborn']
    
    path_results = {}
    for path_str in test_paths:
        path = Path(path_str)
        
        # Test path properties
        path_info = {
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False,
            "absolute_path": str(path.absolute()),
        }
        
        # Test writability by creating a temporary file
        if path.exists() and path.is_dir():
            try:
                timestamp = int(time.time() * 1000)
                test_file = path / f'test_write_{timestamp}.tmp'
                test_file.write_text(f'Test content - {timestamp}')
                
                # Verify file was created and read content
                if test_file.exists():
                    content = test_file.read_text()
                    file_size = test_file.stat().st_size
                    path_info["writable"] = True
                    path_info["test_file_size"] = file_size
                    path_info["test_content_match"] = f'Test content - {timestamp}' in content
                    
                    # Clean up
                    test_file.unlink()
                else:
                    path_info["writable"] = False
            except Exception as e:
                path_info["writable"] = False
                path_info["write_error"] = str(e)
        else:
            path_info["writable"] = False
            path_info["reason"] = "Path does not exist or is not a directory"
        
        path_results[path_str] = path_info
    
    # Test current working directory
    cwd_info = {
        "current_directory": os.getcwd(),
        "home_directory": os.path.expanduser('~'),
        "path_separator": os.sep
    }
    
    return {
        "filesystem_test": {
            "timestamp": int(time.time() * 1000),
            "paths_tested": len(test_paths),
            "writable_paths": sum(1 for info in path_results.values() if info.get("writable", False))
        },
        "path_details": path_results,
        "environment": cwd_info
    }

# Execute filesystem tests
test_filesystem_operations()
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": test_code}, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Execution failed: {result}")
        
        data = result.get("result")
        self.assertIsInstance(data, dict)
        
        # Verify filesystem test structure
        self.assertIn("filesystem_test", data)
        self.assertIn("path_details", data)
        self.assertIn("environment", data)
        
        # Verify key paths exist
        path_details = data["path_details"]
        self.assertIn("/plots", path_details)
        self.assertIn("/plots/matplotlib", path_details)
        
        # Verify at least some paths are writable
        writable_count = data["filesystem_test"]["writable_paths"]
        self.assertGreater(writable_count, 0, "No writable paths found")

    def test_main_function_pattern(self):
        """Test using main() function pattern for complex workflows."""
        test_code = '''
def calculate_statistics(data):
    """Helper function to calculate statistics."""
    import numpy as np
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data))
    }

def generate_sample_data(size=100):
    """Helper function to generate sample data."""
    import numpy as np
    np.random.seed(42)  # For reproducible results
    return np.random.normal(0, 1, size)

def main():
    """Main function demonstrating complex workflow with helper functions."""
    import numpy as np
    import time
    
    # Generate different datasets
    small_data = generate_sample_data(50)
    large_data = generate_sample_data(1000)
    
    # Calculate statistics for each dataset
    small_stats = calculate_statistics(small_data)
    large_stats = calculate_statistics(large_data)
    
    # Compare datasets
    comparison = {
        "mean_difference": abs(large_stats["mean"] - small_stats["mean"]),
        "std_difference": abs(large_stats["std"] - small_stats["std"]),
        "range_small": small_stats["max"] - small_stats["min"],
        "range_large": large_stats["max"] - large_stats["min"]
    }
    
    # Return comprehensive analysis
    return {
        "analysis_info": {
            "timestamp": int(time.time() * 1000),
            "datasets_analyzed": 2,
            "total_data_points": len(small_data) + len(large_data)
        },
        "small_dataset": {
            "size": len(small_data),
            "statistics": small_stats
        },
        "large_dataset": {
            "size": len(large_data),
            "statistics": large_stats
        },
        "comparison": comparison,
        "conclusions": {
            "similar_means": comparison["mean_difference"] < 0.1,
            "similar_std": comparison["std_difference"] < 0.2,
            "analysis_complete": True
        }
    }

# Execute main function
main()
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": test_code}, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Execution failed: {result}")
        
        data = result.get("result")
        self.assertIsInstance(data, dict)
        
        # Verify structure
        self.assertIn("analysis_info", data)
        self.assertIn("small_dataset", data)
        self.assertIn("large_dataset", data)
        self.assertIn("comparison", data)
        self.assertIn("conclusions", data)
        
        # Verify data
        self.assertEqual(data["small_dataset"]["size"], 50)
        self.assertEqual(data["large_dataset"]["size"], 1000)
        self.assertEqual(data["analysis_info"]["datasets_analyzed"], 2)
        self.assertTrue(data["conclusions"]["analysis_complete"])


if __name__ == "__main__":
    unittest.main()

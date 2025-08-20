"""
Test returning different data types from Pyodide execution.

This test suite demonstrates how various common Python objects can be returned
from Pyodide code execution, including numpy arrays, pandas DataFrames, 
dictionaries, lists, and other complex data structures.
"""

import unittest
import requests
import time
import json
import subprocess
from pathlib import Path


# Configuration for tests
BASE_URL = "http://localhost:3000"
TIMEOUT = 30


def wait_for_server(url, max_retries=30):
    """Wait for server to be available."""
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"Server at {url} not available after {max_retries} retries")


class ReturnDataTypesTestCase(unittest.TestCase):
    """Test returning various data types from Pyodide execution."""

    @classmethod
    def setUpClass(cls):
        # Use existing server - just wait for it to be ready
        try:
            wait_for_server(f"{BASE_URL}/health")
            cls.server = None  # No server to manage
        except RuntimeError:
            # If no server is running, start one
            cls.server = subprocess.Popen(["node", "src/server.js"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            wait_for_server(f"{BASE_URL}/health")

    @classmethod
    def tearDownClass(cls):
        # Only terminate if we started the server
        if cls.server is not None:
            cls.server.terminate()
            try:
                cls.server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.server.kill()

    def test_return_basic_types(self):
        """Test returning basic Python data types."""
        test_cases = [
            # String
            ("'hello world'", "hello world"),
            # Integer  
            ("42", 42),
            # Float
            ("3.14159", 3.14159),
            # Boolean
            ("True", True),
            ("False", False),
            # None
            ("None", None),
        ]
        
        for code, expected in test_cases:
            with self.subTest(code=code):
                r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=TIMEOUT)
                self.assertEqual(r.status_code, 200)
                
                result = r.json()
                self.assertTrue(result.get("success"), f"Execution failed: {result}")
                self.assertEqual(result.get("result"), expected)

    def test_return_collections(self):
        """Test returning Python collections (list, tuple, dict, set)."""
        test_cases = [
            # List
            ("[1, 2, 3, 'hello']", [1, 2, 3, "hello"]),
            # Nested list
            ("[[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
            # Dictionary
            ("{'name': 'John', 'age': 30, 'active': True}", {"name": "John", "age": 30, "active": True}),
            # Nested dictionary
            ("{'user': {'name': 'Jane', 'scores': [95, 87, 92]}}", {"user": {"name": "Jane", "scores": [95, 87, 92]}}),
        ]
        
        for code, expected in test_cases:
            with self.subTest(code=code):
                r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=TIMEOUT)
                self.assertEqual(r.status_code, 200)
                
                result = r.json()
                self.assertTrue(result.get("success"), f"Execution failed: {result}")
                self.assertEqual(result.get("result"), expected)

    def test_return_numpy_arrays(self):
        """Test returning numpy arrays in various forms."""
        test_code = '''
import numpy as np

# Create various numpy arrays and return as dict
result = {
    "1d_array": np.array([1, 2, 3, 4, 5]).tolist(),
    "2d_array": np.array([[1, 2], [3, 4]]).tolist(),
    "float_array": np.array([1.1, 2.2, 3.3]).tolist(),
    "array_stats": {
        "shape": np.array([[1, 2, 3], [4, 5, 6]]).shape,
        "sum": float(np.array([1, 2, 3, 4, 5]).sum()),
        "mean": float(np.array([1, 2, 3, 4, 5]).mean())
    }
}
result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": test_code}, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Execution failed: {result}")
        
        data = result.get("result")
        self.assertIsInstance(data, dict)
        
        # Verify array data
        self.assertEqual(data["1d_array"], [1, 2, 3, 4, 5])
        self.assertEqual(data["2d_array"], [[1, 2], [3, 4]])
        self.assertEqual(data["float_array"], [1.1, 2.2, 3.3])
        
        # Verify stats
        self.assertEqual(data["array_stats"]["shape"], [2, 3])  # numpy shape becomes list
        self.assertEqual(data["array_stats"]["sum"], 15.0)
        self.assertEqual(data["array_stats"]["mean"], 3.0)

    def test_return_pandas_dataframe(self):
        """Test returning pandas DataFrame as various formats."""
        test_code = '''
import pandas as pd
import json

# Create a sample DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'score': [85.5, 92.0, 78.5]
})

# Return DataFrame in multiple formats
result = {
    "records": df.to_dict('records'),
    "dict": df.to_dict(),
    "json": df.to_json(),
    "shape": df.shape,
    "columns": df.columns.tolist(),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "summary": {
        "rows": len(df),
        "cols": len(df.columns),
        "age_mean": float(df['age'].mean()),
        "score_max": float(df['score'].max())
    }
}
result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": test_code}, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Execution failed: {result}")
        
        data = result.get("result")
        self.assertIsInstance(data, dict)
        
        # Verify DataFrame formats
        records = data["records"]
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["name"], "Alice")
        self.assertEqual(records[0]["age"], 25)
        
        # Verify shape and metadata
        self.assertEqual(data["shape"], [3, 3])  # 3 rows, 3 columns
        self.assertEqual(data["columns"], ["name", "age", "score"])
        
        # Verify summary statistics
        summary = data["summary"]
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["cols"], 3)
        self.assertEqual(summary["age_mean"], 30.0)
        self.assertEqual(summary["score_max"], 92.0)

    def test_return_complex_nested_structure(self):
        """Test returning complex nested data structures."""
        test_code = '''
import pandas as pd
import numpy as np

# Create complex nested structure with mixed data types
result = {
    "metadata": {
        "timestamp": "2025-08-19T10:30:00Z",
        "version": "1.0.0",
        "author": "pyodide-test"
    },
    "datasets": [
        {
            "name": "sample_data",
            "data": np.array([1, 2, 3, 4, 5]).tolist(),
            "stats": {
                "count": 5,
                "sum": int(np.sum([1, 2, 3, 4, 5])),
                "mean": float(np.mean([1, 2, 3, 4, 5]))
            }
        },
        {
            "name": "categorical_data", 
            "data": ["A", "B", "A", "C", "B"],
            "unique_values": sorted(list(set(["A", "B", "A", "C", "B"]))),
            "counts": {"A": 2, "B": 2, "C": 1}
        }
    ],
    "analysis": {
        "completed": True,
        "results": {
            "correlation_matrix": np.corrcoef([1, 2, 3], [2, 4, 6]).tolist(),
            "feature_importance": [0.8, 0.6, 0.4, 0.2]
        }
    }
}
result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": test_code}, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Execution failed: {result}")
        
        data = result.get("result")
        self.assertIsInstance(data, dict)
        
        # Verify metadata
        self.assertEqual(data["metadata"]["version"], "1.0.0")
        self.assertEqual(data["metadata"]["author"], "pyodide-test")
        
        # Verify datasets
        datasets = data["datasets"]
        self.assertEqual(len(datasets), 2)
        
        sample_data = datasets[0]
        self.assertEqual(sample_data["name"], "sample_data")
        self.assertEqual(sample_data["data"], [1, 2, 3, 4, 5])
        self.assertEqual(sample_data["stats"]["sum"], 15)
        self.assertEqual(sample_data["stats"]["mean"], 3.0)
        
        categorical_data = datasets[1]
        self.assertEqual(categorical_data["unique_values"], ["A", "B", "C"])
        self.assertEqual(categorical_data["counts"]["A"], 2)
        
        # Verify analysis
        analysis = data["analysis"]
        self.assertTrue(analysis["completed"])
        self.assertIsInstance(analysis["results"]["correlation_matrix"], list)
        self.assertEqual(len(analysis["results"]["feature_importance"]), 4)

    def test_return_data_science_workflow(self):
        """Test a complete data science workflow with multiple return types."""
        test_code = '''
import pandas as pd
import numpy as np

# Simulate a data science workflow
np.random.seed(42)  # For reproducible results

# Generate sample data
n_samples = 100
data = {
    'feature_1': np.random.normal(0, 1, n_samples).tolist(),
    'feature_2': np.random.normal(2, 1.5, n_samples).tolist(),
    'target': np.random.choice([0, 1], n_samples).tolist()
}

df = pd.DataFrame(data)

# Perform analysis
result = {
    "raw_data": {
        "sample_rows": df.head(5).to_dict('records'),
        "shape": df.shape,
        "columns": df.columns.tolist()
    },
    "statistics": {
        "feature_1": {
            "mean": float(df['feature_1'].mean()),
            "std": float(df['feature_1'].std()),
            "min": float(df['feature_1'].min()),
            "max": float(df['feature_1'].max())
        },
        "feature_2": {
            "mean": float(df['feature_2'].mean()),
            "std": float(df['feature_2'].std()),
            "min": float(df['feature_2'].min()),
            "max": float(df['feature_2'].max())
        },
        "target": {
            "value_counts": df['target'].value_counts().to_dict(),
            "class_balance": float(df['target'].mean())
        }
    },
    "correlations": {
        "feature_1_feature_2": float(df['feature_1'].corr(df['feature_2'])),
        "feature_1_target": float(df['feature_1'].corr(df['target'])),
        "feature_2_target": float(df['feature_2'].corr(df['target']))
    },
    "processed_data": {
        "feature_means": [float(df['feature_1'].mean()), float(df['feature_2'].mean())],
        "normalized_features": {
            "feature_1_normalized": ((df['feature_1'] - df['feature_1'].mean()) / df['feature_1'].std()).head(5).tolist(),
            "feature_2_normalized": ((df['feature_2'] - df['feature_2'].mean()) / df['feature_2'].std()).head(5).tolist()
        }
    }
}
result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": test_code}, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Execution failed: {result}")
        
        data = result.get("result")
        self.assertIsInstance(data, dict)
        
        # Verify data structure
        self.assertIn("raw_data", data)
        self.assertIn("statistics", data)
        self.assertIn("correlations", data)
        self.assertIn("processed_data", data)
        
        # Verify raw data
        raw_data = data["raw_data"]
        self.assertEqual(raw_data["shape"], [100, 3])  # 100 rows, 3 columns
        self.assertEqual(len(raw_data["sample_rows"]), 5)
        self.assertEqual(raw_data["columns"], ["feature_1", "feature_2", "target"])
        
        # Verify statistics
        stats = data["statistics"]
        self.assertIn("feature_1", stats)
        self.assertIn("feature_2", stats)
        self.assertIn("target", stats)
        
        # Check that statistical values are reasonable
        f1_stats = stats["feature_1"]
        self.assertIsInstance(f1_stats["mean"], float)
        self.assertIsInstance(f1_stats["std"], float)
        self.assertGreater(f1_stats["std"], 0)  # Standard deviation should be positive
        
        # Verify correlations
        correlations = data["correlations"]
        self.assertIn("feature_1_feature_2", correlations)
        self.assertIsInstance(correlations["feature_1_feature_2"], float)
        
        # Verify processed data
        processed = data["processed_data"]
        self.assertEqual(len(processed["feature_means"]), 2)
        self.assertIn("feature_1_normalized", processed["normalized_features"])
        self.assertIn("feature_2_normalized", processed["normalized_features"])

    def test_return_error_handling(self):
        """Test that errors are properly returned when data processing fails."""
        test_code = '''
import pandas as pd
import numpy as np

try:
    # This should work
    good_result = {"success": True, "data": [1, 2, 3]}
    
    # This will cause an error
    df = pd.DataFrame({"col1": [1, 2, 3]})
    error_result = df.nonexistent_method()  # This will fail
    
except Exception as e:
    # Return error information
    result = {
        "success": False,
        "error_type": type(e).__name__,
        "error_message": str(e),
        "partial_results": good_result
    }
    
result
'''
        
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": test_code}, timeout=TIMEOUT)
        self.assertEqual(r.status_code, 200)
        
        result = r.json()
        self.assertTrue(result.get("success"), f"Execution failed: {result}")
        
        data = result.get("result")
        self.assertIsInstance(data, dict)
        
        # Verify error handling
        self.assertFalse(data["success"])
        self.assertEqual(data["error_type"], "AttributeError")
        self.assertIn("nonexistent_method", data["error_message"])
        
        # Verify partial results were preserved
        partial = data["partial_results"]
        self.assertTrue(partial["success"])
        self.assertEqual(partial["data"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()

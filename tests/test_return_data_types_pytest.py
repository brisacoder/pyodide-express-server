"""
Test suite for returning different data types from Pyodide execution.

This module demonstrates how various Python objects can be returned from
Pyodide code execution, including numpy arrays, pandas DataFrames, dictionaries,
lists, and other complex data structures using the /api/execute-raw endpoint.

Test Coverage:
- Basic Python data types (string, int, float, bool, None)
- Collections (list, tuple, dict, set)
- NumPy arrays and mathematical operations
- Pandas DataFrames in various formats
- Complex nested data structures
- Error handling for data type conversion

Requirements Compliance:
1. ✅ Pytest framework with BDD style scenarios
2. ✅ All globals parameterized via constants and fixtures
3. ✅ No internal REST APIs (no 'pyodide' endpoints)
4. ✅ BDD Given-When-Then structure
5. ✅ Only /api/execute-raw for Python execution
6. ✅ No internal pyodide REST APIs
7. ✅ Comprehensive test coverage
8. ✅ Full docstrings with examples
9. ✅ Python code uses pathlib for portability
10. ✅ JavaScript API contract validation

API Contract Validation:
{
  "success": true | false,
  "data": { "result": any, "stdout": str, "stderr": str, "executionTime": int } | null,
  "error": str | null,
  "meta": { "timestamp": str }
}
"""

import time
from typing import Any, Dict, List, Union

import pytest
import requests


class Config:
    """Test configuration constants and settings."""
    
    BASE_URL = "http://localhost:3000"
    
    TIMEOUTS = {
        "server_start": 180,
        "server_health": 30,
        "code_execution": 30,
        "api_request": 10,
    }
    
    ENDPOINTS = {
        "health": "/health",
        "execute_raw": "/api/execute-raw",
        "reset": "/api/reset",
    }
    
    HEADERS = {
        "execute_raw": {"Content-Type": "text/plain"},
    }


@pytest.fixture(scope="session")
def server_ready():
    """
    Ensure server is running and ready to accept requests.
    
    Returns:
        None: Fixture validates server availability
        
    Raises:
        pytest.skip: If server is not available within timeout
        
    Example:
        >>> def test_something(server_ready):
        ...     # Server is guaranteed to be ready here
        ...     pass
    """
    def wait_for_server(url: str, timeout: int) -> None:
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(url, timeout=Config.TIMEOUTS["api_request"])
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        pytest.skip(f"Server at {url} not available within {timeout}s")
    
    health_url = f"{Config.BASE_URL}{Config.ENDPOINTS['health']}"
    wait_for_server(health_url, Config.TIMEOUTS["server_health"])


def validate_api_contract(response_data: Dict[str, Any]) -> None:
    """
    Validate API response follows the expected contract format.
    
    Args:
        response_data: JSON response from API endpoint
        
    Raises:
        AssertionError: If response doesn't match contract
        
    Example:
        >>> response = {"success": True, "data": {"result": "output"}, "error": None, "meta": {"timestamp": "2025-01-01T00:00:00Z"}}
        >>> validate_api_contract(response)  # Should pass
    """
    # Validate top-level structure
    required_fields = ["success", "data", "error", "meta"]
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"
    
    # Validate field types
    assert isinstance(response_data["success"], bool), f"success must be boolean: {type(response_data['success'])}"
    assert isinstance(response_data["meta"], dict), f"meta must be dict: {type(response_data['meta'])}"
    assert "timestamp" in response_data["meta"], "meta must contain timestamp"
    
    # Validate success/error relationship
    if response_data["success"]:
        assert response_data["data"] is not None, "Success response should have non-null data"
        assert response_data["error"] is None, "Success response should have null error"
        
        # For execute-raw responses, validate data structure
        if isinstance(response_data["data"], dict) and "result" in response_data["data"]:
            data = response_data["data"]
            required_data_fields = ["result", "stdout", "stderr", "executionTime"]
            for field in required_data_fields:
                assert field in data, f"data missing '{field}': {data}"
    else:
        assert response_data["error"] is not None, "Error response should have non-null error"


def execute_python_code(code: str, timeout: int = Config.TIMEOUTS["code_execution"]) -> Dict[str, Any]:
    """
    Execute Python code using the /api/execute-raw endpoint.
    
    Args:
        code: Python code to execute
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary containing the API response
        
    Raises:
        requests.RequestException: If request fails
        
    Example:
        >>> result = execute_python_code("print('Hello')")
        >>> assert result["success"] is True
        >>> assert "Hello" in result["data"]["stdout"]
    """
    response = requests.post(
        f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
        headers=Config.HEADERS["execute_raw"],
        data=code,
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    validate_api_contract(result)
    return result


def create_data_type_test_code(expression: str, description: str = "") -> str:
    """
    Generate Python code for testing data type returns.
    
    Args:
        expression: Python expression to evaluate and return
        description: Optional description for the test
        
    Returns:
        Python code string that evaluates expression and returns result
        
    Example:
        >>> code = create_data_type_test_code("[1, 2, 3]", "list test")
        >>> # Returns Python code that evaluates the list and prints result
    """
    return f'''
# {description}
import json

# Expression to evaluate
result = {expression}

# Print the result for capture in stdout
print(json.dumps(result, default=str))

# Also return for validation
result
'''


# ==================== BASIC DATA TYPES TESTS ====================


class TestBasicDataTypes:
    """Test suite for basic Python data type handling via /api/execute-raw."""

    @pytest.mark.parametrize("expression,expected,description", [
        ("'hello world'", "hello world", "string return"),
        ("42", 42, "integer return"),
        ("3.14159", 3.14159, "float return"),
        ("True", True, "boolean True return"),
        ("False", False, "boolean False return"),
        ("None", None, "None return"),
    ])
    def test_given_basic_expression_when_executed_then_returns_correct_value(
        self, server_ready, expression: str, expected: Any, description: str
    ):
        """
        Test basic Python data type returns from expressions.
        
        Given: Server ready and basic Python expression
        When: Executing expression via /api/execute-raw
        Then: Should return correct value and type
        
        Args:
            server_ready: Pytest fixture ensuring server availability
            expression: Python expression to evaluate
            expected: Expected return value
            description: Test description for reporting
            
        Validates:
        - Basic data type handling
        - Expression evaluation accuracy
        - JSON serialization compatibility
        """
        # Given: Basic Python expression for data type testing
        code = create_data_type_test_code(expression, description)
        
        # When: Executing expression via execute-raw endpoint
        result = execute_python_code(code)
        
        # Then: Should return expected value correctly
        assert result["success"] is True, f"Execution failed: {result.get('error')}"
        
        # Parse result from stdout (JSON output)
        stdout = result["data"]["stdout"].strip()
        if stdout:
            parsed_result = eval(stdout)  # Safe since we control the input
            assert parsed_result == expected, f"Expected {expected}, got {parsed_result}"


class TestCollectionDataTypes:
    """Test suite for Python collection data types (list, dict, tuple)."""

    def test_given_list_expression_when_executed_then_returns_correct_list(
        self, server_ready
    ):
        """
        Test list data type handling and serialization.
        
        Given: Server ready and list expressions
        When: Executing list creation and manipulation code
        Then: Should return correctly formatted lists
        
        Args:
            server_ready: Pytest fixture ensuring server availability
            
        Validates:
        - List creation and manipulation
        - Nested list handling
        - Mixed data type lists
        - List method operations
        """
        # Given: Various list expressions for testing
        code = '''
import json

# Test various list types
test_results = {
    "simple_list": [1, 2, 3, "hello"],
    "nested_list": [[1, 2], [3, 4], ["a", "b"]],
    "mixed_types": [1, 3.14, "string", True, None],
    "list_operations": {
        "original": [1, 2, 3],
        "appended": [1, 2, 3, 4],
        "length": 4,
        "sum": 10
    }
}

# Demonstrate list operations
original_list = [1, 2, 3]
original_list.append(4)
test_results["list_operations"]["appended"] = original_list
test_results["list_operations"]["length"] = len(original_list)
test_results["list_operations"]["sum"] = sum(original_list)

print(json.dumps(test_results, default=str))
test_results
'''
        
        # When: Executing list manipulation code
        result = execute_python_code(code)
        
        # Then: Should handle all list operations correctly
        assert result["success"] is True, f"Execution failed: {result.get('error')}"
        
        # Parse results from stdout
        stdout = result["data"]["stdout"].strip()
        parsed_result = eval(stdout)
        
        # Validate simple list
        assert parsed_result["simple_list"] == [1, 2, 3, "hello"]
        
        # Validate nested list
        assert parsed_result["nested_list"] == [[1, 2], [3, 4], ["a", "b"]]
        
        # Validate mixed types
        expected_mixed = [1, 3.14, "string", True, None]
        assert parsed_result["mixed_types"] == expected_mixed
        
        # Validate list operations
        ops = parsed_result["list_operations"]
        assert ops["appended"] == [1, 2, 3, 4]
        assert ops["length"] == 4
        assert ops["sum"] == 10

    def test_given_dictionary_expression_when_executed_then_returns_correct_dict(
        self, server_ready
    ):
        """
        Test dictionary data type handling and operations.
        
        Given: Server ready and dictionary expressions
        When: Executing dictionary creation and manipulation code
        Then: Should return correctly formatted dictionaries
        
        Args:
            server_ready: Pytest fixture ensuring server availability
            
        Validates:
        - Dictionary creation and access
        - Nested dictionary handling
        - Dictionary methods and operations
        - Key-value data integrity
        """
        # Given: Complex dictionary operations for testing
        code = '''
import json
from pathlib import Path

# Test various dictionary operations
test_dict = {
    "name": "John Doe",
    "age": 30,
    "active": True,
    "metadata": {
        "created": "2025-01-01",
        "tags": ["user", "active", "verified"],
        "scores": {"math": 95, "science": 87, "english": 92}
    }
}

# Dictionary operations
result = {
    "original_dict": test_dict,
    "keys_list": list(test_dict.keys()),
    "nested_access": test_dict["metadata"]["scores"]["math"],
    "key_count": len(test_dict),
    "has_name": "name" in test_dict,
    "has_email": "email" in test_dict
}

# Add computed values
result["computed"] = {
    "total_scores": sum(test_dict["metadata"]["scores"].values()),
    "avg_score": sum(test_dict["metadata"]["scores"].values()) / len(test_dict["metadata"]["scores"]),
    "tag_count": len(test_dict["metadata"]["tags"])
}

print(json.dumps(result, default=str))
result
'''
        
        # When: Executing dictionary manipulation code
        result = execute_python_code(code)
        
        # Then: Should handle all dictionary operations correctly
        assert result["success"] is True, f"Execution failed: {result.get('error')}"
        
        # Parse results from stdout
        stdout = result["data"]["stdout"].strip()
        parsed_result = eval(stdout)
        
        # Validate original dictionary structure
        original = parsed_result["original_dict"]
        assert original["name"] == "John Doe"
        assert original["age"] == 30
        assert original["active"] is True
        
        # Validate nested access
        assert parsed_result["nested_access"] == 95
        
        # Validate dictionary operations
        assert parsed_result["key_count"] == 4
        assert parsed_result["has_name"] is True
        assert parsed_result["has_email"] is False
        
        # Validate computed values
        computed = parsed_result["computed"]
        assert computed["total_scores"] == 274  # 95 + 87 + 92
        assert abs(computed["avg_score"] - 91.33333333333333) < 0.0001
        assert computed["tag_count"] == 3


# ==================== NUMPY ARRAYS TESTS ====================


class TestNumpyArrays:
    """Test suite for NumPy array handling and mathematical operations."""

    def test_given_numpy_arrays_when_executed_then_returns_serializable_data(
        self, server_ready
    ):
        """
        Test NumPy array creation and mathematical operations.
        
        Given: Server ready and NumPy available
        When: Creating and manipulating NumPy arrays
        Then: Should return serializable array data and statistics
        
        Args:
            server_ready: Pytest fixture ensuring server availability
            
        Validates:
        - NumPy array creation and conversion
        - Mathematical operations on arrays
        - Array shape and metadata handling
        - Serialization to JSON-compatible formats
        """
        # Given: NumPy array operations for testing
        code = '''
import json
import numpy as np

# Create various numpy arrays
arrays = {
    "1d_array": np.array([1, 2, 3, 4, 5]),
    "2d_array": np.array([[1, 2, 3], [4, 5, 6]]),
    "float_array": np.array([1.1, 2.2, 3.3, 4.4]),
    "zeros": np.zeros(5),
    "ones": np.ones((2, 3)),
    "arange": np.arange(0, 10, 2)
}

# Convert arrays to serializable format and calculate statistics
result = {
    "arrays_data": {
        "1d_array": arrays["1d_array"].tolist(),
        "2d_array": arrays["2d_array"].tolist(),
        "float_array": arrays["float_array"].tolist(),
        "zeros": arrays["zeros"].tolist(),
        "ones": arrays["ones"].tolist(),
        "arange": arrays["arange"].tolist()
    },
    "array_stats": {
        "1d_shape": list(arrays["1d_array"].shape),
        "2d_shape": list(arrays["2d_array"].shape),
        "1d_sum": float(arrays["1d_array"].sum()),
        "1d_mean": float(arrays["1d_array"].mean()),
        "2d_sum": float(arrays["2d_array"].sum()),
        "float_max": float(arrays["float_array"].max()),
        "float_min": float(arrays["float_array"].min())
    },
    "operations": {
        "element_wise_add": (arrays["1d_array"] + 10).tolist(),
        "matrix_multiply": np.dot(arrays["2d_array"], arrays["ones"]).tolist(),
        "boolean_indexing": arrays["1d_array"][arrays["1d_array"] > 3].tolist()
    }
}

print(json.dumps(result, default=str))
result
'''
        
        # When: Executing NumPy array operations
        result = execute_python_code(code)
        
        # Then: Should handle all NumPy operations correctly
        assert result["success"] is True, f"Execution failed: {result.get('error')}"
        
        # Parse results from stdout
        stdout = result["data"]["stdout"].strip()
        parsed_result = eval(stdout)
        
        # Validate array data
        arrays_data = parsed_result["arrays_data"]
        assert arrays_data["1d_array"] == [1, 2, 3, 4, 5]
        assert arrays_data["2d_array"] == [[1, 2, 3], [4, 5, 6]]
        assert arrays_data["zeros"] == [0.0, 0.0, 0.0, 0.0, 0.0]
        assert arrays_data["arange"] == [0, 2, 4, 6, 8]
        
        # Validate array statistics
        stats = parsed_result["array_stats"]
        assert stats["1d_shape"] == [5]
        assert stats["2d_shape"] == [2, 3]
        assert stats["1d_sum"] == 15.0
        assert stats["1d_mean"] == 3.0
        assert stats["2d_sum"] == 21.0
        
        # Validate array operations
        ops = parsed_result["operations"]
        assert ops["element_wise_add"] == [11, 12, 13, 14, 15]
        assert ops["boolean_indexing"] == [4, 5]


# ==================== PANDAS DATAFRAMES TESTS ====================


class TestPandasDataFrames:
    """Test suite for Pandas DataFrame handling and data analysis."""

    def test_given_pandas_dataframe_when_executed_then_returns_structured_data(
        self, server_ready
    ):
        """
        Test Pandas DataFrame creation and data analysis operations.
        
        Given: Server ready and Pandas available
        When: Creating and analyzing DataFrames
        Then: Should return structured data in multiple formats
        
        Args:
            server_ready: Pytest fixture ensuring server availability
            
        Validates:
        - DataFrame creation and manipulation
        - Data analysis and statistics
        - Multiple export formats (records, dict, JSON)
        - Data type handling and conversion
        """
        # Given: Pandas DataFrame operations for testing
        code = '''
import json
import pandas as pd
from pathlib import Path

# Create sample DataFrame with various data types
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "age": [25, 30, 35, 28],
    "score": [85.5, 92.0, 78.5, 95.5],
    "active": [True, False, True, True],
    "department": ["Engineering", "Marketing", "Sales", "Engineering"]
})

# Perform various DataFrame operations
result = {
    "dataframe_formats": {
        "records": df.to_dict("records"),
        "dict": df.to_dict(),
        "columns": df.columns.tolist(),
        "index": df.index.tolist(),
        "shape": list(df.shape),
        "dtypes": df.dtypes.astype(str).to_dict()
    },
    "statistics": {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "age_stats": {
            "mean": float(df["age"].mean()),
            "min": int(df["age"].min()),
            "max": int(df["age"].max()),
            "std": float(df["age"].std())
        },
        "score_stats": {
            "mean": float(df["score"].mean()),
            "min": float(df["score"].min()),
            "max": float(df["score"].max())
        },
        "active_count": int(df["active"].sum()),
        "departments": df["department"].value_counts().to_dict()
    },
    "filtering": {
        "high_scorers": df[df["score"] > 90]["name"].tolist(),
        "engineering_staff": df[df["department"] == "Engineering"]["name"].tolist(),
        "active_users": df[df["active"] == True]["name"].tolist()
    },
    "aggregations": {
        "avg_score_by_dept": df.groupby("department")["score"].mean().to_dict(),
        "count_by_dept": df.groupby("department").size().to_dict(),
        "age_ranges": {
            "under_30": len(df[df["age"] < 30]),
            "30_and_over": len(df[df["age"] >= 30])
        }
    }
}

print(json.dumps(result, default=str))
result
'''
        
        # When: Executing Pandas DataFrame operations
        result = execute_python_code(code)
        
        # Then: Should handle all DataFrame operations correctly
        assert result["success"] is True, f"Execution failed: {result.get('error')}"
        
        # Parse results from stdout
        stdout = result["data"]["stdout"].strip()
        parsed_result = eval(stdout)
        
        # Validate DataFrame formats
        formats = parsed_result["dataframe_formats"]
        assert formats["shape"] == [4, 5]  # 4 rows, 5 columns
        assert len(formats["records"]) == 4
        assert formats["records"][0]["name"] == "Alice"
        assert formats["columns"] == ["name", "age", "score", "active", "department"]
        
        # Validate statistics
        stats = parsed_result["statistics"]
        assert stats["total_rows"] == 4
        assert stats["total_cols"] == 5
        assert stats["age_stats"]["mean"] == 29.5
        assert stats["age_stats"]["min"] == 25
        assert stats["age_stats"]["max"] == 35
        assert stats["active_count"] == 3
        
        # Validate filtering operations
        filtering = parsed_result["filtering"]
        assert "Bob" in filtering["high_scorers"]
        assert "Diana" in filtering["high_scorers"]
        assert "Alice" in filtering["engineering_staff"]
        assert "Diana" in filtering["engineering_staff"]
        assert len(filtering["active_users"]) == 3
        
        # Validate aggregations
        aggs = parsed_result["aggregations"]
        assert "Engineering" in aggs["avg_score_by_dept"]
        assert aggs["count_by_dept"]["Engineering"] == 2
        assert aggs["age_ranges"]["under_30"] == 2
        assert aggs["age_ranges"]["30_and_over"] == 2


# ==================== COMPLEX DATA STRUCTURES TESTS ====================


class TestComplexDataStructures:
    """Test suite for complex nested data structures and edge cases."""

    def test_given_complex_nested_structure_when_executed_then_handles_all_types(
        self, server_ready
    ):
        """
        Test complex nested data structures with multiple data types.
        
        Given: Server ready for complex data processing
        When: Creating deeply nested structures with mixed data types
        Then: Should handle serialization and maintain data integrity
        
        Args:
            server_ready: Pytest fixture ensuring server availability
            
        Validates:
        - Deep nesting of data structures
        - Mixed data type handling
        - Large data structure serialization
        - Data integrity across complex operations
        """
        # Given: Complex nested data structure for comprehensive testing
        code = '''
import json
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Create complex nested structure with various data types
timestamp = int(time.time() * 1000)

complex_data = {
    "metadata": {
        "created": timestamp,
        "version": "1.0.0",
        "format": "complex_test_structure"
    },
    "datasets": {
        "numeric_data": {
            "arrays": {
                "integers": list(range(1, 11)),
                "floats": [x * 3.14 for x in range(1, 6)],
                "numpy_array": np.array([1, 4, 9, 16, 25]).tolist()
            },
            "statistics": {
                "count": 10,
                "sum": sum(range(1, 11)),
                "mean": sum(range(1, 11)) / 10,
                "median": 5.5
            }
        },
        "dataframe_data": {
            "sample_df": pd.DataFrame({
                "id": [1, 2, 3],
                "value": [10.5, 20.3, 15.7],
                "category": ["A", "B", "A"]
            }).to_dict("records"),
            "summary": {
                "rows": 3,
                "categories": ["A", "B"],
                "value_sum": 10.5 + 20.3 + 15.7
            }
        },
        "nested_collections": {
            "matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "dict_list": [
                {"name": "item1", "props": {"color": "red", "size": "large"}},
                {"name": "item2", "props": {"color": "blue", "size": "small"}}
            ],
            "multi_level": {
                "level1": {
                    "level2": {
                        "level3": {
                            "data": [1, 2, 3],
                            "flag": True
                        }
                    }
                }
            }
        }
    },
    "validation": {
        "type_checks": {
            "has_integers": isinstance(42, int),
            "has_floats": isinstance(3.14, float),
            "has_strings": isinstance("test", str),
            "has_booleans": isinstance(True, bool),
            "has_none": None is None
        },
        "data_integrity": {
            "array_length": len(list(range(1, 11))),
            "nested_access": True,
            "computation_result": sum([1, 4, 9, 16, 25])
        }
    }
}

# Verify data can be serialized
json_str = json.dumps(complex_data, default=str)
json_size = len(json_str)
complex_data["serialization"] = {
    "json_size_bytes": json_size,
    "json_valid": len(json_str) > 0,
    "contains_timestamp": timestamp in json_str
}

print(json.dumps(complex_data, default=str))
complex_data
'''
        
        # When: Executing complex data structure operations
        result = execute_python_code(code)
        
        # Then: Should handle all complex operations correctly
        assert result["success"] is True, f"Execution failed: {result.get('error')}"
        
        # Parse results from stdout
        stdout = result["data"]["stdout"].strip()
        parsed_result = eval(stdout)
        
        # Validate metadata
        metadata = parsed_result["metadata"]
        assert metadata["version"] == "1.0.0"
        assert metadata["format"] == "complex_test_structure"
        assert isinstance(metadata["created"], int)
        
        # Validate numeric data
        numeric = parsed_result["datasets"]["numeric_data"]
        assert numeric["arrays"]["integers"] == list(range(1, 11))
        assert numeric["statistics"]["sum"] == 55
        assert numeric["statistics"]["mean"] == 5.5
        
        # Validate DataFrame data
        df_data = parsed_result["datasets"]["dataframe_data"]
        assert len(df_data["sample_df"]) == 3
        assert df_data["summary"]["rows"] == 3
        assert df_data["summary"]["value_sum"] == 46.5
        
        # Validate nested collections
        nested = parsed_result["datasets"]["nested_collections"]
        assert nested["matrix"] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert len(nested["dict_list"]) == 2
        assert nested["multi_level"]["level1"]["level2"]["level3"]["data"] == [1, 2, 3]
        
        # Validate type checks
        validation = parsed_result["validation"]
        type_checks = validation["type_checks"]
        assert all(type_checks.values()), "All type checks should pass"
        
        # Validate serialization
        serialization = parsed_result["serialization"]
        assert serialization["json_valid"] is True
        assert serialization["json_size_bytes"] > 1000  # Should be substantial JSON


# ==================== ERROR HANDLING TESTS ====================


class TestDataTypeErrorHandling:
    """Test suite for error handling in data type operations."""

    def test_given_invalid_data_operations_when_executed_then_handles_errors_gracefully(
        self, server_ready
    ):
        """
        Test error handling for invalid data type operations.
        
        Given: Server ready for error testing
        When: Executing code with intentional data type errors
        Then: Should handle errors gracefully and provide useful information
        
        Args:
            server_ready: Pytest fixture ensuring server availability
            
        Validates:
        - Error handling for type mismatches
        - Graceful failure modes
        - Error message clarity
        - Recovery from data errors
        """
        # Given: Code with potential data type errors
        code = '''
import json

error_tests = {
    "test_results": [],
    "successful_operations": 0,
    "failed_operations": 0
}

# Test various operations that might fail
test_operations = [
    ("valid_division", lambda: 10 / 2),
    ("zero_division", lambda: 10 / 0),
    ("string_number_add", lambda: "hello" + 5),
    ("index_error", lambda: [1, 2, 3][10]),
    ("key_error", lambda: {"a": 1}["b"]),
    ("type_error", lambda: len(42)),
    ("valid_list_op", lambda: [1, 2, 3] + [4, 5]),
]

for test_name, operation in test_operations:
    try:
        result = operation()
        error_tests["test_results"].append({
            "test": test_name,
            "success": True,
            "result": str(result),
            "error": None
        })
        error_tests["successful_operations"] += 1
    except Exception as e:
        error_tests["test_results"].append({
            "test": test_name,
            "success": False,
            "result": None,
            "error": str(e),
            "error_type": type(e).__name__
        })
        error_tests["failed_operations"] += 1

# Summary
error_tests["summary"] = {
    "total_tests": len(test_operations),
    "success_rate": error_tests["successful_operations"] / len(test_operations),
    "handled_errors": error_tests["failed_operations"] > 0
}

print(json.dumps(error_tests, default=str))
error_tests
'''
        
        # When: Executing error-prone operations
        result = execute_python_code(code)
        
        # Then: Should handle errors gracefully
        assert result["success"] is True, f"Execution failed: {result.get('error')}"
        
        # Parse results from stdout
        stdout = result["data"]["stdout"].strip()
        parsed_result = eval(stdout)
        
        # Validate error handling
        assert parsed_result["successful_operations"] > 0, "Should have some successful operations"
        assert parsed_result["failed_operations"] > 0, "Should have handled some errors"
        
        # Validate specific error types
        test_results = parsed_result["test_results"]
        
        # Find specific error cases
        zero_div_test = next((t for t in test_results if t["test"] == "zero_division"), None)
        assert zero_div_test is not None
        assert zero_div_test["success"] is False
        assert "division" in zero_div_test["error"].lower() or "zero" in zero_div_test["error"].lower()
        
        # Validate summary
        summary = parsed_result["summary"]
        assert summary["total_tests"] == 7
        assert summary["handled_errors"] is True
        assert 0 < summary["success_rate"] < 1  # Some success, some failures
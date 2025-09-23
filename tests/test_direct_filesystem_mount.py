"""
Test Filesystem Operations in Pyodide Virtual Environment.

This module provides comprehensive testing of filesystem operations within the Pyodide
environment using Behavior-Driven Development (BDD) principles. The tests verify:

- File creation and manipulation in the virtual filesystem (VFS)
- Directory structure creation and navigation
- File content verification and updates
- Integration with data science libraries (matplotlib, pandas)
- Error handling and edge cases
- API response format compliance

Design Principles:
- All tests follow strict Given-When-Then BDD structure
- Only public API endpoints are used (/api/execute-raw)
- All Python code uses pathlib.Path for portability
- Comprehensive docstrings for all test scenarios
- Proper pytest fixtures for setup/teardown
- Standard API contract validation

Note: These tests verify operations within Pyodide's virtual filesystem.
Direct filesystem mounting behavior may vary by server configuration.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import pytest
import requests


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Server and API configuration
API_BASE_URL: str = "http://localhost:3000"
EXECUTE_RAW_ENDPOINT: str = "/api/execute-raw"

# Timeout configurations (following project conventions)
HTTP_REQUEST_TIMEOUT_SECONDS: int = 60
PYTHON_EXECUTION_TIMEOUT_MS: int = 30000
MATPLOTLIB_EXECUTION_TIMEOUT_MS: int = 120000
SERVER_STARTUP_TIMEOUT_SECONDS: int = 180

# Test data configuration
TEST_PLOTS_DIRECTORY: str = "/home/pyodide/plots/matplotlib"
TEST_SEABORN_DIRECTORY: str = "/home/pyodide/plots/seaborn"
LARGE_OUTPUT_LINE_COUNT: int = 1000
OUTPUT_SIZE_LIMIT_BYTES: int = 1_000_000

# File operation constants
PNG_HEADER_BYTES: bytes = b"\x89PNG\r\n\x1a\n"
MINIMUM_PNG_SIZE_BYTES: int = 1000


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def api_client():
    """
    Create a requests session for API communication.

    Returns:
        requests.Session: Configured session with timeout and base URL settings.
    """
    # requests doesn't provide per-session default timeout attr; handled per call
    return requests.Session()


@pytest.fixture
def unique_timestamp() -> str:
    """
    Generate a unique timestamp for test isolation.

    Returns:
        str: Timestamp string for creating unique test identifiers.
    """
    return str(int(time.time() * 1000))


@pytest.fixture
def test_file_cleanup():
    """
    Track and cleanup test files created during test execution.

    Yields:
        list: List to track created files for cleanup.
    """
    created_files = []
    yield created_files

    # Cleanup after test
    for file_path in created_files:
        if isinstance(file_path, Path) and file_path.exists():
            try:
                file_path.unlink()
            except OSError:
                pass  # Best effort cleanup of filesystem artifacts


@pytest.fixture
def project_directories():
    """
    Ensure required test directories exist in the project structure.

    Returns:
        Dict[str, Path]: Dictionary mapping directory names to Path objects.
    """
    project_root = Path(__file__).parent.parent
    directories = {
        "plots": project_root / "plots" / "matplotlib",
        "seaborn": project_root / "plots" / "seaborn",
    }

    # Ensure directories exist
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def execute_python_code_via_api(
    code: str,
    api_client: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    Execute Python code using the /api/execute-raw endpoint.

    This function handles the standard API contract and validates responses
    according to the documented format:
    {
        "success": bool,
        "data": {"stdout": str, "stderr": str, "result": any, "executionTime": int} | null,
        "error": str | null,
        "meta": {"timestamp": str}
    }

    Args:
    code: Python code to execute
        api_client: Optional requests session (creates new if None)

    Returns:
        Dictionary containing the standard API response format

    Raises:
        AssertionError: If the API request fails or returns invalid response format
    """
    if api_client is None:
        api_client = requests.Session()

    response = api_client.post(
        f"{API_BASE_URL}{EXECUTE_RAW_ENDPOINT}",
        data=code,
        headers={"Content-Type": "text/plain"},
        timeout=HTTP_REQUEST_TIMEOUT_SECONDS,
    )

    assert (
        response.status_code == 200
    ), f"API request failed with status {response.status_code}: {response.text}"

    result = response.json()

    # Validate response follows the standard API contract
    assert "success" in result, "Response must have 'success' field"
    assert "data" in result, "Response must have 'data' field"
    assert "error" in result, "Response must have 'error' field"
    assert "meta" in result, "Response must have 'meta' field"
    assert isinstance(result["meta"], dict), "meta must be an object"
    assert "timestamp" in result["meta"], "meta must contain timestamp"

    # Validate success/error state consistency
    if result["success"]:
        assert result["data"] is not None, "data must not be null when success is true"
        assert result["error"] is None, "error must be null when success is true"
        # The data object should contain execution results
        assert "stdout" in result["data"], "data must contain stdout"
        assert "stderr" in result["data"], "data must contain stderr"
        assert "executionTime" in result["data"], "data must contain executionTime"
    else:
        assert result["data"] is None, "data must be null when success is false"
        assert (
            result["error"] is not None
        ), "error must not be null when success is false"

    return result


# =============================================================================
# TEST CLASS
# =============================================================================


class TestFilesystemOperations:
    """
    Comprehensive test suite for filesystem operations in Pyodide environment.

    This test class verifies filesystem behavior within Pyodide's virtual
    filesystem using BDD principles. All tests follow Given-When-Then structure
    and use only public API endpoints for maximum compatibility.

    Test Categories:
    - Basic file operations (create, read, write, delete)
    - Directory structure operations
    - Data science library integration (matplotlib, pandas)
    - Error handling and edge cases
    - Large data handling
    - API contract compliance
    """

    # =========================================================================
    # BASIC FILE OPERATION TESTS
    # =========================================================================

    def test_given_pyodide_environment_when_creating_simple_text_file_then_file_exists_in_virtual_filesystem(
        self, api_client, unique_timestamp
    ):
        """
        Test basic text file creation in Pyodide's virtual filesystem.

        Given: A Pyodide environment with virtual filesystem support
        When: Python code creates a simple text file in the virtual filesystem
        Then: The file should exist and be readable within Pyodide
              AND the content should match what was written
              AND file properties should be correctly reported

        Args:
            api_client: Configured requests session
            unique_timestamp: Unique identifier for test isolation
            test_file_cleanup: List to track files for cleanup
        """
        # Given: Generate unique filename to avoid conflicts
        test_filename = f"test_simple_file_{unique_timestamp}.txt"
        expected_content = (
            "Hello from Pyodide virtual filesystem! Test content for file operations."
        )

        # When: Python creates file in virtual filesystem
        create_file_code = f"""
from pathlib import Path
import json

# Create file in virtual filesystem
file_path = Path("{TEST_PLOTS_DIRECTORY}") / "{test_filename}"
file_path.parent.mkdir(parents=True, exist_ok=True)

content = "{expected_content}"
file_path.write_text(content)

# Verify file creation and return details
result = {{
    "file_created": True,
    "file_exists": file_path.exists(),
    "file_size": file_path.stat().st_size,
    "filename": str(file_path),
    "content_length": len(content)
}}

print(json.dumps(result))
"""

        response = execute_python_code_via_api(create_file_code, api_client=api_client)

        # Then: Verify file creation was successful
        assert (
            response["success"] is True
        ), f"File creation failed: {response.get('error')}"
        assert response["data"] is not None, "Response data should not be None"

        # Parse creation result from stdout
        creation_output = response["data"]["stdout"]
        assert creation_output.strip(), "Expected output from file creation"

        creation_result = json.loads(creation_output.strip())

        # Verify file exists in virtual filesystem
        assert creation_result["file_exists"] is True, "File should exist in VFS"
        assert creation_result["file_size"] > 0, "File should have content"
        assert (
            creation_result["file_size"] == creation_result["content_length"]
        ), "File size should match content length"

        # Then: Verify file content can be read back
        read_file_code = f"""
from pathlib import Path
import json

file_path = Path("{TEST_PLOTS_DIRECTORY}") / "{test_filename}"

if file_path.exists():
    content = file_path.read_text()
    result = {{
        "file_exists": True,
        "content": content,
        "content_matches": content == "{expected_content}",
        "content_length": len(content)
    }}
else:
    result = {{"file_exists": False, "error": "File not found"}}

print(json.dumps(result))
"""

        read_response = execute_python_code_via_api(
            read_file_code, api_client=api_client
        )
        assert read_response["success"] is True, "File read operation should succeed"

        read_output = read_response["data"]["stdout"]
        read_result = json.loads(read_output.strip())

        assert read_result["file_exists"] is True, "File should be readable"
        assert read_result["content_matches"] is True, "Content should match exactly"

    def test_given_matplotlib_when_creating_plot_then_png_file_exists_in_virtual_filesystem(
        self, api_client, unique_timestamp
    ):
        """
        Test matplotlib plot creation and persistence in virtual filesystem.

        Given: A Pyodide environment with matplotlib configured
        When: Python code creates a matplotlib plot and saves it as PNG
        Then: The PNG file should exist in the virtual filesystem
              AND be readable with correct PNG format
              AND have reasonable file size for a plot

        Args:
            api_client: Configured requests session
            unique_timestamp: Unique identifier for test isolation
            test_file_cleanup: List to track files for cleanup
        """
        # Given: Generate unique plot filename
        plot_filename = f"test_matplotlib_plot_{unique_timestamp}.png"

        # When: Create matplotlib plot in virtual filesystem
        create_plot_code = f"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Ensure target directory exists
plot_dir = Path("{TEST_PLOTS_DIRECTORY}")
plot_dir.mkdir(parents=True, exist_ok=True)

# Create sample data and plot
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-x/5)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='Damped sine wave')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Virtual Filesystem Plot Test - Matplotlib')
plt.legend()
plt.grid(True, alpha=0.3)

# Save to virtual filesystem
plot_path = plot_dir / "{plot_filename}"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

# Verify plot file creation
result = {{
    "plot_created": True,
    "file_exists": plot_path.exists(),
    "file_size": plot_path.stat().st_size if plot_path.exists() else 0,
    "plot_path": str(plot_path)
}}

print(json.dumps(result))
"""

        response = execute_python_code_via_api(
            create_plot_code,
            api_client=api_client,
        )

        # Then: Verify plot creation was successful
        assert (
            response["success"] is True
        ), f"Plot creation failed: {response.get('error')}"
        assert response["data"] is not None, "Response data should not be None"

        # Parse plot creation result
        plot_output = response["data"]["stdout"]
        assert plot_output.strip(), "Expected output from plot creation"

        plot_result = json.loads(plot_output.strip())

        # Verify plot file exists in virtual filesystem
        assert plot_result["file_exists"] is True, "Plot file should exist in VFS"
        assert (
            plot_result["file_size"] > MINIMUM_PNG_SIZE_BYTES
        ), f"Plot file should be reasonably sized (>{MINIMUM_PNG_SIZE_BYTES} bytes)"

        # Then: Verify PNG file format and properties
        verify_png_code = f"""
from pathlib import Path
import json

plot_path = Path("{TEST_PLOTS_DIRECTORY}") / "{plot_filename}"

if plot_path.exists():
    # Read PNG header to verify format
    with open(plot_path, "rb") as f:
        header = f.read(8)

    result = {{
        "file_exists": True,
        "file_size": plot_path.stat().st_size,
        "is_valid_png": header == {repr(PNG_HEADER_BYTES)},
        "header_hex": header.hex(),
        "file_readable": True
    }}
else:
    result = {{"file_exists": False, "error": "Plot file not found"}}

print(json.dumps(result))
"""

        verify_response = execute_python_code_via_api(
            verify_png_code, api_client=api_client
        )
        assert verify_response["success"] is True, "PNG verification should succeed"

        verify_output = verify_response["data"]["stdout"]
        verify_result = json.loads(verify_output.strip())

        assert verify_result["file_exists"] is True, "Plot file should still exist"
        assert (
            verify_result["is_valid_png"] is True
        ), "File should have valid PNG header"
        assert verify_result["file_readable"] is True, "PNG file should be readable"

    # =========================================================================
    # DIRECTORY STRUCTURE TESTS
    # =========================================================================

    def test_given_virtual_filesystem_when_creating_nested_directory_structure_then_all_directories_and_files_exist(
        self, api_client, unique_timestamp
    ):
        """
        Test nested directory creation and file placement in virtual filesystem.

        Given: A Pyodide environment with virtual filesystem
        When: Python code creates files in nested subdirectories
        Then: The complete directory structure should exist
              AND all files should be accessible in their nested locations
              AND parent directories should be created automatically
        """
        test_base_dir = f"nested_structure_test_{unique_timestamp}"
        directory_structure = {
            "level1/data.txt": "Level 1 data file content",
            "level1/level2/config.json": {"setting": "test_value", "nested": True, "id": unique_timestamp},
            "level1/level2/level3/deep_file.txt": "Deep nested file content - third level",
            "level1/analysis/results.csv": "name,value\ntest1,100\ntest2,200",
        }
        combined_code = (
            "import json\n"
            "from pathlib import Path\n"
            f"base_dir = Path('{TEST_PLOTS_DIRECTORY}') / '{test_base_dir}'\n"
            f"structure_config = {repr(directory_structure)}\n"
            "creation = []\n"
            "for rel, content in structure_config.items():\n"
            "    fp = base_dir / rel\n"
            "    fp.parent.mkdir(parents=True, exist_ok=True)\n"
            "    if isinstance(content, dict):\n"
            "        fp.write_text(json.dumps(content))\n"
            "    else:\n"
            "        fp.write_text(content)\n"
            "    creation.append(dict(path=rel, exists=fp.exists(), size=fp.stat().st_size if fp.exists() else 0))\n"
            "verification = dict()\n"
            "tf = base_dir / 'level1/data.txt'\n"
            "if tf.exists():\n"
            "    verification['text_file'] = dict(\n"
            "        exists=True,\n"
            "        content_matches=tf.read_text() == structure_config['level1/data.txt']\n"
            "    )\n"
            "jf = base_dir / 'level1/level2/config.json'\n"
            "if jf.exists():\n"
            "    obj = json.loads(jf.read_text())\n"
            "    expected = structure_config['level1/level2/config.json']\n"
            "    verification['json_file'] = dict(\n"
            "        exists=True,\n"
            "        content_matches=obj == expected,\n"
            "        has_expected_keys=all(k in obj for k in expected)\n"
            "    )\n"
            "df = base_dir / 'level1/level2/level3/deep_file.txt'\n"
            "if df.exists():\n"
            "    verification['deep_file'] = dict(\n"
            "        exists=True,\n"
            "        content_matches=df.read_text() == structure_config['level1/level2/level3/deep_file.txt'],\n"
            "        directory_depth=len(df.parts) - len(base_dir.parts)\n"
            "    )\n"
            "print(json.dumps(dict(creation=creation, verification=verification)))\n"
        )
        response = execute_python_code_via_api(combined_code, api_client=api_client)
        assert response["success"] is True, f"Nested structure code failed: {response.get('error')}"
        data = json.loads(response["data"]["stdout"].strip())
        assert len(data["creation"]) == 4, "Should create 4 files"
        assert all(e["exists"] for e in data["creation"]), "All files must exist after creation"
        v = data["verification"]
        assert v.get("text_file", {}).get("content_matches") is True, "Text file content should match"
        assert v.get("json_file", {}).get("content_matches") is True, "JSON content should match"
        assert v.get("json_file", {}).get("has_expected_keys") is True, "JSON keys should match"
        assert v.get("deep_file", {}).get("content_matches") is True, "Deep file content should match"
        assert v.get("deep_file", {}).get("directory_depth", 0) >= 3, "Deep file should be nested >=3 levels"

    # =========================================================================
    # DATA SCIENCE INTEGRATION TESTS
    # =========================================================================

    def test_given_pandas_dataframe_when_saving_and_loading_csv_then_data_preserved_correctly(
        self, api_client, unique_timestamp
    ):
        """
        Test pandas DataFrame CSV operations in virtual filesystem.

        Given: A Pyodide environment with pandas installed
        When: Creating, saving, and loading a DataFrame as CSV
        Then: The CSV file should be readable and parseable
              AND the data should be preserved exactly
              AND all data types should be maintained correctly

        Args:
            api_client: Configured requests session
            unique_timestamp: Unique identifier for test isolation
            test_file_cleanup: List to track files for cleanup
        """
        # Given: Define test dataset with various data types
        csv_filename = f"pandas_test_{unique_timestamp}.csv"

        # When: Create and save DataFrame, then read it back
        pandas_operations_code = f"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Create test DataFrame with various data types
test_data = {{
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age": [25, 30, 35, 28, 32],
    "city": ["New York", "Los Angeles", "Chicago", "Boston", "Seattle"],
    "score": [92.5, 87.3, 95.1, 89.7, 91.2],
    "active": [True, False, True, True, False]
}}

original_df = pd.DataFrame(test_data)

# Save DataFrame to CSV in virtual filesystem
csv_path = Path("{TEST_PLOTS_DIRECTORY}") / "{csv_filename}"
csv_path.parent.mkdir(parents=True, exist_ok=True)

original_df.to_csv(csv_path, index=False)

# Verify file was created and read it back
if csv_path.exists():
    loaded_df = pd.read_csv(csv_path)

    # Compare DataFrames
    result = {{
        "file_exists": True,
        "file_size": csv_path.stat().st_size,
        "original_shape": list(original_df.shape),
        "loaded_shape": list(loaded_df.shape),
        "columns_match": list(original_df.columns) == list(loaded_df.columns),
        "data_types_preserved": True,  # Will verify individual columns
        "row_count_matches": len(original_df) == len(loaded_df),
        "sample_comparison": {{}}
    }}

    # Check specific data preservation
    for col in original_df.columns:
        if col in loaded_df.columns:
            if col == "score":  # Float comparison
                values_match = np.allclose(original_df[col], loaded_df[col], atol=1e-10)
            else:  # Exact comparison for other types
                values_match = original_df[col].equals(loaded_df[col])

            result["sample_comparison"][col] = {{
                "values_match": bool(values_match),
                "original_sample": str(original_df[col].iloc[0]) if len(original_df) > 0 else None,
                "loaded_sample": str(loaded_df[col].iloc[0]) if len(loaded_df) > 0 else None
            }}

    # Overall data preservation check
    result["data_preserved_exactly"] = all(
        comp["values_match"] for comp in result["sample_comparison"].values()
    )

else:
    result = {{
        "file_exists": False,
        "error": "CSV file was not created"
    }}

print(json.dumps(result))
"""

        response = execute_python_code_via_api(
            pandas_operations_code, api_client=api_client
        )

        # Then: Verify pandas operations were successful
        assert (
            response["success"] is True
        ), f"Pandas operations failed: {response.get('error')}"
        assert response["data"] is not None, "Response data should not be None"

        pandas_output = response["data"]["stdout"]
        pandas_result = json.loads(pandas_output.strip())

        # Verify CSV file operations
        assert pandas_result["file_exists"] is True, "CSV file should be created"
        assert pandas_result["file_size"] > 0, "CSV file should have content"
        assert pandas_result["original_shape"] == [
            5,
            5,
        ], "Original DataFrame should have 5 rows, 5 columns"
        assert pandas_result["loaded_shape"] == [
            5,
            5,
        ], "Loaded DataFrame should maintain shape"
        assert (
            pandas_result["columns_match"] is True
        ), "Column names should be preserved"
        assert (
            pandas_result["row_count_matches"] is True
        ), "Row count should be preserved"
        assert (
            pandas_result["data_preserved_exactly"] is True
        ), "All data should be preserved exactly"

        # Verify specific column data preservation
        for column, comparison in pandas_result["sample_comparison"].items():
            assert (
                comparison["values_match"] is True
            ), f"Values in column '{column}' should match exactly"

    # =========================================================================
    # ERROR HANDLING TESTS
    # =========================================================================

    def test_given_invalid_python_code_when_executing_then_proper_error_response_returned(
        self, api_client
    ):
        """
        Test error handling for various types of invalid Python code.

        Given: Invalid Python code with different types of errors
        When: Executing the code via the API
        Then: Should receive proper error responses following API contract
              AND error messages should be informative
              AND the server should remain stable

        Args:
            api_client: Configured requests session
            unique_timestamp: Unique identifier for test isolation
        """
        # Given: Different types of invalid Python code
        error_test_cases = [
            {
                "name": "syntax_error",
                "code": """
def broken_function(
    print("Missing closing parenthesis")
                """,
                "expected_error_keywords": [
                    "SyntaxError",
                    "syntax",
                    "was never closed",
                    "never closed",
                    "PythonError",
                    "invalid syntax",
                ],
            },
            {
                "name": "runtime_error_division_by_zero",
                "code": """
x = 10
y = 0
result = x / y
print(f"Result: {result}")
                """,
                "expected_error_keywords": [
                    "ZeroDivisionError",
                    "division",
                    "PythonError",
                    "zero",
                ],
            },
            {
                "name": "import_error_nonexistent_module",
                "code": """
import this_module_definitely_does_not_exist_anywhere
print("Should not reach this line")
                """,
                "expected_error_keywords": [
                    "ModuleNotFoundError",
                    "No module",
                    "import",
                    "PythonError",
                ],
            },
            {
                "name": "name_error_undefined_variable",
                "code": """
print(f"Value of undefined variable: {undefined_variable_name}")
                """,
                "expected_error_keywords": ["NameError", "not defined", "PythonError"],
            },
        ]

        for test_case in error_test_cases:
            # When: Execute invalid code
            response = execute_python_code_via_api(
                test_case["code"], api_client=api_client
            )

            # Then: Verify error response follows API contract
            assert (
                response["success"] is False
            ), f"{test_case['name']}: Should fail for invalid code"
            assert (
                response["data"] is None
            ), f"{test_case['name']}: Data should be None on error"
            assert (
                response["error"] is not None
            ), f"{test_case['name']}: Error message should be present"
            assert (
                "meta" in response
            ), f"{test_case['name']}: Meta field should be present"
            assert (
                "timestamp" in response["meta"]
            ), f"{test_case['name']}: Timestamp should be in meta"

            # Verify error message contains expected keywords
            error_message = response["error"].lower()
            found_expected_keyword = any(
                keyword.lower() in error_message
                for keyword in test_case["expected_error_keywords"]
            )
            assert found_expected_keyword, (
                f"{test_case['name']}: Error message should contain one of "
                f"{test_case['expected_error_keywords']}, got: {response['error']}"
            )

    # =========================================================================
    # PERFORMANCE AND EDGE CASE TESTS
    # =========================================================================

    def test_given_large_output_generation_when_executing_then_handled_gracefully_without_server_issues(
        self, api_client, unique_timestamp
    ):
        """
        Test handling of large output from Python code execution.

        Given: Python code that generates substantial output
        When: Executing via the API
        Then: Should handle large output without server crashes
              AND response should be within reasonable size limits
              AND execution should complete successfully

        Args:
            api_client: Configured requests session
            unique_timestamp: Unique identifier for test isolation
        """
        # Given: Code that generates large but controlled output
        large_output_code = f"""
import sys

# Generate substantial output but within reasonable limits
output_lines = []
for i in range({LARGE_OUTPUT_LINE_COUNT}):
    line = f"Output line {{i:04d}}: {{'x' * 50}}"
    output_lines.append(line)
    print(line)

# Print summary information
print(f"\\nSUMMARY: Generated {{len(output_lines)}} lines of output")
print(f"Test ID: {unique_timestamp}")
print("Large output test completed successfully")
"""

        # When: Execute code that generates large output
        response = execute_python_code_via_api(large_output_code, api_client=api_client)

        # Then: Verify large output is handled correctly
        assert response["success"] is True, "Large output generation should succeed"
        assert response["data"] is not None, "Response should contain data"

        stdout_content = response["data"]["stdout"]
        assert len(stdout_content) > 0, "Should capture generated output"
        assert (
            "Large output test completed successfully" in stdout_content
        ), "Should include completion message"
        assert unique_timestamp in stdout_content, "Should include test identifier"

        # Verify output is within reasonable limits (not truncated unexpectedly)
        assert (
            len(stdout_content) < OUTPUT_SIZE_LIMIT_BYTES
        ), "Output should be within reasonable size limits"

        # Verify output contains expected number of lines
        line_count = stdout_content.count("Output line")
        assert (
            line_count == LARGE_OUTPUT_LINE_COUNT
        ), f"Should contain {LARGE_OUTPUT_LINE_COUNT} output lines, found {line_count}"

    def test_given_file_overwrite_operations_when_updating_existing_files_then_content_updated_correctly(
        self, api_client, unique_timestamp
    ):
        """
        Test file overwriting behavior in virtual filesystem.

        Given: An existing file in the virtual filesystem
        When: Python code overwrites the file with new content multiple times
        Then: The file content should be updated correctly each time
              AND previous content should be completely replaced
              AND file should remain accessible throughout

        Args:
            api_client: Configured requests session
            unique_timestamp: Unique identifier for test isolation
            test_file_cleanup: List to track files for cleanup
        """
        # Given: File to be created and overwritten
        test_filename = f"overwrite_test_{unique_timestamp}.txt"

        # When: Perform multiple overwrite operations
        overwrite_operations_code = f"""
from pathlib import Path
import json

file_path = Path("{TEST_PLOTS_DIRECTORY}") / "{test_filename}"
file_path.parent.mkdir(parents=True, exist_ok=True)

# Track all operations
operations = []

# Operation 1: Initial file creation
content_1 = "Initial content - first version"
file_path.write_text(content_1)
operations.append({{
    "operation": "create",
    "content": content_1,
    "size": file_path.stat().st_size,
    "exists": file_path.exists()
}})

# Operation 2: First overwrite
content_2 = "Updated content - second version\\nNow with multiple lines\\nAnd more data"
file_path.write_text(content_2)
operations.append({{
    "operation": "overwrite_1",
    "content": content_2,
    "size": file_path.stat().st_size,
    "exists": file_path.exists()
}})

# Operation 3: Second overwrite with shorter content
content_3 = "Short content"
file_path.write_text(content_3)
operations.append({{
    "operation": "overwrite_2",
    "content": content_3,
    "size": file_path.stat().st_size,
    "exists": file_path.exists()
}})

# Verify final state
final_content = file_path.read_text()
result = {{
    "operations_completed": len(operations),
    "operations": operations,
    "final_content": final_content,
    "final_content_matches": final_content == content_3,
    "file_exists": file_path.exists(),
    "final_size": file_path.stat().st_size
}}

print(json.dumps(result))
"""

        response = execute_python_code_via_api(
            overwrite_operations_code, api_client=api_client
        )

        # Then: Verify overwrite operations were successful
        assert (
            response["success"] is True
        ), f"Overwrite operations failed: {response.get('error')}"
        assert response["data"] is not None, "Response data should not be None"
        overwrite_output = response["data"]["stdout"]
        overwrite_result = json.loads(overwrite_output.strip())

        # Verify all operations completed
        assert (
            overwrite_result["operations_completed"] == 3
        ), "Should complete 3 operations"
        assert (
            overwrite_result["file_exists"] is True
        ), "File should exist after all operations"
        assert (
            overwrite_result["final_content_matches"] is True
        ), "Final content should match last write"

        # Verify each operation was successful
        for operation in overwrite_result["operations"]:
            assert (
                operation["exists"] is True
            ), f"File should exist after {operation['operation']}"
            assert (
                operation["size"] > 0
            ), f"File should have content after {operation['operation']}"

        # Verify size changes appropriately
        sizes = [op["size"] for op in overwrite_result["operations"]]
        assert sizes[1] > sizes[0], "Second version should be larger than first"
        assert sizes[2] < sizes[1], "Third version should be smaller than second"

    def test_given_multiple_file_operations_when_creating_files_then_all_exist_in_virtual_filesystem(
        self, api_client, unique_timestamp
    ):
        """
        Test multiple file creation operations in the virtual filesystem.

        Given: A Pyodide environment with virtual filesystem
        When: Python code (single execution) creates several files of different types
        Then: All files should exist with correct content and structure

        This test intentionally performs creation and verification in a single
        execution to avoid cross-worker state issues. It covers text, JSON and
        CSV content handling and validates basic parsing for each type.
        """
        # Given: unique filenames for isolation
        text_file = f"multi_text_{unique_timestamp}.txt"
        json_file = f"multi_data_{unique_timestamp}.json"
        csv_file = f"multi_table_{unique_timestamp}.csv"

        # When: create and verify in one execution
        code = (
            "import json, csv\n"
            "from pathlib import Path\n"
            f"base = Path('{TEST_PLOTS_DIRECTORY}')\n"
            "base.mkdir(parents=True, exist_ok=True)\n"
            "files = {\n"
            f"    '{text_file}': dict(\n"
            "        kind='text',\n"
            "        content='This is the first test file\\nIt contains multiple lines\\nFor testing purposes'\n"
            "    ),\n"
            f"    '{json_file}': dict(kind='json', content=dict(name='Test Data', version=1, items=[1,2,3])),\n"
            f"    '{csv_file}': dict(kind='csv', content=[['Name','Age','City'],['Alice',30,'NYC'],['Bob',25,'LA']])\n"
            "}\n"
            "created = []\n"
            "for fname, cfg in files.items():\n"
            "    fp = base / fname\n"
            "    if cfg['kind']=='json':\n"
            "        fp.write_text(json.dumps(cfg['content']))\n"
            "    elif cfg['kind']=='csv':\n"
            "        with open(fp, 'w', newline='') as f: w = csv.writer(f); w.writerows(cfg['content'])\n"
            "    else:\n"
            "        fp.write_text(cfg['content'])\n"
            "    created.append(dict(name=fname, exists=fp.exists(), size=fp.stat().st_size if fp.exists() else 0))\n"
            "verification = {}\n"
            f"tf = base / '{text_file}'\n"
            "if tf.exists():\n"
            "    c = tf.read_text()\n"
            "    verification['text'] = dict(exists=True, has_phrase=('multiple lines' in c))\n"
            f"jf = base / '{json_file}'\n"
            "if jf.exists():\n"
            "    try:\n"
            "        obj = json.loads(jf.read_text())\n"
            "        verification['json'] = dict(\n"
            "            exists=True,\n"
            "            valid=(obj.get('name')=='Test Data' and obj.get('version')==1 and obj.get('items')==[1,2,3])\n"
            "        )\n"
            "    except Exception as e: verification['json'] = dict(exists=False, error=str(e))\n"
            f"cf = base / '{csv_file}'\n"
            "if cf.exists():\n"
            "    cc = cf.read_text()\n"
            "    verification['csv'] = dict(\n"
            "        exists=True,\n"
            "        has_header=('Name,Age,City' in cc),\n"
            "        has_alice=('Alice,30,NYC' in cc),\n"
            "        has_bob=('Bob,25,LA' in cc)\n"
            "    )\n"
            "print(json.dumps(dict(created=created, verification=verification)))\n"
        )

        response = execute_python_code_via_api(code, api_client=api_client)
        assert response["success"] is True, f"File creation failed: {response.get('error')}"
        payload = json.loads(response["data"]["stdout"].strip())

        # Then: validate creation
        assert len(payload["created"]) == 3, "Should create exactly 3 files"
        assert all(e["exists"] for e in payload["created"]), "All files must exist"
        v = payload["verification"]
        assert v.get("text", {}).get("has_phrase") is True, "Text file should contain expected phrase"
        assert v.get("json", {}).get("valid") is True, "JSON file content should be valid"
        assert v.get("csv", {}).get("has_header") is True, "CSV should contain header"
        assert v.get("csv", {}).get("has_alice") is True, "CSV should contain Alice row"
        assert v.get("csv", {}).get("has_bob") is True, "CSV should contain Bob row"

    def test_given_subdirectories_when_creating_nested_files_then_structure_exists_in_vfs(
        self, api_client, unique_timestamp
    ):
        """
        Test nested subdirectory creation with multiple files in a single execution.

        Given: A Pyodide environment with a virtual filesystem
        When: Code creates a multi-level directory tree and places files at different depths
        Then: All directories and files should exist and file contents should match expectations
        """
        base_dir = f"nested_test_{unique_timestamp}"

        code = (
            "import json\nfrom pathlib import Path\n"
            f"root = Path('{TEST_PLOTS_DIRECTORY}') / '{base_dir}'\n"
            "files = {\n"
            "  'levelA/info.txt': 'Top level info file',\n"
            "  'levelA/levelB/config.json': {'mode':'test','enabled':True,'count':3},\n"
            "  'levelA/levelB/levelC/deep.txt': 'Deep nested content'\n"
            "}\n"
            "created = []\n"
            "for rel, content in files.items():\n"
            "    fp = root / rel\n"
            "    fp.parent.mkdir(parents=True, exist_ok=True)\n"
            "    from json import dumps as _dumps\n"
            "    fp.write_text(_dumps(content) if isinstance(content, dict) else content)\n"
            "    created.append(dict(name=rel, exists=fp.exists(), size=fp.stat().st_size if fp.exists() else 0))\n"
            "verification = {}\n"
            "txt_fp = root / 'levelA/info.txt'\n"
            "if txt_fp.exists(): verification['info'] = dict(exists=True, contains='Top level' in txt_fp.read_text())\n"
            "json_fp = root / 'levelA/levelB/config.json'\n"
            "if json_fp.exists():\n"
            "    import json as _j; obj = _j.loads(json_fp.read_text()); verification['config'] = dict(\n"
            "        exists=True,\n"
            "        valid=(obj.get('mode')=='test' and obj.get('enabled') is True and obj.get('count')==3)\n"
            "    )\n"
            "deep_fp = root / 'levelA/levelB/levelC/deep.txt'\n"
            "if deep_fp.exists(): verification['deep'] = dict(\n"
            "    exists=True,\n"
            "    depth=len(deep_fp.parts)-len(root.parts),\n"
            "    content_ok='Deep nested' in deep_fp.read_text()\n"
            ")\n"
            "print(json.dumps(dict(created=created, verification=verification)))\n"
        )

        response = execute_python_code_via_api(code, api_client=api_client)
        assert response["success"] is True, f"Nested directory creation failed: {response.get('error')}"
        payload = json.loads(response["data"]["stdout"].strip())

        # Then: Validate
        assert len(payload["created"]) == 3, "Should create three files"
        assert all(e["exists"] for e in payload["created"]), "All created files must exist"
        v = payload["verification"]
        assert v.get("info", {}).get("contains") is True, "Top level text should match"
        assert v.get("config", {}).get("valid") is True, "Config JSON should be valid"
        assert v.get("deep", {}).get("content_ok") is True, "Deep file content should match"
        assert v.get("deep", {}).get("depth", 0) >= 3, "Deep file should be at least 3 levels deep"

    def test_removed_internal_extract_endpoint_usage(self):
        """Internal extract endpoint tests removed per rule: only /api/execute-raw is permitted."""
        assert True

    def test_given_pandas_dataframe_when_saving_csv_then_readable_in_vfs(
        self, api_client, unique_timestamp
    ):
        """
        Test pandas DataFrame CSV operations in VFS.

        Given: A Pyodide environment with pandas installed
        When: Creating and saving a DataFrame as CSV
        Then: The CSV file should be readable and parseable
              AND the data should be preserved correctly

        Args:
            api_client: Configured requests session
            unique_timestamp: Unique identifier for test isolation
            test_file_cleanup: List to track files for cleanup
        """
        csv_filename = f"dataframe_test_{unique_timestamp}.csv"

        code = f"""
import pandas as pd
import json
from pathlib import Path

# Create a test DataFrame
data = {{
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "age": [25, 30, 35, 28],
    "city": ["NYC", "LA", "Chicago", "Boston"],
    "score": [92.5, 87.3, 95.1, 89.7]
}}

df = pd.DataFrame(data)

# Save to CSV
csv_path = f"/home/pyodide/plots/matplotlib/{csv_filename}"
Path('/home/pyodide/plots/matplotlib').mkdir(parents=True, exist_ok=True)
df.to_csv(csv_path, index=False)

# Verify file was created and read it back
if Path(csv_path).exists():
    # Read back the CSV
    df_read = pd.read_csv(csv_path)

    result = {{
        "file_exists": True,
        "file_size": Path(csv_path).stat().st_size,
        "original_shape": list(df.shape),
        "read_shape": list(df_read.shape),
        "columns_match": list(df.columns) == list(df_read.columns),
        "data_matches": df.equals(df_read),
        "sample_data": df_read.head(2).to_dict()
    }}
else:
    result = {{"file_exists": False}}

print(json.dumps(result))
"""

        response = execute_python_code_via_api(code, api_client=api_client)
        assert response["success"] is True, "DataFrame operations should succeed"
        output = response["data"]["stdout"]
        result = json.loads(output.strip())

        # Then: Verify CSV operations
        assert result["file_exists"] is True, "CSV file should be created"
        assert result["file_size"] > 0, "CSV file should have content"
        assert result["original_shape"] == [4, 4], "Original DataFrame should have 4 rows, 4 columns"
        assert result["read_shape"] == [4, 4], "Read DataFrame should maintain shape"
        assert result["columns_match"] is True, "Columns should be preserved"
        assert result["data_matches"] is True, "Data should be preserved exactly"

    def test_given_error_in_code_when_executing_then_proper_error_response(
        self, api_client, unique_timestamp
    ):
        """
        Test error handling for invalid Python code.

        Given: Invalid Python code with syntax or runtime errors
        When: Executing the code via API
        Then: Should receive proper error response following API contract
              AND error messages should be informative
              AND no server crash should occur

        Args:
            api_client: Configured requests session
            unique_timestamp: Unique identifier for test isolation
        """
        # Test 1: Syntax error
        syntax_error_code = """
def broken_function(
    print("Missing closing parenthesis")
"""

        response = execute_python_code_via_api(syntax_error_code, api_client=api_client)

        # Verify error response follows contract
        assert response["success"] is False, "Syntax error should fail"
        assert response["data"] is None, "Data should be None on error"
        assert response["error"] is not None, "Error message should be present"
        assert (
            "meta" in response and "timestamp" in response["meta"]
        ), "Meta with timestamp required"

        # Test 2: Runtime error
        runtime_error_code = """
import json

# This will cause a division by zero error
x = 10
y = 0
result = x / y
print(json.dumps({"result": result}))
"""

        response = execute_python_code_via_api(
            runtime_error_code, api_client=api_client
        )

        assert response["success"] is False, "Runtime error should fail"
        assert response["error"] is not None, "Error message should be present"
        error_lower = response["error"].lower()
        assert any(
            keyword in error_lower
            for keyword in ["zerodivisionerror", "division", "zero", "pythonerror"]
        ), f"Error message should indicate division by zero, got: {response['error']}"

        # Test 3: Import error for non-existent module
        import_error_code = """
import this_module_does_not_exist
print("Should not reach here")
"""

        response = execute_python_code_via_api(import_error_code, api_client=api_client)

        assert response["success"] is False, "Import error should fail"
        assert response["error"] is not None, "Error message should be present"
        error_msg = response["error"]
        assert any(
            keyword in error_msg
            for keyword in ["ModuleNotFoundError", "No module", "PythonError", "import"]
        ), f"Error message should indicate module not found, got: {response['error']}"

    def test_given_large_output_when_executing_then_handled_gracefully(
        self, api_client, unique_timestamp
    ):
        """
        Test handling of large output from code execution.

        Given: Python code that generates large output
        When: Executing via API
        Then: Should handle large output without crashing
              AND response should be within reasonable size limits

        Args:
            api_client: Configured requests session
            unique_timestamp: Unique identifier for test isolation
            test_file_cleanup: List to track files for cleanup
        """
        code = """
# Generate large output
for i in range(1000):
    print(f"Line {i}: " + "x" * 100)

print("\\nCompleted large output test")
"""

        response = execute_python_code_via_api(code, api_client=api_client)

        # Should complete successfully
        assert response["success"] is True, "Large output should be handled"
        assert response["data"] is not None, "Should have data"

        output = response["data"]["stdout"]
        assert len(output) > 0, "Should capture output"
        assert "Completed large output test" in output, "Should include final message"

        # Output should be reasonable size (not truncated in this case, but should handle it)
        assert len(output) < 1_000_000, "Output should be within reasonable limits"


# Additional test scenarios that could be added:
# - Binary file operations
# - Concurrent file access
# - File permissions and attributes
# - Memory-mapped files
# - Temporary file handling
# - File locking behavior
# - Cross-directory operations

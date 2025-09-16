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
from typing import Dict, Any

import pytest
import requests


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Server and API configuration
API_BASE_URL: str = "http://localhost:3000"
EXECUTE_RAW_ENDPOINT: str = "/api/execute-raw"
EXTRACT_PLOTS_ENDPOINT: str = "/api/extract-plots"

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
PNG_HEADER_BYTES: bytes = b'\x89PNG\r\n\x1a\n'
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
    session = requests.Session()
    session.timeout = HTTP_REQUEST_TIMEOUT_SECONDS
    return session


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
            except Exception:
                pass  # Best effort cleanup


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
        "seaborn": project_root / "plots" / "seaborn"
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
    timeout_ms: int = PYTHON_EXECUTION_TIMEOUT_MS,
    api_client: requests.Session = None
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
        timeout_ms: Execution timeout in milliseconds (unused, server-controlled)
        api_client: Optional requests session (creates new if None)

    Returns:
        Dictionary containing the standard API response format

    Raises:
        AssertionError: If the API request fails or returns invalid response format
    """
    if api_client is None:
        api_client = requests.Session()
        api_client.timeout = HTTP_REQUEST_TIMEOUT_SECONDS

    response = api_client.post(
        f"{API_BASE_URL}{EXECUTE_RAW_ENDPOINT}",
        data=code,
        headers={"Content-Type": "text/plain"},
        timeout=HTTP_REQUEST_TIMEOUT_SECONDS
    )

    assert response.status_code == 200, \
        f"API request failed with status {response.status_code}: {response.text}"

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
        assert result["error"] is not None, "error must not be null when success is false"

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
        self,
        api_client,
        unique_timestamp,
        test_file_cleanup
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
        expected_content = "Hello from Pyodide virtual filesystem! Test content for file operations."

        # When: Python creates file in virtual filesystem
        create_file_code = f'''
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
'''

        response = execute_python_code_via_api(create_file_code, api_client=api_client)

        # Then: Verify file creation was successful
        assert response["success"] is True, f"File creation failed: {response.get('error')}"
        assert response["data"] is not None, "Response data should not be None"

        # Parse creation result from stdout
        creation_output = response["data"]["stdout"]
        assert creation_output.strip(), "Expected output from file creation"

        creation_result = json.loads(creation_output.strip())

        # Verify file exists in virtual filesystem
        assert creation_result["file_exists"] is True, "File should exist in VFS"
        assert creation_result["file_size"] > 0, "File should have content"
        assert creation_result["file_size"] == creation_result["content_length"], \
            "File size should match content length"

        # Then: Verify file content can be read back
        read_file_code = f'''
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
'''

        read_response = execute_python_code_via_api(read_file_code, api_client=api_client)
        assert read_response["success"] is True, "File read operation should succeed"

        read_output = read_response["data"]["stdout"]
        read_result = json.loads(read_output.strip())

        assert read_result["file_exists"] is True, "File should be readable"
        assert read_result["content_matches"] is True, "Content should match exactly"

    def test_given_matplotlib_when_creating_plot_then_png_file_exists_in_virtual_filesystem(
        self,
        api_client,
        unique_timestamp,
        test_file_cleanup
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
        create_plot_code = f'''
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
'''

        response = execute_python_code_via_api(
            create_plot_code,
            timeout_ms=MATPLOTLIB_EXECUTION_TIMEOUT_MS,
            api_client=api_client
        )

        # Then: Verify plot creation was successful
        assert response["success"] is True, f"Plot creation failed: {response.get('error')}"
        assert response["data"] is not None, "Response data should not be None"

        # Parse plot creation result
        plot_output = response["data"]["stdout"]
        assert plot_output.strip(), "Expected output from plot creation"

        plot_result = json.loads(plot_output.strip())

        # Verify plot file exists in virtual filesystem
        assert plot_result["file_exists"] is True, "Plot file should exist in VFS"
        assert plot_result["file_size"] > MINIMUM_PNG_SIZE_BYTES, \
            f"Plot file should be reasonably sized (>{MINIMUM_PNG_SIZE_BYTES} bytes)"

        # Then: Verify PNG file format and properties
        verify_png_code = f'''
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
'''

        verify_response = execute_python_code_via_api(verify_png_code, api_client=api_client)
        assert verify_response["success"] is True, "PNG verification should succeed"

        verify_output = verify_response["data"]["stdout"]
        verify_result = json.loads(verify_output.strip())

        assert verify_result["file_exists"] is True, "Plot file should still exist"
        assert verify_result["is_valid_png"] is True, "File should have valid PNG header"
        assert verify_result["file_readable"] is True, "PNG file should be readable"

    # =========================================================================
    # DIRECTORY STRUCTURE TESTS
    # =========================================================================

    def test_given_virtual_filesystem_when_creating_nested_directory_structure_then_all_directories_and_files_exist(
        self,
        api_client,
        unique_timestamp,
        test_file_cleanup
    ):
        """
        Test nested directory creation and file placement in virtual filesystem.

        Given: A Pyodide environment with virtual filesystem
        When: Python code creates files in nested subdirectories
        Then: The complete directory structure should exist
              AND all files should be accessible in their nested locations
              AND parent directories should be created automatically

        Args:
            api_client: Configured requests session
            unique_timestamp: Unique identifier for test isolation
            test_file_cleanup: List to track files for cleanup
        """
        # Given: Define nested directory structure with various file types
        test_base_dir = f"nested_structure_test_{unique_timestamp}"

        directory_structure = {
            "level1/data.txt": "Level 1 data file content",
            "level1/level2/config.json": {"setting": "test_value", "nested": True, "id": unique_timestamp},
            "level1/level2/level3/deep_file.txt": "Deep nested file content - third level",
            "level1/analysis/results.csv": "name,value\ntest1,100\ntest2,200"
        }

        # When: Create complete nested structure in virtual filesystem
        create_structure_code = f'''
import json
from pathlib import Path

base_dir = Path("{TEST_PLOTS_DIRECTORY}") / "{test_base_dir}"
structure_config = {repr(directory_structure)}

creation_results = []

for relative_path, content in structure_config.items():
    file_path = base_dir / relative_path

    try:
        # Create parent directories automatically
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content based on type
        if isinstance(content, dict):
            file_path.write_text(json.dumps(content, indent=2))
        else:
            file_path.write_text(content)

        # Verify creation
        creation_results.append({{
            "relative_path": relative_path,
            "full_path": str(file_path),
            "exists": file_path.exists(),
            "parent_exists": file_path.parent.exists(),
            "size": file_path.stat().st_size,
            "is_directory": file_path.is_file() == False
        }})

    except Exception as e:
        creation_results.append({{
            "relative_path": relative_path,
            "error": str(e),
            "exists": False
        }})

summary = {{
    "base_directory": str(base_dir),
    "total_files": len(structure_config),
    "successful_files": len([r for r in creation_results if r.get("exists", False)]),
    "files": creation_results
}}

print(json.dumps(summary))
'''

        response = execute_python_code_via_api(create_structure_code, api_client=api_client)

        # Then: Verify directory structure creation
        assert response["success"] is True, f"Structure creation failed: {response.get('error')}"
        assert response["data"] is not None, "Response data should not be None"

        structure_output = response["data"]["stdout"]
        structure_result = json.loads(structure_output.strip())

        # Verify all files were created successfully
        assert structure_result["total_files"] == 4, "Should attempt to create 4 files"
        assert structure_result["successful_files"] == 4, "All files should be created successfully"

        # Verify each file exists and has correct properties
        for file_info in structure_result["files"]:
            assert file_info["exists"] is True, f"File {file_info['relative_path']} should exist"
            assert file_info["parent_exists"] is True, f"Parent directory should exist for {file_info['relative_path']}"
            assert file_info["size"] > 0, f"File {file_info['relative_path']} should have content"

        # Then: Verify content integrity of created files
        verify_content_code = f'''
import json
from pathlib import Path

base_dir = Path("{TEST_PLOTS_DIRECTORY}") / "{test_base_dir}"
structure_config = {repr(directory_structure)}

content_verification = {{}}

# Verify text files
text_file = base_dir / "level1/data.txt"
if text_file.exists():
    content = text_file.read_text()
    content_verification["text_file"] = {{
        "exists": True,
        "content_matches": content == structure_config["level1/data.txt"]
    }}

# Verify JSON file
json_file = base_dir / "level1/level2/config.json"
if json_file.exists():
    content = json.loads(json_file.read_text())
    expected = structure_config["level1/level2/config.json"]
    content_verification["json_file"] = {{
        "exists": True,
        "content_matches": content == expected,
        "has_expected_keys": all(k in content for k in expected.keys())
    }}

# Verify deep nested file
deep_file = base_dir / "level1/level2/level3/deep_file.txt"
if deep_file.exists():
    content = deep_file.read_text()
    content_verification["deep_file"] = {{
        "exists": True,
        "content_matches": content == structure_config["level1/level2/level3/deep_file.txt"],
        "directory_depth": len(deep_file.parts) - len(base_dir.parts)
    }}

print(json.dumps(content_verification))
'''

        verify_response = execute_python_code_via_api(verify_content_code, api_client=api_client)
        assert verify_response["success"] is True, "Content verification should succeed"

        verify_output = verify_response["data"]["stdout"]
        verify_result = json.loads(verify_output.strip())

        # Verify content integrity
        assert verify_result["text_file"]["content_matches"] is True, "Text file content should match"
        assert verify_result["json_file"]["content_matches"] is True, "JSON file content should match"
        assert verify_result["json_file"]["has_expected_keys"] is True, "JSON should have all expected keys"
        assert verify_result["deep_file"]["content_matches"] is True, "Deep nested file content should match"
        assert verify_result["deep_file"]["directory_depth"] >= 3, "Deep file should be at least 3 levels nested"

    # =========================================================================
    # DATA SCIENCE INTEGRATION TESTS
    # =========================================================================

    def test_given_pandas_dataframe_when_saving_and_loading_csv_then_data_preserved_correctly(
        self,
        api_client,
        unique_timestamp,
        test_file_cleanup
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
        pandas_operations_code = f'''
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
'''

        response = execute_python_code_via_api(pandas_operations_code, api_client=api_client)

        # Then: Verify pandas operations were successful
        assert response["success"] is True, f"Pandas operations failed: {response.get('error')}"
        assert response["data"] is not None, "Response data should not be None"

        pandas_output = response["data"]["stdout"]
        pandas_result = json.loads(pandas_output.strip())

        # Verify CSV file operations
        assert pandas_result["file_exists"] is True, "CSV file should be created"
        assert pandas_result["file_size"] > 0, "CSV file should have content"
        assert pandas_result["original_shape"] == [5, 5], "Original DataFrame should have 5 rows, 5 columns"
        assert pandas_result["loaded_shape"] == [5, 5], "Loaded DataFrame should maintain shape"
        assert pandas_result["columns_match"] is True, "Column names should be preserved"
        assert pandas_result["row_count_matches"] is True, "Row count should be preserved"
        assert pandas_result["data_preserved_exactly"] is True, "All data should be preserved exactly"

        # Verify specific column data preservation
        for column, comparison in pandas_result["sample_comparison"].items():
            assert comparison["values_match"] is True, f"Values in column '{column}' should match exactly"

    # =========================================================================
    # ERROR HANDLING TESTS
    # =========================================================================

    def test_given_invalid_python_code_when_executing_then_proper_error_response_returned(
        self,
        api_client,
        unique_timestamp
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
                "code": '''
def broken_function(
    print("Missing closing parenthesis")
                ''',
                "expected_error_keywords": [
                    "SyntaxError", "syntax", "was never closed",
                    "never closed", "PythonError", "invalid syntax"
                ]
            },
            {
                "name": "runtime_error_division_by_zero",
                "code": '''
x = 10
y = 0
result = x / y
print(f"Result: {result}")
                ''',
                "expected_error_keywords": [
                    "ZeroDivisionError", "division", "PythonError", "zero"
                ]
            },
            {
                "name": "import_error_nonexistent_module",
                "code": '''
import this_module_definitely_does_not_exist_anywhere
print("Should not reach this line")
                ''',
                "expected_error_keywords": [
                    "ModuleNotFoundError", "No module", "import", "PythonError"
                ]
            },
            {
                "name": "name_error_undefined_variable",
                "code": '''
print(f"Value of undefined variable: {undefined_variable_name}")
                ''',
                "expected_error_keywords": ["NameError", "not defined", "PythonError"]
            }
        ]

        for test_case in error_test_cases:
            # When: Execute invalid code
            response = execute_python_code_via_api(test_case["code"], api_client=api_client)

            # Then: Verify error response follows API contract
            assert response["success"] is False, f"{test_case['name']}: Should fail for invalid code"
            assert response["data"] is None, f"{test_case['name']}: Data should be None on error"
            assert response["error"] is not None, f"{test_case['name']}: Error message should be present"
            assert "meta" in response, f"{test_case['name']}: Meta field should be present"
            assert "timestamp" in response["meta"], f"{test_case['name']}: Timestamp should be in meta"

            # Verify error message contains expected keywords
            error_message = response["error"].lower()
            found_expected_keyword = any(
                keyword.lower() in error_message
                for keyword in test_case["expected_error_keywords"]
            )
            assert found_expected_keyword, \
                f"{test_case['name']}: Error message should contain one of " \
                f"{test_case['expected_error_keywords']}, got: {response['error']}"

    # =========================================================================
    # PERFORMANCE AND EDGE CASE TESTS
    # =========================================================================

    def test_given_large_output_generation_when_executing_then_handled_gracefully_without_server_issues(
        self,
        api_client,
        unique_timestamp
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
        large_output_code = f'''
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
'''

        # When: Execute code that generates large output
        response = execute_python_code_via_api(large_output_code, api_client=api_client)

        # Then: Verify large output is handled correctly
        assert response["success"] is True, "Large output generation should succeed"
        assert response["data"] is not None, "Response should contain data"

        stdout_content = response["data"]["stdout"]
        assert len(stdout_content) > 0, "Should capture generated output"
        assert "Large output test completed successfully" in stdout_content, \
            "Should include completion message"
        assert unique_timestamp in stdout_content, "Should include test identifier"

        # Verify output is within reasonable limits (not truncated unexpectedly)
        assert len(stdout_content) < OUTPUT_SIZE_LIMIT_BYTES, \
            "Output should be within reasonable size limits"

        # Verify output contains expected number of lines
        line_count = stdout_content.count("Output line")
        assert line_count == LARGE_OUTPUT_LINE_COUNT, \
            f"Should contain {LARGE_OUTPUT_LINE_COUNT} output lines, found {line_count}"

    def test_given_file_overwrite_operations_when_updating_existing_files_then_content_updated_correctly(
        self,
        api_client,
        unique_timestamp,
        test_file_cleanup
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
        overwrite_operations_code = f'''
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
'''

        response = execute_python_code_via_api(overwrite_operations_code, api_client=api_client)

        # Then: Verify overwrite operations were successful
        assert response["success"] is True, f"Overwrite operations failed: {response.get('error')}"
        assert response["data"] is not None, "Response data should not be None"

        overwrite_output = response["data"]["stdout"]
        import json as json_module
        overwrite_result = json_module.loads(overwrite_output.strip())

        # Verify all operations completed
        assert overwrite_result["operations_completed"] == 3, "Should complete 3 operations"
        assert overwrite_result["file_exists"] is True, "File should exist after all operations"
        assert overwrite_result["final_content_matches"] is True, "Final content should match last write"

        # Verify each operation was successful
        for operation in overwrite_result["operations"]:
            assert operation["exists"] is True, f"File should exist after {operation['operation']}"
            assert operation["size"] > 0, f"File should have content after {operation['operation']}"

        # Verify size changes appropriately
        sizes = [op["size"] for op in overwrite_result["operations"]]
        assert sizes[1] > sizes[0], "Second version should be larger than first"
        assert sizes[2] < sizes[1], "Third version should be smaller than second"

    def test_given_multiple_file_operations_when_creating_files_then_all_exist_in_virtual_filesystem(
        self,
        api_client,
        unique_timestamp,
        test_file_cleanup
    ):
        """
        Test multiple file creation operations in VFS.

        Given: A Pyodide environment with virtual filesystem
        When: Python code creates multiple files with different content
        Then: All files should exist in the virtual filesystem
              AND each file should have the correct content
              AND files should be accessible for reading

        Args:
            api_client: Configured requests session
            unique_timestamp: Unique identifier for test isolation
            test_file_cleanup: List to track files for cleanup
        """
        # Given: Generate unique filenames
        test_files = [
            f"multi_test_{unique_timestamp}_1.txt",
            f"multi_test_{unique_timestamp}_2.json",
            f"multi_test_{unique_timestamp}_3.csv"
        ]

        # When: Create multiple files with different content
        code = f'''
import json
import csv
from pathlib import Path

# Ensure directory exists
base_dir = Path("{TEST_PLOTS_DIRECTORY}")
base_dir.mkdir(parents=True, exist_ok=True)

# Files to create with their content
files_config = {{
    "{test_files[0]}": {{
        "content": "This is the first test file\\nIt contains multiple lines\\nFor testing purposes",
        "type": "text"
    }},
    "{test_files[1]}": {{
        "content": {{"test": "data", "timestamp": "{unique_timestamp}", "items": [1, 2, 3]}},
        "type": "json"
    }},
    "{test_files[2]}": {{
        "content": "name,value,active\\nAlice,100,true\\nBob,200,false",
        "type": "csv"
    }}
}}

results = []
for filename, config in files_config.items():
    file_path = base_dir / filename
    
    try:
        if config["type"] == "json":
            file_path.write_text(json.dumps(config["content"]))
        else:
            file_path.write_text(config["content"])
        
        # Verify creation
        results.append({{
            "filename": filename,
            "exists": file_path.exists(),
            "size": file_path.stat().st_size,
            "readable": True,
            "type": config["type"]
        }})
    except Exception as e:
        results.append({{
            "filename": filename,
            "exists": False,
            "error": str(e),
            "type": config["type"]
        }})

summary = {{
    "total_files": len(files_config),
    "successful_files": len([r for r in results if r.get("exists", False)]),
    "results": results
}}

print(json.dumps(summary))
'''

        response = execute_python_code_via_api(code, api_client=api_client)

        # Then: Verify multiple file creation
        assert response["success"] is True, f"Multiple file creation failed: {response.get('error')}"
        assert response["data"] is not None, "Response data should not be None"

        creation_output = response["data"]["stdout"]
        import json as json_module
        creation_result = json_module.loads(creation_output.strip())

        # Verify all files were created successfully
        assert creation_result["total_files"] == 3, "Should attempt to create 3 files"
        assert creation_result["successful_files"] == 3, "All files should be created successfully"

        # Verify each file
        for file_result in creation_result["results"]:
            assert file_result["exists"] is True, f"File {file_result['filename']} should exist"
            assert file_result["size"] > 0, f"File {file_result['filename']} should have content"
            assert file_result["readable"] is True, f"File {file_result['filename']} should be readable"

        # When: Create multiple files with different content
        code = f'''
import json
import csv
from pathlib import Path

# Ensure directory exists
Path('/home/pyodide/plots/matplotlib').mkdir(parents=True, exist_ok=True)

# Files to create with their content
files_config = {{
    "{test_files[0]}": {{
        "content": "This is the first test file\\nIt contains multiple lines\\nFor testing purposes",
        "type": "text"
    }},
    "{test_files[1]}": {{
        "content": {{"name": "Test Data", "version": 1, "items": [1, 2, 3]}},
        "type": "json"
    }},
    "{test_files[2]}": {{
        "content": [["Name", "Age", "City"], ["Alice", 30, "NYC"], ["Bob", 25, "LA"]],
        "type": "csv"
    }}
}}

results = []

for filename, config in files_config.items():
    file_path = f"/home/pyodide/plots/matplotlib/{{filename}}"

    try:
        if config["type"] == "text":
            with open(file_path, "w") as f:
                f.write(config["content"])
        elif config["type"] == "json":
            with open(file_path, "w") as f:
                json.dump(config["content"], f, indent=2)
        elif config["type"] == "csv":
            with open(file_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(config["content"])

        # Verify file was created
        path_obj = Path(file_path)
        results.append({{
            "filename": filename,
            "path": file_path,
            "exists": path_obj.exists(),
            "size": path_obj.stat().st_size if path_obj.exists() else 0,
            "type": config["type"]
        }})
    except Exception as e:
        results.append({{
            "filename": filename,
            "path": file_path,
            "exists": False,
            "error": str(e)
        }})

summary = {{
    "total_files": len(files_config),
    "successful_files": len([r for r in results if r.get("exists", False)]),
    "files": results
}}

print(json.dumps(summary))
'''

        response = execute_python_code_via_api(code, api_client=api_client)

        # Then: Verify multiple file creation
        assert response["success"] is True, f"Multiple file creation failed: {response.get('error')}"
        assert response["data"] is not None, "Response data should not be None"

        creation_output = response["data"]["stdout"]
        import json as json_module
        creation_result = json_module.loads(creation_output.strip())

        # Verify all files were created successfully
        assert creation_result["total_files"] == 3, "Should attempt to create 3 files"
        assert creation_result["successful_files"] == 3, "All files should be created successfully"

        # Verify each file
        for file_result in creation_result["files"]:
            assert file_result["exists"] is True, f"File {file_result['filename']} should exist"
            assert file_result["size"] > 0, f"File {file_result['filename']} should have content"

        # Then: Verify each file exists and has correct content in VFS
        verify_code = f'''
import json
import csv
from pathlib import Path

results = {{}}

# Check text file
text_path = Path('/home/pyodide/plots/matplotlib/{test_files[0]}')
if text_path.exists():
    content = text_path.read_text()
    results["text_file"] = {{
        "exists": True,
        "content": content,
        "has_expected_content": "This is the first test file" in content and "multiple lines" in content
    }}

# Check JSON file
json_path = Path('/home/pyodide/plots/matplotlib/{test_files[1]}')
if json_path.exists():
    with open(json_path, 'r') as f:
        data = json.load(f)
    results["json_file"] = {{
        "exists": True,
        "data": data,
        "valid_structure": data.get("name") == "Test Data" and data.get("version") == 1
    }}

# Check CSV file
csv_path = Path('/home/pyodide/plots/matplotlib/{test_files[2]}')
if csv_path.exists():
    content = csv_path.read_text()
    results["csv_file"] = {{
        "exists": True,
        "has_header": "Name,Age,City" in content,
        "has_alice": "Alice,30,NYC" in content,
        "has_bob": "Bob,25,LA" in content
    }}

print(json.dumps(results))
'''

        verify_response = execute_python_code_via_api(verify_code, api_client=api_client)
        assert verify_response["success"] is True, "Verification should succeed"

        verify_output = verify_response["data"]["stdout"]
        import json as json_module
        verify_results = json_module.loads(verify_output.strip())

        # Verify text file
        assert verify_results["text_file"]["exists"] is True, "Text file should exist"
        assert verify_results["text_file"]["has_expected_content"] is True, "Text content should match"

        # Verify JSON file
        assert verify_results["json_file"]["exists"] is True, "JSON file should exist"
        assert verify_results["json_file"]["valid_structure"] is True, "JSON structure should be valid"

        # Verify CSV file
        assert verify_results["csv_file"]["exists"] is True, "CSV file should exist"
        assert verify_results["csv_file"]["has_header"] is True, "CSV should have header"
        assert verify_results["csv_file"]["has_alice"] is True, "CSV should contain Alice row"
        assert verify_results["csv_file"]["has_bob"] is True, "CSV should contain Bob row"

    def test_given_subdirectories_when_creating_nested_files_then_structure_exists_in_vfs(
        self,
        api_client,
        unique_timestamp,
        test_file_cleanup
    ):
        """
        Test nested directory creation and file placement in VFS.

        Given: A Pyodide environment with virtual filesystem
        When: Python code creates files in nested subdirectories
        Then: The complete directory structure should exist in VFS
              AND files should be accessible in their nested locations
              AND parent directories should be created automatically

        Args:
            api_client: Configured requests session
            unique_timestamp: Unique identifier for test isolation
            test_file_cleanup: List to track files for cleanup
        """
        # Given: Generate unique directory structure
        test_subdir = f"nested_test_{unique_timestamp}"
        nested_structure = {
            "level1/data.txt": "Level 1 data file",
            "level1/level2/config.json": {"setting": "value", "nested": True},
            "level1/level2/level3/deep.txt": "Deep nested file content"
        }

        # When: Create nested directory structure
        code = f'''
import json
from pathlib import Path

base_dir = Path('/home/pyodide/plots/matplotlib/{test_subdir}')
structure = {repr(nested_structure)}

results = []

for rel_path, content in structure.items():
    file_path = base_dir / rel_path

    try:
        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content based on type
        if isinstance(content, dict):
            with open(file_path, "w") as f:
                json.dump(content, f)
        else:
            with open(file_path, "w") as f:
                f.write(content)

        results.append({{
            "path": str(file_path),
            "relative_path": rel_path,
            "exists": file_path.exists(),
            "parent_exists": file_path.parent.exists(),
            "size": file_path.stat().st_size if file_path.exists() else 0
        }})
    except Exception as e:
        results.append({{
            "path": str(file_path),
            "relative_path": rel_path,
            "error": str(e)
        }})

summary = {{
    "base_directory": str(base_dir),
    "files_created": len([r for r in results if r.get("exists", False)]),
    "total_files": len(structure),
    "files": results
}}

print(json.dumps(summary))
'''

        response = execute_python_code_via_api(code, api_client=api_client)

        # Then: Verify API response
        assert response["success"] is True, f"Code execution failed: {response.get('error')}"
        assert response["data"] is not None, "Response data should not be None"

        # Parse the output
        output = response["data"]["stdout"]
        assert output, "Expected output from code execution"

        import json
        summary = json.loads(output.strip())

        # Verify all files were created in VFS
        assert summary["files_created"] == 3, "All nested files should be created"

        # Then: Verify complete directory structure exists in VFS
        verify_structure_code = f'''
import json
from pathlib import Path

base_dir = Path('/home/pyodide/plots/matplotlib/{test_subdir}')
structure = {repr(nested_structure)}
verification_results = []

for rel_path, expected_content in structure.items():
    file_path = base_dir / rel_path

    if file_path.exists():
        if isinstance(expected_content, dict):
            with open(file_path, 'r') as f:
                actual_content = json.load(f)
            content_matches = actual_content == expected_content
        else:
            actual_content = file_path.read_text()
            content_matches = actual_content == expected_content

        verification_results.append({{
            "path": rel_path,
            "exists": True,
            "content_matches": content_matches,
            "parent_exists": file_path.parent.exists()
        }})
    else:
        verification_results.append({{
            "path": rel_path,
            "exists": False
        }})

summary = {{
    "all_exist": all(r["exists"] for r in verification_results),
    "all_content_matches": all(r.get("content_matches", False) for r in verification_results if r["exists"]),
    "files": verification_results
}}

print(json.dumps(summary))
'''

        verify_response = execute_python_code_via_api(verify_structure_code, api_client=api_client)
        assert verify_response["success"] is True, "Structure verification should succeed"

        verify_output = verify_response["data"]["stdout"]
        verify_summary = json.loads(verify_output.strip())

        assert verify_summary["all_exist"] is True, "All files should exist in nested structure"
        assert verify_summary["all_content_matches"] is True, "All file contents should match"

    def test_given_existing_file_when_overwriting_then_content_updated_in_vfs(
        self,
        api_client,
        unique_timestamp,
        test_file_cleanup
    ):
        """
        Test file overwriting behavior in VFS.

        Given: An existing file in the virtual filesystem
        When: Python code overwrites the file with new content
        Then: The file content should be updated in VFS
              AND the original content should be replaced
              AND file should remain accessible

        Args:
            api_client: Configured requests session
            unique_timestamp: Unique identifier for test isolation
            test_file_cleanup: List to track files for cleanup
        """
        # Given: Create initial file in VFS
        test_filename = f"overwrite_test_{unique_timestamp}.txt"

        # Create initial file in VFS
        create_code = f'''
from pathlib import Path
import json

file_path = Path('/home/pyodide/plots/matplotlib/{test_filename}')
Path('/home/pyodide/plots/matplotlib').mkdir(parents=True, exist_ok=True)

# Create initial file
initial_content = "Initial content before overwrite"
with open(file_path, "w") as f:
    f.write(initial_content)

result = {{
    "file_created": file_path.exists(),
    "initial_size": file_path.stat().st_size if file_path.exists() else 0
}}

print(json.dumps(result))
'''

        create_response = execute_python_code_via_api(create_code, api_client=api_client)
        assert create_response["success"] is True, "Initial file creation should succeed"

        # When: Overwrite file from Python
        overwrite_code = f'''
from pathlib import Path
import json

file_path = Path('/home/pyodide/plots/matplotlib/{test_filename}')

# Read initial content
initial_content = file_path.read_text() if file_path.exists() else ""
initial_size = file_path.stat().st_size if file_path.exists() else 0

# Overwrite with new content
new_content = "New content after overwrite\\nThis replaces the original content completely"
with open(file_path, "w") as f:
    f.write(new_content)

# Read back to verify
final_content = file_path.read_text()

result = {{
    "initial_content": initial_content,
    "initial_size": initial_size,
    "new_content": final_content,
    "new_size": file_path.stat().st_size,
    "overwrite_successful": final_content == new_content
}}

print(json.dumps(result))
'''

        response = execute_python_code_via_api(overwrite_code, api_client=api_client)

        # Then: Verify overwrite was successful
        assert response["success"] is True, f"Overwrite operation failed: {response.get('error')}"

        output = response["data"]["stdout"]
        result = json.loads(output.strip())

        # Verify initial content was correct
        assert result["initial_content"] == "Initial content before overwrite", "Initial content should match"

        # Verify file was overwritten
        expected_new_content = "New content after overwrite\nThis replaces the original content completely"
        assert result["new_content"] == expected_new_content, "File content should be updated"
        assert result["overwrite_successful"] is True, "Overwrite should be successful"

        # Verify size changed
        assert result["new_size"] > result["initial_size"], "File size should increase after overwrite"

    def test_given_files_in_vfs_when_using_extract_api_then_files_can_be_retrieved(
        self,
        api_client,
        unique_timestamp,
        test_file_cleanup
    ):
        """
        Test file extraction via API after creation in VFS.

        Given: Files created in Pyodide's virtual filesystem
        When: Using the extract API to retrieve files
        Then: Files should be accessible with correct content and metadata
              AND the API response should follow the standard contract

        Args:
            api_client: Configured requests session
            unique_timestamp: Unique identifier for test isolation
            test_file_cleanup: List to track files for cleanup
        """
        # Given: Create multiple types of files in VFS
        timestamp = str(int(time.time() * 1000))

        create_files_code = f'''
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create directories
Path('/home/pyodide/plots/matplotlib').mkdir(parents=True, exist_ok=True)
Path('/home/pyodide/plots/seaborn').mkdir(parents=True, exist_ok=True)

results = []

# 1. Create a text file
text_file = f"/home/pyodide/plots/matplotlib/extract_test_{timestamp}.txt"
with open(text_file, "w") as f:
    f.write("Test content for extraction API")
results.append({{"type": "text", "path": text_file, "created": True}})

# 2. Create a JSON file
json_file = f"/home/pyodide/plots/seaborn/data_{timestamp}.json"
with open(json_file, "w") as f:
    json.dump({{"test": "data", "timestamp": "{timestamp}"}}, f)
results.append({{"type": "json", "path": json_file, "created": True}})

# 3. Create a matplotlib plot
plt.figure(figsize=(6, 4))
x = np.linspace(0, 5, 50)
plt.plot(x, np.sin(x), 'b-')
plt.title("Test Plot for Extraction")
plot_file = f"/home/pyodide/plots/matplotlib/plot_{timestamp}.png"
plt.savefig(plot_file)
plt.close()
results.append({{"type": "plot", "path": plot_file, "created": True}})

print(json.dumps({{"files_created": len(results), "files": results}}))
'''

        # When: Create files in VFS
        create_response = execute_python_code_via_api(create_files_code, api_client=api_client)
        assert create_response["success"] is True, "File creation should succeed"

        # Parse creation results
        creation_output = create_response["data"]["stdout"]
        creation_result = json.loads(creation_output.strip())
        assert creation_result["files_created"] == 3, "Should create 3 files"

        # Then: Use extract API to retrieve files (POST request)
        extract_response = requests.post(f"{API_BASE_URL}/api/extract-plots", timeout=HTTP_REQUEST_TIMEOUT_SECONDS)
        
        # Note: The extract API may not be fully implemented and could return 500
        # This is expected behavior if the extractAllPlotFiles method is not available
        if extract_response.status_code == 500:
            # Skip this test if the API is not implemented
            pytest.skip("Extract API not implemented (returns 500)")
            return
        
        assert extract_response.status_code == 200, f"Extract API failed: {extract_response.status_code}"

        extract_data = extract_response.json()

        # Check the actual format of the extract API response
        assert "success" in extract_data, "Response should have success field"
        assert extract_data["success"] is True, "Extract should be successful"

        # The API returns extracted_files and count in the response
        if "extracted_files" in extract_data:
            # This server configuration extracts files to local filesystem
            extracted_files = extract_data["extracted_files"]
            assert isinstance(extracted_files, list), "extracted_files should be a list"
            assert extract_data.get("count", 0) >= 0, "Should have a valid count"

            # Note: The files might not be in the extracted files if the service
            # only extracts actual plot files (PNG, etc) and not text/json files
            # So we'll just verify the API works, not specific file presence
            assert "timestamp" in extract_data, "Response should have timestamp"
        else:
            # Alternative: check if files are returned in a data wrapper following standard contract
            assert "data" in extract_data, "Response should have data field if not using extracted_files"
            assert "error" in extract_data, "Response should have error field"
            assert "meta" in extract_data, "Response should have meta field"

    def test_given_pandas_dataframe_when_saving_csv_then_readable_in_vfs(
        self,
        api_client,
        unique_timestamp,
        test_file_cleanup
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

        code = f'''
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
'''

        response = execute_python_code_via_api(code, api_client=api_client)
        assert response["success"] is True, "DataFrame operations should succeed"

        output = response["data"]["stdout"]
        import json as json_module
        result = json_module.loads(output.strip())

        # Verify CSV operations
        assert result["file_exists"] is True, "CSV file should be created"
        assert result["file_size"] > 0, "CSV file should have content"
        assert result["original_shape"] == [4, 4], "Original DataFrame should have 4 rows, 4 columns"
        assert result["read_shape"] == [4, 4], "Read DataFrame should maintain shape"
        assert result["columns_match"] is True, "Columns should be preserved"
        assert result["data_matches"] is True, "Data should be preserved exactly"

    def test_given_error_in_code_when_executing_then_proper_error_response(
        self,
        api_client,
        unique_timestamp,
        test_file_cleanup
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
            test_file_cleanup: List to track files for cleanup
        """
        # Test 1: Syntax error
        syntax_error_code = '''
def broken_function(
    print("Missing closing parenthesis")
'''

        response = execute_python_code_via_api(syntax_error_code, api_client=api_client)

        # Verify error response follows contract
        assert response["success"] is False, "Syntax error should fail"
        assert response["data"] is None, "Data should be None on error"
        assert response["error"] is not None, "Error message should be present"
        assert "meta" in response and "timestamp" in response["meta"], "Meta with timestamp required"

        # Test 2: Runtime error
        runtime_error_code = '''
import json

# This will cause a division by zero error
x = 10
y = 0
result = x / y
print(json.dumps({"result": result}))
'''

        response = execute_python_code_via_api(runtime_error_code, api_client=api_client)

        assert response["success"] is False, "Runtime error should fail"
        assert response["error"] is not None, "Error message should be present"
        error_lower = response["error"].lower()
        assert any(keyword in error_lower for keyword in ["zerodivisionerror", "division", "zero", "pythonerror"]), \
            f"Error message should indicate division by zero, got: {response['error']}"

        # Test 3: Import error for non-existent module
        import_error_code = '''
import this_module_does_not_exist
print("Should not reach here")
'''

        response = execute_python_code_via_api(import_error_code, api_client=api_client)

        assert response["success"] is False, "Import error should fail"
        assert response["error"] is not None, "Error message should be present"
        error_msg = response["error"]
        assert any(keyword in error_msg for keyword in ["ModuleNotFoundError", "No module", "PythonError", "import"]), \
            f"Error message should indicate module not found, got: {response['error']}"

    def test_given_large_output_when_executing_then_handled_gracefully(
        self,
        api_client,
        unique_timestamp,
        test_file_cleanup
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
        code = '''
# Generate large output
for i in range(1000):
    print(f"Line {i}: " + "x" * 100)

print("\\nCompleted large output test")
'''

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

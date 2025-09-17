"""BDD-style integration tests for complex data flow scenarios.

This module contains pytest-based integration tests following the Given-When-Then
pattern. Tests use only public API endpoints and avoid internal pyodide APIs.
"""

import tempfile
from pathlib import Path
from typing import Dict

import pytest
import requests

# Import configuration (use try/except for different import contexts)
try:
    from .conftest import Config
    # Map old constants to new Config class
    BASE_URL = Config.BASE_URL
    ENDPOINTS = Config.ENDPOINTS
    EXECUTE_TIMEOUT = Config.TIMEOUTS["code_execution"]
    UPLOAD_TIMEOUT = Config.TIMEOUTS["file_upload"]
    FILE_OPERATION_TIMEOUT = Config.TIMEOUTS["file_operation"]
    PACKAGE_INSTALL_TIMEOUT = Config.TIMEOUTS["package_install"]
    RESET_TIMEOUT = Config.TIMEOUTS["reset_operation"]
    TEST_CSV_CONTENT = Config.TEST_DATA["csv_content"]
    TEST_FILES = Config.TEST_DATA["test_files"]
    PACKAGES = Config.TEST_DATA["packages"]
except ImportError:
    from conftest import Config
    # Map old constants to new Config class
    BASE_URL = Config.BASE_URL
    ENDPOINTS = Config.ENDPOINTS
    EXECUTE_TIMEOUT = Config.TIMEOUTS["code_execution"]
    UPLOAD_TIMEOUT = Config.TIMEOUTS["file_upload"]
    FILE_OPERATION_TIMEOUT = Config.TIMEOUTS["file_operation"]
    PACKAGE_INSTALL_TIMEOUT = Config.TIMEOUTS["package_install"]
    RESET_TIMEOUT = Config.TIMEOUTS["reset_operation"]
    TEST_CSV_CONTENT = Config.TEST_DATA["csv_content"]
    TEST_FILES = Config.TEST_DATA["test_files"]
    PACKAGES = Config.TEST_DATA["packages"]


# ===== Helper Functions =====

def create_temp_csv_file(content: str, filename: str) -> Path:
    """Create a temporary CSV file with given content."""
    temp_file = Path(tempfile.mktemp(suffix=".csv"))
    temp_file.write_text(content, encoding='utf-8')
    return temp_file


def upload_csv_file(file_path: Path, filename: str) -> Dict:
    """Upload a CSV file and return the response data."""
    with open(file_path, "rb") as fh:
        response = requests.post(
            f"{BASE_URL}{ENDPOINTS['upload_csv']}",
            files={"file": (filename, fh, "text/csv")},
            timeout=UPLOAD_TIMEOUT
        )
    response.raise_for_status()
    return response.json()


def execute_python_code(code: str) -> Dict:
    """Execute Python code using the execute-raw endpoint."""
    response = requests.post(
        f"{BASE_URL}{ENDPOINTS['execute_raw']}",
        data=code,
        headers={"Content-Type": "text/plain"},
        timeout=EXECUTE_TIMEOUT
    )
    response.raise_for_status()
    return {"success": True, "result": response.text.strip()}


def delete_uploaded_file(filename: str) -> None:
    """Delete an uploaded file."""
    try:
        requests.delete(
            f"{BASE_URL}{ENDPOINTS['uploaded_files']}/{filename}",
            timeout=FILE_OPERATION_TIMEOUT
        )
    except requests.RequestException:
        pass  # File might not exist


# ===== Integration Test Scenarios =====

class TestDataConsistency:
    """Test data consistency and JSON parsing in file operations."""

    def test_csv_upload_and_processing_workflow(self, server, base_url):
        """
        GIVEN: A simple CSV file with basic data
        WHEN: The file is uploaded and processed with execute-raw
        THEN: The file can be read and processed correctly via Python
        """
        # GIVEN: Prepare test data
        test_content = TEST_CSV_CONTENT["simple"]
        temp_file = create_temp_csv_file(test_content, TEST_FILES["simple"])

        try:
            # WHEN: Upload the CSV file
            upload_data = upload_csv_file(temp_file, TEST_FILES["simple"])
            uploaded_filename = upload_data["data"]["file"]["sanitizedOriginal"]

            # AND: Process the file using execute-raw endpoint
            python_code = f"""
import pandas as pd
import json

# Read the uploaded file
try:
    df = pd.read_csv('/home/pyodide/uploads/{uploaded_filename}')
    result = {{
        "shape": list(df.shape),
        "columns": list(df.columns),
        "values": df.to_dict('records')
    }}
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""

            # THEN: Execute code and verify results
            exec_result = execute_python_code(python_code)
            assert exec_result["success"]

            # Parse the JSON output
            import json
            result_data = json.loads(exec_result["result"])
            assert "error" not in result_data
            assert result_data["shape"] == [2, 3]  # 2 rows, 3 columns
            assert set(result_data["columns"]) == {"name", "value", "category"}

        finally:
            # Cleanup
            temp_file.unlink(missing_ok=True)
            delete_uploaded_file(uploaded_filename)


class TestCSVProcessingEdgeCases:
    """Test CSV file processing with various edge cases."""

    @pytest.mark.parametrize("content_key,expected_rows", [
        ("quotes", 1),
        ("unicode", 2),
        ("empty_fields", 3),
    ])
    def test_csv_edge_cases_processing(self, server, base_url, content_key, expected_rows):
        """
        GIVEN: CSV files with various edge cases (quotes, unicode, empty fields)
        WHEN: Files are uploaded and processed with pandas
        THEN: All files are processed correctly without errors
        """
        # GIVEN: Prepare test data
        test_content = TEST_CSV_CONTENT[content_key]
        temp_file = create_temp_csv_file(test_content, TEST_FILES[content_key])

        try:
            # WHEN: Upload the CSV file
            upload_data = upload_csv_file(temp_file, TEST_FILES[content_key])
            uploaded_filename = upload_data["data"]["file"]["sanitizedOriginal"]

            # AND: Process the file using execute-raw
            python_code = f"""
import pandas as pd
import json

try:
    df = pd.read_csv('/home/pyodide/uploads/{uploaded_filename}')
    result = {{
        "success": True,
        "shape": list(df.shape),
        "columns": list(df.columns)
    }}
    print(json.dumps(result))
except Exception as e:
    result = {{"success": False, "error": str(e)}}
    print(json.dumps(result))
"""

            # THEN: Verify successful processing
            exec_result = execute_python_code(python_code)
            assert exec_result["success"]

            import json
            result_data = json.loads(exec_result["result"])
            assert result_data["success"], f"Processing failed: {result_data.get('error')}"
            assert result_data["shape"][0] == expected_rows

        finally:
            # Cleanup
            temp_file.unlink(missing_ok=True)
            delete_uploaded_file(uploaded_filename)


class TestConcurrentOperations:
    """Test multiple operations happening in sequence."""

    def test_sequential_file_operations(self, server, base_url):
        """
        GIVEN: A CSV file is uploaded
        WHEN: Multiple operations are performed on the same file sequentially
        THEN: All operations complete successfully without interference
        """
        # GIVEN: Prepare and upload test data
        test_content = "x,y\n1,2\n3,4\n"
        temp_file = create_temp_csv_file(test_content, TEST_FILES["concurrent"])

        try:
            upload_data = upload_csv_file(temp_file, TEST_FILES["concurrent"])
            uploaded_filename = upload_data["data"]["file"]["sanitizedOriginal"]

            # WHEN: Execute multiple operations using the same file
            operations = [
                f"import pandas as pd; df = pd.read_csv('/home/pyodide/uploads/{uploaded_filename}'); "
                + "print(df.shape[0])",
                f"import pandas as pd; df = pd.read_csv('/home/pyodide/uploads/{uploaded_filename}'); "
                + "print(df.sum().sum())",
                f"import pandas as pd; df = pd.read_csv('/home/pyodide/uploads/{uploaded_filename}'); "
                + "print(list(df.columns))",
            ]

            # THEN: All operations should succeed
            for code in operations:
                exec_result = execute_python_code(code)
                assert exec_result["success"], f"Operation failed: {code}"
                assert exec_result["result"], "No output from operation"

        finally:
            # Cleanup
            temp_file.unlink(missing_ok=True)
            delete_uploaded_file(uploaded_filename)


class TestStatePersistence:
    """Test that packages persist after reset while maintaining isolated execution."""

    def test_package_persistence_after_reset(self, server, base_url):
        """
        GIVEN: A Python package is installed and variables are set
        WHEN: The environment is reset
        THEN: Packages persist but variables are isolated between executions
        """
        # GIVEN: Install a package
        install_response = requests.post(
            f"{BASE_URL}{ENDPOINTS['install_package']}",
            json={"package": PACKAGES["beautifulsoup4"]},
            timeout=PACKAGE_INSTALL_TIMEOUT
        )
        assert install_response.status_code == 200

        # WHEN: Verify package is available
        package_test_code = "import bs4; print('package_available')"
        exec_result = execute_python_code(package_test_code)
        assert exec_result["success"]
        assert "package_available" in exec_result["result"]

        # AND: Test that variables don't persist between executions (isolated execution)
        variable_set_code = "test_var = 'isolated_value'; print('variable_set')"
        exec_result = execute_python_code(variable_set_code)
        assert exec_result["success"]
        assert "variable_set" in exec_result["result"]

        # AND: Verify variable doesn't exist in separate execution (this is correct behavior)
        variable_check_code = """
try:
    print(test_var)
except NameError:
    print('variable_not_found')
"""
        exec_result = execute_python_code(variable_check_code)
        assert exec_result["success"]
        assert "variable_not_found" in exec_result["result"]

        # WHEN: Reset the environment
        reset_response = requests.post(f"{BASE_URL}{ENDPOINTS['reset']}", timeout=RESET_TIMEOUT)
        assert reset_response.status_code == 200
        assert reset_response.json().get("success")

        # THEN: Verify package is still available after reset (packages should persist)
        package_retest_code = "import bs4; print('still_available')"
        exec_result = execute_python_code(package_retest_code)
        assert exec_result["success"]
        assert "still_available" in exec_result["result"]

        # AND: Verify reset completed successfully by checking we can still execute code
        basic_test_code = "result = 2 + 2; print(result)"
        exec_result = execute_python_code(basic_test_code)
        assert exec_result["success"]
        assert "4" in exec_result["result"]


class TestComplexDataFlow:
    """Test complex data processing workflows."""

    def test_multi_file_data_processing_workflow(self, server, base_url):
        """
        GIVEN: Multiple CSV files with related data
        WHEN: Files are uploaded and processed together in a complex workflow
        THEN: Data can be merged and computed correctly
        """
        # GIVEN: Prepare multiple related CSV files
        files_data = [
            (TEST_FILES["data1"], "id,value\n1,10\n2,20\n"),
            (TEST_FILES["data2"], "id,score\n1,90\n2,85\n"),
        ]

        uploaded_files = []
        temp_files = []

        try:
            # WHEN: Upload multiple files
            for filename, content in files_data:
                temp_file = create_temp_csv_file(content, filename)
                temp_files.append(temp_file)

                upload_data = upload_csv_file(temp_file, filename)
                uploaded_files.append(upload_data["data"]["file"]["sanitizedOriginal"])

            # AND: Perform complex data processing
            complex_code = f'''
import pandas as pd
import json

# Load both datasets
df1 = pd.read_csv('/home/pyodide/uploads/{uploaded_files[0]}')
df2 = pd.read_csv('/home/pyodide/uploads/{uploaded_files[1]}')

# Merge the datasets
merged = pd.merge(df1, df2, on='id')

# Calculate some metrics
total_value = int(merged['value'].sum())
avg_score = float(merged['score'].mean())
record_count = len(merged)

# Return results as a dictionary
result = {{
    "total_value": total_value,
    "avg_score": avg_score,
    "record_count": record_count,
    "columns": list(merged.columns)
}}

print(json.dumps(result))
'''

            # THEN: Verify the computed results are correct
            exec_result = execute_python_code(complex_code)
            assert exec_result["success"]

            import json
            result_data = json.loads(exec_result["result"])
            assert result_data["total_value"] == 30  # 10 + 20
            assert result_data["avg_score"] == 87.5  # (90 + 85) / 2
            assert result_data["record_count"] == 2
            assert set(result_data["columns"]) == {"id", "value", "score"}

        finally:
            # Cleanup
            for temp_file in temp_files:
                temp_file.unlink(missing_ok=True)
            for uploaded_file in uploaded_files:
                delete_uploaded_file(uploaded_file)


class TestErrorRecovery:
    """Test that errors don't break subsequent operations."""

    def test_error_recovery_workflow(self, server, base_url):
        """
        GIVEN: Python code that will cause errors
        WHEN: Errors occur during execution
        THEN: Subsequent operations continue to work correctly
        """
        # GIVEN & WHEN: Execute code that will fail (undefined variable)
        error_code1 = "print(undefined_variable)"
        try:
            execute_python_code(error_code1)
        except Exception:
            pass  # Expected to fail

        # THEN: Execute code that will succeed after the error
        success_code1 = "result = 2 + 2; print(result)"
        exec_result = execute_python_code(success_code1)
        assert exec_result["success"]
        assert "4" in exec_result["result"]

        # WHEN: Try syntax error
        error_code2 = "if True\n    print('missing colon')"
        try:
            execute_python_code(error_code2)
        except Exception:
            pass  # Expected to fail

        # THEN: Verify system still works
        success_code2 = "print('system_recovered')"
        exec_result = execute_python_code(success_code2)
        assert exec_result["success"]
        assert "system_recovered" in exec_result["result"]


# ===== Additional BDD Test Scenarios =====

class TestExecuteRawEndpoint:
    """Test the execute-raw endpoint specifically for comprehensive coverage."""

    def test_execute_raw_basic_functionality(self, server, base_url):
        """
        GIVEN: Simple Python code
        WHEN: Code is executed via execute-raw endpoint
        THEN: Correct output is returned as plain text
        """
        # GIVEN: Simple Python code
        code = "print('Hello from execute-raw!')"

        # WHEN: Execute via execute-raw
        exec_result = execute_python_code(code)

        # THEN: Verify correct output
        assert exec_result["success"]
        assert "Hello from execute-raw!" in exec_result["result"]

    def test_execute_raw_with_imports(self, server, base_url):
        """
        GIVEN: Python code with standard library imports
        WHEN: Code is executed via execute-raw endpoint
        THEN: Imports work correctly and output is returned
        """
        # GIVEN: Code with imports
        code = """
import json
import math

data = {"value": math.pi}
print(json.dumps(data))
"""

        # WHEN: Execute via execute-raw
        exec_result = execute_python_code(code)

        # THEN: Verify successful execution
        assert exec_result["success"]
        import json
        result_data = json.loads(exec_result["result"])
        assert "value" in result_data
        assert abs(result_data["value"] - 3.14159) < 0.001

    def test_execute_raw_multiline_output(self, server, base_url):
        """
        GIVEN: Python code that produces multiple lines of output
        WHEN: Code is executed via execute-raw endpoint
        THEN: All output lines are returned correctly
        """
        # GIVEN: Code with multiple print statements
        code = """
for i in range(3):
    print(f"Line {i+1}")
print("Final line")
"""

        # WHEN: Execute via execute-raw
        exec_result = execute_python_code(code)

        # THEN: Verify all output is captured
        assert exec_result["success"]
        result_lines = exec_result["result"].split('\n')
        assert "Line 1" in result_lines
        assert "Line 2" in result_lines
        assert "Line 3" in result_lines
        assert "Final line" in result_lines


if __name__ == "__main__":
    pytest.main([__file__])

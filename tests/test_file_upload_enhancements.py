#!/usr/bin/env python3
"""
BDD Test Suite for File Upload System Enhancements.

This module provides comprehensive BDD-style tests for the file upload system,
focusing on recent enhancements including temp filename generation, file-type-specific
analysis, and cross-platform compatibility using only the /api/execute-raw endpoint.

Test Categories:
    - File upload workflow validation
    - Temporary filename generation
    - Cross-platform Python code execution
    - API contract compliance
    - Error handling and edge cases

Examples:
    Run all tests:
        $ uv run pytest tests/test_file_upload_enhancements.py -v

    Run specific test:
        $ uv run pytest tests/test_file_upload_enhancements.py::test_upload_file_with_enhanced_temp_naming -v

    Run with coverage:
        $ uv run pytest tests/test_file_upload_enhancements.py --cov=src --cov-report=html
"""

import json
import tempfile
from pathlib import Path
from typing import List

import pytest
import requests

from conftest import Config


@pytest.fixture
def api_session():
    """
    Create a configured requests session for API testing.

    Returns:
        requests.Session: Configured session with timeout and base settings

    Examples:
        >>> def test_upload(api_session):
        ...     response = api_session.post('/api/upload', files={'file': file_data})
    """
    session = requests.Session()
    session.timeout = Config.DEFAULT_TIMEOUT
    return session


@pytest.fixture
def temp_file_tracker():
    """
    Track temporary files for cleanup after tests.

    Yields:
        List[Path]: List to track temporary files that need cleanup

    Examples:
        >>> def test_file_creation(temp_file_tracker):
        ...     temp_file = Path(tempfile.mktemp())
        ...     temp_file_tracker.append(temp_file)
        ...     # File will be cleaned up automatically
    """
    temp_files = []
    yield temp_files

    # Cleanup
    for temp_file in temp_files:
        if isinstance(temp_file, Path) and temp_file.exists():
            try:
                if temp_file.is_file():
                    temp_file.unlink()
                elif temp_file.is_dir():
                    temp_file.rmdir()
            except (OSError, PermissionError):
                pass  # File may have been cleaned up by system


@pytest.fixture
def uploaded_file_tracker(api_session):
    """
    Track uploaded files for cleanup via API after tests.

    Args:
        api_session: API session fixture for making cleanup requests

    Yields:
        List[str]: List to track uploaded filenames that need cleanup

    Examples:
        >>> def test_upload(api_session, uploaded_file_tracker):
        ...     response = api_session.post('/api/upload', files={'file': data})
        ...     uploaded_file_tracker.append(response.json()['filename'])
        ...     # File will be cleaned up automatically via API
    """
    uploaded_files = []
    yield uploaded_files

    # Cleanup uploaded files via API
    for filename in uploaded_files:
        try:
            api_session.delete(f"{Config.BASE_URL}/api/uploaded-files/{filename}")
        except requests.RequestException:
            pass  # File may already be deleted


class TestFileUploadWorkflow:
    """BDD tests for basic file upload workflow functionality."""

    def test_upload_json_file_successfully(
        self,
        api_session: requests.Session,
        temp_file_tracker: List[Path],
        uploaded_file_tracker: List[str],
    ):
        """
        Test successful JSON file upload with proper response validation.

        Given: A valid JSON file is prepared for upload
        When: The file is uploaded via /api/upload endpoint
        Then: The upload succeeds and returns proper response format

        Args:
            api_session: Configured requests session
            temp_file_tracker: Temporary file cleanup tracker
            uploaded_file_tracker: Uploaded file cleanup tracker

        Examples:
            Test uploads a JSON file and validates the response structure
        """
        # Given: A valid JSON file is prepared for upload
        test_data = {
            "test_key": "test_value",
            "numbers": [1, 2, 3],
            "nested": {"key": "value"},
        }

        temp_file = Path(tempfile.mktemp(suffix=".json"))
        temp_file_tracker.append(temp_file)

        with temp_file.open("w", encoding="utf-8") as f:
            json.dump(test_data, f)

        # When: The file is uploaded via /api/upload endpoint
        with temp_file.open("rb") as f:
            files = {"file": ("test_data.json", f, "application/json")}
            response = api_session.post(f"{Config.BASE_URL}/api/upload", files=files)

        # Then: The upload succeeds and returns proper API contract response format
        assert response.status_code == 200
        response_data = response.json()

        # Validate API contract structure
        assert response_data.get("success") is True
        assert "data" in response_data
        assert "error" in response_data
        assert "meta" in response_data
        assert response_data.get("error") is None
        assert "timestamp" in response_data.get("meta", {})

        # Validate file upload data structure
        file_data = response_data.get("data", {}).get("file", {})
        assert "storedFilename" in file_data
        assert "originalName" in file_data
        assert file_data.get("originalName") == "test_data.json"

        # Track for cleanup using the stored filename
        uploaded_file_tracker.append(file_data["storedFilename"])

    def test_upload_multiple_files_with_same_name(
        self,
        api_session: requests.Session,
        temp_file_tracker: List[Path],
        uploaded_file_tracker: List[str],
    ):
        """
        Test uploading multiple files with identical names generates unique filenames.

        Given: Multiple files with identical names are prepared
        When: Files are uploaded sequentially
        Then: Each upload generates unique server-side filenames

        Args:
            api_session: Configured requests session
            temp_file_tracker: Temporary file cleanup tracker
            uploaded_file_tracker: Uploaded file cleanup tracker

        Examples:
            Test uploads two files named 'identical_name.json' and verifies
            unique filename generation on the server side
        """
        # Given: Multiple files with identical names are prepared
        content = '{"test": "data"}'
        uploaded_names = []

        for _ in range(2):
            temp_file = Path(tempfile.mktemp(suffix="_identical_name.json"))
            temp_file_tracker.append(temp_file)

            with temp_file.open("w", encoding="utf-8") as f:
                f.write(content)

            # When: Files are uploaded sequentially
            with temp_file.open("rb") as f:
                files = {"file": ("identical_name.json", f, "application/json")}
                response = api_session.post(
                    f"{Config.BASE_URL}/api/upload", files=files
                )

            assert response.status_code == 200
            response_data = response.json()

            # Validate API contract
            assert response_data.get("success") is True
            assert "data" in response_data
            assert "error" in response_data
            assert response_data.get("error") is None

            # Extract filename from new API contract structure
            file_data = response_data.get("data", {}).get("file", {})
            stored_filename = file_data.get("storedFilename")
            assert stored_filename is not None

            uploaded_names.append(stored_filename)
            uploaded_file_tracker.append(stored_filename)

        # Then: Each upload generates unique server-side filenames
        assert len(uploaded_names) == 2
        assert (
            uploaded_names[0] != uploaded_names[1]
        ), "Files with identical names should generate unique server filenames"


class TestPythonCodeExecution:
    """BDD tests for Python code execution using /api/execute-raw endpoint."""

    def test_execute_file_processing_with_pathlib(
        self,
        api_session: requests.Session,
        temp_file_tracker: List[Path],
        uploaded_file_tracker: List[str],
    ):
        """
        Test Python code execution for file processing using pathlib for portability.

        Given: A CSV file is uploaded and Python code uses pathlib for file operations
        When: Python code is executed via /api/execute-raw endpoint
        Then: Code executes successfully with proper cross-platform file handling

        Args:
            api_session: Configured requests session
            temp_file_tracker: Temporary file cleanup tracker
            uploaded_file_tracker: Uploaded file cleanup tracker

        Examples:
            Test uploads a CSV file and executes Python code that processes it
            using pathlib for cross-platform compatibility
        """
        # Given: A CSV file is uploaded
        csv_content = (
            "name,age,city\nAlice,30,New York\nBob,25,London\nCharlie,35,Tokyo"
        )
        temp_file = Path(tempfile.mktemp(suffix=".csv"))
        temp_file_tracker.append(temp_file)

        with temp_file.open("w", encoding="utf-8") as f:
            f.write(csv_content)

        # Upload the CSV file
        with temp_file.open("rb") as f:
            files = {"file": ("test_data.csv", f, "text/csv")}
            upload_response = api_session.post(
                f"{Config.BASE_URL}/api/upload", files=files
            )

        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        file_data = upload_data.get("data", {}).get("file", {})
        uploaded_filename = file_data.get("storedFilename")
        assert uploaded_filename is not None
        uploaded_file_tracker.append(uploaded_filename)

        # Given: Python code uses pathlib for file operations
        python_code = f"""
from pathlib import Path
import pandas as pd

# Use pathlib for cross-platform file handling
uploads_dir = Path('/home/pyodide/uploads')
csv_file = uploads_dir / '{uploaded_filename}'

# Verify file exists and read it
if csv_file.exists():
    df = pd.read_csv(csv_file)
    print(f"Successfully loaded CSV with {{len(df)}} rows")
    print(f"Columns: {{list(df.columns)}}")
    print(f"First row data: {{df.iloc[0].to_dict()}}")

    # Create output directory using pathlib
    output_dir = Path('/home/pyodide/plots/processed_data')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save processed data
    output_file = output_dir / 'processed_data.json'
    result_data = {{
        'row_count': len(df),
        'columns': list(df.columns),
        'summary': df.describe().to_dict()
    }}

    import json
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2)

    print(f"Processed data saved to: {{output_file}}")
    print("SUCCESS: File processing completed with pathlib")
else:
    print(f"ERROR: File {{csv_file}} not found")
        """

        # When: Python code is executed via /api/execute-raw endpoint
        execute_response = api_session.post(
            f"{Config.BASE_URL}/api/execute-raw",
            data=python_code,
            headers={"Content-Type": "text/plain"},
        )

        # Then: Code executes successfully with proper cross-platform file handling
        assert execute_response.status_code == 200

        # Validate API contract for execute-raw
        response_data = execute_response.json()
        assert response_data.get("success") is True
        assert "data" in response_data
        stdout = response_data.get("data", {}).get("stdout", "")
        assert "Successfully loaded CSV with 3 rows" in stdout
        assert "SUCCESS: File processing completed with pathlib" in stdout
        assert "ERROR" not in stdout

    def test_execute_data_visualization_with_pathlib(
        self, api_session: requests.Session
    ):
        """
        Test Python code execution for data visualization using pathlib.

        Given: Python code creates a matplotlib plot using pathlib for file operations
        When: Code is executed via /api/execute-raw endpoint
        Then: Plot is created successfully with cross-platform file paths

        Args:
            api_session: Configured requests session

        Examples:
            Test executes Python code that creates a matplotlib plot and saves it
            using pathlib for cross-platform path handling
        """
        # Given: Python code creates a matplotlib plot using pathlib
        python_code = """
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('File Upload Enhancement Test - Sine Wave')
plt.legend()
plt.grid(True, alpha=0.3)

# Use pathlib for cross-platform file operations
plots_dir = Path('/home/pyodide/plots/matplotlib')
plots_dir.mkdir(parents=True, exist_ok=True)
plot_file = plots_dir / 'file_upload_test_sine.png'

# Save plot
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"SUCCESS: Plot saved to: {plot_file}")
print(f"Plot file exists: {plot_file.exists()}")

# Verify file was created
if plot_file.exists():
    file_size = plot_file.stat().st_size
    print(f"Plot file size: {file_size} bytes")
    print("SUCCESS: Data visualization completed with pathlib")
else:
    print("ERROR: Plot file was not created")
        """

        # When: Code is executed via /api/execute-raw endpoint
        response = api_session.post(
            f"{Config.BASE_URL}/api/execute-raw",
            data=python_code,
            headers={"Content-Type": "text/plain"},
        )

        # Then: Plot is created successfully with cross-platform file paths
        assert response.status_code == 200

        response_data = response.json()
        assert response_data.get("success") is True
        stdout = response_data.get("data", {}).get("stdout", "")
        assert "SUCCESS: Plot saved to:" in stdout
        assert "SUCCESS: Data visualization completed with pathlib" in stdout
        assert "Plot file exists: True" in stdout
        assert "ERROR" not in stdout


class TestAPIContractCompliance:
    """BDD tests for API contract compliance and response validation."""

    def test_execute_raw_returns_json_with_api_contract(
        self, api_session: requests.Session
    ):
        """
        Test that /api/execute-raw returns JSON response following API contract.

        Given: Simple Python code is prepared for execution
        When: Code is sent to /api/execute-raw endpoint
        Then: Response follows the API contract with success, data, error, meta fields

        Args:
            api_session: Configured requests session

        Examples:
            Test verifies that execute-raw endpoint returns proper JSON response
            with the expected API contract structure including stdout, stderr, result
        """
        # Given: Simple Python code is prepared for execution
        python_code = """print("Hello from file upload test!")
result = 2 + 2
print(f"Calculation result: {result}")"""

        # When: Code is sent to /api/execute-raw endpoint
        response = api_session.post(
            f"{Config.BASE_URL}/api/execute-raw",
            data=python_code,
            headers={"Content-Type": "text/plain"},
        )

        # Then: Response follows the API contract with success, data, error, meta fields
        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("application/json")

        response_data = response.json()

        # Validate API contract structure
        assert response_data.get("success") is True
        assert "data" in response_data
        assert "error" in response_data
        assert "meta" in response_data
        assert response_data.get("error") is None
        assert "timestamp" in response_data.get("meta", {})

        # Validate execute-raw specific data structure
        data = response_data.get("data", {})
        assert "result" in data
        assert "stdout" in data
        assert "stderr" in data
        assert "executionTime" in data

        # Validate Python code output
        stdout = data.get("stdout", "")
        assert "Hello from file upload test!" in stdout
        assert "Calculation result: 4" in stdout

    def test_upload_endpoint_returns_json_with_success_field(
        self,
        api_session: requests.Session,
        temp_file_tracker: List[Path],
        uploaded_file_tracker: List[str],
    ):
        """
        Test that /api/upload returns JSON response with success field.

        Given: A valid file is prepared for upload
        When: File is uploaded via /api/upload endpoint
        Then: Response is JSON with success field and proper structure

        Args:
            api_session: Configured requests session
            temp_file_tracker: Temporary file cleanup tracker
            uploaded_file_tracker: Uploaded file cleanup tracker

        Examples:
            Test validates the JSON response structure from upload endpoint
            includes required success field and filename information
        """
        # Given: A valid file is prepared for upload
        test_content = "test,data\n1,2\n3,4"
        temp_file = Path(tempfile.mktemp(suffix=".csv"))
        temp_file_tracker.append(temp_file)

        with temp_file.open("w", encoding="utf-8") as f:
            f.write(test_content)

        # When: File is uploaded via /api/upload endpoint
        with temp_file.open("rb") as f:
            files = {"file": ("api_test.csv", f, "text/csv")}
            response = api_session.post(f"{Config.BASE_URL}/api/upload", files=files)

        # Then: Response is JSON with proper API contract structure
        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("application/json")

        response_data = response.json()
        assert isinstance(response_data, dict)

        # Validate API contract structure
        assert "success" in response_data
        assert response_data["success"] is True
        assert "data" in response_data
        assert "error" in response_data
        assert "meta" in response_data
        assert response_data.get("error") is None
        assert "timestamp" in response_data.get("meta", {})

        # Validate file data structure
        file_data = response_data.get("data", {}).get("file", {})
        assert "storedFilename" in file_data
        assert isinstance(file_data["storedFilename"], str)
        assert "originalName" in file_data
        assert file_data.get("originalName") == "api_test.csv"

        # Track for cleanup
        uploaded_file_tracker.append(file_data["storedFilename"])


class TestErrorHandlingAndEdgeCases:
    """BDD tests for error handling and edge cases in file upload system."""

    def test_execute_raw_handles_python_syntax_errors(
        self, api_session: requests.Session
    ):
        """
        Test that /api/execute-raw properly handles Python syntax errors.

        Given: Python code with syntax errors is prepared
        When: Invalid code is sent to /api/execute-raw endpoint
        Then: Appropriate error response is returned

        Args:
            api_session: Configured requests session

        Examples:
            Test sends Python code with syntax errors and validates
            that the server handles it gracefully
        """
        # Given: Python code with syntax errors is prepared
        invalid_python_code = """
print("Missing closing quote)
def invalid_function(
    return "syntax error"
        """

        # When: Invalid code is sent to /api/execute-raw endpoint
        response = api_session.post(
            f"{Config.BASE_URL}/api/execute-raw",
            data=invalid_python_code,
            headers={"Content-Type": "text/plain"},
        )

        # Then: Appropriate error response is returned
        # The server should return an error response with API contract
        if response.status_code == 200:
            response_data = response.json()
            # For successful HTTP status, error should be in the response structure
            assert response_data.get(
                "success"
            ) is False or "error" in response_data.get("data", {}).get("stderr", "")
        else:
            # For HTTP error status, should return proper error response
            assert response.status_code >= 400

    def test_execute_raw_with_file_not_found_error(self, api_session: requests.Session):
        """
        Test /api/execute-raw handles file not found errors gracefully.

        Given: Python code attempts to access non-existent file
        When: Code is executed via /api/execute-raw endpoint
        Then: Error is handled and reported appropriately

        Args:
            api_session: Configured requests session

        Examples:
            Test executes Python code that tries to read a non-existent file
            and validates error handling behavior
        """
        # Given: Python code attempts to access non-existent file
        python_code = """
from pathlib import Path

# Try to access non-existent file
nonexistent_file = Path('/home/pyodide/uploads/does_not_exist.csv')
print(f"Checking file: {nonexistent_file}")

if nonexistent_file.exists():
    print("File exists")
    with nonexistent_file.open('r') as f:
        content = f.read()
        print(f"Content: {content}")
else:
    print("File does not exist - this is expected")

# Try to read non-existent file (should raise error)
try:
    with open('/home/pyodide/uploads/definitely_does_not_exist.txt', 'r') as f:
        content = f.read()
except FileNotFoundError as e:
    print(f"Caught expected error: {e}")
    print("SUCCESS: Error handled correctly")
        """

        # When: Code is executed via /api/execute-raw endpoint
        response = api_session.post(
            f"{Config.BASE_URL}/api/execute-raw",
            data=python_code,
            headers={"Content-Type": "text/plain"},
        )

        # Then: Error is handled and reported appropriately
        assert response.status_code == 200
        response_data = response.json()
        assert response_data.get("success") is True
        stdout = response_data.get("data", {}).get("stdout", "")
        assert "File does not exist - this is expected" in stdout
        assert "SUCCESS: Error handled correctly" in stdout


class TestCrossPlatformCompatibility:
    """BDD tests for cross-platform compatibility of Python code execution."""

    def test_pathlib_usage_for_cross_platform_paths(
        self, api_session: requests.Session
    ):
        """
        Test that Python code uses pathlib for cross-platform path handling.

        Given: Python code uses pathlib for all file operations
        When: Code is executed via /api/execute-raw endpoint
        Then: Paths work correctly on both Windows and Linux systems

        Args:
            api_session: Configured requests session

        Examples:
            Test validates that pathlib is used consistently for all file
            operations to ensure cross-platform compatibility
        """
        # Given: Python code uses pathlib for all file operations
        python_code = """
from pathlib import Path
import os
import sys

print(f"Python version: {sys.version}")
print(f"Operating system: {os.name}")
print(f"Platform: {sys.platform}")

# Test pathlib cross-platform functionality
base_paths = ['/home/pyodide/uploads', '/home/pyodide/plots', '/plots/matplotlib']

for path_str in base_paths:
    path_obj = Path(path_str)
    print(f"Path: {path_obj}")
    print(f"  - Is absolute: {path_obj.is_absolute()}")
    print(f"  - Parts: {path_obj.parts}")
    print(f"  - Parent: {path_obj.parent}")

    # Create directory structure
    path_obj.mkdir(parents=True, exist_ok=True)
    print(f"  - Exists after mkdir: {path_obj.exists()}")

    # Test file creation
    test_file = path_obj / 'cross_platform_test.txt'
    with test_file.open('w', encoding='utf-8') as f:
        f.write(f"Created on {sys.platform} at {path_obj}")

    print(f"  - Test file exists: {test_file.exists()}")

    # Read back the content
    if test_file.exists():
        with test_file.open('r', encoding='utf-8') as f:
            content = f.read()
        print(f"  - Content: {content}")

print("SUCCESS: All pathlib operations completed successfully")
        """

        # When: Code is executed via /api/execute-raw endpoint
        response = api_session.post(
            f"{Config.BASE_URL}/api/execute-raw",
            data=python_code,
            headers={"Content-Type": "text/plain"},
        )

        # Then: Paths work correctly on both Windows and Linux systems
        assert response.status_code == 200
        response_data = response.json()
        assert response_data.get("success") is True
        stdout = response_data.get("data", {}).get("stdout", "")
        assert "SUCCESS: All pathlib operations completed successfully" in stdout
        assert "Is absolute: True" in stdout
        assert "Test file exists: True" in stdout

    def test_encoding_handling_for_cross_platform_files(
        self, api_session: requests.Session
    ):
        """
        Test proper encoding handling for cross-platform file operations.

        Given: Python code handles various text encodings explicitly
        When: Code processes files with different encodings
        Then: All text is handled correctly across platforms

        Args:
            api_session: Configured requests session

        Examples:
            Test creates and reads files with explicit UTF-8 encoding
            to ensure consistent behavior across Windows and Linux
        """
        # Given: Python code handles various text encodings explicitly
        python_code = """
from pathlib import Path

# Test UTF-8 encoding handling
test_dir = Path('/home/pyodide/plots/encoding_test')
test_dir.mkdir(parents=True, exist_ok=True)

# Test various text content including special characters
test_content = {
    'ascii': 'Hello World!',
    'utf8_basic': 'Café, naïve, résumé',
    'utf8_extended': '你好世界, 🚀 🐍 ✅',
    'numbers': '123.45, €100, $50'
}

success_count = 0

for test_name, content in test_content.items():
    test_file = test_dir / f'{test_name}_test.txt'

    try:
        # Write with explicit UTF-8 encoding
        with test_file.open('w', encoding='utf-8') as f:
            f.write(content)

        # Read back with explicit UTF-8 encoding
        with test_file.open('r', encoding='utf-8') as f:
            read_content = f.read()

        # Verify content matches
        if read_content == content:
            print(f"✅ {test_name}: Content matches ({len(content)} chars)")
            success_count += 1
        else:
            print(f"❌ {test_name}: Content mismatch")
            print(f"   Original: {repr(content)}")
            print(f"   Read back: {repr(read_content)}")

    except Exception as e:
        print(f"❌ {test_name}: Exception occurred - {e}")

print(f"\\nResults: {success_count}/{len(test_content)} tests passed")

if success_count == len(test_content):
    print("SUCCESS: All encoding tests passed")
else:
    print("WARNING: Some encoding tests failed")
        """

        # When: Code processes files with different encodings
        response = api_session.post(
            f"{Config.BASE_URL}/api/execute-raw",
            data=python_code,
            headers={"Content-Type": "text/plain"},
        )

        # Then: All text is handled correctly across platforms
        assert response.status_code == 200
        response_data = response.json()
        assert response_data.get("success") is True
        stdout = response_data.get("data", {}).get("stdout", "")
        assert "SUCCESS: All encoding tests passed" in stdout
        assert "4/4 tests passed" in stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

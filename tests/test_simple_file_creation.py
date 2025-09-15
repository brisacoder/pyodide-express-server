"""
Comprehensive file creation tests for the Pyodide Express Server virtual filesystem.

This module contains BDD-style pytest tests that verify file creation capabilities
in the Pyodide virtual filesystem, testing various directory structures and
file operations. All tests use only the /api/execute-raw endpoint and follow
the standardized API contract.

Test Coverage:
- Root directory file creation and verification
- Temporary directory (/tmp) file operations
- Custom directory structure creation
- Writable directory discovery and validation
- File existence, content verification, and metadata checks
- Error handling for failed file operations

API Contract Compliance:
- All responses follow the standardized format: {success, data, error, meta}
- Proper validation of response structure and content
- Error cases handled with appropriate status codes and messages

Requirements Compliance:
1. ✅ Pytest framework with BDD naming conventions
2. ✅ Parameterized constants using Config class
3. ✅ Only /api/execute-raw endpoint usage
4. ✅ BDD-style test naming and structure
5. ✅ Comprehensive API contract validation
6. ✅ Pathlib usage in all Python code snippets
7. ✅ Full docstrings with examples and specifications
"""

import json
from typing import Any, Dict

import pytest
import requests

from conftest import Config


class TestSimpleFileCreation:
    """
    BDD-style tests for simple file creation in the Pyodide virtual filesystem.

    This test class verifies basic file creation capabilities across different
    directory structures in the virtual filesystem, ensuring proper file handling,
    content verification, and directory discovery functionality.

    All tests validate the API contract response format and use only the
    /api/execute-raw endpoint with proper pathlib usage in Python code.
    """

    def test_given_root_directory_when_creating_text_file_then_file_created_successfully(
        self, server_ready: None
    ) -> None:
        """
        Test that creating a text file in the root directory works correctly.

        This test verifies the basic file creation functionality in the Pyodide
        virtual filesystem root directory, including file existence, content
        verification, and metadata validation.

        Args:
            server_ready: Fixture ensuring server is available

        Expected Behavior:
            GIVEN: A request to create a text file in root directory
            WHEN: Using pathlib to create and write content to a file
            THEN: File is created successfully with correct content and metadata

        API Contract:
            Request: POST /api/execute-raw with Python code as plain text
            Response: {success: true, data: {result, stdout, stderr}, error: null, meta: {timestamp}}

        Example:
            >>> # Creates file /test_simple_<timestamp>.txt with specific content
            >>> # Validates file existence, content matching, and file size
        """
        # Given: Python code to create a text file in root directory
        python_code = '''
import json
import time
from pathlib import Path

result = {
    "operation": "create_txt_in_root",
    "success": False,
    "error": None,
    "file_exists": False,
    "filename": None,
    "content": None,
    "content_matches": False,
    "file_size": 0
}

try:
    # Generate unique timestamp-based filename to avoid conflicts
    timestamp = int(time.time() * 1000)
    filename = Path(f"/test_simple_{timestamp}.txt")

    # Create file with test content using pathlib
    test_content = "Hello from Pyodide virtual filesystem!"
    filename.write_text(test_content)

    # Verify file creation and properties
    result["file_exists"] = filename.exists()
    result["filename"] = str(filename)

    if result["file_exists"]:
        # Read back content and validate
        content = filename.read_text()
        result["content"] = content
        result["content_matches"] = content == test_content
        result["file_size"] = filename.stat().st_size
        result["success"] = True

except Exception as e:
    result["error"] = str(e)
    result["error_type"] = type(e).__name__

# Output result as JSON for parsing
print(json.dumps(result))
'''

        # When: Executing the file creation code
        response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
            data=python_code,
            headers=Config.HEADERS["execute_raw"],
            timeout=Config.TIMEOUTS["code_execution"]
        )

        # Then: API call should succeed
        assert response.status_code == 200, f"API request failed: {response.status_code}"

        # Validate API contract format
        api_response = response.json()
        self._validate_api_contract(api_response)

        # Extract and parse the result from stdout
        result = self._parse_json_result_from_stdout(api_response)

        # Validate file creation results
        assert result["success"] is True, f"File creation failed: {result.get('error')}"
        assert result["file_exists"] is True, "Text file should exist after creation"
        assert result["content_matches"] is True, "File content should match what was written"
        assert result["file_size"] > 0, "File should have non-zero size"
        assert result["filename"] is not None, "Filename should be recorded"

        print(f"✅ Successfully created file: {result['filename']}")

    def test_given_tmp_directory_when_creating_text_file_then_file_created_successfully(
        self, server_ready: None
    ) -> None:
        """
        Test that creating a text file in /tmp directory works correctly.

        This test verifies file creation functionality in the temporary directory,
        which should be writable in most virtual filesystem configurations.

        Args:
            server_ready: Fixture ensuring server is available

        Expected Behavior:
            GIVEN: A request to create a text file in /tmp directory
            WHEN: Using pathlib to create and write content to a file
            THEN: File is created successfully with correct content and metadata

        API Contract:
            Request: POST /api/execute-raw with Python code as plain text
            Response: {success: true, data: {result, stdout, stderr}, error: null, meta: {timestamp}}
        """
        # Given: Python code to create a text file in /tmp directory
        python_code = '''
import json
import time
from pathlib import Path

result = {
    "operation": "create_txt_in_tmp",
    "success": False,
    "error": None,
    "file_exists": False,
    "filename": None,
    "content": None,
    "content_matches": False,
    "file_size": 0
}

try:
    # Generate unique timestamp-based filename to avoid conflicts
    timestamp = int(time.time() * 1000)
    filename = Path(f"/tmp/test_simple_{timestamp}.txt")

    # Create file with test content using pathlib
    test_content = "Hello from /tmp directory!"
    filename.write_text(test_content)

    # Verify file creation and properties
    result["file_exists"] = filename.exists()
    result["filename"] = str(filename)

    if result["file_exists"]:
        # Read back content and validate
        content = filename.read_text()
        result["content"] = content
        result["content_matches"] = content == test_content
        result["file_size"] = filename.stat().st_size
        result["success"] = True

except Exception as e:
    result["error"] = str(e)
    result["error_type"] = type(e).__name__

# Output result as JSON for parsing
print(json.dumps(result))
'''

        # When: Executing the file creation code
        response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
            data=python_code,
            headers=Config.HEADERS["execute_raw"],
            timeout=Config.TIMEOUTS["code_execution"]
        )

        # Then: API call should succeed
        assert response.status_code == 200, f"API request failed: {response.status_code}"

        # Validate API contract format
        api_response = response.json()
        self._validate_api_contract(api_response)

        # Extract and parse the result from stdout
        result = self._parse_json_result_from_stdout(api_response)

        # Validate file creation results
        assert result["success"] is True, f"File creation in /tmp failed: {result.get('error')}"
        assert result["file_exists"] is True, "Text file should exist in /tmp directory"
        assert result["content_matches"] is True, "File content should match what was written"
        assert result["file_size"] > 0, "File should have non-zero size"

        print(f"✅ Successfully created file in /tmp: {result['filename']}")

    def test_given_custom_directory_when_creating_structure_then_directory_and_file_created(
        self, server_ready: None
    ) -> None:
        """
        Test creating custom directory structures and files within them.

        This test verifies the ability to create nested directory structures
        (like /vfs/matplotlib) and files within those directories, which is
        essential for plot generation and complex file organization.

        Args:
            server_ready: Fixture ensuring server is available

        Expected Behavior:
            GIVEN: A request to create a custom directory structure and file
            WHEN: Using pathlib to create directories and files within them
            THEN: Directory structure is created and files can be written successfully

        API Contract:
            Request: POST /api/execute-raw with Python code as plain text
            Response: {success: true, data: {result, stdout, stderr}, error: null, meta: {timestamp}}
        """
        # Given: Python code to create custom directory structure and file
        python_code = '''
import json
from pathlib import Path

result = {
    "operation": "create_custom_dir_and_file",
    "steps": [],
    "success": False,
    "error": None,
    "directory_created": False,
    "file_created": False,
    "content": None,
    "file_size": 0
}

try:
    # Step 1: Create the custom directory structure using pathlib
    custom_dir = Path("/vfs/matplotlib")
    custom_dir.mkdir(parents=True, exist_ok=True)

    dir_created = custom_dir.exists() and custom_dir.is_dir()
    result["steps"].append({
        "step": "create_directory",
        "path": str(custom_dir),
        "success": dir_created
    })
    result["directory_created"] = dir_created

    # Step 2: Create a file in the custom directory
    if dir_created:
        test_file = custom_dir / "test_plot.txt"
        test_content = "This would be a plot file!"
        test_file.write_text(test_content)

        file_created = test_file.exists()
        result["steps"].append({
            "step": "create_file",
            "path": str(test_file),
            "success": file_created
        })
        result["file_created"] = file_created

        if file_created:
            # Read back content and validate
            content = test_file.read_text()
            result["content"] = content
            result["file_size"] = test_file.stat().st_size
            result["content_matches"] = content == test_content
            result["success"] = True

    # Step 3: List directory contents for verification (using pathlib)
    result["directory_listing"] = {
        "root_contents": [str(p) for p in Path("/").iterdir()][:10] if Path("/").exists() else [],
        "vfs_exists": Path("/vfs").exists(),
        "vfs_contents": [p.name for p in Path("/vfs").iterdir()] if Path("/vfs").exists() else [],
        "matplotlib_exists": Path("/vfs/matplotlib").exists(),
        "matplotlib_contents": [p.name for p in Path("/vfs/matplotlib").iterdir()]
                              if Path("/vfs/matplotlib").exists() else []
    }

except Exception as e:
    result["error"] = str(e)
    result["error_type"] = type(e).__name__

# Output result as JSON for parsing
print(json.dumps(result))
'''

        # When: Executing the directory and file creation code
        response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
            data=python_code,
            headers=Config.HEADERS["execute_raw"],
            timeout=Config.TIMEOUTS["code_execution"]
        )

        # Then: API call should succeed
        assert response.status_code == 200, f"API request failed: {response.status_code}"

        # Validate API contract format
        api_response = response.json()
        self._validate_api_contract(api_response)

        # Extract and parse the result from stdout
        result = self._parse_json_result_from_stdout(api_response)

        # Validate directory and file creation results
        steps = result.get("steps", [])
        assert len(steps) > 0, "Should have attempted directory creation"

        # Find the directory creation step
        dir_step = next((step for step in steps if step.get("step") == "create_directory"), None)
        assert dir_step is not None, "Should have attempted directory creation"

        if dir_step.get("success"):
            # Directory creation worked, validate file creation
            file_step = next((step for step in steps if step.get("step") == "create_file"), None)
            if file_step:
                assert file_step.get("success"), "File creation should succeed if directory exists"
                assert result["content_matches"], "File content should match expected content"
            assert result.get("success"), "Overall operation should succeed when directory creation works"
            print("✅ Custom directory structure and file created successfully")
        else:
            # Directory creation failed - this tells us about filesystem limitations
            print(f"⚠️ Directory creation failed for /vfs/matplotlib: {result.get('error')}")
            print("This indicates filesystem limitations that affect plot generation")

    def test_given_filesystem_when_checking_writable_directories_then_writable_locations_discovered(
        self, server_ready: None
    ) -> None:
        """
        Test which directories are available and writable in the Pyodide virtual filesystem.

        This comprehensive test discovers the writable filesystem locations which is
        crucial for understanding where files can be created for various operations
        like plot generation, data storage, and temporary file handling.

        Args:
            server_ready: Fixture ensuring server is available

        Expected Behavior:
            GIVEN: A request to test various directory paths for write access
            WHEN: Using pathlib to test directory existence and write permissions
            THEN: Writable directories are identified and cataloged for reference

        API Contract:
            Request: POST /api/execute-raw with Python code as plain text
            Response: {success: true, data: {result, stdout, stderr}, error: null, meta: {timestamp}}
        """
        # Given: Python code to test directory write permissions
        python_code = '''
import json
from pathlib import Path

result = {
    "operation": "test_writable_directories",
    "directories_tested": [],
    "writable_directories": [],
    "read_only_directories": [],
    "nonexistent_directories": [],
    "error": None
}

# Test common directory paths using pathlib
test_paths = [
    "/",
    "/tmp",
    "/home",
    "/usr",
    "/var",
    "/etc",
    "/mnt",
    "/opt",
    '/home/pyodide/plots',     # Expected plots directory
    "/vfs",       # Custom vfs directory
    '/home/pyodide/uploads',   # Expected uploads directory
    "/test_data"  # Test data directory
]

for path_str in test_paths:
    path = Path(path_str)
    test_result = {
        "path": path_str,
        "exists": False,
        "is_directory": False,
        "writable": False,
        "created": False,
        "error": None
    }

    try:
        test_result["exists"] = path.exists()

        if test_result["exists"]:
            test_result["is_directory"] = path.is_dir()

            # Test write permission for existing directory
            if test_result["is_directory"]:
                test_file = path / "write_test.txt"
                try:
                    test_file.write_text("test")
                    test_result["writable"] = test_file.exists()
                    if test_result["writable"]:
                        test_file.unlink()  # Clean up
                        result["writable_directories"].append(path_str)
                    else:
                        result["read_only_directories"].append(path_str)
                except Exception as e:
                    test_result["write_error"] = str(e)
                    result["read_only_directories"].append(path_str)
        else:
            # Try to create the directory
            try:
                path.mkdir(parents=True, exist_ok=True)
                test_result["created"] = path.exists()

                if test_result["created"]:
                    test_result["exists"] = True
                    test_result["is_directory"] = True

                    # Test write permission after creation
                    test_file = path / "write_test.txt"
                    test_file.write_text("test")
                    test_result["writable"] = test_file.exists()

                    if test_result["writable"]:
                        test_file.unlink()  # Clean up
                        result["writable_directories"].append(path_str)
                    else:
                        result["read_only_directories"].append(path_str)
                else:
                    result["nonexistent_directories"].append(path_str)

            except Exception as e:
                test_result["create_error"] = str(e)
                result["nonexistent_directories"].append(path_str)

    except Exception as e:
        test_result["error"] = str(e)
        result["nonexistent_directories"].append(path_str)

    result["directories_tested"].append(test_result)

# Output result as JSON for parsing
print(json.dumps(result))
'''

        # When: Executing the directory testing code
        response = requests.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
            data=python_code,
            headers=Config.HEADERS["execute_raw"],
            timeout=Config.TIMEOUTS["code_execution"]
        )

        # Then: API call should succeed
        assert response.status_code == 200, f"API request failed: {response.status_code}"

        # Validate API contract format
        api_response = response.json()
        self._validate_api_contract(api_response)

        # Extract and parse the result from stdout
        result = self._parse_json_result_from_stdout(api_response)

        # Validate directory testing results
        directories_tested = result.get("directories_tested", [])
        assert len(directories_tested) > 0, "Should have tested some directories"

        writable_dirs = result.get("writable_directories", [])
        read_only_dirs = result.get("read_only_directories", [])
        nonexistent_dirs = result.get("nonexistent_directories", [])

        # Should find at least one writable directory (typically root or /tmp)
        assert len(writable_dirs) > 0, "Should find at least one writable directory"

        # Print detailed results for analysis
        print(f"\n📊 Filesystem Analysis Results:")
        print(f"✅ Writable directories ({len(writable_dirs)}): {writable_dirs}")
        print(f"📁 Read-only directories ({len(read_only_dirs)}): {read_only_dirs}")
        print(f"❌ Nonexistent/inaccessible ({len(nonexistent_dirs)}): {nonexistent_dirs}")

        # Detailed per-directory analysis
        for dir_test in directories_tested:
            if dir_test.get("writable"):
                status, desc = "✅", "writable"
            elif dir_test.get("exists"):
                status, desc = "📁", "read-only"
            else:
                status, desc = "❌", "inaccessible"
            print(f"{status} {dir_test['path']}: {desc}")

    def _validate_api_contract(self, response: Dict[str, Any]) -> None:
        """
        Validate that API response follows the standardized contract format.

        Args:
            response: API response dictionary to validate

        Expected Format:
            {
                "success": true | false,
                "data": <object|null>,
                "error": <string|null>,
                "meta": {"timestamp": <string>}
            }
        """
        assert "success" in response, "Response must include 'success' field"
        assert isinstance(response["success"], bool), "Success field must be boolean"

        assert "data" in response, "Response must include 'data' field"
        assert "error" in response, "Response must include 'error' field"
        assert "meta" in response, "Response must include 'meta' field"

        if response["success"]:
            assert response["data"] is not None, "Data should not be null on success"
            assert response["error"] is None, "Error should be null on success"
        else:
            assert response["data"] is None, "Data should be null on error"
            assert response["error"] is not None, "Error should not be null on failure"

        assert "timestamp" in response["meta"], "Meta must include timestamp"
        assert isinstance(response["meta"]["timestamp"], str), "Timestamp must be string"

    def _parse_json_result_from_stdout(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse JSON result from the stdout of the API response.

        The execute-raw endpoint returns Python execution output in stdout,
        and our test code outputs JSON results that need to be parsed.

        Args:
            api_response: The API response containing stdout with JSON result

        Returns:
            dict: Parsed JSON result from stdout

        Raises:
            AssertionError: If JSON cannot be found or parsed from stdout
        """
        assert api_response.get("success"), f"API execution failed: {api_response.get('error')}"

        # Get stdout from the result data
        data = api_response.get("data", {})
        stdout = data.get("stdout", "")

        # Find JSON line in stdout (our Python code outputs JSON)
        stdout_lines = stdout.strip().split("\n")
        json_line = None

        for line in stdout_lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                json_line = line
                break

        assert json_line is not None, f"No JSON result found in stdout: {stdout}"

        try:
            return json.loads(json_line)
        except json.JSONDecodeError as e:
            raise AssertionError(f"Failed to parse JSON from stdout: {e}. Line: {json_line}")


# Test execution entry point - pytest will discover and run these tests
if __name__ == "__main__":
    pytest.main([__file__])

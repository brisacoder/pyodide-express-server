"""
Comprehensive filesystem persistence tests for Pyodide Express Server.

This module tests filesystem operations including file creation, persistence,
virtual filesystem behavior, and directory operations using pathlib within
the Pyodide environment. Uses only the /api/execute-raw endpoint.
"""

import requests
import time
from pathlib import Path


class TestFilesystemOperations:
    """Test filesystem operations and persistence in Pyodide environment.

    This test class validates that file operations work correctly through
    the /api/execute-raw endpoint, including mounted directory persistence,
    virtual filesystem behavior, and comprehensive file operations.
    """

    def test_given_mounted_directory_when_file_created_then_appears_in_local_filesystem(
        self, server, base_url, api_timeout
    ):
        """Test file creation and persistence in mounted directories.

        This test validates that files created in mounted directories (like /plots)
        within Pyodide appear in the corresponding local filesystem directories,
        demonstrating proper filesystem mounting functionality.

        Args:
            server: Server fixture ensuring the service is running
            base_url: Base URL for API requests
            api_timeout: Default timeout for API requests

        Input:
            - File creation in mounted directory /plots/matplotlib/
            - Text content written to file
            - pathlib-based file operations

        Expected Output:
            - File exists both in Pyodide and local filesystem
            - Content matches between Pyodide and local file
            - Successful API response following contract
            - Local file can be read and verified

        Example:
            API Response:
            {
                "success": true,
                "data": {
                    "result": {
                        "stdout": "...",
                        "stderr": "",
                        "executionTime": 123
                    }
                },
                "error": null,
                "meta": {"timestamp": "..."}
            }
        """
        # Create unique filename for test isolation
        timestamp = int(time.time() * 1000)
        filename = f"simple_test_{timestamp}.txt"

        # Python code to create files using pathlib
        create_code = f'''
from pathlib import Path

# Create file in plots directory (should appear locally)
plots_file = Path("/plots/matplotlib/{filename}")
plots_file.parent.mkdir(parents=True, exist_ok=True)

# Write test content
test_content = "Test content from Pyodide filesystem"
plots_file.write_text(test_content)

print("Created file:", plots_file)
print("File exists in Pyodide:", plots_file.exists())
print("File content:", plots_file.read_text())

print("Test completed successfully for {filename}")
'''

        # Execute code via /api/execute-raw endpoint
        response = requests.post(
            f"{base_url}/api/execute-raw",
            data=create_code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout
        )

        assert response.status_code == 200

        # Validate API contract
        result = response.json()
        assert result.get("success") is True, f"API call failed: {result}"
        assert "data" in result, f"Missing 'data' field in response: {result}"
        assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
        assert "meta" in result, f"Missing 'meta' field in response: {result}"
        assert "timestamp" in result["meta"], f"Missing timestamp in meta: {result}"

        # Validate execution results
        result_data = result["data"]["result"]
        assert "stdout" in result_data, f"Missing stdout in result: {result_data}"
        stdout = result_data["stdout"]

        # Validate file operations in Pyodide
        assert "Created file:" in stdout
        assert "File exists in Pyodide: True" in stdout
        assert "Test content from Pyodide filesystem" in stdout

        # Check if file appears in local filesystem (mounted to pyodide_data/plots/)
        local_file = Path(__file__).parent.parent / "pyodide_data" / "plots" / "matplotlib" / filename
        assert local_file.exists(), f"File should appear in local filesystem: {local_file}"

        # Verify content matches between Pyodide and local
        local_content = local_file.read_text()
        assert local_content == "Test content from Pyodide filesystem", \
            f"Content mismatch: expected 'Test content from Pyodide filesystem', got '{local_content}'"

        # Clean up test file
        local_file.unlink()

        print(f"✅ Test passed: File {filename} created and persisted locally")

    def test_given_virtual_directory_when_file_created_then_does_not_appear_locally(
        self, server, base_url, api_timeout
    ):
        """Test virtual filesystem behavior - files stay within Pyodide only.

        This test validates that files created in virtual directories (like /tmp)
        exist within the Pyodide environment but do not persist to the local
        filesystem, demonstrating proper virtual filesystem isolation.

        Args:
            server: Server fixture ensuring the service is running
            base_url: Base URL for API requests
            api_timeout: Default timeout for API requests

        Input:
            - File creation in virtual directory /tmp/
            - Text content written to virtual file
            - pathlib-based file operations

        Expected Output:
            - File exists only within Pyodide environment
            - File does NOT appear in local filesystem
            - Content accessible within Pyodide session
            - Successful API response following contract

        Example:
            Expected behavior:
            - /tmp/file.txt exists in Pyodide: True
            - /tmp/file.txt exists locally: False
            - Virtual filesystem isolation maintained
        """
        # Create unique filename for test isolation
        timestamp = int(time.time() * 1000)
        filename = f"tmp_test_{timestamp}.txt"

        # Python code to create files in virtual filesystem
        create_code = f'''
from pathlib import Path

# Create file in /tmp (virtual filesystem)
tmp_file = Path("/tmp/{filename}")
tmp_file.parent.mkdir(parents=True, exist_ok=True)

# Write test content to virtual file
virtual_content = "Temporary content in virtual filesystem"
tmp_file.write_text(virtual_content)

print(f"Created virtual file: {{tmp_file}}")
print(f"Virtual file exists: {{tmp_file.exists()}}")
print(f"Virtual file content: {{tmp_file.read_text()}}")

# Return result dictionary for validation
result = {{
    "tmp_exists": tmp_file.exists(),
    "tmp_content": tmp_file.read_text(),
    "filename": "{filename}",
    "full_path": str(tmp_file)
}}

print(f"Virtual filesystem result: {{result}}")
result
'''

        response = requests.post(
            f"{base_url}/api/execute-raw",
            data=create_code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout
        )

        assert response.status_code == 200

        # Validate API contract
        result = response.json()
        assert result.get("success") is True, f"API call failed: {result}"
        assert "data" in result, f"Missing 'data' field in response: {result}"
        assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
        assert "meta" in result, f"Missing 'meta' field in response: {result}"
        assert "timestamp" in result["meta"], f"Missing timestamp in meta: {result}"

        # Validate execution results
        result_data = result["data"]["result"]
        assert "stdout" in result_data, f"Missing stdout in result: {result_data}"
        stdout = result_data["stdout"]

        # Validate virtual file operations
        assert "Created virtual file:" in stdout
        assert "Virtual file exists: True" in stdout
        assert "Temporary content in virtual filesystem" in stdout

        # Verify /tmp files do NOT appear in local filesystem
        local_tmp = Path(__file__).parent.parent / "tmp" / filename
        assert not local_tmp.exists(), \
            f"/tmp files should not appear in local filesystem: {local_tmp}"

        # Also check common tmp locations don't exist
        alt_tmp_locations = [
            Path(__file__).parent.parent / "tmp",
            Path("/tmp") / filename if Path("/tmp").exists() else None
        ]

        for tmp_path in alt_tmp_locations:
            if tmp_path and tmp_path != local_tmp:
                assert not (tmp_path / filename if tmp_path.is_dir() else tmp_path).exists(), \
                    f"Virtual file should not exist at: {tmp_path}"

        print("✅ Test passed: /tmp file exists in Pyodide but not locally")

    def test_given_directory_operations_when_nested_paths_created_then_structure_persists(
        self, server, base_url, api_timeout
    ):
        """Test complex directory structure creation and persistence.

        This test validates that nested directory structures created within
        Pyodide persist properly to the local filesystem, demonstrating
        comprehensive filesystem mounting and directory operations.

        Args:
            server: Server fixture ensuring the service is running
            base_url: Base URL for API requests
            api_timeout: Default timeout for API requests

        Input:
            - Nested directory structure creation
            - Multiple files in different subdirectories
            - pathlib-based directory operations

        Expected Output:
            - All directories created both in Pyodide and locally
            - All files accessible with correct content
            - Directory structure matches between environments
            - Successful API response following contract

        Example:
            Directory structure created:
            /plots/
            ├── subdir1/
            │   └── file1.txt
            └── subdir2/
                ├── file2.txt
                └── nested/
                    └── file3.txt
        """
        # Create unique identifier for test isolation
        timestamp = int(time.time() * 1000)

        # Python code to create nested directory structure
        structure_code = f'''
from pathlib import Path

# Create nested directory structure
base_dir = Path("/plots/filesystem_test_{timestamp}")
subdirs = [
    base_dir / "subdir1",
    base_dir / "subdir2",
    base_dir / "subdir2" / "nested"
]

# Create directories and files
created_files = []
for i, subdir in enumerate(subdirs, 1):
    subdir.mkdir(parents=True, exist_ok=True)

    # Create a file in each directory
    file_path = subdir / f"test_file_{{i}}.txt"
    content = f"Content for file {{i}} in {{subdir.name}} - timestamp {timestamp}"
    file_path.write_text(content)
    created_files.append({{
        "path": str(file_path),
        "exists": file_path.exists(),
        "content": file_path.read_text(),
        "parent_exists": file_path.parent.exists()
    }})

    print(f"Created: {{file_path}} -> {{file_path.exists()}}")

# Verify directory structure
structure_info = {{
    "base_dir": str(base_dir),
    "base_exists": base_dir.exists(),
    "subdirs_count": len(subdirs),
    "files_created": len(created_files),
    "all_files_exist": all(f["exists"] for f in created_files),
    "files": created_files
}}

print(f"Directory structure: {{structure_info}}")
structure_info
'''

        response = requests.post(
            f"{base_url}/api/execute-raw",
            data=structure_code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout * 2
        )

        assert response.status_code == 200

        # Validate API contract
        result = response.json()
        assert result.get("success") is True, f"API call failed: {result}"
        assert "data" in result, f"Missing 'data' field in response: {result}"
        assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
        assert "meta" in result, f"Missing 'meta' field in response: {result}"
        assert "timestamp" in result["meta"], f"Missing timestamp in meta: {result}"

        # Validate execution results
        result_data = result["data"]["result"]
        assert "stdout" in result_data, f"Missing stdout in result: {result_data}"
        stdout = result_data["stdout"]

        # Validate directory structure creation
        assert "Created:" in stdout
        assert f"filesystem_test_{timestamp}" in stdout

        # Verify local filesystem persistence (mounted to pyodide_data/plots/)
        local_base = Path(__file__).parent.parent / "pyodide_data" / "plots" / f"filesystem_test_{timestamp}"
        assert local_base.exists(), f"Base directory should exist locally: {local_base}"

        # Check subdirectories exist locally
        expected_subdirs = [
            local_base / "subdir1",
            local_base / "subdir2",
            local_base / "subdir2" / "nested"
        ]

        for subdir in expected_subdirs:
            assert subdir.exists(), f"Subdirectory should exist locally: {subdir}"

        # Check files exist locally with correct content
        expected_files = [
            (local_base / "subdir1" / "test_file_1.txt", "Content for file 1 in subdir1"),
            (local_base / "subdir2" / "test_file_2.txt", "Content for file 2 in subdir2"),
            (local_base / "subdir2" / "nested" / "test_file_3.txt", "Content for file 3 in nested")
        ]

        for file_path, expected_content_prefix in expected_files:
            assert file_path.exists(), f"File should exist locally: {file_path}"
            content = file_path.read_text()
            assert expected_content_prefix in content, \
                f"File content mismatch: {content}"

        # Clean up test directory
        import shutil
        shutil.rmtree(local_base)

        print("✅ Test passed: Complex directory structure created and persisted")

    def test_given_file_operations_when_read_write_performed_then_data_integrity_maintained(
        self, server, base_url, api_timeout
    ):
        """Test comprehensive file operations with data integrity verification.

        This test validates various file operations including creation, writing,
        reading, appending, and modification while ensuring data integrity
        is maintained between Pyodide and local filesystem.

        Args:
            server: Server fixture ensuring the service is running
            base_url: Base URL for API requests
            api_timeout: Default timeout for API requests

        Input:
            - File creation with initial content
            - File modification and appending
            - File reading and content verification
            - pathlib-based file operations

        Expected Output:
            - All file operations succeed
            - Content integrity maintained
            - Local filesystem reflects all changes
            - Successful API response following contract

        Example:
            Operations performed:
            1. Create file with initial content
            2. Append additional content
            3. Verify content matches expectations
            4. Check local filesystem synchronization
        """
        # Create unique identifier for test isolation
        timestamp = int(time.time() * 1000)
        filename = f"file_ops_test_{timestamp}.txt"

        # Python code for comprehensive file operations
        file_ops_code = f'''
from pathlib import Path
import json

# File operations test
test_file = Path("/plots/matplotlib/{filename}")
test_file.parent.mkdir(parents=True, exist_ok=True)

operations = []

# 1. Create file with initial content
initial_content = "Initial content - Line 1\\nInitial content - Line 2\\n"
test_file.write_text(initial_content)
operations.append({{
    "operation": "write_initial",
    "content_length": len(initial_content),
    "file_exists": test_file.exists()
}})
print(f"Step 1: Created file with {{len(initial_content)}} characters")

# 2. Append additional content
append_content = "Appended content - Line 3\\nAppended content - Line 4\\n"
with test_file.open("a") as f:
    f.write(append_content)
operations.append({{
    "operation": "append",
    "appended_length": len(append_content),
    "total_size": test_file.stat().st_size
}})
print(f"Step 2: Appended {{len(append_content)}} characters")

# 3. Read and verify content
final_content = test_file.read_text()
expected_lines = ["Initial content - Line 1", "Initial content - Line 2",
                 "Appended content - Line 3", "Appended content - Line 4"]
actual_lines = [line.strip() for line in final_content.split("\\n") if line.strip()]

operations.append({{
    "operation": "read_verify",
    "expected_lines": len(expected_lines),
    "actual_lines": len(actual_lines),
    "content_match": actual_lines == expected_lines,
    "final_content": final_content
}})
print(f"Step 3: Verified {{len(actual_lines)}} lines of content")

# 4. File metadata
file_stats = {{
    "size": test_file.stat().st_size,
    "exists": test_file.exists(),
    "is_file": test_file.is_file(),
    "name": test_file.name,
    "parent": str(test_file.parent)
}}

result = {{
    "filename": "{filename}",
    "operations": operations,
    "file_stats": file_stats,
    "final_content": final_content,
    "content_lines": actual_lines,
    "all_operations_success": all(op.get("file_exists", True) for op in operations)
}}

print(f"File operations completed: {{json.dumps(result, indent=2)}}")
result
'''

        response = requests.post(
            f"{base_url}/api/execute-raw",
            data=file_ops_code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout * 2
        )

        assert response.status_code == 200

        # Validate API contract
        result = response.json()
        assert result.get("success") is True, f"API call failed: {result}"
        assert "data" in result, f"Missing 'data' field in response: {result}"
        assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
        assert "meta" in result, f"Missing 'meta' field in response: {result}"
        assert "timestamp" in result["meta"], f"Missing timestamp in meta: {result}"

        # Validate execution results
        result_data = result["data"]["result"]
        assert "stdout" in result_data, f"Missing stdout in result: {result_data}"
        stdout = result_data["stdout"]

        # Validate file operations
        assert "Step 1: Created file" in stdout
        assert "Step 2: Appended" in stdout
        assert "Step 3: Verified" in stdout

        # Verify local filesystem synchronization (mounted to pyodide_data/plots/)
        local_file = Path(__file__).parent.parent / "pyodide_data" / "plots" / "matplotlib" / filename
        assert local_file.exists(), f"File should exist locally: {local_file}"

        # Verify content integrity
        local_content = local_file.read_text()
        expected_content = (
            "Initial content - Line 1\n"
            "Initial content - Line 2\n"
            "Appended content - Line 3\n"
            "Appended content - Line 4\n"
        )
        assert local_content == expected_content, \
            f"Content mismatch:\nExpected: {repr(expected_content)}\nActual: {repr(local_content)}"

        # Clean up test file
        local_file.unlink()

        print("✅ Test passed: File operations completed with data integrity")

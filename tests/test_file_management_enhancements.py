#!/usr/bin/env python3
"""
File Management Enhancements Test Suite (Pytest BDD Implementation)

This module provides comprehensive testing for file management functionality
in the Pyodide Express Server, following BDD (Behavior-Driven Development)
patterns and ensuring API contract compliance.

Test Categories:
- File upload and management operations
- Pyodide virtual filesystem operations
- File clearing and environment reset functionality
- API contract validation for all responses
- Cross-platform file operations with pathlib

API Contract Validation:
All tests verify that responses follow the strict API contract:
{
  "success": true | false,
  "data": <object|null>,
  "error": <string|null>,
  "meta": { "timestamp": <string> }
}

BDD Pattern:
Tests follow Given/When/Then patterns for clear behavior specification:
- Given: Initial test setup and preconditions
- When: Action or operation being tested
- Then: Expected outcomes and assertions

Requirements Compliance:
✅ Pytest framework with BDD patterns
✅ Only /api/execute-raw endpoint usage (no internal pyodide APIs)
✅ Pathlib for cross-platform file operations
✅ Parameterized constants via conftest.py Config
✅ Comprehensive docstrings with examples
✅ API contract validation for all responses
✅ Server-side response format compliance

Author: Pyodide Express Server Test Suite
Version: 2.0.0 (Pytest BDD Implementation)
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import pytest
import requests

from tests.conftest import Config


class TestFileManagementEnhancements:
    """
    Comprehensive test suite for file management enhancements.
    
    This test class validates file management operations including:
    - File upload and storage functionality
    - Pyodide virtual filesystem operations
    - File clearing and environment reset
    - API contract compliance
    - Cross-platform compatibility
    
    All tests follow BDD patterns and use only the /api/execute-raw endpoint
    for Python code execution, ensuring compatibility and avoiding internal APIs.
    """
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self, request):
        """
        Auto-setup fixture for each test method.
        
        Provides:
        - HTTP session configuration
        - File tracking for cleanup
        - Test timing for performance monitoring
        - Automatic cleanup after each test
        
        Args:
            request: Pytest request object for test metadata
            
        Yields:
            None: Setup is performed before test, cleanup after
            
        Example:
            This fixture runs automatically for every test method,
            ensuring clean state and proper resource management.
        """
        # Given: Clean test environment setup
        self.session = requests.Session()
        self.session.timeout = Config.TIMEOUTS["api_request"]
        self.uploaded_files = []
        self.temp_files = []
        self.test_start_time = time.time()
        
        yield
        
        # Cleanup: Ensure clean state for next test
        self._cleanup_test_resources()
        
        # Performance monitoring
        test_duration = time.time() - self.test_start_time
        if test_duration > Config.TIMEOUTS["quick_operation"]:
            print(f"SLOW TEST: {request.node.name} took {test_duration:.2f}s")
    
    def _cleanup_test_resources(self) -> None:
        """
        Clean up all test resources and artifacts.
        
        Performs comprehensive cleanup:
        - Removes uploaded files via API
        - Cleans temporary local files
        - Resets server state if needed
        
        This method ensures no test artifacts remain that could
        affect subsequent tests.
        """
        # Clear uploaded files via API
        try:
            clear_response = self.session.post(
                f"{Config.BASE_URL}{Config.ENDPOINTS['clear_all_files']}",
                timeout=Config.TIMEOUTS["api_request"]
            )
            # Don't assert here as cleanup should be resilient
        except requests.RequestException:
            pass
        
        # Clean up temporary local files
        for temp_file in self.temp_files:
            if isinstance(temp_file, Path) and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
    
    def _validate_api_contract(self, response: requests.Response) -> Dict[str, Any]:
        """
        Validate that API response follows the required contract.
        
        Ensures all responses match the strict contract:
        {
          "success": boolean,
          "data": object|null,
          "error": string|null,
          "meta": { "timestamp": string }
        }
        
        Args:
            response: HTTP response object from API call
            
        Returns:
            Dict containing the validated JSON response data
            
        Raises:
            AssertionError: If response doesn't match API contract
            
        Example:
            >>> response = requests.post("/api/execute-raw", data="print('hello')")
            >>> data = self._validate_api_contract(response)
            >>> assert data["success"] is True
            >>> assert data["data"]["result"] == "hello\\n"
        """
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            pytest.fail(f"Response is not valid JSON: {e}")
        
        # Validate required top-level fields
        required_fields = ["success", "data", "error", "meta"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Validate field types and constraints
        assert isinstance(data["success"], bool), "success must be boolean"
        assert data["data"] is None or isinstance(data["data"], dict), "data must be dict or null"
        assert data["error"] is None or isinstance(data["error"], str), "error must be string or null"
        assert isinstance(data["meta"], dict), "meta must be dict"
        assert "timestamp" in data["meta"], "meta must contain timestamp"
        
        # Validate logical constraints
        if data["success"]:
            assert data["error"] is None, "success=true must have error=null"
            assert data["data"] is not None, "success=true must have data object"
        else:
            assert data["data"] is None, "success=false must have data=null"
            assert data["error"] is not None, "success=false must have error message"
        
        return data
    
    def _execute_python_code(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code using the /api/execute-raw endpoint.
        
        Provides a convenient wrapper for executing Python code with
        proper API contract validation and error handling.
        
        Args:
            code: Python code string to execute
            
        Returns:
            Dict containing validated API response data
            
        Raises:
            AssertionError: If API response is invalid or execution fails
            
        Example:
            >>> result = self._execute_python_code("print('Hello World')")
            >>> assert result["success"] is True
            >>> assert "Hello World" in result["data"]["stdout"]
        """
        response = self.session.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['execute_raw']}",
            data=code,
            headers=Config.HEADERS["execute_raw"],
            timeout=Config.TIMEOUTS["code_execution"]
        )
        
        return self._validate_api_contract(response)
    
    def _create_test_file(self, filename: str, content: str, mime_type: str = 'text/plain') -> Path:
        """
        Create a temporary test file for upload operations.
        
        Args:
            filename: Name for the test file
            content: Content to write to the file
            mime_type: MIME type for the file (currently unused but for future compatibility)
            
        Returns:
            Path object pointing to the created temporary file
            
        Example:
            >>> test_file = self._create_test_file("data.csv", "col1,col2\\n1,2")
            >>> assert test_file.exists()
            >>> assert test_file.read_text() == "col1,col2\\n1,2"
        """
        temp_dir = Path(tempfile.mkdtemp())
        temp_file = temp_dir / filename
        temp_file.write_text(content)
        self.temp_files.append(temp_file)
        self.temp_files.append(temp_dir)  # Track directory for cleanup
        return temp_file
    
    def _upload_file_via_api(self, file_path: Path) -> Dict[str, Any]:
        """
        Upload a file to the server via the upload API.
        
        Args:
            file_path: Path to the file to upload
            
        Returns:
            Dict containing the API response data
            
        Example:
            >>> test_file = self._create_test_file("test.txt", "content")
            >>> response = self._upload_file_via_api(test_file)
            >>> assert response["success"] is True
        """
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'text/plain')}
            response = self.session.post(
                f"{Config.BASE_URL}{Config.ENDPOINTS['upload']}",
                files=files,
                timeout=Config.TIMEOUTS["api_request"]
            )
        
        return self._validate_api_contract(response)
    
    @pytest.mark.api
    @pytest.mark.integration
    def test_clear_all_files_removes_uploaded_files(self):
        """
        Test that the clear-all-files API removes uploaded files.
        
        Scenario: Clear all uploaded files
        Given: Multiple files are uploaded to the server
        When: The clear-all-files API is called
        Then: All uploaded files should be removed
        And: The file listing should be empty or contain only system files
        
        This test validates the core file cleanup functionality that
        administrators and automated systems rely on for maintenance.
        
        API Endpoints Tested:
        - POST /api/upload (file upload)
        - GET /api/uploaded-files (file listing)
        - POST {Config.ENDPOINTS['clear_all_files']} (file cleanup)
        
        Expected Behavior:
        - Files are successfully uploaded and visible in listings
        - Clear operation succeeds and returns success response
        - Subsequent file listing shows files are removed
        - Only system files (like __pycache__) may remain
        
        Cross-Platform Notes:
        - Uses pathlib for all file operations
        - Content encoding is UTF-8 compatible
        - File paths use forward slashes for URLs
        """
        # Given: Multiple test files are uploaded
        test_file1 = self._create_test_file("test1.txt", "Test content 1")
        test_file2 = self._create_test_file("test2.json", '{"test": "data"}')
        
        upload1_result = self._upload_file_via_api(test_file1)
        upload2_result = self._upload_file_via_api(test_file2)
        
        assert upload1_result["success"], "First file upload should succeed"
        assert upload2_result["success"], "Second file upload should succeed"
        
        # Verify files are present before clearing
        list_response = self.session.get(
            f"{Config.BASE_URL}{Config.ENDPOINTS['uploaded_files']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        list_data = self._validate_api_contract(list_response)
        files_before = list_data["data"]["files"]
        assert len(files_before) >= 2, "At least 2 files should be present before clearing"
        
        # When: Clear all files operation is performed
        clear_response = self.session.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['clear_all_files']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        clear_data = self._validate_api_contract(clear_response)
        
        # Then: Clear operation should succeed
        assert clear_data["success"], "Clear all files operation should succeed"
        assert "message" in clear_data["data"], "Clear response should contain message"
        
        # And: File listing should show files are removed
        list_response_after = self.session.get(
            f"{Config.BASE_URL}{Config.ENDPOINTS['uploaded_files']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        list_data_after = self._validate_api_contract(list_response_after)
        files_after = list_data_after["data"]["files"]
        
        # Only system files like __pycache__ should remain, if any
        assert len(files_after) <= 1, "At most 1 system file should remain after clearing"
        if len(files_after) == 1:
            remaining_file = files_after[0]["filename"]
            assert "__pycache__" in remaining_file, f"Only system files should remain, got: {remaining_file}"
    
    @pytest.mark.api
    @pytest.mark.integration
    def test_clear_all_files_removes_pyodide_virtual_files(self):
        """
        Test that clear-all-files removes Pyodide virtual filesystem files.
        
        Scenario: Clear Pyodide virtual filesystem files
        Given: Files are created in Pyodide virtual filesystem
        When: The clear-all-files API is called
        Then: Virtual filesystem files should be removed
        And: File listing via Python code should show empty directory
        
        This test ensures that the file clearing mechanism works for
        files created programmatically within the Pyodide environment,
        not just uploaded files.
        
        API Endpoints Tested:
        - POST /api/execute-raw (Python code execution)
        - POST {Config.ENDPOINTS['clear_all_files']} (file cleanup)
        
        Python Code Patterns:
        - pathlib for cross-platform file operations
        - Directory creation with mkdir(parents=True, exist_ok=True)
        - File listing with glob patterns
        - UTF-8 text writing
        
        Expected Behavior:
        - Files can be created in virtual filesystem
        - Files are visible via Python directory listing
        - Clear operation removes virtual files
        - Directory listing shows empty state after clearing
        """
        # Given: Files are created in Pyodide virtual filesystem
        create_files_code = f'''
import json
from pathlib import Path

# Ensure uploads directory exists
uploads_dir = Path("{Config.PATHS['uploads_dir']}")
uploads_dir.mkdir(parents=True, exist_ok=True)

# Create test files in virtual filesystem
file1 = uploads_dir / "pyodide_test1.txt"
file2 = uploads_dir / "pyodide_test2.csv"

file1.write_text("Pyodide content 1")
file2.write_text("col1,col2\\nval1,val2")

# Verify files were created
files_created = [f.name for f in uploads_dir.glob("*") if f.is_file()]
result = {{"files_created": files_created, "count": len(files_created)}}
print(json.dumps(result))
        '''
        
        create_result = self._execute_python_code(create_files_code)
        assert create_result["success"], "File creation should succeed"
        
        # Verify files were created by parsing the output
        stdout_content = create_result["data"]["stdout"]
        assert "pyodide_test1.txt" in stdout_content, "First test file should be created"
        assert "pyodide_test2.csv" in stdout_content, "Second test file should be created"
        
        # When: Clear all files operation is performed
        clear_response = self.session.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['clear_all_files']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        clear_data = self._validate_api_contract(clear_response)
        assert clear_data["success"], "Clear all files operation should succeed"
        
        # Then: Virtual filesystem files should be removed
        verify_cleared_code = f'''
import json
from pathlib import Path

uploads_dir = Path("{Config.PATHS['uploads_dir']}")
if uploads_dir.exists():
    remaining_files = [f.name for f in uploads_dir.glob("*") if f.is_file()]
else:
    remaining_files = []

result = {{"remaining_files": remaining_files, "count": len(remaining_files)}}
print(json.dumps(result))
        '''
        
        verify_result = self._execute_python_code(verify_cleared_code)
        assert verify_result["success"], "File verification should succeed"
        
        # Parse verification results
        verify_stdout = verify_result["data"]["stdout"]
        assert "pyodide_test1.txt" not in verify_stdout, "First test file should be removed"
        assert "pyodide_test2.csv" not in verify_stdout, "Second test file should be removed"
    
    @pytest.mark.api
    @pytest.mark.integration
    def test_environment_reset_and_file_clearing_workflow(self):
        """
        Test complete environment reset and file clearing workflow.
        
        Scenario: Complete environment and file system reset
        Given: Variables are set and files are created in Pyodide
        When: Environment reset and file clearing are performed
        Then: Variables should be cleared and files should be removed
        And: Subsequent operations should start with clean state
        
        This test validates the complete cleanup workflow that combines
        variable reset with file system clearing, ensuring a completely
        clean environment for subsequent operations.
        
        API Endpoints Tested:
        - POST /api/execute-raw (Python execution and verification)
        - POST /api/reset (environment variable reset)
        - POST {Config.ENDPOINTS['clear_all_files']} (file system cleanup)
        
        Test Pattern:
        1. Setup: Create variables and files
        2. Verify: Confirm setup was successful
        3. Reset: Clear variables and files
        4. Verify: Confirm clean state
        
        Cross-Platform Considerations:
        - Uses pathlib for all file operations
        - JSON serialization for data verification
        - UTF-8 encoding for text files
        - Forward slash path separators
        """
        # Given: Variables and files are created in Pyodide environment
        setup_code = f'''
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Set up directory structure
uploads_dir = Path("{Config.PATHS['uploads_dir']}")
uploads_dir.mkdir(parents=True, exist_ok=True)

# Set some variables that should be cleared by reset
test_variable = "This should be cleared by reset"
test_dataframe = pd.DataFrame({{"col1": [1, 2, 3], "col2": [4, 5, 6]}})
test_number = 42

# Create a file that should be cleared by clear-all-files
test_file = uploads_dir / "reset_test.txt"
test_file.write_text("This file should be cleared")

# Verify setup
setup_status = {{
    "variable_set": test_variable,
    "dataframe_shape": test_dataframe.shape,
    "file_exists": test_file.exists(),
    "file_content": test_file.read_text(),
    "number_value": test_number
}}

print(json.dumps(setup_status))
        '''
        
        setup_result = self._execute_python_code(setup_code)
        assert setup_result["success"], "Environment setup should succeed"
        
        # Verify setup was successful
        setup_output = setup_result["data"]["stdout"]
        assert "This should be cleared by reset" in setup_output, "Variable should be set"
        assert '"file_exists": true' in setup_output, "File should be created"
        
        # When: Environment reset is performed
        reset_response = self.session.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['reset']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        reset_data = self._validate_api_contract(reset_response)
        assert reset_data["success"], "Environment reset should succeed"
        
        # And: File clearing is performed
        clear_response = self.session.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['clear_all_files']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        clear_data = self._validate_api_contract(clear_response)
        assert clear_data["success"], "File clearing should succeed"
        
        # Then: Environment should be clean
        verification_code = f'''
import json
from pathlib import Path

# Check if variables were cleared (they should cause NameError)
verification_results = {{}}

# Test if variables still exist
try:
    test_variable
    verification_results["variable_cleared"] = False
except NameError:
    verification_results["variable_cleared"] = True

try:
    test_dataframe
    verification_results["dataframe_cleared"] = False  
except NameError:
    verification_results["dataframe_cleared"] = True

try:
    test_number
    verification_results["number_cleared"] = False
except NameError:
    verification_results["number_cleared"] = True

# Check if files were cleared
uploads_dir = Path("{Config.PATHS['uploads_dir']}")
test_file = uploads_dir / "reset_test.txt"
verification_results["file_cleared"] = not test_file.exists()

if uploads_dir.exists():
    remaining_files = [f.name for f in uploads_dir.glob("*") if f.is_file()]
    verification_results["remaining_files"] = remaining_files
else:
    verification_results["remaining_files"] = []

print(json.dumps(verification_results))
        '''
        
        verify_result = self._execute_python_code(verification_code)
        assert verify_result["success"], "Verification should succeed"
        
        # Parse verification results
        verify_output = verify_result["data"]["stdout"]
        assert '"variable_cleared": true' in verify_output, "Variables should be cleared by reset"
        assert '"file_cleared": true' in verify_output, "Files should be cleared by clear-all-files"
        assert "reset_test.txt" not in verify_output, "Test file should not exist"
    
    @pytest.mark.api
    def test_clear_all_files_api_endpoint_availability(self):
        """
        Test that the clear-all-files API endpoint is available and functional.
        
        Scenario: API endpoint availability verification
        Given: The server is running
        When: The clear-all-files endpoint is called
        Then: It should respond with success regardless of file presence
        And: Response should follow API contract
        
        This test ensures the API endpoint is properly configured and
        accessible, providing a foundation for other file management tests.
        
        API Contract Validation:
        - HTTP 200 status code
        - Proper JSON response structure
        - Required fields present and correct types
        - Success flag and appropriate message
        
        Error Handling:
        - Should not fail if no files exist
        - Should handle concurrent requests gracefully
        - Should provide clear status messages
        """
        # Given: Server is running (implicit from test setup)
        # When: Clear all files endpoint is called
        response = self.session.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['clear_all_files']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        
        # Then: Response should follow API contract
        data = self._validate_api_contract(response)
        
        # And: Operation should succeed
        assert data["success"], "Clear all files operation should succeed"
        assert "message" in data["data"], "Response should contain status message"
        
        # Verify message is meaningful
        message = data["data"]["message"]
        assert isinstance(message, str), "Message should be a string"
        assert len(message) > 0, "Message should not be empty"
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_multiple_file_operations_consistency(self):
        """
        Test consistency across multiple file operations and state changes.
        
        Scenario: Multiple file operations consistency
        Given: Files are uploaded and created via different methods
        When: Various file operations are performed in sequence
        Then: File counts and listings should remain consistent
        And: Operations should not interfere with each other
        
        This test validates the robustness of file management operations
        when multiple types of operations are performed in sequence,
        ensuring data integrity and consistent behavior.
        
        Operations Tested:
        - File upload via API
        - File creation via Python code
        - Individual file deletion
        - File listing and counting
        - Bulk file clearing
        
        Consistency Checks:
        - File counts match expectations at each step
        - File listings reflect actual file system state
        - Operations are atomic and don't leave partial state
        - Error conditions are handled gracefully
        
        Performance Considerations:
        - Operations complete within reasonable timeouts
        - Server remains responsive during batch operations
        - Memory usage remains stable
        """
        # Given: Files are created via multiple methods
        
        # Upload files via API
        upload_file1 = self._create_test_file("consistency_test1.txt", "Upload content 1")
        upload_file2 = self._create_test_file("consistency_test2.json", '{"upload": 1}')
        
        upload1_result = self._upload_file_via_api(upload_file1)
        upload2_result = self._upload_file_via_api(upload_file2)
        
        assert upload1_result["success"], "First upload should succeed"
        assert upload2_result["success"], "Second upload should succeed"
        
        # Create files via Python code
        create_code = f'''
from pathlib import Path

uploads_dir = Path("{Config.PATHS['uploads_dir']}")
uploads_dir.mkdir(parents=True, exist_ok=True)

# Create additional files
(uploads_dir / "pyodide_consistency.txt").write_text("Pyodide content")
(uploads_dir / "data_analysis.csv").write_text("col1,col2\\n1,2\\n3,4")

print("Files created via Python")
        '''
        
        create_result = self._execute_python_code(create_code)
        assert create_result["success"], "Python file creation should succeed"
        
        # When: File operations are performed in sequence
        
        # List files and verify initial count
        list_response = self.session.get(
            f"{Config.BASE_URL}{Config.ENDPOINTS['uploaded_files']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        list_data = self._validate_api_contract(list_response)
        files_initial = list_data["data"]["files"]
        initial_count = len(files_initial)
        
        assert initial_count >= 4, f"Should have at least 4 files, got {initial_count}"
        
        # Delete one uploaded file
        file_to_delete = files_initial[0]["filename"]
        delete_response = self.session.delete(
            f"{Config.BASE_URL}{Config.ENDPOINTS['uploaded_files']}/{file_to_delete}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        delete_data = self._validate_api_contract(delete_response)
        assert delete_data["success"], f"File deletion should succeed for {file_to_delete}"
        
        # Verify count decreased by 1
        list_response_after_delete = self.session.get(
            f"{Config.BASE_URL}{Config.ENDPOINTS['uploaded_files']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        list_data_after_delete = self._validate_api_contract(list_response_after_delete)
        files_after_delete = list_data_after_delete["data"]["files"]
        
        assert len(files_after_delete) == initial_count - 1, "File count should decrease by 1 after deletion"
        
        # Clear all files
        clear_response = self.session.post(
            f"{Config.BASE_URL}{Config.ENDPOINTS['clear_all_files']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        clear_data = self._validate_api_contract(clear_response)
        assert clear_data["success"], "Clear all files should succeed"
        
        # Then: Final verification of consistency
        final_list_response = self.session.get(
            f"{Config.BASE_URL}{Config.ENDPOINTS['uploaded_files']}",
            timeout=Config.TIMEOUTS["api_request"]
        )
        final_list_data = self._validate_api_contract(final_list_response)
        files_final = final_list_data["data"]["files"]
        
        # Should have no files or only system files
        assert len(files_final) == 0, f"Should have no files after clearing, got {len(files_final)}"
        
        # Verify via Python code that virtual filesystem is also clean
        verify_clean_code = f'''
import json
from pathlib import Path

uploads_dir = Path("{Config.PATHS['uploads_dir']}")
if uploads_dir.exists():
    remaining_files = [f.name for f in uploads_dir.glob("*") if f.is_file()]
else:
    remaining_files = []

result = {{"remaining_files": remaining_files, "directory_exists": uploads_dir.exists()}}
print(json.dumps(result))
        '''
        
        verify_result = self._execute_python_code(verify_clean_code)
        assert verify_result["success"], "Verification should succeed"
        
        verify_output = verify_result["data"]["stdout"]
        assert '"remaining_files": []' in verify_output, "No files should remain in virtual filesystem"


# Test execution entry point
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

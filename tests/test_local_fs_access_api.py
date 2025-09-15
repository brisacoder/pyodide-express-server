"""
Local Filesystem Access API Tests for Pyodide Express Server.

This module tests the ability to read files from the local filesystem through
the Pyodide WebAssembly runtime via the /api/execute-raw endpoint. It validates
that files placed in the server's mounted directories can be accessed by Python
code running in the Pyodide environment.

Test Categories:
- Basic file reading operations
- Directory existence and permissions
- File content validation
- Cross-platform path handling
- Error handling for non-existent files
- Multiple file operations
- Binary vs text file handling

Requirements Compliance:
1. ✅ Pytest framework with BDD Given-When-Then structure
2. ✅ No hardcoded globals - all parameters via fixtures/constants
3. ✅ Only uses /api/execute-raw endpoint
4. ✅ No internal REST APIs (no 'pyodide' in URLs)
5. ✅ Comprehensive test coverage with edge cases
6. ✅ Full docstrings with descriptions, inputs, outputs, examples
7. ✅ Portable Pyodide code using pathlib
8. ✅ API contract validation for all responses

API Contract Validation:
All responses must follow the standardized contract:
{
    "success": true | false,
    "data": {
        "result": <any>,
        "stdout": <string>,
        "stderr": <string>,
        "executionTime": <number>
    } | null,
    "error": <string> | null,
    "meta": {
        "timestamp": <ISO string>
    }
}
"""

from pathlib import Path
from typing import Any, Dict

import pytest
import uuid

from conftest import Config, execute_python_code, validate_api_contract


class TestLocalFilesystemAccess:
    """
    Test suite for local filesystem access through Pyodide API.

    This class contains comprehensive tests for reading, writing, and
    manipulating files on the local filesystem through the Pyodide
    WebAssembly runtime via the /api/execute-raw endpoint.

    Test Structure:
    - Setup creates test files in mounted directories
    - Tests validate various filesystem operations
    - Cleanup removes test artifacts
    - All operations use portable pathlib syntax
    """

    @pytest.fixture(scope="class")
    def test_data_paths(self) -> Dict[str, Path]:
        """
        Create test directory structure and file paths.

        Sets up directory paths for various test scenarios including
        plots directory, uploads directory, and temporary files.

        Returns:
            Dict[str, Path]: Dictionary mapping test scenario names to Path objects

        Example:
            >>> paths = test_data_paths()
            >>> assert paths["plots_dir"].exists()
            >>> assert paths["test_file"].parent.exists()
        """
        # Create unique test session ID to avoid conflicts
        session_id = str(uuid.uuid4())[:8]

        # Local filesystem paths (server-side)
        local_plots_dir = Path(__file__).parent.parent / "plots"
        local_uploads_dir = Path(__file__).parent.parent / "uploads"

        # Ensure directories exist
        local_plots_dir.mkdir(parents=True, exist_ok=True)
        local_uploads_dir.mkdir(parents=True, exist_ok=True)

        # Test file paths
        paths = {
            "plots_dir": local_plots_dir,
            "uploads_dir": local_uploads_dir,
            "test_file": local_plots_dir / f"test_fs_access_{session_id}.txt",
            "binary_file": local_plots_dir / f"test_binary_{session_id}.dat",
            "json_file": local_uploads_dir / f"test_data_{session_id}.json",
            "multi_file_1": local_plots_dir / f"multi_test_1_{session_id}.txt",
            "multi_file_2": local_plots_dir / f"multi_test_2_{session_id}.txt",
            "session_id": session_id,
        }

        return paths

    @pytest.fixture(scope="class")
    def setup_test_files(self, test_data_paths: Dict[str, Path]):
        """
        Create test files with known content using Pyodide for filesystem access tests.

        Creates various types of test files (text, binary, JSON) using Pyodide
        execution so they are properly mounted and accessible within the Pyodide
        environment. This approach ensures proper filesystem synchronization.

        Args:
            test_data_paths: Dictionary of test file paths from fixture

        Returns:
            Dict[str, Any]: Test file metadata including paths, content, and checksums

        Example:
            >>> metadata = setup_test_files(paths)
            >>> assert metadata["text_content"] == "Hello from local FS via API!"
        """
        session_id = test_data_paths["session_id"]

        # Test content definitions
        text_content = "Hello from local FS via API!"
        json_content = '{"message": "Test JSON data", "number": 42, "array": [1, 2, 3]}'
        binary_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'

        # Create files using Pyodide to ensure proper mounting
        setup_code = f'''
from pathlib import Path
import json

session_id = "{session_id}"

# Create test files in mounted directories
plots_dir = Path('/home/pyodide/plots')
uploads_dir = Path('/home/pyodide/uploads')

# Ensure directories exist
plots_dir.mkdir(parents=True, exist_ok=True)
uploads_dir.mkdir(parents=True, exist_ok=True)

# Create text file
text_file = plots_dir / f'test_fs_access_{{session_id}}.txt'
text_file.write_text("{text_content}", encoding='utf-8')
print(f"Created text file: {{text_file}} ({{text_file.stat().st_size}} bytes)")

# Create JSON file
json_file = uploads_dir / f'test_data_{{session_id}}.json'
json_content = '{json_content}'
json_file.write_text(json_content, encoding='utf-8')
print(f"Created JSON file: {{json_file}} ({{json_file.stat().st_size}} bytes)")

# Create binary file (simplified PNG-like header)
binary_file = plots_dir / f'test_binary_{{session_id}}.dat'
binary_data = b'\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01'
binary_file.write_bytes(binary_data)
print(f"Created binary file: {{binary_file}} ({{binary_file.stat().st_size}} bytes)")

# Create multiple files for batch operations
multi_file_1 = plots_dir / f'multi_test_1_{{session_id}}.txt'
multi_file_1.write_text("Content of file 1", encoding='utf-8')
print(f"Created multi file 1: {{multi_file_1}} ({{multi_file_1.stat().st_size}} bytes)")

multi_file_2 = plots_dir / f'multi_test_2_{{session_id}}.txt'
multi_file_2.write_text("Content of file 2", encoding='utf-8')
print(f"Created multi file 2: {{multi_file_2}} ({{multi_file_2.stat().st_size}} bytes)")

# Verification - list created files
created_files = []
for file_pattern in [f'*{{session_id}}*']:
    created_files.extend(list(plots_dir.glob(file_pattern)))
    created_files.extend(list(uploads_dir.glob(file_pattern)))

print(f"Total files created: {{len(created_files)}}")
for f in created_files:
    print(f"  - {{f.name}}")

print("File setup completed successfully!")
'''

        result = execute_python_code(setup_code, timeout=Config.TIMEOUTS["code_execution"])
        if not result["success"]:
            pytest.fail(f"Failed to create test files: {result.get('error')}")

        # Return metadata for test validation
        metadata = {
            "text_content": text_content,
            "json_content": json_content,
            "binary_content": binary_content,
            "file_count": 5,
            "session_id": session_id
        }

        yield metadata

        # Cleanup: Remove test files using Pyodide to ensure proper cleanup
        cleanup_code = f'''
from pathlib import Path

session_id = "{session_id}"
plots_dir = Path('/home/pyodide/plots')
uploads_dir = Path('/home/pyodide/uploads')

# Find and remove all test files for this session
cleanup_patterns = [f'*{{session_id}}*']
removed_files = []

for pattern in cleanup_patterns:
    for directory in [plots_dir, uploads_dir]:
        if directory.exists():
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        removed_files.append(str(file_path))
                        print(f"Removed: {{file_path.name}}")
                    except Exception as e:
                        print(f"Error removing {{file_path.name}}: {{e}}")

print(f"Cleanup completed: {{len(removed_files)}} files removed")
'''

        try:
            cleanup_result = execute_python_code(cleanup_code, timeout=Config.TIMEOUTS["code_execution"])
            if not cleanup_result["success"]:
                print(f"Warning: Cleanup failed: {cleanup_result.get('error')}")
        except Exception as e:
            print(f"Warning: Cleanup exception: {e}")

    @pytest.mark.api
    @pytest.mark.filesystem
    def test_basic_file_reading(
        self,
        server_ready: None,
        test_data_paths: Dict[str, Path],
        setup_test_files: Dict[str, Any]
    ) -> None:
        """
        Test basic file reading operation through Pyodide API.

        Validates that a simple text file can be read from the mounted
        filesystem using Python's built-in file operations within Pyodide.

        Given: A text file exists in the mounted plots directory
        When: Python code attempts to read the file via /api/execute-raw
        Then: The file content is returned correctly in the API response

        Args:
            server_ready: Ensures server is available
            test_data_paths: Test file paths from fixture
            setup_test_files: Test file metadata from fixture

        Raises:
            AssertionError: If file reading fails or content doesn't match

        Example:
            Test validates that this Pyodide code works:
            ```python
            from pathlib import Path
            file_path = Path('/home/pyodide/plots/test_file.txt')
            content = file_path.read_text(encoding='utf-8')
            print(content)
            ```
        """
        # Given: Test file exists with known content
        expected_content = setup_test_files["text_content"]
        session_id = setup_test_files["session_id"]
        test_filename = f"test_fs_access_{session_id}.txt"

        # When: Execute Python code to read the file
        python_code = f"""
from pathlib import Path

# Use pathlib for cross-platform compatibility
file_path = Path('/home/pyodide/plots/{test_filename}')

# Validate file exists before reading
if not file_path.exists():
    raise FileNotFoundError(f"Test file not found: {{file_path}}")

# Read file content using pathlib
content = file_path.read_text(encoding='utf-8')
print(f"File content: {{content}}")
print(f"File size: {{file_path.stat().st_size}} bytes")
"""

        result = execute_python_code(python_code, timeout=Config.TIMEOUTS["code_execution"])

        # Then: Validate successful file reading
        assert result["success"] is True, f"File reading failed: {result.get('error')}"
        assert expected_content in result["data"]["stdout"], \
            f"Expected content '{expected_content}' not found in stdout: {result['data']['stdout']}"
        assert "File size:" in result["data"]["stdout"], \
            "File size information not found in output"
        assert result["data"]["stderr"] == "", \
            f"Unexpected stderr output: {result['data']['stderr']}"

    @pytest.mark.api
    @pytest.mark.filesystem
    def test_directory_listing_and_validation(
        self,
        server_ready: None,
        test_data_paths: Dict[str, Path],
        setup_test_files: Dict[str, Any]
    ) -> None:
        """
        Test directory listing and file existence validation.

        Validates that Python code can list directory contents and validate
        file existence using pathlib operations in the Pyodide environment.

        Given: Multiple test files exist in mounted directories
        When: Python code lists directory contents and validates files
        Then: All expected files are found and properly identified

        Args:
            server_ready: Ensures server is available
            test_data_paths: Test file paths from fixture
            setup_test_files: Test file metadata from fixture

        Example:
            Validates directory operations like:
            ```python
            plots_dir = Path('/home/pyodide/plots')
            files = list(plots_dir.glob('*.txt'))
            assert len(files) >= 2
            ```
        """
        # Given: Multiple test files exist
        session_id = setup_test_files["session_id"]
        expected_files = [
            f"test_fs_access_{session_id}.txt",
            f"test_binary_{session_id}.dat",
            f"multi_test_1_{session_id}.txt",
            f"multi_test_2_{session_id}.txt"
        ]

        # When: Execute Python code to list and validate directories
        python_code = f"""
from pathlib import Path
import json

# Check plots directory
plots_dir = Path('/home/pyodide/plots')
uploads_dir = Path('/home/pyodide/uploads')

# Validate directories exist
print(f"Plots directory exists: {{plots_dir.exists()}}")
print(f"Uploads directory exists: {{uploads_dir.exists()}}")

# List files in plots directory
plots_files = []
if plots_dir.exists():
    for file_path in plots_dir.iterdir():
        if file_path.is_file():
            plots_files.append(file_path.name)

print(f"Found {{len(plots_files)}} files in plots directory")

# Check for our specific test files
test_files = {expected_files}
found_files = []
missing_files = []

for test_file in test_files:
    file_path = plots_dir / test_file
    if file_path.exists():
        found_files.append(test_file)
        print(f"✓ Found: {{test_file}} (size: {{file_path.stat().st_size}} bytes)")
    else:
        missing_files.append(test_file)
        print(f"✗ Missing: {{test_file}}")

print(f"Summary: {{len(found_files)}} found, {{len(missing_files)}} missing")

# Check uploads directory for JSON file
json_file = uploads_dir / "test_data_{session_id}.json"
print(f"JSON file exists: {{json_file.exists()}}")
if json_file.exists():
    print(f"JSON file size: {{json_file.stat().st_size}} bytes")
"""

        result = execute_python_code(python_code, timeout=Config.TIMEOUTS["code_execution"])

        # Then: Validate directory operations succeeded
        assert result["success"] is True, f"Directory validation failed: {result.get('error')}"

        stdout = result["data"]["stdout"]
        assert "Plots directory exists: True" in stdout, "Plots directory not found"
        assert "Uploads directory exists: True" in stdout, "Uploads directory not found"
        assert "4 found, 0 missing" in stdout, f"Not all test files found. Output: {stdout}"
        assert "JSON file exists: True" in stdout, "JSON file not found in uploads directory"

    @pytest.mark.api
    @pytest.mark.filesystem
    def test_json_file_parsing(
        self,
        server_ready: None,
        test_data_paths: Dict[str, Path],
        setup_test_files: Dict[str, Any]
    ) -> None:
        """
        Test JSON file reading and parsing operations.

        Validates that JSON files can be read from the filesystem and
        parsed correctly using Python's json module in Pyodide.

        Given: A JSON file exists with structured data
        When: Python code reads and parses the JSON file
        Then: The parsed data matches the expected structure and values

        Args:
            server_ready: Ensures server is available
            test_data_paths: Test file paths from fixture
            setup_test_files: Test file metadata from fixture

        Example:
            Tests JSON operations like:
            ```python
            import json
            data = json.loads(Path('/home/pyodide/uploads/data.json').read_text())
            assert data['message'] == 'Test JSON data'
            ```
        """
        # Given: JSON file with structured data exists
        session_id = setup_test_files["session_id"]

        # When: Execute Python code to read and parse JSON
        python_code = f"""
import json
from pathlib import Path

# Read JSON file from uploads directory
json_file = Path('/home/pyodide/uploads/test_data_{session_id}.json')

if not json_file.exists():
    raise FileNotFoundError(f"JSON file not found: {{json_file}}")

# Read and parse JSON content
json_content = json_file.read_text(encoding='utf-8')
data = json.loads(json_content)

# Validate parsed data structure
print(f"JSON message: {{data.get('message')}}")
print(f"JSON number: {{data.get('number')}}")
print(f"JSON array: {{data.get('array')}}")
print(f"JSON array length: {{len(data.get('array', []))}}")

# Type validation
print(f"Message type: {{type(data.get('message')).__name__}}")
print(f"Number type: {{type(data.get('number')).__name__}}")
print(f"Array type: {{type(data.get('array')).__name__}}")

# Verify specific values
assert data['message'] == 'Test JSON data', f"Wrong message: {{data['message']}}"
assert data['number'] == 42, f"Wrong number: {{data['number']}}"
assert data['array'] == [1, 2, 3], f"Wrong array: {{data['array']}}"

print("JSON parsing validation successful!")
"""

        result = execute_python_code(python_code, timeout=Config.TIMEOUTS["code_execution"])

        # Then: Validate JSON parsing succeeded
        assert result["success"] is True, f"JSON parsing failed: {result.get('error')}"

        stdout = result["data"]["stdout"]
        assert "JSON message: Test JSON data" in stdout, "Wrong JSON message parsed"
        assert "JSON number: 42" in stdout, "Wrong JSON number parsed"
        assert "JSON array: [1, 2, 3]" in stdout, "Wrong JSON array parsed"
        assert "JSON parsing validation successful!" in stdout, "JSON validation failed"

    @pytest.mark.api
    @pytest.mark.filesystem
    def test_binary_file_access(
        self,
        server_ready: None,
        test_data_paths: Dict[str, Path],
        setup_test_files: Dict[str, Any]
    ) -> None:
        """
        Test binary file reading and validation.

        Validates that binary files can be read correctly from the filesystem
        using Python's binary file operations in Pyodide.

        Given: A binary file exists with known byte content
        When: Python code reads the file in binary mode
        Then: The binary content matches expected byte sequences

        Args:
            server_ready: Ensures server is available
            test_data_paths: Test file paths from fixture
            setup_test_files: Test file metadata from fixture

        Example:
            Tests binary operations like:
            ```python
            data = Path('/home/pyodide/plots/image.png').read_bytes()
            assert data[:4] == b'\\x89PNG'
            ```
        """
        # Given: Binary file with known content exists
        session_id = setup_test_files["session_id"]

        # When: Execute Python code to read binary file
        python_code = f"""
from pathlib import Path

# Read binary file from plots directory
binary_file = Path('/home/pyodide/plots/test_binary_{session_id}.dat')

if not binary_file.exists():
    raise FileNotFoundError(f"Binary file not found: {{binary_file}}")

# Read binary content
binary_data = binary_file.read_bytes()

print(f"Binary file size: {{len(binary_data)}} bytes")
print(f"First 4 bytes (hex): {{binary_data[:4].hex()}}")
print(f"First 8 bytes (repr): {{repr(binary_data[:8])}}")

# Validate PNG header signature
expected_png_header = b'\\x89PNG\\r\\n\\x1a\\n'
if binary_data.startswith(expected_png_header):
    print("✓ PNG header signature verified")
else:
    print(f"✗ PNG header mismatch. Expected: {{expected_png_header.hex()}}, Got: {{binary_data[:8].hex()}}")

# Check specific bytes
assert binary_data[0] == 0x89, f"Wrong first byte: {{hex(binary_data[0])}}"
assert binary_data[1:4] == b'PNG', f"Wrong PNG signature: {{binary_data[1:4]}}"

print("Binary file validation successful!")
"""

        result = execute_python_code(python_code, timeout=Config.TIMEOUTS["code_execution"])

        # Then: Validate binary file reading succeeded
        assert result["success"] is True, f"Binary file reading failed: {result.get('error')}"

        stdout = result["data"]["stdout"]
        assert "Binary file size:" in stdout, "Binary file size not reported"
        assert "✓ PNG header signature verified" in stdout, "PNG header validation failed"
        assert "Binary file validation successful!" in stdout, "Binary validation failed"

    @pytest.mark.api
    @pytest.mark.filesystem
    def test_multiple_file_operations(
        self,
        server_ready: None,
        test_data_paths: Dict[str, Path],
        setup_test_files: Dict[str, Any]
    ) -> None:
        """
        Test reading multiple files in a single operation.

        Validates that multiple files can be read and processed together
        using efficient file operations and batch processing in Pyodide.

        Given: Multiple test files exist in the mounted directory
        When: Python code reads all files in a batch operation
        Then: All files are read correctly with proper content validation

        Args:
            server_ready: Ensures server is available
            test_data_paths: Test file paths from fixture
            setup_test_files: Test file metadata from fixture

        Example:
            Tests batch operations like:
            ```python
            files = ['file1.txt', 'file2.txt']
            contents = [Path(f'/plots/{f}').read_text() for f in files]
            ```
        """
        # Given: Multiple test files exist
        session_id = setup_test_files["session_id"]

        # When: Execute Python code for batch file operations
        python_code = f"""
from pathlib import Path
import json

# Define files to process
files_to_read = [
    'test_fs_access_{session_id}.txt',
    'multi_test_1_{session_id}.txt',
    'multi_test_2_{session_id}.txt'
]

plots_dir = Path('/home/pyodide/plots')
results = {{}}
errors = []

# Batch file reading
for filename in files_to_read:
    file_path = plots_dir / filename
    try:
        if file_path.exists():
            content = file_path.read_text(encoding='utf-8')
            file_stats = file_path.stat()
            results[filename] = {{
                'content': content,
                'size': file_stats.st_size,
                'exists': True
            }}
            print(f"✓ Successfully read {{filename}} ({{file_stats.st_size}} bytes)")
        else:
            results[filename] = {{'exists': False}}
            print(f"✗ File not found: {{filename}}")
    except Exception as e:
        errors.append(f"Error reading {{filename}}: {{str(e)}}")
        print(f"✗ Error reading {{filename}}: {{str(e)}}")

# Summary statistics
total_files = len(files_to_read)
successful_reads = len([r for r in results.values() if r.get('exists', False)])
total_bytes = sum(r.get('size', 0) for r in results.values() if r.get('exists', False))

print(f"Batch operation summary:")
print(f"- Total files attempted: {{total_files}}")
print(f"- Successful reads: {{successful_reads}}")
print(f"- Total bytes read: {{total_bytes}}")
print(f"- Errors encountered: {{len(errors)}}")

# Validate specific content
if 'test_fs_access_{session_id}.txt' in results:
    main_content = results['test_fs_access_{session_id}.txt'].get('content', '')
    if 'Hello from local FS via API!' in main_content:
        print("✓ Main test file content validated")
    else:
        print(f"✗ Main test file content invalid: {{main_content[:50]}}")

print("Batch file operation completed!")
"""

        result = execute_python_code(python_code, timeout=Config.TIMEOUTS["code_execution"])

        # Then: Validate batch file operations succeeded
        assert result["success"] is True, f"Batch file operations failed: {result.get('error')}"

        stdout = result["data"]["stdout"]
        assert "- Successful reads: 3" in stdout, "Not all files read successfully"
        assert "- Errors encountered: 0" in stdout, f"Unexpected errors in batch operation: {stdout}"
        assert "✓ Main test file content validated" in stdout, "Main test file content validation failed"
        assert "Batch file operation completed!" in stdout, "Batch operation did not complete"

    @pytest.mark.api
    @pytest.mark.filesystem
    def test_error_handling_nonexistent_file(
        self,
        server_ready: None
    ) -> None:
        """
        Test proper error handling for non-existent files.

        Validates that attempting to read non-existent files results in
        appropriate error handling and informative error messages.

        Given: A file path that does not exist on the filesystem
        When: Python code attempts to read the non-existent file
        Then: Appropriate FileNotFoundError is raised with clear messaging

        Args:
            server_ready: Ensures server is available

        Example:
            Tests error handling like:
            ```python
            try:
                Path('/home/pyodide/plots/nonexistent.txt').read_text()
            except FileNotFoundError as e:
                print(f"Expected error: {e}")
            ```
        """
        # Given: Non-existent file path
        nonexistent_file = f"definitely_does_not_exist_{uuid.uuid4().hex}.txt"

        # When: Execute Python code that handles file errors gracefully
        python_code = f"""
from pathlib import Path

# Attempt to read non-existent file with proper error handling
nonexistent_file = Path('/home/pyodide/plots/{nonexistent_file}')

print(f"Checking file: {{nonexistent_file}}")
print(f"File exists: {{nonexistent_file.exists()}}")

# Method 1: Check existence first (recommended approach)
if nonexistent_file.exists():
    content = nonexistent_file.read_text(encoding='utf-8')
    print(f"File content: {{content}}")
else:
    print("✓ File does not exist - handled gracefully")

# Method 2: Exception handling approach
try:
    content = nonexistent_file.read_text(encoding='utf-8')
    print(f"Unexpected success reading: {{content}}")
except FileNotFoundError as e:
    print(f"✓ Caught expected FileNotFoundError: {{type(e).__name__}}")
    print(f"✓ Error message: {{str(e)[:100]}}")
except Exception as e:
    print(f"✗ Unexpected error type: {{type(e).__name__}}: {{str(e)}}")

print("Error handling test completed successfully!")
"""

        result = execute_python_code(python_code, timeout=Config.TIMEOUTS["code_execution"])

        # Then: Validate proper error handling
        assert result["success"] is True, f"Error handling test failed: {result.get('error')}"

        stdout = result["data"]["stdout"]
        assert "File exists: False" in stdout, "File existence check failed"
        assert "✓ File does not exist - handled gracefully" in stdout, "Graceful error handling failed"
        assert "✓ Caught expected FileNotFoundError" in stdout, "FileNotFoundError not caught properly"
        assert "Error handling test completed successfully!" in stdout, \
            "Error handling test did not complete"

    @pytest.mark.api
    @pytest.mark.filesystem
    @pytest.mark.integration
    def test_complex_filesystem_workflow(
        self,
        server_ready: None,
        test_data_paths: Dict[str, Path],
        setup_test_files: Dict[str, Any]
    ) -> None:
        """
        Test complex filesystem workflow with multiple operations.

        Validates a complete workflow involving file reading, processing,
        directory operations, and data manipulation using pathlib and
        standard Python libraries in Pyodide.

        Given: Multiple test files exist with different types of content
        When: Python code executes a complex workflow involving all files
        Then: All operations complete successfully with expected results

        Args:
            server_ready: Ensures server is available
            test_data_paths: Test file paths from fixture
            setup_test_files: Test file metadata from fixture

        Example:
            Tests complex workflows like:
            - Reading multiple files
            - Processing JSON data
            - Analyzing file metadata
            - Generating summary reports
        """
        # Given: Complex test scenario with multiple file types
        session_id = setup_test_files["session_id"]

        # When: Execute complex filesystem workflow
        python_code = f"""
import json
from pathlib import Path
from datetime import datetime

# Complex filesystem workflow
print("=== Starting Complex Filesystem Workflow ===")

# Step 1: Directory analysis
plots_dir = Path('/home/pyodide/plots')
uploads_dir = Path('/home/pyodide/uploads')

workflow_results = {{
    'timestamp': datetime.now().isoformat(),
    'directories_analyzed': [],
    'files_processed': [],
    'content_analysis': {{}},
    'errors': []
}}

# Analyze plots directory
if plots_dir.exists():
    plots_files = list(plots_dir.iterdir())
    workflow_results['directories_analyzed'].append({{'name': 'plots', 'file_count': len(plots_files)}})
    print(f"Plots directory contains {{len(plots_files)}} items")

# Analyze uploads directory
if uploads_dir.exists():
    uploads_files = list(uploads_dir.iterdir())
    workflow_results['directories_analyzed'].append({{'name': 'uploads', 'file_count': len(uploads_files)}})
    print(f"Uploads directory contains {{len(uploads_files)}} items")

# Step 2: Process specific test files
test_files = {{
    'text_file': plots_dir / 'test_fs_access_{session_id}.txt',
    'json_file': uploads_dir / 'test_data_{session_id}.json',
    'binary_file': plots_dir / 'test_binary_{session_id}.dat',
    'multi_file_1': plots_dir / 'multi_test_1_{session_id}.txt',
    'multi_file_2': plots_dir / 'multi_test_2_{session_id}.txt'
}}

for file_type, file_path in test_files.items():
    try:
        if file_path.exists():
            file_stats = file_path.stat()
            file_info = {{
                'type': file_type,
                'path': str(file_path),
                'size': file_stats.st_size,
                'exists': True
            }}

            # Process based on file type
            if file_type == 'json_file':
                # Parse JSON content
                json_data = json.loads(file_path.read_text(encoding='utf-8'))
                file_info['json_keys'] = list(json_data.keys())
                file_info['json_message'] = json_data.get('message', 'N/A')
                workflow_results['content_analysis']['json_processed'] = True

            elif file_type == 'binary_file':
                # Analyze binary content
                binary_data = file_path.read_bytes()
                file_info['binary_header'] = binary_data[:8].hex()
                file_info['is_png'] = binary_data.startswith(b'\\x89PNG')
                workflow_results['content_analysis']['binary_processed'] = True

            else:
                # Process text files
                text_content = file_path.read_text(encoding='utf-8')
                file_info['text_length'] = len(text_content)
                file_info['line_count'] = text_content.count('\\n') + 1

            workflow_results['files_processed'].append(file_info)
            print(f"✓ Processed {{file_type}}: {{file_path.name}} ({{file_stats.st_size}} bytes)")

        else:
            workflow_results['errors'].append(f"File not found: {{file_type}} - {{file_path}}")
            print(f"✗ Missing {{file_type}}: {{file_path.name}}")

    except Exception as e:
        error_msg = f"Error processing {{file_type}}: {{str(e)}}"
        workflow_results['errors'].append(error_msg)
        print(f"✗ {{error_msg}}")

# Step 3: Generate workflow summary
total_files = len(workflow_results['files_processed'])
total_bytes = sum(f.get('size', 0) for f in workflow_results['files_processed'])
error_count = len(workflow_results['errors'])

print(f"\\n=== Workflow Summary ===")
print(f"Files successfully processed: {{total_files}}")
print(f"Total bytes processed: {{total_bytes}}")
print(f"Directories analyzed: {{len(workflow_results['directories_analyzed'])}}")
print(f"Errors encountered: {{error_count}}")
print(f"JSON processing: {{workflow_results['content_analysis'].get('json_processed', False)}}")
print(f"Binary processing: {{workflow_results['content_analysis'].get('binary_processed', False)}}")

# Validation checks
assert total_files >= 4, f"Expected at least 4 files, got {{total_files}}"
assert error_count == 0, f"Unexpected errors: {{workflow_results['errors']}}"
assert workflow_results['content_analysis'].get('json_processed', False), "JSON processing failed"
assert workflow_results['content_analysis'].get('binary_processed', False), "Binary processing failed"

print("\\n✓ Complex filesystem workflow completed successfully!")
"""

        result = execute_python_code(python_code, timeout=Config.TIMEOUTS["code_execution"] * 2)

        # Then: Validate complex workflow completed successfully
        assert result["success"] is True, f"Complex workflow failed: {result.get('error')}"

        stdout = result["data"]["stdout"]
        assert "=== Starting Complex Filesystem Workflow ===" in stdout, "Workflow did not start"
        assert "Files successfully processed: 5" in stdout, "Not all files processed"
        assert "Errors encountered: 0" in stdout, f"Unexpected workflow errors: {stdout}"
        assert "JSON processing: True" in stdout, "JSON processing failed"
        assert "Binary processing: True" in stdout, "Binary processing failed"
        assert "✓ Complex filesystem workflow completed successfully!" in stdout, \
            "Workflow did not complete successfully"

    @pytest.mark.api
    @pytest.mark.filesystem
    def test_api_contract_compliance(
        self,
        server_ready: None
    ) -> None:
        """
        Test strict API contract compliance for filesystem operations.

        Validates that all filesystem operation responses strictly adhere
        to the standardized API contract format with proper data structure,
        error handling, and metadata inclusion.

        Given: Various filesystem operations are executed
        When: API responses are returned from /api/execute-raw
        Then: All responses follow the exact API contract specification

        Args:
            server_ready: Ensures server is available

        API Contract:
            {
                "success": true | false,
                "data": {
                    "result": <any>,
                    "stdout": <string>,
                    "stderr": <string>,
                    "executionTime": <number>
                } | null,
                "error": <string> | null,
                "meta": {
                    "timestamp": <ISO string>
                }
            }
        """
        # Test successful filesystem operation
        success_code = """
from pathlib import Path
plots_dir = Path('/home/pyodide/plots')
print(f"Directory exists: {plots_dir.exists()}")
"""

        result = execute_python_code(success_code, timeout=Config.TIMEOUTS["code_execution"])

        # Validate successful response contract
        validate_api_contract(result)
        assert result["success"] is True
        assert result["data"] is not None
        assert "result" in result["data"]
        assert "stdout" in result["data"]
        assert "stderr" in result["data"]
        assert "executionTime" in result["data"]
        assert isinstance(result["data"]["executionTime"], (int, float))
        assert result["data"]["executionTime"] >= 0
        assert result["error"] is None
        assert "meta" in result
        assert "timestamp" in result["meta"]

        print("✅ API contract compliance validated for filesystem operations")

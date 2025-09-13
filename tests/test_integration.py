"""BDD-style integration tests with proper test isolation and fixtures.

Tests complex integration scenarios using only public APIs.
Each test follows the Given-When-Then pattern typical of BDD.
Only uses /api/execute-raw for Python code execution.
"""

# pylint: disable=redefined-outer-name,unused-argument
# ^ Normal for pytest: fixtures are used as parameters and may not be directly referenced

import os
import tempfile
import time
from pathlib import Path
from typing import Generator, Tuple

import pytest
import requests

# Global configuration
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30
REQUEST_TIMEOUT = 10


def wait_for_server(url: str, timeout: int = 120):
    """Poll ``url`` until it responds or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
        except (requests.RequestException, OSError):
            pass  # Server not ready yet
        time.sleep(1)
    raise RuntimeError(f"Server at {url} did not start in time")


# ==================== FIXTURES ====================


@pytest.fixture(scope="session")
def server_ready():
    """Ensure server is ready before running any tests."""
    wait_for_server(f"{BASE_URL}/health")
    return True


@pytest.fixture
def execution_timeout():
    """Provide a default timeout for Python code execution."""
    return DEFAULT_TIMEOUT


@pytest.fixture
def sample_csv_content():
    """Provide sample CSV content for testing."""
    return "name,value,category\nitem1,1,A\nitem2,2,B\nitem3,3,C\n"


@pytest.fixture
def uploaded_file(
    sample_csv_content, execution_timeout
) -> Generator[Tuple[str, str], None, None]:
    """Upload a test CSV file and return (filename, vfs_path).
    
    Yields (filename, vfs_path) for the uploaded file.
    Automatically cleans up the file after the test.
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(sample_csv_content)
        tmp_path = tmp.name

    # Upload file
    with open(tmp_path, "rb") as fh:
        r = requests.post(
            f"{BASE_URL}/api/upload",
            files={"file": ("test_data.csv", fh, "text/csv")},
            timeout=execution_timeout,
        )

    # Clean up temp file
    os.unlink(tmp_path)

    # Extract file references
    assert r.status_code == 200
    upload_data = r.json()
    assert upload_data.get("success") is True

    file_info = upload_data["data"]["file"]
    filename = Path(file_info["vfsPath"]).name
    vfs_path = file_info["vfsPath"]

    # Yield for test to use
    yield filename, vfs_path

    try:
        # Delete from server
        requests.delete(f"{BASE_URL}/api/uploaded-files/{filename}", timeout=REQUEST_TIMEOUT)
    except (requests.RequestException, OSError):
        pass  # Ignore cleanup errors


# ==================== DATA CONSISTENCY TESTS ====================


class TestDataConsistency:
    """Test data processing consistency and file handling."""

    def test_basic_execute_raw_functionality(self, server_ready, execution_timeout):
        """Given: The server is running
        When: I execute simple Python code via execute-raw
        Then: It should return the correct result
        """
        # Given - Simple Python calculation
        python_code = '''
# Basic calculation
result = 2 + 2
print(f"Calculation result: {result}")
result
        '''

        # When
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=python_code,
            headers={"Content-Type": "text/plain"},
            timeout=execution_timeout,
        )

        # Then
        assert response.status_code == 200
        result = response.json()
        assert result.get("success") is True, f"Basic execution failed: {result.get('error')}"
        assert result.get("result") == 4
        assert "Calculation result: 4" in result.get("stdout", "")

    def test_csv_processing_with_execute_raw(self, server_ready, uploaded_file, execution_timeout):
        """Given: A CSV file is uploaded to the server
        When: I process it using execute-raw endpoint
        Then: The data should be properly accessible and processable
        """
        # Given
        filename, vfs_path = uploaded_file

        # When - First install pandas, then execute Python code to read and process the CSV
        install_code = '''
import micropip
await micropip.install("pandas")
print("Pandas installed successfully")
"pandas_installed"
        '''
        
        # Install pandas first
        install_response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=install_code,
            headers={"Content-Type": "text/plain"},
            timeout=120,  # Package installation takes longer
        )
        assert install_response.status_code == 200
        install_result = install_response.json()
        assert install_result.get("success") is True, f"Failed to install pandas: {install_result.get('error')}"
        
        python_code = f'''
import pandas as pd
from pathlib import Path

# Read the uploaded CSV file
df = pd.read_csv(Path("{vfs_path}"))

# Verify data structure and content
result = {{
    "success": True,
    "shape": list(df.shape),
    "columns": list(df.columns),
    "first_row": df.iloc[0].to_dict() if len(df) > 0 else {{}},
    "data_types": str(df.dtypes.to_dict())  # Convert to string for JSON serialization
}}
print(f"Processed CSV with shape: {{result['shape']}}")
result
        '''

        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=python_code,
            headers={"Content-Type": "text/plain"},
            timeout=execution_timeout,
        )

        # Then
        assert response.status_code == 200
        result = response.json()
        assert result.get("success") is True
        
        # Verify the data was processed correctly
        exec_result = result.get("result")
        assert exec_result["success"] is True
        assert exec_result["shape"] == [3, 3]  # 3 rows, 3 columns
        assert set(exec_result["columns"]) == {"name", "value", "category"}
        assert exec_result["first_row"]["name"] == "item1"

    def test_csv_processing_edge_cases(self, server_ready, execution_timeout):
        """Given: CSV files with various edge cases
        When: I upload and process them using execute-raw
        Then: They should be handled correctly without errors
        """
        # Given - Test cases with different CSV edge cases
        test_cases = [
            # CSV with commas in quoted fields
            ('quotes.csv', 'name,description,value\n"Smith, John","A person named ""John""",42\n'),
            # CSV with different encodings and special characters
            ('unicode.csv', 'name,value\nCafé,123\nNaïve,456\n'),
            # CSV with empty fields
            ('empty_fields.csv', 'name,value,category\nitem1,,A\n,2,\n,,\n'),
            # CSV with very long lines
            ('long_lines.csv', 'name,value\n' + 'x' * 100 + ',123\n'),
        ]
        
        uploaded_files = []
        try:
            for filename, content in test_cases:
                # Given - Create and upload each test file
                with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                with open(tmp_path, "rb") as fh:
                    r = requests.post(
                        f"{BASE_URL}/api/upload",
                        files={"file": (filename, fh, "text/csv")},
                        timeout=execution_timeout
                    )
                
                if r.status_code == 200:
                    upload_data = r.json()
                    file_info = upload_data["data"]["file"]
                    vfs_path = file_info["vfsPath"]
                    server_filename = Path(vfs_path).name
                    
                    uploaded_files.append({
                        'filename': server_filename,
                        'vfs_path': vfs_path,
                        'temp_path': tmp_path
                    })
                    
                    # When - Test that the file can be read and processed
                    python_code = f'''
import pandas as pd
from pathlib import Path

try:
    df = pd.read_csv(Path("{vfs_path}"))
    result = {{
        "success": True, 
        "shape": list(df.shape), 
        "columns": list(df.columns),
        "filename": "{filename}"
    }}
    print(f"Successfully processed {{result['filename']}} with shape {{result['shape']}}")
except Exception as e:
    result = {{"success": False, "error": str(e), "filename": "{filename}"}}
    print(f"Error processing {{result['filename']}}: {{result['error']}}")
result
                    '''
                    
                    # When
                    response = requests.post(
                        f"{BASE_URL}/api/execute-raw",
                        data=python_code,
                        headers={"Content-Type": "text/plain"},
                        timeout=execution_timeout
                    )
                    
                    # Then
                    assert response.status_code == 200
                    exec_result = response.json()
                    assert exec_result.get("success") is True
                    
                    # Verify the processing result
                    result_data = exec_result.get("result")
                    assert result_data["success"] is True, f"Failed to process {filename}: {result_data.get('error', 'Unknown error')}"
                    
                os.unlink(tmp_path)
        
        finally:
            # Clean up all uploaded files
            for file_info in uploaded_files:
                try:
                    requests.delete(f"{BASE_URL}/api/uploaded-files/{file_info['filename']}", timeout=REQUEST_TIMEOUT)
                except:
                    pass


# ==================== CONCURRENT OPERATIONS TESTS ====================


class TestConcurrentOperations:
    """Test multiple operations and file handling scenarios."""

    def test_multiple_file_operations_sequence(self, server_ready, execution_timeout):
        """Given: Multiple files are uploaded
        When: I perform sequential operations on them
        Then: All operations should succeed without interference
        """
        # Given - Upload a file
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write("x,y\n1,2\n3,4\n5,6\n")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, "rb") as fh:
                r = requests.post(
                    f"{BASE_URL}/api/upload",
                    files={"file": ("concurrent.csv", fh, "text/csv")},
                    timeout=execution_timeout
                )
            assert r.status_code == 200
            upload_data = r.json()
            file_info = upload_data["data"]["file"]
            vfs_path = file_info["vfsPath"]
            filename = Path(vfs_path).name
            
            # When - Execute multiple operations using the same file
            operations = [
                f'''
import pandas as pd
from pathlib import Path
df = pd.read_csv(Path("{vfs_path}"))
result = df.shape[0]
print(f"Row count: {{result}}")
result
                ''',
                f'''
import pandas as pd
from pathlib import Path
df = pd.read_csv(Path("{vfs_path}"))
result = df.sum().sum()
print(f"Sum of all values: {{result}}")
result
                ''',
                f'''
import pandas as pd
from pathlib import Path
df = pd.read_csv(Path("{vfs_path}"))
result = list(df.columns)
print(f"Columns: {{result}}")
result
                ''',
            ]
            
            # Then - All operations should succeed
            for i, code in enumerate(operations):
                response = requests.post(
                    f"{BASE_URL}/api/execute-raw",
                    data=code,
                    headers={"Content-Type": "text/plain"},
                    timeout=execution_timeout
                )
                assert response.status_code == 200
                result = response.json()
                assert result.get("success") is True, f"Operation {i+1} failed: {result.get('error', 'Unknown error')}"
            
            # Verify file is still accessible after multiple operations
            verification_code = f'''
import pandas as pd
from pathlib import Path
df = pd.read_csv(Path("{vfs_path}"))
result = {{"accessible": True, "shape": list(df.shape)}}
print(f"File still accessible: {{result}}")
result
            '''
            
            response = requests.post(
                f"{BASE_URL}/api/execute-raw",
                data=verification_code,
                headers={"Content-Type": "text/plain"},
                timeout=execution_timeout
            )
            assert response.status_code == 200
            result = response.json()
            assert result.get("success") is True
            exec_result = result.get("result")
            assert exec_result["accessible"] is True
            assert exec_result["shape"] == [3, 2]
            
            # Clean up
            r = requests.delete(f"{BASE_URL}/api/uploaded-files/{filename}", timeout=REQUEST_TIMEOUT)
            assert r.status_code == 200
            
        finally:
            os.unlink(tmp_path)

# ==================== STATE PERSISTENCE TESTS ====================


class TestStatePersistence:
    """Test execution context isolation and package persistence."""

    def test_execution_context_isolation(self, server_ready, execution_timeout):
        """Given: Variables are defined in one execution
        When: I execute code in separate requests
        Then: Variables should not persist between executions (isolation)
        """
        # Given - Define a variable in first execution
        define_code = '''
test_var = "isolated_value"
print(f"Defined variable: {test_var}")
"variable_defined"
        '''
        
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=define_code,
            headers={"Content-Type": "text/plain"},
            timeout=execution_timeout,
        )
        assert response.status_code == 200
        result = response.json()
        assert result.get("success") is True
        assert result.get("result") == "variable_defined"

        # When - Check for the variable in a second request
        check_code = '''
try:
    # This should fail as variables don't persist
    test_var
    result = "defined"
except NameError:
    result = "undefined"
print(f"Variable check result: {result}")
result
        '''
        
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=check_code,
            headers={"Content-Type": "text/plain"},
            timeout=execution_timeout,
        )
        assert response.status_code == 200
        result = response.json()
        assert result.get("success") is True

        # Then - The variable should be undefined (proper isolation)
        exec_result = result.get("result")
        assert exec_result == "undefined", "Variables should not persist between executions"

    def test_package_persistence_and_reset(self, server_ready, execution_timeout):
        """Given: A package is installed
        When: I reset the environment and check package availability
        Then: Packages should persist after reset while maintaining isolation
        """
        # Given - Install a package
        install_code = '''
import micropip
await micropip.install("beautifulsoup4")
print("Package installation completed")
"installation_complete"
        '''
        
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=install_code,
            headers={"Content-Type": "text/plain"},
            timeout=120,  # Package installation takes longer
        )
        assert response.status_code == 200
        result = response.json()
        assert result.get("success") is True
        
        # Verify package is available
        verify_code = '''
try:
    import bs4
    result = "package_available"
    print("Package successfully imported")
except ImportError:
    result = "package_not_found"
    print("Package import failed")
result
        '''
        
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=verify_code,
            headers={"Content-Type": "text/plain"},
            timeout=execution_timeout,
        )
        assert response.status_code == 200
        result = response.json()
        assert result.get("success") is True
        assert result.get("result") == "package_available"
        
        # When - Reset the environment (if reset endpoint exists)
        try:
            reset_response = requests.post(f"{BASE_URL}/api/reset", timeout=60)
            if reset_response.status_code == 200:
                # Then - Verify package is still available after reset
                response = requests.post(
                    f"{BASE_URL}/api/execute-raw",
                    data=verify_code,
                    headers={"Content-Type": "text/plain"},
                    timeout=execution_timeout,
                )
                assert response.status_code == 200
                result = response.json()
                assert result.get("success") is True
                assert result.get("result") == "package_available", "Packages should persist after reset"
        except requests.exceptions.RequestException:
            # Reset endpoint might not exist, skip this part
            pass

# ==================== COMPLEX DATA FLOW TESTS ====================


class TestComplexDataFlow:
    """Test complex data processing workflows with multiple files."""

    def test_multi_file_data_processing_workflow(self, server_ready, execution_timeout):
        """Given: Multiple CSV files are uploaded
        When: I perform complex data processing across files
        Then: The workflow should complete successfully with correct results
        """
        # Given - Upload multiple CSV files
        files_data = [
            ("data1.csv", "id,value\n1,10\n2,20\n3,30\n"),
            ("data2.csv", "id,score\n1,90\n2,85\n3,95\n"),
        ]
        
        uploaded = []
        try:
            for filename, content in files_data:
                with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                with open(tmp_path, "rb") as fh:
                    r = requests.post(
                        f"{BASE_URL}/api/upload",
                        files={"file": (filename, fh, "text/csv")},
                        timeout=execution_timeout
                    )
                assert r.status_code == 200
                upload_data = r.json()
                file_info = upload_data["data"]["file"]
                uploaded.append({
                    'filename': Path(file_info["vfsPath"]).name,
                    'vfs_path': file_info["vfsPath"],
                    'temp_path': tmp_path
                })
                os.unlink(tmp_path)
            
            # When - Perform complex data processing
            complex_code = f'''
import pandas as pd
from pathlib import Path

# Load both datasets
df1 = pd.read_csv(Path("{uploaded[0]['vfs_path']}"))
df2 = pd.read_csv(Path("{uploaded[1]['vfs_path']}"))

print(f"Loaded df1 with shape: {{df1.shape}}")
print(f"Loaded df2 with shape: {{df2.shape}}")

# Merge the datasets
merged = pd.merge(df1, df2, on='id')
print(f"Merged data shape: {{merged.shape}}")

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

print(f"Processing complete: {{result}}")
result
            '''
            
            response = requests.post(
                f"{BASE_URL}/api/execute-raw",
                data=complex_code,
                headers={"Content-Type": "text/plain"},
                timeout=execution_timeout
            )
            
            # Then
            assert response.status_code == 200
            result = response.json()
            assert result.get("success") is True
            
            # Verify the computed results are correct
            result_data = result.get("result")
            assert isinstance(result_data, dict)
            assert result_data["total_value"] == 60  # 10 + 20 + 30
            assert result_data["avg_score"] == 90.0  # (90 + 85 + 95) / 3
            assert result_data["record_count"] == 3
            assert set(result_data["columns"]) == {"id", "value", "score"}
            
        finally:
            # Clean up
            for file_info in uploaded:
                try:
                    requests.delete(f"{BASE_URL}/api/uploaded-files/{file_info['filename']}", timeout=REQUEST_TIMEOUT)
                except:
                    pass


# ==================== ERROR RECOVERY TESTS ====================


class TestErrorRecovery:
    """Test error handling and system recovery."""

    def test_system_recovery_after_errors(self, server_ready, execution_timeout):
        """Given: Python execution errors occur
        When: I execute subsequent code
        Then: The system should recover and process new requests correctly
        """
        # Given - Execute code that will fail (undefined variable)
        error_code = '''
# This will cause a NameError
undefined_variable
        '''
        
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=error_code,
            headers={"Content-Type": "text/plain"},
            timeout=execution_timeout
        )
        assert response.status_code == 200
        result = response.json()
        # Error should be handled gracefully
        assert result.get("success") is False or "undefined_variable" in str(result.get("error", ""))
        
        # When - Execute code that will succeed after the error
        success_code = '''
result = 2 + 2
print(f"Calculation result: {result}")
result
        '''
        
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=success_code,
            headers={"Content-Type": "text/plain"},
            timeout=execution_timeout
        )
        
        # Then
        assert response.status_code == 200
        result = response.json()
        assert result.get("success") is True
        assert result.get("result") == 4
        
        # Try syntax error
        syntax_error_code = '''
# This has invalid Python syntax
if True
    print('missing colon')
        '''
        
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=syntax_error_code,
            headers={"Content-Type": "text/plain"},
            timeout=execution_timeout
        )
        assert response.status_code == 200
        result = response.json()
        # Syntax error should be handled
        assert result.get("success") is False or "syntax" in str(result.get("error", "")).lower()
        
        # Verify system still works after syntax error
        recovery_code = '''
result = "system_recovered"
print(f"Recovery test: {result}")
result
        '''
        
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=recovery_code,
            headers={"Content-Type": "text/plain"},
            timeout=execution_timeout
        )
        assert response.status_code == 200
        result = response.json()
        assert result.get("success") is True
        assert result.get("result") == "system_recovered"

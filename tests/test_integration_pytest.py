"""
Comprehensive Integration Tests for Pyodide Express Server

This module contains comprehensive integration tests written in BDD (Behavior-Driven Development) 
style using pytest. These tests validate complex integration scenarios and data flow workflows 
using only the public API endpoints.

Key Features:
- BDD-style test structure with Given-When-Then patterns
- Comprehensive test coverage for integration scenarios
- Uses only /api/execute-raw for Python code execution
- Avoids internal Pyodide APIs for better API abstraction
- Proper test isolation with automatic cleanup
- Global configuration for timeouts and constants
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Generator

import pytest
import requests


# ==================== GLOBAL CONFIGURATION ====================

# API Configuration
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30
EXECUTION_TIMEOUT = 30000
MAX_CODE_LENGTH = 50000
MAX_FILE_SIZE_MB = 10
PACKAGE_INSTALL_TIMEOUT = 120

# Path Configuration
UPLOADS_PATH = "uploads"  # Pyodide filesystem path
TEST_CSV_CONTENT = "name,value,category\nitem1,1,A\nitem2,2,B\nitem3,3,C\n"
COMPLEX_CSV_CONTENT = "id,value\n1,10\n2,20\n3,30\n"
SECONDARY_CSV_CONTENT = "id,score\n1,90\n2,85\n3,75\n"

# Edge Case Test Data
EDGE_CASE_CSVS = {
    'quotes.csv': 'name,description,value\n"Smith, John","A person named ""John""",42\n',
    'unicode.csv': 'name,value\nCafé,123\nNaïve,456\n',
    'empty_fields.csv': 'name,value,category\nitem1,,A\n,2,\n,,\n',
    'long_lines.csv': 'name,value\n' + 'x' * 1000 + ',123\n'
}


# ==================== UTILITY FUNCTIONS ====================

def wait_for_server(url: str, timeout: int = 120) -> bool:
    """
    Wait for the server to be ready for accepting requests.
    
    Args:
        url: The health check URL to poll
        timeout: Maximum time to wait in seconds
        
    Returns:
        bool: True if server is ready, False if timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except (requests.RequestException, OSError):
            pass  # Server not ready yet
        time.sleep(1)
    return False


def get_execution_output(response: Dict) -> str:
    """
    Extract output from API response, checking both result and stdout fields.
    
    Args:
        response: API response dictionary
        
    Returns:
        str: The execution output as a string
    """
    # Check result field first (for expressions)
    result = response.get("result")
    if result is not None:
        return str(result)
    
    # Fall back to stdout (for print statements)
    stdout = response.get("stdout", "").strip()
    if stdout:
        return stdout
    
    # If neither has content, return empty string
    return ""


def execute_python_code(code: str, timeout: int = DEFAULT_TIMEOUT) -> Dict:
    """
    Execute Python code using the /api/execute-raw endpoint.
    
    Args:
        code: Python code to execute
        timeout: Request timeout in seconds
        
    Returns:
        dict: JSON response from the API
    """
    response = requests.post(
        f"{BASE_URL}/api/execute-raw",
        data=code,
        headers={"Content-Type": "text/plain"},
        timeout=timeout
    )
    response.raise_for_status()  # Raise exception for HTTP errors
    return response.json()


def upload_csv_file(filename: str, content: str, timeout: int = DEFAULT_TIMEOUT) -> Dict:
    """
    Upload a CSV file to the server.
    
    Args:
        filename: Name for the uploaded file
        content: CSV content to upload
        timeout: Request timeout in seconds
        
    Returns:
        dict: Upload response data
    """
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        with open(tmp_path, "rb") as fh:
            response = requests.post(
                f"{BASE_URL}/api/upload",
                files={"file": (filename, fh, "text/csv")},
                timeout=timeout
            )
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    finally:
        os.unlink(tmp_path)


def list_uploaded_files(timeout: int = DEFAULT_TIMEOUT) -> List[Dict]:
    """
    Get list of uploaded files from the server.
    
    Args:
        timeout: Request timeout in seconds
        
    Returns:
        list: List of file information dictionaries
    """
    response = requests.get(f"{BASE_URL}/api/uploaded-files", timeout=timeout)
    data = response.json()
    return data.get("data", {}).get("files", [])


def delete_uploaded_file(filename: str, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """
    Delete an uploaded file from the server.
    
    Args:
        filename: Name of file to delete
        timeout: Request timeout in seconds
        
    Returns:
        bool: True if deletion successful
    """
    try:
        response = requests.delete(f"{BASE_URL}/api/uploaded-files/{filename}", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


# ==================== PYTEST FIXTURES ====================

@pytest.fixture(scope="session")
def server_ready():
    """
    Ensure the server is ready before running any tests.
    
    This session-scoped fixture waits for the server to be available
    and is used by all tests that need server access.
    """
    assert wait_for_server(f"{BASE_URL}/health"), "Server is not ready for testing"
    yield True


@pytest.fixture
def base_url():
    """Provide the base URL for API requests."""
    return BASE_URL


@pytest.fixture
def default_timeout():
    """Provide the default timeout for API requests."""
    return DEFAULT_TIMEOUT


@pytest.fixture
def uploaded_test_file(server_ready) -> Generator[Tuple[str, str], None, None]:
    """
    Upload a test CSV file and provide cleanup.
    
    This fixture uploads a test CSV file and yields both the original filename
    and the server-side filename for use in tests. It automatically cleans up
    the file after the test completes.
    
    Yields:
        tuple: (original_filename, server_filename)
    """
    # Given: A test CSV file needs to be uploaded
    filename = "test_integration.csv"
    upload_response = upload_csv_file(filename, TEST_CSV_CONTENT)
    
    assert upload_response.get("success"), f"Upload failed: {upload_response}"
    
    file_info = upload_response["data"]["file"]
    original_filename = file_info["originalName"]
    server_filename = file_info["sanitizedOriginal"]
    
    # Yield for test usage
    yield original_filename, server_filename
    
    # Cleanup after test
    delete_uploaded_file(server_filename)


@pytest.fixture
def multiple_test_files(server_ready) -> Generator[List[Tuple[str, str]], None, None]:
    """
    Upload multiple test CSV files for complex integration scenarios.
    
    This fixture uploads multiple CSV files with different data structures
    and provides cleanup for all files after test completion.
    
    Yields:
        list: List of (original_filename, server_filename) tuples
    """
    uploaded_files = []
    
    # Upload primary data file
    response1 = upload_csv_file("data1.csv", COMPLEX_CSV_CONTENT)
    assert response1.get("success"), f"Upload 1 failed: {response1}"
    
    # Upload secondary data file
    response2 = upload_csv_file("data2.csv", SECONDARY_CSV_CONTENT)
    assert response2.get("success"), f"Upload 2 failed: {response2}"
    
    # Extract file information
    file1_info = response1["data"]["file"]
    file2_info = response2["data"]["file"]
    
    uploaded_files = [
        (file1_info["originalName"], file1_info["sanitizedOriginal"]),
        (file2_info["originalName"], file2_info["sanitizedOriginal"])
    ]
    
    yield uploaded_files
    
    # Cleanup all uploaded files
    for _, server_filename in uploaded_files:
        delete_uploaded_file(server_filename)


# ==================== DATA CONSISTENCY TESTS ====================

class TestDataConsistency:
    """Test data consistency and JSON parsing across API boundaries."""
    
    def test_given_uploaded_file_when_checking_file_info_then_returns_valid_json(
        self, server_ready, uploaded_test_file
    ):
        """
        Scenario: Verify file information returns proper JSON structures
        
        Given: A CSV file has been uploaded to the server
        When: I request file information through the API
        Then: The response should contain valid JSON objects, not Python strings
        And: File existence information should be properly typed as booleans
        """
        # Given
        original_filename, server_filename = uploaded_test_file
        
        # When: I check if the file exists using Python code
        check_code = f"""
import os
from pathlib import Path

filename = "{server_filename}"
file_path = Path("uploads") / filename

result = {{
    "filename": filename,
    "exists": file_path.exists(),
    "is_file": file_path.is_file() if file_path.exists() else False,
    "size": file_path.stat().st_size if file_path.exists() else 0
}}
print(f"File check result: {{result}}")
result
"""
        
        response = execute_python_code(check_code)
        
        # Then
        assert response.get("success"), f"Code execution failed: {response}"
        
        # Verify the result is accessible in both stdout and result fields
        output = get_execution_output(response)
        
        assert output, "Should have output from execution"
        assert server_filename in output, "Should contain the filename"
        assert "exists" in output or "File check result:" in output, "Should contain existence check"
    
    def test_given_csv_data_when_processing_with_basic_python_then_returns_consistent_data(
        self, server_ready, uploaded_test_file
    ):
        """
        Scenario: Verify CSV processing with basic Python returns consistent data
        
        Given: A CSV file with known data has been uploaded
        When: I process it with basic Python CSV module
        Then: The data types and structure should be consistent and predictable
        """
        # Given
        _, server_filename = uploaded_test_file
        
        # When: I process the CSV with basic Python
        csv_code = f"""
import csv
from pathlib import Path

csv_path = Path("{UPLOADS_PATH}") / "{server_filename}"
rows = []

with open(csv_path, 'r') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

result = {{
    "total_rows": len(rows),
    "columns": list(rows[0].keys()) if rows else [],
    "sample_data": rows[:2] if len(rows) >= 2 else rows,
    "file_exists": csv_path.exists()
}}
result
"""
        
        response = execute_python_code(csv_code)
        
        # Then
        assert response.get("success"), f"CSV processing failed: {response}"
        
        # Verify output contains expected structure
        output = get_execution_output(response)
        assert "total_rows" in output, "Should contain row count information"
        assert "columns" in output, "Should contain column information"
        assert "3" in output, "Should reference correct row count"
        assert "file_exists" in output, "Should confirm file exists"


# ==================== CSV PROCESSING EDGE CASES ====================

class TestCsvProcessingEdgeCases:
    """Test CSV file processing with various edge cases and special characters."""
    
    @pytest.mark.parametrize("csv_name,csv_content", list(EDGE_CASE_CSVS.items()))
    def test_given_edge_case_csv_when_uploading_and_processing_then_handles_correctly(
        self, server_ready, csv_name, csv_content
    ):
        """
        Scenario: Process CSV files with edge cases
        
        Given: A CSV file with edge case content (quotes, unicode, empty fields, long lines)
        When: I upload and process it with pandas
        Then: The system should handle it gracefully without errors
        And: The data should be readable and processable
        """
        # Given: Upload the edge case CSV
        upload_response = upload_csv_file(csv_name, csv_content)
        assert upload_response.get("success"), f"Upload failed for {csv_name}: {upload_response}"
        
        server_filename = upload_response["data"]["file"]["sanitizedOriginal"]
        
        try:
            # When: Process the CSV with basic Python
            process_code = f"""
import csv
from pathlib import Path

try:
    csv_path = Path("/uploads/{server_filename}")
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    result = {{
        "success": True,
        "row_count": len(rows),
        "columns": rows[0] if rows else [],
        "has_data": len(rows) > 1,  # Account for header row
        "csv_type": "{csv_name}"
    }}
except Exception as e:
    result = {{
        "success": False,
        "error": str(e),
        "csv_type": "{csv_name}"
    }}
result
"""
            
            response = execute_python_code(process_code)
            
            # Then
            assert response.get("success"), f"Code execution failed for {csv_name}: {response}"
            
            # Verify the CSV was processed successfully
            output = response.get("stdout", "")
            assert "success" in output, f"Should contain success indicator for {csv_name}"
            assert csv_name in output, f"Should contain CSV type identifier for {csv_name}"
            
        finally:
            # Cleanup
            delete_uploaded_file(server_filename)


# ==================== CONCURRENT OPERATIONS TESTS ====================

class TestConcurrentOperations:
    """Test multiple operations in sequence to verify system stability."""
    
    def test_given_uploaded_file_when_multiple_sequential_operations_then_all_succeed(
        self, server_ready, uploaded_test_file
    ):
        """
        Scenario: Perform multiple operations on the same file sequentially
        
        Given: A CSV file has been uploaded and is available
        When: I perform multiple different operations on the same file
        Then: All operations should succeed without interference
        And: Each operation should return consistent results
        """
        # Given
        _, server_filename = uploaded_test_file
        
        # When: Execute multiple operations sequentially
        operations = [
            # Operation 1: Get row count
            f"""
import csv
with open("/uploads/{server_filename}", 'r') as f:
    row_count = len(list(csv.reader(f))) - 1  # Subtract header
f"Row count: {{row_count}}"
""",
            # Operation 2: Calculate sum of values
            f"""
import csv
with open("/uploads/{server_filename}", 'r') as f:
    reader = csv.DictReader(f)
    total = sum(int(row['value']) for row in reader)
f"Value sum: {{total}}"
""",
            # Operation 3: List columns
            f"""
import csv
with open("/uploads/{server_filename}", 'r') as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
f"Columns: {{list(columns)}}"
""",
            # Operation 4: Get unique categories
            f"""
import csv
with open("/uploads/{server_filename}", 'r') as f:
    reader = csv.DictReader(f)
    categories = sorted(set(row['category'] for row in reader))
f"Categories: {{categories}}"
"""
        ]
        
        results = []
        for i, code in enumerate(operations):
            response = execute_python_code(code)
            
            # Then: Each operation should succeed
            assert response.get("success"), f"Operation {i+1} failed: {response}"
            
            output = get_execution_output(response)
            results.append(output)
        
        # Verify expected results
        assert "Row count: 3" in results[0], "Should have correct row count"
        assert "Value sum: 6" in results[1], "Should have correct sum (1+2+3=6)"
        assert "'name', 'value', 'category'" in results[2] or "name" in results[2], "Should have correct columns"
        assert "'A', 'B', 'C'" in results[3] or "A" in results[3], "Should have correct categories"
    
    def test_given_multiple_files_when_concurrent_access_then_no_interference(
        self, server_ready, multiple_test_files
    ):
        """
        Scenario: Access multiple files concurrently without interference
        
        Given: Multiple CSV files have been uploaded
        When: I access different files in rapid succession
        Then: Each file should be accessible independently
        And: There should be no cross-contamination between file operations
        """
        # Given
        files = multiple_test_files
        file1_name = files[0][1]  # server filename for data1.csv
        file2_name = files[1][1]  # server filename for data2.csv
        
        # When: Access files in rapid succession
        rapid_operations = [
            f'''
import csv
with open("/uploads/{file1_name}", 'r') as f:
    rows = list(csv.reader(f))
f"File1 shape: ({len(rows)-1}, {len(rows[0]) if rows else 0})"
''',
            f'''
import csv  
with open("/uploads/{file2_name}", 'r') as f:
    rows = list(csv.reader(f))
f"File2 shape: ({len(rows)-1}, {len(rows[0]) if rows else 0})"
''',
            f'''
import csv
with open("/uploads/{file1_name}", 'r') as f:
    reader = csv.DictReader(f)
    columns = list(reader.fieldnames)
f"File1 columns: {columns}"
''',
            f'''
import csv
with open("/uploads/{file2_name}", 'r') as f:
    reader = csv.DictReader(f) 
    columns = list(reader.fieldnames)
f"File2 columns: {columns}"
'''
        ]
        
        for operation in rapid_operations:
            response = execute_python_code(operation)
            
            # Then: Each operation should succeed
            assert response.get("success"), f"Rapid operation failed: {response}"
            
            output = get_execution_output(response)
            
            # Verify correct file is being accessed
            if "File1" in output:
                assert "(3, 2)" in output or "id" in output, \
                    f"File1 operation should access correct file: {output}"
            elif "File2" in output:
                assert "(3, 2)" in output or "score" in output, \
                    f"File2 operation should access correct file: {output}"


# ==================== COMPLEX DATA FLOW TESTS ====================

class TestComplexDataFlow:
    """Test complex data processing workflows combining multiple operations."""
    
    def test_given_multiple_datasets_when_merging_and_analyzing_then_produces_correct_results(
        self, server_ready, multiple_test_files
    ):
        """
        Scenario: Complex data processing workflow with multiple datasets
        
        Given: Multiple CSV files with relatable data have been uploaded
        When: I merge the datasets and perform complex analysis
        Then: The analysis should produce correct and consistent results
        And: The workflow should handle the entire process without errors
        """
        # Given
        files = multiple_test_files
        file1_name = files[0][1]  # data1.csv (id, value)
        file2_name = files[1][1]  # data2.csv (id, score)
        
        # When: Perform complex data analysis
        complex_analysis_code = f"""
import csv

# Load both datasets
data1 = []
with open("/uploads/{file1_name}", 'r') as f:
    reader = csv.DictReader(f)
    data1 = list(reader)

data2 = []
with open("/uploads/{file2_name}", 'r') as f:
    reader = csv.DictReader(f)
    data2 = list(reader)

# Merge datasets on 'id' column (simple join)
merged = []
for row1 in data1:
    for row2 in data2:
        if row1['id'] == row2['id']:
            merged_row = {{**row1, **row2}}
            merged.append(merged_row)

# Perform complex calculations
total_value = sum(int(row['value']) for row in merged)
scores = [int(row['score']) for row in merged]
avg_score = sum(scores) / len(scores) if scores else 0
max_score = max(scores) if scores else 0
min_score = min(scores) if scores else 0
record_count = len(merged)

# Calculate correlation (simplified)
values = [int(row['value']) for row in merged]
if len(values) > 1 and len(scores) > 1:
    # Simple correlation calculation
    mean_val = sum(values) / len(values)
    mean_score = sum(scores) / len(scores)
    numerator = sum((v - mean_val) * (s - mean_score) for v, s in zip(values, scores))
    val_sq = sum((v - mean_val) ** 2 for v in values)
    score_sq = sum((s - mean_score) ** 2 for s in scores)
    correlation = numerator / (val_sq * score_sq) ** 0.5 if val_sq * score_sq > 0 else 0
else:
    correlation = 0

# Create summary statistics
summary = {{
    "total_value": int(total_value),
    "avg_score": round(avg_score, 2),
    "max_score": int(max_score),
    "min_score": int(min_score),
    "record_count": record_count,
    "correlation": round(correlation, 3),
    "columns": list(merged[0].keys()) if merged else [],
    "data_quality": "complete"
}}

f"Analysis complete: {{summary}}"
"""
        
        response = execute_python_code(complex_analysis_code)
        
        # Then
        assert response.get("success"), f"Complex analysis failed: {response}"
        
        output = get_execution_output(response)
        
        # Verify expected calculations
        assert "total_value': 60" in output, "Should have correct total value (10+20+30=60)"
        assert "avg_score': 83.33" in output, "Should have correct average score ((90+85+75)/3≈83.33)"
        assert "max_score': 90" in output, "Should have correct max score"
        assert "min_score': 75" in output, "Should have correct min score"
        assert "record_count': 3" in output, "Should have correct record count"
        assert "data_quality': 'complete'" in output, "Should have complete data quality"
    
    def test_given_dataset_when_advanced_statistical_analysis_then_produces_insights(
        self, server_ready, uploaded_test_file
    ):
        """
        Scenario: Advanced statistical analysis on uploaded dataset
        
        Given: A dataset with numerical data has been uploaded
        When: I perform advanced statistical analysis including distributions and trends
        Then: The analysis should produce meaningful statistical insights
        And: All calculations should be mathematically correct
        """
        # Given
        _, server_filename = uploaded_test_file
        
        # When: Perform advanced statistical analysis
        statistical_code = f"""
import csv
import math

# Load and analyze dataset
data = []
with open("/uploads/{server_filename}", 'r') as f:
    reader = csv.DictReader(f)
    data = [int(row['value']) for row in reader]

# Basic statistics
mean_val = sum(data) / len(data)
sorted_data = sorted(data)
median_val = sorted_data[len(sorted_data)//2] if len(sorted_data) % 2 == 1 else (sorted_data[len(sorted_data)//2-1] + sorted_data[len(sorted_data)//2]) / 2
variance = sum((x - mean_val) ** 2 for x in data) / len(data)
std_val = math.sqrt(variance)

basic_stats = {{
    "mean": round(mean_val, 3),
    "median": median_val,
    "std": round(std_val, 3),
    "variance": round(variance, 3),
    "min": min(data),
    "max": max(data),
    "range": max(data) - min(data)
}}

# Distribution analysis
distribution = {{
    "is_uniform": std_val < 1.0,  # Low standard deviation
    "has_outliers": any(abs(x - mean_val) > 2 * std_val for x in data),
    "value_counts": {{str(val): data.count(val) for val in set(data)}}
}}

# Advanced metrics
coefficient_of_variation = std_val / mean_val if mean_val != 0 else 0
unique_values = len(set(data))

advanced = {{
    "coefficient_of_variation": round(coefficient_of_variation, 3),
    "unique_values": unique_values,
    "missing_values": 0  # No missing values in our test data
}}

results = {{
    "basic_stats": basic_stats,
    "distribution": distribution,
    "advanced": advanced
}}

f"Statistical analysis: {{results}}"
"""
        
        response = execute_python_code(statistical_code)
        
        # Then
        assert response.get("success"), f"Statistical analysis failed: {response}"
        
        output = get_execution_output(response)
        
        # Verify statistical calculations are reasonable
        assert "mean': 2.0" in output, "Should have correct mean (1+2+3)/3=2"
        assert "median': 2" in output, "Should have correct median"
        assert "min': 1" in output, "Should have correct minimum"
        assert "max': 3" in output, "Should have correct maximum"
        assert "range': 2" in output, "Should have correct range (3-1=2)"
        assert "unique_values': 3" in output, "Should have 3 unique values"


# ==================== ERROR RECOVERY TESTS ====================

class TestErrorRecovery:
    """Test system resilience and error recovery capabilities."""
    
    def test_given_invalid_python_code_when_executing_then_recovers_gracefully(
        self, server_ready
    ):
        """
        Scenario: System recovery after Python execution errors
        
        Given: The system is running normally
        When: I execute invalid Python code that causes errors
        Then: The system should handle errors gracefully
        And: Subsequent valid code should execute successfully
        """
        # Given: System is ready
        
        # When: Execute code that will cause a NameError
        invalid_code1 = "undefined_variable_that_does_not_exist"
        response1 = execute_python_code(invalid_code1)
        
        # Then: Should handle error gracefully
        assert not response1.get("success"), "Should fail for undefined variable"
        assert "error" in response1 or "stderr" in response1, "Should provide error information"
        
        # When: Execute code that will cause a SyntaxError
        invalid_code2 = "if True\n    print('missing colon')"
        response2 = execute_python_code(invalid_code2)
        
        # Then: Should handle syntax error gracefully
        assert not response2.get("success"), "Should fail for syntax error"
        
        # When: Execute valid code after errors
        valid_code = "result = 2 + 2; f'Result: {result}'"
        response3 = execute_python_code(valid_code)
        
        # Then: Should execute successfully
        assert response3.get("success"), f"Should recover and execute valid code: {response3}"
        result_output = get_execution_output(response3)
        assert "Result: 4" in result_output, "Should produce correct result"
    
    def test_given_file_operations_when_errors_occur_then_system_remains_stable(
        self, server_ready
    ):
        """
        Scenario: File operation error recovery
        
        Given: The system is handling file operations
        When: File-related errors occur (missing files, permission issues)
        Then: The system should remain stable and continue functioning
        And: Error messages should be informative and helpful
        """
        # Given: System is ready
        
        # When: Try to access a non-existent file
        missing_file_code = '''
import csv
try:
    with open("/uploads/nonexistent_file.csv", 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    result = "File read successfully"
except FileNotFoundError as e:
    result = f"Expected error: File not found - {type(e).__name__}"
except Exception as e:
    result = f"Unexpected error: {type(e).__name__}"

result
'''
        
        response1 = execute_python_code(missing_file_code)
        
        # Then: Should handle missing file gracefully
        assert response1.get("success"), f"Should execute error handling code: {response1}"
        output1 = get_execution_output(response1)
        assert "Expected error" in output1, "Should catch FileNotFoundError appropriately"
        
        # When: Execute normal operation after error
        normal_code = "import os; f'Current working directory: {os.getcwd()}'"
        response2 = execute_python_code(normal_code)
        
        # Then: Should continue working normally
        assert response2.get("success"), f"Should work normally after file error: {response2}"
        output2 = get_execution_output(response2)
        assert "Current working directory" in output2, "Should provide directory info"


# ==================== EXECUTION CONTEXT ISOLATION TESTS ====================

class TestExecutionContextIsolation:
    """Test that execution contexts are properly isolated between requests."""
    
    def test_given_variable_definition_when_separate_execution_then_context_isolated(
        self, server_ready
    ):
        """
        Scenario: Execution context isolation between requests
        
        Given: I define a variable in one execution context
        When: I try to access it in a separate execution request
        Then: The variable should not be accessible (contexts are isolated)
        And: Each execution should have a clean, independent environment
        """
        # Given: Define a variable in first execution
        define_code = "isolation_test_variable = 'should_not_persist'"
        response1 = execute_python_code(define_code)
        
        assert response1.get("success"), f"Variable definition should succeed: {response1}"
        
        # When: Try to access the variable in a separate execution
        access_code = """
try:
    # Try to access the variable from previous execution
    value = isolation_test_variable
    result = f"Variable accessible: {value}"
except NameError:
    result = "Variable not accessible - contexts are isolated"
except Exception as e:
    result = f"Unexpected error: {type(e).__name__}"

result
"""
        
        response2 = execute_python_code(access_code)
        
        # Then: Variable should not be accessible
        assert response2.get("success"), f"Context check should execute: {response2}"
        output2 = get_execution_output(response2)
        assert "Variable not accessible" in output2, \
            "Execution contexts should be isolated between requests"
    
    def test_given_import_statements_when_multiple_executions_then_imports_available(
        self, server_ready
    ):
        """
        Scenario: Standard library imports should be available in each execution
        
        Given: Standard Python libraries should be available
        When: I import and use standard libraries in separate executions
        Then: Each execution should have access to standard libraries
        And: Import statements should work consistently across executions
        """
        # Given & When: Test multiple standard library imports in separate executions
        import_tests = [
            "import os; f'OS name: {os.name}'",
            "import json; f'JSON loaded: {bool(json)}'",
            "import math; f'Pi value: {round(math.pi, 2)}'",
            "from pathlib import Path; f'Path available: {bool(Path)}'",
            "import datetime; f'Current year type: {type(datetime.datetime.now().year).__name__}'"
        ]
        
        for i, code in enumerate(import_tests):
            response = execute_python_code(code)
            
            # Then: Each import should work
            assert response.get("success"), f"Import test {i+1} should succeed: {response}"
            
            output = get_execution_output(response)
            
            # Verify expected outputs
            if "OS name" in code:
                assert "OS name:" in output, "Should show OS information"
            elif "JSON loaded" in code:
                assert "JSON loaded: True" in output, "Should confirm JSON is available"
            elif "Pi value" in code:
                assert "Pi value: 3.14" in output, "Should show correct Pi value"
            elif "Path available" in code:
                assert "Path available: True" in output, "Should confirm Path is available"
            elif "Current year" in code:
                assert "int" in output, "Should show year as integer type"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
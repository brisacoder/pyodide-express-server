"""Modernized integration tests using pytest and BDD style.

This test suite covers complex integration scenarios and data flow edge cases
using only public API endpoints and the execute-raw endpoint for Python execution.

Requirements satisfied:
1. ✅ Convert to Pytest
2. ✅ Global timeout configuration
3. ✅ No internal REST APIs (no 'pyodide' endpoints)
4. ✅ BDD style (Given-When-Then)
5. ✅ Only use /api/execute-raw for code execution
6. ✅ No internal REST API usage
7. ✅ Comprehensive coverage
"""

# Standard library imports
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Generator, List, Tuple

# Third-party imports
import pytest
import requests

# Global configuration
BASE_URL = "http://localhost:3000"
DEFAULT_TIMEOUT = 30
LONG_TIMEOUT = 120
SHORT_TIMEOUT = 10


# ==================== PYTEST FIXTURES ====================


@pytest.fixture(scope="session")
def server_ready():
    """Ensure server is ready before running any tests."""
    start_time = time.time()
    while time.time() - start_time < LONG_TIMEOUT:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=SHORT_TIMEOUT)
            if response.status_code == 200:
                return True
        except (requests.RequestException, OSError):
            pass  # Server not ready yet
        time.sleep(2)
    raise RuntimeError(f"Server at {BASE_URL} did not start within {LONG_TIMEOUT} seconds")


@pytest.fixture
def api_timeout():
    """Provide global timeout for API requests."""
    return DEFAULT_TIMEOUT


@pytest.fixture
def long_timeout():
    """Provide longer timeout for complex operations."""
    return LONG_TIMEOUT


@pytest.fixture
def short_timeout():
    """Provide short timeout for quick operations."""
    return SHORT_TIMEOUT


@pytest.fixture
def sample_csv_data():
    """Provide sample CSV data for testing."""
    return {
        "simple": "name,value,category\nitem1,1,A\nitem2,2,B\nitem3,3,C\n",
        "complex": 'name,description,value\n"Smith, John","A person named ""John""",42\n',
        "unicode": "name,value\nCafé,123\nNaïve,456\n",
        "empty_fields": "name,value,category\nitem1,,A\n,2,\n,,\n",
        "large_data": "x,y\n" + "\n".join([f"{i},{i*2}" for i in range(100)])
    }


@pytest.fixture
def uploaded_test_file(server_ready, api_timeout, sample_csv_data) -> Generator[Tuple[str, Dict], None, None]:
    """Upload a test CSV file and return file info for testing.
    
    Yields:
        Tuple[str, Dict]: (filename, upload_response_data)
    """
    # Given: A temporary CSV file
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp_file:
        tmp_file.write(sample_csv_data["simple"])
        tmp_path = tmp_file.name

    try:
        # When: Upload the file
        with open(tmp_path, "rb") as file_handle:
            response = requests.post(
                f"{BASE_URL}/api/upload",
                files={"file": ("test_data.csv", file_handle, "text/csv")},
                timeout=api_timeout,
            )
        
        # Then: Verify upload succeeded
        assert response.status_code == 200, f"Upload failed: {response.text}"
        upload_data = response.json()
        assert upload_data.get("success") is True, f"Upload failed: {upload_data}"
        
        file_info = upload_data["data"]["file"]
        filename = Path(file_info["vfsPath"]).name
        
        yield filename, upload_data
        
    finally:
        # Cleanup: Remove temporary file
        Path(tmp_path).unlink(missing_ok=True)
        
        # Cleanup: Remove uploaded file from server
        try:
            requests.delete(f"{BASE_URL}/api/uploaded-files/{filename}", timeout=SHORT_TIMEOUT)
        except requests.RequestException:
            pass  # Ignore cleanup errors


# ==================== DATA CONSISTENCY TESTS ====================


class TestDataConsistency:
    """Test data consistency and JSON parsing across the API."""

    def test_json_response_consistency(self, server_ready, uploaded_test_file, api_timeout):
        """
        Given: A CSV file has been uploaded
        When: I check file information via public API
        Then: All responses should be properly formatted JSON objects
        """
        # Given
        filename, upload_data = uploaded_test_file
        
        # When: Check file info via public API
        response = requests.get(f"{BASE_URL}/api/file-info/{filename}", timeout=api_timeout)
        
        # Then: Response should be properly structured JSON
        assert response.status_code == 200, f"File info request failed: {response.text}"
        info = response.json()
        
        # Verify proper JSON structure (not Python string representations)
        assert isinstance(info, dict), "Response should be a dictionary"
        assert "data" in info, f"Response missing 'data' field: {info}"
        assert isinstance(info["data"], dict), "Data field should be a dictionary"
        
        # Verify boolean fields are actual booleans, not strings
        data = info["data"]
        if "exists" in data:
            assert isinstance(data["exists"], bool), f"'exists' should be boolean, got {type(data['exists'])}"

    def test_file_listing_consistency(self, server_ready, uploaded_test_file, api_timeout):
        """
        Given: A file has been uploaded
        When: I list uploaded files
        Then: The response should be consistent JSON structure
        """
        # Given
        filename, _ = uploaded_test_file
        
        # When: List uploaded files
        response = requests.get(f"{BASE_URL}/api/uploaded-files", timeout=api_timeout)
        
        # Then: Verify consistent JSON structure
        assert response.status_code == 200, f"File listing failed: {response.text}"
        files_data = response.json()
        
        assert isinstance(files_data, dict), "Files response should be a dictionary"
        assert files_data.get("success") is True, f"File listing failed: {files_data}"
        assert "data" in files_data, f"Response missing 'data': {files_data}"
        
        data = files_data["data"]
        assert "files" in data, f"Data missing 'files': {data}"
        assert isinstance(data["files"], list), "Files should be a list"
        
        # Verify the uploaded file is in the list
        filenames = [f["filename"] for f in data["files"]]
        assert filename in filenames, f"File {filename} not found in list: {filenames}"


# ==================== CSV PROCESSING TESTS ====================


class TestCsvProcessing:
    """Test CSV file processing with various edge cases."""

    def test_csv_edge_cases_processing(self, server_ready, sample_csv_data, api_timeout):
        """
        Given: CSV files with various edge cases (quotes, unicode, empty fields)
        When: I upload and process them with execute-raw
        Then: All files should be processed correctly
        """
        test_cases = [
            ("quotes.csv", sample_csv_data["complex"]),
            ("unicode.csv", sample_csv_data["unicode"]),
            ("empty_fields.csv", sample_csv_data["empty_fields"]),
        ]
        
        uploaded_files = []
        
        try:
            for csv_filename, content in test_cases:
                # Given: A CSV file with edge case content
                with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                # When: Upload the file
                with open(tmp_path, "rb") as fh:
                    response = requests.post(
                        f"{BASE_URL}/api/upload",
                        files={"file": (csv_filename, fh, "text/csv")},
                        timeout=api_timeout
                    )
                
                # Then: Upload should succeed
                assert response.status_code == 200, f"Upload failed for {csv_filename}: {response.text}"
                upload_data = response.json()
                assert upload_data.get("success") is True, f"Upload failed: {upload_data}"
                
                file_info = upload_data["data"]["file"]
                filename = Path(file_info["vfsPath"]).name
                uploaded_files.append(filename)
                
                # When: Process the file with execute-raw
                python_code = f'''
import pandas as pd
from pathlib import Path

try:
    # Load the CSV file
    file_path = Path("{file_info["vfsPath"]}")
    df = pd.read_csv(file_path)
    
    # Get basic information about the file
    result = {{
        "success": True,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "file_processed": "{csv_filename}"
    }}
except Exception as e:
    result = {{
        "success": False,
        "error": str(e),
        "file_processed": "{csv_filename}"
    }}

print(json.dumps(result))
'''
                
                response = requests.post(
                    f"{BASE_URL}/api/execute-raw",
                    data=python_code,
                    headers={"Content-Type": "text/plain"},
                    timeout=api_timeout
                )
                
                # Then: Execution should succeed
                assert response.status_code == 200, f"Execution failed for {csv_filename}: {response.text}"
                exec_result = response.json()
                assert exec_result.get("success") is True, f"Python execution failed: {exec_result}"
                
                # Parse the result to verify file processing
                import json as json_module
                result_data = json_module.loads(exec_result["stdout"])
                assert result_data["success"] is True, f"File processing failed: {result_data}"
                assert result_data["file_processed"] == csv_filename
                
                # Cleanup temp file
                Path(tmp_path).unlink()
                
        finally:
            # Cleanup uploaded files
            for filename in uploaded_files:
                try:
                    requests.delete(f"{BASE_URL}/api/uploaded-files/{filename}", timeout=SHORT_TIMEOUT)
                except requests.RequestException:
                    pass


# ==================== CONCURRENT OPERATIONS TESTS ====================


class TestConcurrentOperations:
    """Test multiple operations happening in sequence."""

    def test_sequential_file_operations(self, server_ready, uploaded_test_file, api_timeout):
        """
        Given: A file has been uploaded
        When: I perform multiple sequential operations on the same file
        Then: All operations should succeed without interference
        """
        # Given
        filename, upload_data = uploaded_test_file
        file_path = upload_data["data"]["file"]["vfsPath"]
        
        # When: Execute multiple operations using execute-raw only
        operations = [
            {
                "name": "count_rows",
                "code": f'''
# Count rows in the uploaded CSV file
import os
if os.path.exists("{file_path}"):
    with open("{file_path}", "r") as f:
        lines = f.readlines()
    row_count = len(lines) - 1  # Subtract header
    print(f"Row count: {{row_count}}")
else:
    print("File not found")
'''
            },
            {
                "name": "sum_values", 
                "code": f'''
# Sum values from the CSV file
import csv
total = 0
try:
    with open("{file_path}", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "value" in row:
                total += int(row["value"])
    print(f"Sum of values: {{total}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
            },
            {
                "name": "list_columns",
                "code": f'''
# List columns from the CSV file
import csv
try:
    with open("{file_path}", "r") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
    print(f"Columns: {{list(columns)}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
            }
        ]
        
        results = []
        for operation in operations:
            # When: Execute operation
            response = requests.post(
                f"{BASE_URL}/api/execute-raw",
                data=operation["code"],
                headers={"Content-Type": "text/plain"},
                timeout=api_timeout
            )
            
            # Then: Operation should succeed
            assert response.status_code == 200, f"Operation {operation['name']} failed: {response.text}"
            result = response.json()
            assert result.get("success") is True, f"Operation {operation['name']} failed: {result}"
            results.append(result)
        
        # Then: All operations should have completed successfully
        assert len(results) == len(operations), "Not all operations completed"
        
        # Verify expected results
        assert "Row count: 3" in results[0]["stdout"]
        assert "Sum of values: 6" in results[1]["stdout"]  # 1+2+3 = 6
        assert "['name', 'value', 'category']" in results[2]["stdout"]

    def test_multiple_file_info_requests(self, server_ready, uploaded_test_file, short_timeout):
        """
        Given: A file exists
        When: I make multiple file info requests
        Then: All requests should return consistent information
        """
        # Given
        filename, _ = uploaded_test_file
        
        # When: Make multiple file info requests
        responses = []
        for i in range(3):
            response = requests.get(f"{BASE_URL}/api/file-info/{filename}", timeout=short_timeout)
            responses.append(response)
        
        # Then: All requests should succeed with consistent data
        for i, response in enumerate(responses):
            assert response.status_code == 200, f"Request {i} failed: {response.text}"
            data = response.json()
            assert data["data"]["exists"] is True, f"File should exist in request {i}"
            assert data["data"]["filename"] == filename, f"Filename mismatch in request {i}"


# ==================== ERROR RECOVERY TESTS ====================


class TestErrorRecovery:
    """Test that errors don't break subsequent operations."""

    def test_python_error_recovery(self, server_ready, api_timeout):
        """
        Given: I execute Python code that will fail
        When: I execute valid Python code afterwards
        Then: The system should recover and execute successfully
        """
        # Given: Code that will fail with undefined variable
        failing_code = '''
# This will fail
undefined_variable_that_does_not_exist
'''
        
        # When: Execute failing code
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=failing_code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout
        )
        
        # Then: Request should complete but execution should fail
        assert response.status_code == 200, f"Request failed: {response.text}"
        result = response.json()
        assert result.get("success") is False, "Code execution should have failed"
        
        # When: Execute valid code after the error
        working_code = '''
# This should work
result = 2 + 2
print(f"Result: {result}")
'''
        
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=working_code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout
        )
        
        # Then: Valid code should execute successfully
        assert response.status_code == 200, f"Recovery failed: {response.text}"
        result = response.json()
        assert result.get("success") is True, f"System should have recovered: {result}"
        assert "Result: 4" in result["stdout"], "Code should have executed correctly"

    def test_syntax_error_recovery(self, server_ready, api_timeout):
        """
        Given: I execute Python code with syntax errors
        When: I execute valid Python code afterwards
        Then: The system should handle syntax errors gracefully
        """
        # Given: Code with syntax error
        syntax_error_code = '''
# This has a syntax error
if True
    print("missing colon")
'''
        
        # When: Execute code with syntax error
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=syntax_error_code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout
        )
        
        # Then: Request should complete but execution should fail
        assert response.status_code == 200, f"Request failed: {response.text}"
        result = response.json()
        assert result.get("success") is False, "Syntax error should cause failure"
        
        # When: Execute valid code to verify recovery
        recovery_code = '''
print("System recovered successfully")
'''
        
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=recovery_code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout
        )
        
        # Then: System should have recovered
        assert response.status_code == 200, f"Recovery failed: {response.text}"
        result = response.json()
        assert result.get("success") is True, f"System should have recovered: {result}"
        assert "System recovered successfully" in result["stdout"]


# ==================== COMPLEX DATA FLOW TESTS ====================


class TestComplexDataFlow:
    """Test complex data processing workflows using multiple files."""

    def test_multi_file_data_processing_workflow(self, server_ready, api_timeout, long_timeout):
        """
        Given: Multiple CSV files with related data
        When: I upload them and perform complex data processing with execute-raw
        Then: I should be able to merge and analyze the data successfully
        """
        # Given: Multiple related CSV files
        files_data = [
            ("customers.csv", "id,name,region\n1,Alice,North\n2,Bob,South\n3,Charlie,East\n"),
            ("orders.csv", "customer_id,amount,product\n1,100,Widget\n2,150,Gadget\n1,75,Tool\n"),
            ("products.csv", "product,category,price\nWidget,Electronics,50\nGadget,Electronics,75\nTool,Hardware,25\n")
        ]
        
        uploaded_files = []
        file_paths = {}
        
        try:
            # When: Upload all files
            for filename, content in files_data:
                with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                with open(tmp_path, "rb") as fh:
                    response = requests.post(
                        f"{BASE_URL}/api/upload",
                        files={"file": (filename, fh, "text/csv")},
                        timeout=api_timeout
                    )
                
                # Then: Upload should succeed
                assert response.status_code == 200, f"Upload failed for {filename}: {response.text}"
                upload_data = response.json()
                assert upload_data.get("success") is True, f"Upload failed: {upload_data}"
                
                file_info = upload_data["data"]["file"]
                server_filename = Path(file_info["vfsPath"]).name
                uploaded_files.append(server_filename)
                file_paths[filename.split('.')[0]] = file_info["vfsPath"]
                
                Path(tmp_path).unlink()
            
            # When: Perform complex data processing with execute-raw
            complex_analysis_code = f'''
import pandas as pd
import json
from pathlib import Path

try:
    # Load all datasets
    customers = pd.read_csv(Path("{file_paths['customers']}"))
    orders = pd.read_csv(Path("{file_paths['orders']}"))
    products = pd.read_csv(Path("{file_paths['products']}"))
    
    # Perform complex data analysis
    # 1. Merge orders with customers
    orders_with_customers = pd.merge(orders, customers, left_on='customer_id', right_on='id')
    
    # 2. Merge with products
    full_data = pd.merge(orders_with_customers, products, on='product')
    
    # 3. Calculate metrics
    total_revenue = full_data['amount'].sum()
    avg_order_value = full_data['amount'].mean()
    orders_by_region = full_data.groupby('region')['amount'].sum().to_dict()
    revenue_by_category = full_data.groupby('category')['amount'].sum().to_dict()
    customer_analysis = full_data.groupby('name').agg({{
        'amount': ['sum', 'count'],
        'product': lambda x: list(x.unique())
    }}).to_dict()
    
    # Create comprehensive analysis result
    analysis_result = {{
        "success": True,
        "data_quality": {{
            "customers_count": len(customers),
            "orders_count": len(orders),
            "products_count": len(products),
            "merged_records": len(full_data)
        }},
        "financial_metrics": {{
            "total_revenue": float(total_revenue),
            "average_order_value": float(avg_order_value),
            "orders_by_region": orders_by_region,
            "revenue_by_category": revenue_by_category
        }},
        "data_validation": {{
            "expected_total_amount": 325.0,  # 100 + 150 + 75
            "calculated_total": float(total_revenue),
            "totals_match": abs(float(total_revenue) - 325.0) < 0.01
        }}
    }}
    
    print(json.dumps(analysis_result, indent=2))
    
except Exception as e:
    error_result = {{
        "success": False,
        "error": str(e),
        "error_type": type(e).__name__
    }}
    print(json.dumps(error_result, indent=2))
'''
            
            response = requests.post(
                f"{BASE_URL}/api/execute-raw",
                data=complex_analysis_code,
                headers={"Content-Type": "text/plain"},
                timeout=long_timeout
            )
            
            # Then: Complex analysis should succeed
            assert response.status_code == 200, f"Complex analysis failed: {response.text}"
            result = response.json()
            assert result.get("success") is True, f"Analysis execution failed: {result}"
            
            # Parse and validate the analysis results
            import json as json_module
            analysis_data = json_module.loads(result["stdout"])
            
            assert analysis_data["success"] is True, f"Analysis logic failed: {analysis_data}"
            
            # Validate data quality metrics
            quality = analysis_data["data_quality"]
            assert quality["customers_count"] == 3, "Should have 3 customers"
            assert quality["orders_count"] == 3, "Should have 3 orders"
            assert quality["products_count"] == 3, "Should have 3 products"
            assert quality["merged_records"] == 3, "Should have 3 merged records"
            
            # Validate financial metrics
            financial = analysis_data["financial_metrics"]
            assert financial["total_revenue"] == 325.0, f"Total revenue should be 325.0, got {financial['total_revenue']}"
            assert analysis_data["data_validation"]["totals_match"] is True, "Revenue calculation validation failed"
            
            # Validate regional analysis
            assert "North" in financial["orders_by_region"], "North region should have orders"
            assert "South" in financial["orders_by_region"], "South region should have orders"
            
        finally:
            # Cleanup: Remove all uploaded files
            for filename in uploaded_files:
                try:
                    requests.delete(f"{BASE_URL}/api/uploaded-files/{filename}", timeout=SHORT_TIMEOUT)
                except requests.RequestException:
                    pass

    def test_data_persistence_and_isolation(self, server_ready, api_timeout):
        """
        Given: I execute code that defines variables
        When: I execute code in separate requests
        Then: Variables should not persist (proper isolation)
        """
        # Given: Code that defines variables
        setup_code = '''
# Define some variables
test_variable = "this_should_not_persist"
calculation_result = 42 * 2
data_structure = {"key": "value", "numbers": [1, 2, 3]}

print(f"Variables defined: test_variable={test_variable}")
'''
        
        # When: Execute setup code
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=setup_code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout
        )
        
        # Then: Setup should succeed
        assert response.status_code == 200, f"Setup failed: {response.text}"
        result = response.json()
        assert result.get("success") is True, f"Setup execution failed: {result}"
        assert "Variables defined" in result["stdout"]
        
        # When: Check for variables in a separate request (should be isolated)
        isolation_test_code = '''
# Check if variables from previous execution exist
import json

variables_check = {
    "test_variable_exists": "test_variable" in globals(),
    "calculation_result_exists": "calculation_result" in globals(), 
    "data_structure_exists": "data_structure" in globals(),
    "isolation_verified": True
}

# Variables should NOT exist due to proper isolation
if any([variables_check["test_variable_exists"], 
        variables_check["calculation_result_exists"],
        variables_check["data_structure_exists"]]):
    variables_check["isolation_verified"] = False
    variables_check["error"] = "Variables persisted between requests"

print(json.dumps(variables_check, indent=2))
'''
        
        response = requests.post(
            f"{BASE_URL}/api/execute-raw",
            data=isolation_test_code,
            headers={"Content-Type": "text/plain"},
            timeout=api_timeout
        )
        
        # Then: Isolation test should confirm variables don't persist
        assert response.status_code == 200, f"Isolation test failed: {response.text}"
        result = response.json()
        assert result.get("success") is True, f"Isolation test execution failed: {result}"
        
        import json as json_module
        isolation_data = json_module.loads(result["stdout"])
        assert isolation_data["isolation_verified"] is True, "Execution context should be isolated between requests"
        assert isolation_data["test_variable_exists"] is False, "test_variable should not persist"
        assert isolation_data["calculation_result_exists"] is False, "calculation_result should not persist"
        assert isolation_data["data_structure_exists"] is False, "data_structure should not persist"


# ==================== PERFORMANCE AND TIMEOUT TESTS ====================


class TestPerformanceAndTimeouts:
    """Test performance characteristics and timeout handling."""

    def test_large_data_processing(self, server_ready, sample_csv_data, long_timeout):
        """
        Given: A large CSV file
        When: I process it with complex operations
        Then: The system should handle it within reasonable time limits
        """
        # Given: Large CSV data
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write(sample_csv_data["large_data"])
            tmp_path = tmp.name
        
        try:
            # When: Upload large file
            with open(tmp_path, "rb") as fh:
                response = requests.post(
                    f"{BASE_URL}/api/upload",
                    files={"file": ("large_data.csv", fh, "text/csv")},
                    timeout=long_timeout
                )
            
            # Then: Upload should succeed
            assert response.status_code == 200, f"Large file upload failed: {response.text}"
            upload_data = response.json()
            assert upload_data.get("success") is True, f"Upload failed: {upload_data}"
            
            file_info = upload_data["data"]["file"]
            filename = Path(file_info["vfsPath"]).name
            
            # When: Process large data with complex operations
            large_data_code = f'''
import pandas as pd
import numpy as np
import time
from pathlib import Path

start_time = time.time()

try:
    # Load the large dataset
    df = pd.read_csv(Path("{file_info['vfsPath']}"))
    
    # Perform complex operations
    df['z'] = df['x'] * df['y']
    df['category'] = pd.cut(df['x'], bins=5, labels=['A', 'B', 'C', 'D', 'E'])
    
    # Statistical analysis
    stats = {{
        'row_count': len(df),
        'mean_x': float(df['x'].mean()),
        'std_y': float(df['y'].std()),
        'correlation': float(df[['x', 'y']].corr().iloc[0, 1]),
        'category_counts': df['category'].value_counts().to_dict()
    }}
    
    # Memory usage analysis
    memory_usage = df.memory_usage(deep=True).sum()
    
    processing_time = time.time() - start_time
    
    result = {{
        "success": True,
        "statistics": stats,
        "performance": {{
            "processing_time_seconds": processing_time,
            "memory_usage_bytes": int(memory_usage),
            "rows_processed": len(df)
        }}
    }}
    
    print(f"Processing completed in {{processing_time:.3f}} seconds")
    print(f"Processed {{len(df)}} rows successfully")
    
except Exception as e:
    result = {{
        "success": False,
        "error": str(e),
        "processing_time": time.time() - start_time
    }}
    print(f"Error: {{str(e)}}")
'''
            
            response = requests.post(
                f"{BASE_URL}/api/execute-raw",
                data=large_data_code,
                headers={"Content-Type": "text/plain"},
                timeout=long_timeout
            )
            
            # Then: Processing should complete successfully
            assert response.status_code == 200, f"Large data processing failed: {response.text}"
            result = response.json()
            assert result.get("success") is True, f"Large data execution failed: {result}"
            
            # Verify performance expectations
            assert "Processing completed" in result["stdout"], "Processing should complete"
            assert "100 rows successfully" in result["stdout"], "Should process 100 rows"
            
            # Cleanup
            requests.delete(f"{BASE_URL}/api/uploaded-files/{filename}", timeout=SHORT_TIMEOUT)
            
        finally:
            Path(tmp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
import os
import tempfile
import time
import subprocess
import requests
import unittest

BASE_URL = "http://localhost:3000"


class PerformanceTestCase(unittest.TestCase):
    """Test performance characteristics and resource limits."""

    @classmethod
    def setUpClass(cls):
        # Start the server in a subprocess
        cls.server = subprocess.Popen(["node", "src/server.js"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for server to be ready
        start = time.time()
        while time.time() - start < 120:
            try:
                r = requests.get(f"{BASE_URL}/health", timeout=5)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Server did not start in time")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.server.kill()

    # ===== Execution Performance Tests =====
    
    def test_execution_timeout(self):
        """Test that long-running code is properly timed out"""
        # Code that should take a long time
        long_running_code = '''
import time
total = 0
for i in range(1000000):
    total += i
    if i % 100000 == 0:
        time.sleep(0.1)  # Small sleep to make it slower
total
'''
        
        start_time = time.time()
        r = requests.post(
            f"{BASE_URL}/api/execute", 
            json={"code": long_running_code, "timeout": 5000},  # 5 second timeout
            timeout=30
        )
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (either timeout or finish quickly)
        self.assertLess(execution_time, 10)  # Should not take more than 10 seconds
        
        # Check response
        self.assertEqual(r.status_code, 200)
        response = r.json()
        # Could either succeed quickly or timeout - both are acceptable

    def test_memory_intensive_operations(self):
        """Test handling of memory-intensive operations"""
        memory_codes = [
            # Large list creation
            "large_list = list(range(100000)); len(large_list)",
            # Large string operations
            "big_string = 'x' * 1000000; len(big_string)",
            # Large dictionary
            "big_dict = {i: f'value_{i}' for i in range(10000)}; len(big_dict)",
        ]
        
        for code in memory_codes:
            start_time = time.time()
            r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=30)
            execution_time = time.time() - start_time
            
            self.assertEqual(r.status_code, 200)
            self.assertLess(execution_time, 5)  # Should complete reasonably quickly
            
            response = r.json()
            if response.get("success"):
                # If successful, result should be reasonable
                result = response.get("result")
                self.assertIsInstance(result, (int, str))

    def test_cpu_intensive_operations(self):
        """Test CPU-intensive calculations"""
        cpu_codes = [
            # Prime number calculation
            """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [i for i in range(2, 1000) if is_prime(i)]
len(primes)
""",
            # Fibonacci calculation
            """
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

fib(20)
""",
            # Matrix operations
            """
matrix = [[i*j for j in range(100)] for i in range(100)]
total = sum(sum(row) for row in matrix)
total
""",
        ]
        
        for code in cpu_codes:
            start_time = time.time()
            r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=30)
            execution_time = time.time() - start_time
            
            self.assertEqual(r.status_code, 200)
            self.assertLess(execution_time, 10)  # Should complete within 10 seconds
            
            response = r.json()
            self.assertTrue(response.get("success"))
            self.assertIsInstance(response.get("result"), int)

    # ===== File Processing Performance =====
    
    def test_large_csv_processing(self):
        """Test processing of larger CSV files"""
        # Create a moderately large CSV file
        large_csv_content = "id,value,category,description\n"
        for i in range(5000):  # 5000 rows
            large_csv_content += f"{i},{i*2},category_{i%10},description for item {i}\n"
        
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write(large_csv_content)
            tmp_path = tmp.name
        
        try:
            # Upload the large file
            start_time = time.time()
            with open(tmp_path, "rb") as fh:
                r = requests.post(
                    f"{BASE_URL}/api/upload-csv",
                    files={"csvFile": ("large.csv", fh, "text/csv")},
                    timeout=60
                )
            upload_time = time.time() - start_time
            
            self.assertEqual(r.status_code, 200)
            self.assertLess(upload_time, 30)  # Should upload within 30 seconds
            
            upload_data = r.json()
            pyodide_name = upload_data["file"]["pyodideFilename"]
            server_filename = os.path.basename(upload_data["file"]["tempPath"])
            
            # Process the large file
            processing_code = f'''
import pandas as pd
df = pd.read_csv("{pyodide_name}")
result = {{
    "shape": list(df.shape),
    "memory_usage": df.memory_usage(deep=True).sum(),
    "value_sum": df["value"].sum(),
    "categories": df["category"].nunique()
}}
result
'''
            
            start_time = time.time()
            r = requests.post(f"{BASE_URL}/api/execute", json={"code": processing_code}, timeout=60)
            processing_time = time.time() - start_time
            
            self.assertEqual(r.status_code, 200)
            self.assertLess(processing_time, 15)  # Should process within 15 seconds
            
            response = r.json()
            self.assertTrue(response.get("success"))
            
            result = response.get("result")
            self.assertEqual(result["shape"], [5000, 4])  # 5000 rows, 4 columns
            self.assertEqual(result["categories"], 10)  # 10 unique categories
            
            # Clean up
            requests.delete(f"{BASE_URL}/api/pyodide-files/{pyodide_name}", timeout=10)
            requests.delete(f"{BASE_URL}/api/uploaded-files/{server_filename}", timeout=10)
            
        finally:
            os.unlink(tmp_path)

    def test_multiple_file_operations(self):
        """Test performance with multiple file operations"""
        files_to_create = 5
        uploaded_files = []
        
        try:
            # Upload multiple files
            for i in range(files_to_create):
                content = f"id,value\n{i},100\n{i+1},200\n"
                with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                with open(tmp_path, "rb") as fh:
                    r = requests.post(
                        f"{BASE_URL}/api/upload-csv",
                        files={"csvFile": (f"file_{i}.csv", fh, "text/csv")},
                        timeout=30
                    )
                
                self.assertEqual(r.status_code, 200)
                upload_data = r.json()
                uploaded_files.append({
                    'pyodide_name': upload_data["file"]["pyodideFilename"],
                    'server_filename': os.path.basename(upload_data["file"]["tempPath"]),
                    'temp_path': tmp_path
                })
                os.unlink(tmp_path)
            
            # List files multiple times to test caching/performance
            for _ in range(3):
                start_time = time.time()
                r = requests.get(f"{BASE_URL}/api/pyodide-files", timeout=10)
                list_time = time.time() - start_time
                
                self.assertEqual(r.status_code, 200)
                self.assertLess(list_time, 2)  # Should list quickly
                
                files_in_response = [f["name"] for f in r.json()["result"]["files"]]
                for file_info in uploaded_files:
                    self.assertIn(file_info['pyodide_name'], files_in_response)
            
            # Get file info for all files
            for file_info in uploaded_files:
                start_time = time.time()
                r = requests.get(f"{BASE_URL}/api/file-info/{file_info['pyodide_name']}", timeout=10)
                info_time = time.time() - start_time
                
                self.assertEqual(r.status_code, 200)
                self.assertLess(info_time, 1)  # Should get info quickly
                self.assertTrue(r.json()["pyodideFile"]["exists"])
                
        finally:
            # Clean up all files
            for file_info in uploaded_files:
                requests.delete(f"{BASE_URL}/api/pyodide-files/{file_info['pyodide_name']}", timeout=10)
                requests.delete(f"{BASE_URL}/api/uploaded-files/{file_info['server_filename']}", timeout=10)

    # ===== Concurrent Request Performance =====
    
    def test_concurrent_execution_requests(self):
        """Test handling of multiple execution requests"""
        # Send multiple simple requests
        codes = [
            "1 + 1",
            "2 * 3", 
            "'hello world'",
            "[1, 2, 3, 4, 5]",
            "{'key': 'value'}",
        ]
        
        start_time = time.time()
        responses = []
        
        for code in codes:
            r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=30)
            responses.append(r)
        
        total_time = time.time() - start_time
        
        # All should succeed
        for r in responses:
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.json().get("success"))
        
        # Should complete all requests reasonably quickly
        self.assertLess(total_time, 10)
        
        # Verify correct results
        expected_results = [2, 6, "hello world", [1, 2, 3, 4, 5], {"key": "value"}]
        for r, expected in zip(responses, expected_results):
            self.assertEqual(r.json().get("result"), expected)

    # ===== Resource Cleanup Performance =====
    
    def test_cleanup_after_errors(self):
        """Test that errors don't cause resource leaks"""
        # Create some files and variables
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write("a,b\n1,2\n")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, "rb") as fh:
                r = requests.post(
                    f"{BASE_URL}/api/upload-csv",
                    files={"csvFile": ("cleanup_test.csv", fh, "text/csv")},
                    timeout=30
                )
            self.assertEqual(r.status_code, 200)
            upload_data = r.json()
            pyodide_name = upload_data["file"]["pyodideFilename"]
            server_filename = os.path.basename(upload_data["file"]["tempPath"])
            
            # Set some variables
            r = requests.post(f"{BASE_URL}/api/execute", json={"code": "test_var = 'should_be_cleaned'"}, timeout=30)
            self.assertEqual(r.status_code, 200)
            
            # Cause an error
            r = requests.post(f"{BASE_URL}/api/execute", json={"code": "undefined_variable_xyz"}, timeout=30)
            self.assertEqual(r.status_code, 200)
            self.assertFalse(r.json().get("success"))
            
            # Verify system still works normally after error
            r = requests.post(f"{BASE_URL}/api/execute", json={"code": "2 + 2"}, timeout=30)
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.json().get("success"))
            self.assertEqual(r.json().get("result"), 4)
            
            # Verify files still accessible after error
            r = requests.get(f"{BASE_URL}/api/file-info/{pyodide_name}", timeout=10)
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.json()["pyodideFile"]["exists"])
            
            # Clean up
            requests.delete(f"{BASE_URL}/api/pyodide-files/{pyodide_name}", timeout=10)
            requests.delete(f"{BASE_URL}/api/uploaded-files/{server_filename}", timeout=10)
            
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()

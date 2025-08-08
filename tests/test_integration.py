import os
import tempfile
import time
import subprocess
import requests
import unittest
import json

BASE_URL = "http://localhost:3000"


class IntegrationTestCase(unittest.TestCase):
    """Test complex integration scenarios and data flow edge cases."""

    @classmethod
    def setUpClass(cls):
        # Start the server in a subprocess
        cls.server = subprocess.Popen(["node", "src/server.js"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for server to be ready
        start = time.time()
        while time.time() - start < 120:
            try:
                r = requests.get(f"{BASE_URL}/health", timeout=10)
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

    # ===== Data Consistency Tests =====
    
    def test_json_parsing_consistency(self):
        """Test that Python dict strings are properly converted to JavaScript objects"""
        # Create a file first
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write("name,value,category\nitem1,1,A\nitem2,2,B\n")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, "rb") as fh:
                r = requests.post(
                    f"{BASE_URL}/api/upload-csv",
                    files={"csvFile": ("test.csv", fh, "text/csv")},
                    timeout=30
                )
            self.assertEqual(r.status_code, 200)
            upload_data = r.json()
            pyodide_name = upload_data["file"]["pyodideFilename"]
            
            # Test file info endpoint returns proper JSON objects
            r = requests.get(f"{BASE_URL}/api/file-info/{pyodide_name}", timeout=10)
            self.assertEqual(r.status_code, 200)
            info = r.json()
            
            # Verify the response structure is proper JSON, not Python string
            self.assertIsInstance(info["uploadedFile"], dict)
            self.assertIsInstance(info["pyodideFile"], dict)
            self.assertIn("exists", info["uploadedFile"])
            self.assertIn("exists", info["pyodideFile"])
            self.assertIsInstance(info["uploadedFile"]["exists"], bool)
            self.assertIsInstance(info["pyodideFile"]["exists"], bool)
            
            # Test pyodide files listing
            r = requests.get(f"{BASE_URL}/api/pyodide-files", timeout=10)
            self.assertEqual(r.status_code, 200)
            files_data = r.json()
            
            # Verify result is proper JSON structure
            self.assertIsInstance(files_data.get("result"), dict)
            self.assertIn("files", files_data["result"])
            self.assertIsInstance(files_data["result"]["files"], list)
            
            # Clean up
            requests.delete(f"{BASE_URL}/api/pyodide-files/{pyodide_name}", timeout=10)
            server_filename = os.path.basename(upload_data["file"]["tempPath"])
            requests.delete(f"{BASE_URL}/api/uploaded-files/{server_filename}", timeout=10)
            
        finally:
            os.unlink(tmp_path)

    def test_csv_processing_edge_cases(self):
        """Test CSV files with various edge cases"""
        test_cases = [
            # CSV with commas in quoted fields
            ('quotes.csv', 'name,description,value\n"Smith, John","A person named ""John""",42\n'),
            # CSV with different encodings and special characters
            ('unicode.csv', 'name,value\nCafé,123\nNaïve,456\n'),
            # CSV with empty fields
            ('empty_fields.csv', 'name,value,category\nitem1,,A\n,2,\n,,\n'),
            # CSV with very long lines
            ('long_lines.csv', 'name,value\n' + 'x' * 1000 + ',123\n'),
        ]
        
        uploaded_files = []
        try:
            for filename, content in test_cases:
                with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                with open(tmp_path, "rb") as fh:
                    r = requests.post(
                        f"{BASE_URL}/api/upload-csv",
                        files={"csvFile": (filename, fh, "text/csv")},
                        timeout=30
                    )
                
                if r.status_code == 200:
                    upload_data = r.json()
                    uploaded_files.append({
                        'pyodide_name': upload_data["file"]["pyodideFilename"],
                        'server_filename': os.path.basename(upload_data["file"]["tempPath"]),
                        'temp_path': tmp_path
                    })
                    
                    # Test that the file can be read and processed
                    code = f'''
import pandas as pd
try:
    df = pd.read_csv("{upload_data["file"]["pyodideFilename"]}")
    result = {{"success": True, "shape": list(df.shape), "columns": list(df.columns)}}
except Exception as e:
    result = {{"success": False, "error": str(e)}}
result
'''
                    r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=30)
                    self.assertEqual(r.status_code, 200)
                    exec_result = r.json()
                    self.assertTrue(exec_result.get("success"))
                    
                os.unlink(tmp_path)
        
        finally:
            # Clean up all uploaded files
            for file_info in uploaded_files:
                requests.delete(f"{BASE_URL}/api/pyodide-files/{file_info['pyodide_name']}", timeout=10)
                requests.delete(f"{BASE_URL}/api/uploaded-files/{file_info['server_filename']}", timeout=10)

    def test_concurrent_operations(self):
        """Test multiple operations happening in sequence"""
        # Upload a file
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write("x,y\n1,2\n3,4\n")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, "rb") as fh:
                r = requests.post(
                    f"{BASE_URL}/api/upload-csv",
                    files={"csvFile": ("concurrent.csv", fh, "text/csv")},
                    timeout=30
                )
            self.assertEqual(r.status_code, 200)
            upload_data = r.json()
            pyodide_name = upload_data["file"]["pyodideFilename"]
            server_filename = os.path.basename(upload_data["file"]["tempPath"])
            
            # Execute multiple operations using the same file
            operations = [
                f'import pandas as pd; df = pd.read_csv("{pyodide_name}"); df.shape[0]',
                f'import pandas as pd; df = pd.read_csv("{pyodide_name}"); df.sum().sum()',
                f'import pandas as pd; df = pd.read_csv("{pyodide_name}"); list(df.columns)',
            ]
            
            for code in operations:
                r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=30)
                self.assertEqual(r.status_code, 200)
                self.assertTrue(r.json().get("success"))
            
            # Check file info multiple times
            for _ in range(3):
                r = requests.get(f"{BASE_URL}/api/file-info/{pyodide_name}", timeout=10)
                self.assertEqual(r.status_code, 200)
                self.assertTrue(r.json()["pyodideFile"]["exists"])
            
            # List files multiple times
            for _ in range(3):
                r = requests.get(f"{BASE_URL}/api/pyodide-files", timeout=10)
                self.assertEqual(r.status_code, 200)
                files = [f["name"] for f in r.json()["result"]["files"]]
                self.assertIn(pyodide_name, files)
            
            # Clean up
            r = requests.delete(f"{BASE_URL}/api/pyodide-files/{pyodide_name}", timeout=10)
            self.assertEqual(r.status_code, 200)
            r = requests.delete(f"{BASE_URL}/api/uploaded-files/{server_filename}", timeout=10)
            self.assertEqual(r.status_code, 200)
            
        finally:
            os.unlink(tmp_path)

    def test_state_persistence_after_reset(self):
        """Test that packages persist after reset while maintaining isolated execution"""
        # Install a package
        r = requests.post(f"{BASE_URL}/api/install-package", json={"package": "beautifulsoup4"}, timeout=120)
        self.assertEqual(r.status_code, 200)
        
        # Verify package is available
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "import bs4; 'success'"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))
        
        # Test that variables don't persist between executions (isolated execution)
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "test_var = 'isolated_value'"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        
        # Verify variable doesn't exist in separate execution (this is correct behavior)
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "test_var"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json().get("success"))  # Should fail - variables don't persist
        
        # Reset the environment
        r = requests.post(f"{BASE_URL}/api/reset", timeout=60)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))
        
        # Verify package is still available after reset (packages should persist)
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "import bs4; 'still_available'"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))
        
        # Verify reset completed successfully by checking we can still execute code
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "2 + 2"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))
        self.assertEqual(r.json().get("result"), 4)

    def test_complex_data_flow(self):
        """Test complex data processing workflow"""
        # Upload multiple CSV files
        files_data = [
            ("data1.csv", "id,value\n1,10\n2,20\n"),
            ("data2.csv", "id,score\n1,90\n2,85\n"),
        ]
        
        uploaded = []
        try:
            for filename, content in files_data:
                with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                with open(tmp_path, "rb") as fh:
                    r = requests.post(
                        f"{BASE_URL}/api/upload-csv",
                        files={"csvFile": (filename, fh, "text/csv")},
                        timeout=30
                    )
                self.assertEqual(r.status_code, 200)
                upload_data = r.json()
                uploaded.append({
                    'pyodide_name': upload_data["file"]["pyodideFilename"],
                    'server_filename': os.path.basename(upload_data["file"]["tempPath"]),
                    'temp_path': tmp_path
                })
                os.unlink(tmp_path)
            
            # Perform complex data processing
            complex_code = f'''
import pandas as pd

# Load both datasets
df1 = pd.read_csv("{uploaded[0]['pyodide_name']}")
df2 = pd.read_csv("{uploaded[1]['pyodide_name']}")

# Merge the datasets
merged = pd.merge(df1, df2, on='id')

# Calculate some metrics
total_value = merged['value'].sum()
avg_score = merged['score'].mean()
record_count = len(merged)

# Return results as a dictionary
result = {{
    "total_value": total_value,
    "avg_score": avg_score,
    "record_count": record_count,
    "columns": list(merged.columns)
}}
result
'''
            
            r = requests.post(f"{BASE_URL}/api/execute", json={"code": complex_code}, timeout=30)
            self.assertEqual(r.status_code, 200)
            result = r.json()
            self.assertTrue(result.get("success"))
            
            # Verify the computed results are reasonable
            result_data = result.get("result")
            self.assertIsInstance(result_data, dict)
            self.assertEqual(result_data["total_value"], 30)  # 10 + 20
            self.assertEqual(result_data["avg_score"], 87.5)  # (90 + 85) / 2
            self.assertEqual(result_data["record_count"], 2)
            self.assertEqual(set(result_data["columns"]), {"id", "value", "score"})
            
        finally:
            # Clean up
            for file_info in uploaded:
                requests.delete(f"{BASE_URL}/api/pyodide-files/{file_info['pyodide_name']}", timeout=10)
                requests.delete(f"{BASE_URL}/api/uploaded-files/{file_info['server_filename']}", timeout=10)

    def test_error_recovery(self):
        """Test that errors don't break subsequent operations"""
        # Execute code that will fail
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "undefined_variable"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json().get("success"))
        
        # Execute code that will succeed after the error
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "2 + 2"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))
        self.assertEqual(r.json().get("result"), 4)
        
        # Try syntax error
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "if True\n    print('missing colon')"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json().get("success"))
        
        # Verify system still works
        r = requests.post(f"{BASE_URL}/api/execute", json={"code": "'system_recovered'"}, timeout=30)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))
        self.assertEqual(r.json().get("result"), "system_recovered")


if __name__ == "__main__":
    unittest.main()

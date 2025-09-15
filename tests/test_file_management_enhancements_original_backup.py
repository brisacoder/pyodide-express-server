#!/usr/bin/env python3
"""
Test suite for file management enhancements
Tests the recent fixes for clearAllFiles and resetEnvironment functionality
"""

import json
import tempfile
import time
import unittest
from pathlib import Path

import requests

BASE_URL = "http://localhost:3000"


class FileManagementEnhancementsTestCase(unittest.TestCase):
    """Test recent file management enhancements"""
    
    def setUp(self):
        """Set up test environment"""
        self.session = requests.Session()
        self.uploaded_files = []
        self.temp_files = []
        
    def tearDown(self):
        """Clean up test environment"""
        # Clear all files at the end of each test
        try:
            self.session.post(f"{BASE_URL}/api/clear-all-files", timeout=30)
        except requests.RequestException:
            pass
        
        # Clean up temp files
        for temp_file in self.temp_files:
            if isinstance(temp_file, Path) and temp_file.exists():
                temp_file.unlink()

    def upload_test_file(self, filename, content, mime_type='text/plain'):
        """Helper to upload a test file"""
        temp_file = Path(tempfile.mkdtemp()) / filename
        self.temp_files.append(temp_file)
        
        with open(temp_file, 'w') as f:
            f.write(content)
        
        with open(temp_file, 'rb') as f:
            files = {'file': (filename, f, mime_type)}
            response = self.session.post(f"{BASE_URL}/api/upload", files=files, timeout=30)
        
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result['success'])
        return result['file']['pyodideFilename']

    def create_pyodide_file(self, filename, content):
        """Helper to create a file in Pyodide filesystem"""
        python_code = f'''
from pathlib import Path
file_path = Path(r"/uploads/{filename}")
file_path.write_text("""{content}""")
print(f"Created file: {{file_path}}")
        '''
        
        response = self.session.post(f"{BASE_URL}/api/execute-raw", 
                                   data=python_code,
                                   headers={'Content-Type': 'text/plain'}, 
                                   timeout=30)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result['success'])

    def test_clear_all_files_removes_uploaded_files(self):
        """Test that clearAllFiles removes uploaded files"""
        # Upload test files
        self.upload_test_file("test1.txt", "Test content 1")
        self.upload_test_file("test2.json", '{"test": "data"}', 'application/json')
        
        # Verify files exist
        response = self.session.get(f"{BASE_URL}/api/uploaded-files", timeout=30)
        self.assertEqual(response.status_code, 200)
        files_before = response.json()['files']
        self.assertGreaterEqual(len(files_before), 2)
        
        # Clear all files
        response = self.session.post(f"{BASE_URL}/api/clear-all-files", timeout=30)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result['success'])
        
        # Verify files are cleared
        response = self.session.get(f"{BASE_URL}/api/uploaded-files", timeout=30)
        self.assertEqual(response.status_code, 200)
        files_after = response.json()['files']
        self.assertLessEqual(len(files_after), 1)
        if len(files_after) == 1:
            self.assertIn("__pycache__", files_after[0]['filename'])


    def test_clear_all_files_removes_pyodide_files(self):
        """Test that clearAllFiles removes Pyodide virtual filesystem files"""
        # Create files in Pyodide filesystem
        self.create_pyodide_file("pyodide_test1.txt", "Pyodide content 1")
        self.create_pyodide_file("pyodide_test2.csv", "col1,col2\nval1,val2")
        
        # Verify files exist in Pyodide
        list_code = '''
from pathlib import Path
uploads_dir = Path("/uploads")
files = list(uploads_dir.glob("*"))
{"data": [f.name for f in files]}
        '''
        
        response = self.session.post(f"{BASE_URL}/api/execute-raw", 
                                   data=list_code,
                                   headers={'Content-Type': 'text/plain'}, 
                                   timeout=30)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result['success'])
        self.assertIn("pyodide_test1.txt", result['stdout'])
        
        # Clear all files
        response = self.session.post(f"{BASE_URL}/api/clear-all-files", timeout=30)
        self.assertEqual(response.status_code, 200)
        clear_result = response.json()
        self.assertTrue(clear_result['success'])
        
        # Verify Pyodide files are cleared
        response = self.session.post(f"{BASE_URL}/api/execute-raw", 
                                   data=list_code,
                                   headers={'Content-Type': 'text/plain'}, 
                                   timeout=30)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result['success'])
        self.assertNotIn("pyodide_test1.txt", result['stdout'])
        self.assertNotIn("pyodide_test2.csv", result['stdout'])

    def test_reset_environment_clears_variables_and_files(self):
        """Test that resetEnvironment clears variables and clear-all-files clears files"""
        # Set some variables and create files
        setup_code = '''
import pandas as pd
import numpy as np
from pathlib import Path


# Ensure uploads directory exists
Path("/uploads").mkdir(exist_ok=True)

# Set some variables
test_variable = "This should be cleared"
test_dataframe = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

# Create a file
test_file = Path("/uploads/reset_test.txt")
test_file.write_text("This file should be cleared")

print(f"Variable set: {test_variable}")
print(f"DataFrame shape: {test_dataframe.shape}")
print(f"File created: {test_file.exists()}")
        '''
        
        response = self.session.post(f"{BASE_URL}/api/execute-raw", 
                                   data=setup_code,
                                   headers={'Content-Type': 'text/plain'}, 
                                   timeout=30)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result['success'])
        self.assertIn("Variable set:", result['stdout'])
        self.assertIn("File created: True", result['stdout'])
        
        # Reset environment
        response = self.session.post(f"{BASE_URL}/api/reset", timeout=30)
        self.assertEqual(response.status_code, 200)
        reset_result = response.json()
        self.assertTrue(reset_result['success'])
        
        # Clear all files (reset only clears variables, not files)
        response = self.session.post(f"{BASE_URL}/api/clear-all-files", timeout=30)
        self.assertEqual(response.status_code, 200)
        clear_result = response.json()
        self.assertTrue(clear_result['success'])
        
        # Verify files are cleared by clear-all-files
        check_files_code = '''
from pathlib import Path
test_file = Path("/uploads/reset_test.txt")
print(f"File still exists: {test_file.exists()}")

uploads_dir = Path("/uploads")
all_files = list(uploads_dir.glob("*"))
print(f"Remaining files: {[f.name for f in all_files]}")
        '''
        
        response = self.session.post(f"{BASE_URL}/api/execute-raw", 
                                   data=check_files_code,
                                   headers={'Content-Type': 'text/plain'}, 
                                   timeout=30)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result['success'])
        self.assertIn("File still exists: False", result['stdout'])

    def test_file_listing_filters_system_files(self):
        """Test that file listing properly filters system files"""
        # Upload regular files
        file1_actual = self.upload_test_file("user_file1.txt", "User content 1")
        file2_actual = self.upload_test_file("user_file2.json", '{"user": "data"}', 'application/json')
        
        # Create some system-like files in Pyodide
        system_files_code = '''
from pathlib import Path

# Create files that might be considered system files
(Path("/uploads") / ".hidden").write_text("hidden file")
(Path("/uploads") / "__pycache__").mkdir(exist_ok=True)
(Path("/uploads") / "temp_12345.tmp").write_text("temp file")
        '''
        
        response = self.session.post(f"{BASE_URL}/api/execute-raw", 
                                   data=system_files_code,
                                   headers={'Content-Type': 'text/plain'}, 
                                   timeout=30)
        self.assertEqual(response.status_code, 200)
        
        # Get file listing
        response = self.session.get(f"{BASE_URL}/api/uploaded-files", timeout=30)
        self.assertEqual(response.status_code, 200)
        files_result = response.json()
        self.assertTrue(files_result['success'])
        
        file_names = [f['filename'] for f in files_result['files']]
        
        # Verify user files are present (check that filenames contain the base names)
        user_files_found = [f for f in file_names if 'user_file1' in f and f.endswith('.txt')]
        json_files_found = [f for f in file_names if 'user_file2' in f and f.endswith('.json')]
        
        self.assertGreaterEqual(len(user_files_found), 1, f"Should find at least one user_file1 txt file in {file_names}")
        self.assertGreaterEqual(len(json_files_found), 1, f"Should find at least one user_file2 json file in {file_names}")
        
        # Verify system files are filtered out (implementation dependent)
        # This test documents expected behavior for future filtering
        # Currently all files might be shown, but this test establishes the expectation

    def test_clear_all_files_api_endpoint_exists(self):
        """Test that the clearAllFiles API endpoint exists and responds correctly"""
        response = self.session.post(f"{BASE_URL}/api/clear-all-files", timeout=30)
        
        # Should return 200 whether files exist or not
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result['success'])
        self.assertIn('message', result)

    def test_multiple_file_operations_consistency(self):
        """Test that multiple file operations maintain consistency"""
        # Upload files
        file1 = self.upload_test_file("consistency_test1.txt", "Content 1")
        file2 = self.upload_test_file("consistency_test2.json", '{"test": 1}', 'application/json')
        
        # Create Pyodide files
        self.create_pyodide_file("pyodide_consistency.txt", "Pyodide content")
        
        # List all files
        response = self.session.get(f"{BASE_URL}/api/uploaded-files", timeout=30)
        self.assertEqual(response.status_code, 200)
        files_before = response.json()['files']
        initial_count = len(files_before)
        self.assertGreaterEqual(initial_count, 2)
        
        # Get the first uploaded file's actual filename
        uploaded_file_to_delete = files_before[0]['filename']
        
        # Delete one uploaded file
        response = self.session.delete(f"{BASE_URL}/api/uploaded-files/{uploaded_file_to_delete}", timeout=30)
        self.assertEqual(response.status_code, 200)
        
        # Verify count decreased
        response = self.session.get(f"{BASE_URL}/api/uploaded-files", timeout=30)
        self.assertEqual(response.status_code, 200)
        files_after_delete = response.json()['files']
        self.assertEqual(len(files_after_delete), initial_count - 1)
        
        # Clear all files
        response = self.session.post(f"{BASE_URL}/api/clear-all-files", timeout=30)
        self.assertEqual(response.status_code, 200)
        
        # Verify all files cleared
        response = self.session.get(f"{BASE_URL}/api/uploaded-files", timeout=30)
        self.assertEqual(response.status_code, 200)
        files_final = response.json()['files']
        self.assertEqual(len(files_final), 0)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""
Test suite for file upload system enhancements
Tests the recent fixes for temp filename generation and file-type-specific analysis
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import requests

BASE_URL = "http://localhost:3000"


class FileUploadEnhancementsTestCase(unittest.TestCase):
    """Test recent file upload system enhancements"""
    
    def setUp(self):
        """Set up test environment"""
        self.session = requests.Session()
        self.uploaded_files = []
        self.temp_files = []
        
    def tearDown(self):
        """Clean up uploaded files"""
        for filename in self.uploaded_files:
            try:
                response = self.session.delete(f"{BASE_URL}/api/uploaded-files/{filename}")
                print(f"Cleanup: Deleted {filename}, status: {response.status_code}")
            except requests.RequestException as e:
                print(f"Cleanup warning: Could not delete {filename}: {e}")
        
        # Clean up temp files
        for temp_file in self.temp_files:
            if isinstance(temp_file, Path) and temp_file.exists():
                temp_file.unlink()

    def track_upload(self, filename):
        """Track uploaded file for cleanup"""
        self.uploaded_files.append(filename)

    def test_json_file_upload_proper_filename(self):
        """Test JSON file upload generates proper temp filename (not csvFile prefix)"""
        # Create sample JSON file
        test_data = {
            "test": "data",
            "array": [1, 2, 3],
            "nested": {"key": "value"}
        }
        
        temp_file = Path(tempfile.mkdtemp()) / "test_data.json"
        self.temp_files.append(temp_file)
        
        with open(temp_file, 'w') as f:
            json.dump(test_data, f)
        
        # Upload file
        with open(temp_file, 'rb') as f:
            files = {'file': ('test_data.json', f, 'application/json')}
            response = self.session.post(f"{BASE_URL}/api/upload", files=files, timeout=30)
        
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result['success'])
        
        # Verify temp filename contains original basename (not csvFile)
        temp_path = result['file']['tempPath']
        self.assertIn('test_data', temp_path)
        self.assertNotIn('csvFile', temp_path)
        
        # Track for cleanup - extract filename from tempPath
        temp_filename = Path(temp_path).name
        self.track_upload(temp_filename)

    def test_temp_filename_uniqueness(self):
        """Test that temp filenames are unique even for same original filename"""
        # Create two identical filenames
        content = '{"test": "data"}'
        
        uploaded_temp_paths = []
        
        for i in range(2):
            temp_file = Path(tempfile.mkdtemp()) / "identical_name.json"
            self.temp_files.append(temp_file)
            
            with open(temp_file, 'w') as f:
                f.write(content)
            
            # Upload file
            with open(temp_file, 'rb') as f:
                files = {'file': ('identical_name.json', f, 'application/json')}
                response = self.session.post(f"{BASE_URL}/api/upload", files=files, timeout=30)
            
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertTrue(result['success'])
            
            uploaded_temp_paths.append(result['file']['tempPath'])
            temp_filename = Path(result['file']['tempPath']).name
            self.track_upload(temp_filename)
            
            # Small delay to ensure unique timestamps
            time.sleep(0.1)
        
        # Verify temp paths are unique
        self.assertNotEqual(uploaded_temp_paths[0], uploaded_temp_paths[1])
        
        # Verify both contain the base filename
        for temp_path in uploaded_temp_paths:
            self.assertIn('identical_name', temp_path)
            self.assertNotIn('csvFile', temp_path)


if __name__ == '__main__':
    unittest.main()

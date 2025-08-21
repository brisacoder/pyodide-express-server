#!/usr/bin/env python3
"""
Debug test for file upload functionality
"""

import json
import tempfile
from pathlib import Path

import requests

BASE_URL = "http://localhost:3000"

def test_file_upload():
    """Debug the file upload functionality"""
    
    # Create a simple JSON test file
    test_data = {"test": "data", "array": [1, 2, 3]}
    
    # Create temp file
    temp_dir = Path(tempfile.mkdtemp())
    temp_file = temp_dir / "test_data.json"
    
    print(f"Creating temp file: {temp_file}")
    
    with open(temp_file, 'w') as f:
        json.dump(test_data, f)
    
    print(f"File created with size: {temp_file.stat().st_size} bytes")
    
    # Upload file
    try:
        with open(temp_file, 'rb') as f:
            files = {'file': ('test_data.json', f, 'application/json')}
            print("Sending POST request...")
            response = requests.post(f"{BASE_URL}/api/upload-csv", files=files, timeout=30)
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Temp path: {result['file']['tempPath']}")
            print(f"Analysis: {result['analysis']['message']}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Exception occurred: {e}")
    
    finally:
        # Cleanup
        if temp_file.exists():
            temp_file.unlink()
        temp_dir.rmdir()

if __name__ == "__main__":
    test_file_upload()

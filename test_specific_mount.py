#!/usr/bin/env python3

import requests

code = '''
import os

result = {}

# Test writing to matplotlib subdirectory specifically  
try:
    test_file = "/plots/matplotlib/mount_test_specific.txt"
    with open(test_file, "w") as f:
        f.write("testing matplotlib subdirectory mount")
    result["matplotlib_write"] = "success" 
    result["matplotlib_file_exists"] = os.path.exists(test_file)
except Exception as e:
    result["matplotlib_write"] = f"failed: {str(e)}"

# Also test the root plots directory
try:
    test_file_root = "/plots/mount_test_root.txt"
    with open(test_file_root, "w") as f:
        f.write("testing root plots mount")
    result["root_write"] = "success"
    result["root_file_exists"] = os.path.exists(test_file_root)
except Exception as e:
    result["root_write"] = f"failed: {str(e)}"

result
'''

try:
    r = requests.post('http://localhost:3000/api/execute', json={'code': code}, timeout=60)
    print('Status:', r.status_code)
    print('Response:', r.json())
except Exception as e:
    print('Error:', e)

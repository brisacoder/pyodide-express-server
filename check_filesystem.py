#!/usr/bin/env python3

import requests

code = '''
import os

result = {
    "current_dir": os.getcwd(),
    "root_contents": [],
    "plots_exists": False,
    "plots_contents": [],
}

try:
    result["root_contents"] = os.listdir("/")
except Exception as e:
    result["root_list_error"] = str(e)

try:
    result["plots_exists"] = os.path.exists("/plots")
    if result["plots_exists"]:
        result["plots_contents"] = os.listdir("/plots")
except Exception as e:
    result["plots_check_error"] = str(e)

# Test writing to /plots
try:
    test_file = "/plots/mount_test.txt"
    with open(test_file, "w") as f:
        f.write("test content")
    result["write_test"] = "success"
    result["file_exists_after_write"] = os.path.exists(test_file)
    if result["file_exists_after_write"]:
        with open(test_file, "r") as f:
            result["file_content"] = f.read()
except Exception as e:
    result["write_test"] = f"failed: {str(e)}"

result
'''

try:
    r = requests.post('http://localhost:3000/api/execute', json={'code': code}, timeout=60)
    print('Status:', r.status_code)
    print('Response:', r.json())
except Exception as e:
    print('Error:', e)

#!/usr/bin/env python3

import requests

code = '''
import js

result = {
    "js_available": True,
    "has_pyodide": hasattr(js, "pyodide"),
}

if hasattr(js, "pyodide"):
    result["has_mountNodeFS"] = hasattr(js.pyodide, "mountNodeFS")
    result["pyodide_version"] = getattr(js.pyodide, "version", "unknown")
else:
    result["has_mountNodeFS"] = False

result
'''

try:
    r = requests.post('http://localhost:3000/api/execute', json={'code': code}, timeout=60)
    print('Status:', r.status_code)
    print('Response:', r.json())
except Exception as e:
    print('Error:', e)

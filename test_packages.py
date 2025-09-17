#!/usr/bin/env python3
import requests

def test_what_packages_available():
    """Test what packages are already available"""
    code = '''
import sys
print(f"Python version: {sys.version}")

# Check what packages are available
import importlib

packages_to_check = [
    'matplotlib', 'numpy', 'pandas', 'seaborn', 
    'scipy', 'sklearn', 'plotly', 'micropip'
]

available = []
missing = []

for pkg in packages_to_check:
    try:
        importlib.import_module(pkg)
        available.append(pkg)
        print(f"✅ {pkg} is available")
    except ImportError:
        missing.append(pkg)
        print(f"❌ {pkg} is missing")

print(f"\\nAvailable packages: {available}")
print(f"Missing packages: {missing}")

result = {
    "available": available,
    "missing": missing,
    "python_version": sys.version
}
print(f"Final result: {result}")
'''

    try:
        response = requests.post(
            "http://localhost:3000/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Error: {data.get('error')}")
            
            if 'data' in data and data['data']:
                result_data = data['data']
                if 'stdout' in result_data:
                    print(f"Stdout: {result_data['stdout']}")
                if 'stderr' in result_data:
                    print(f"Stderr: {result_data['stderr']}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_what_packages_available()
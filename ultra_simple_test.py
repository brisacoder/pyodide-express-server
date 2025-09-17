#!/usr/bin/env python3
"""
Ultra simple test
"""
import requests

def test_basic():
    code = '''
print("Hello from Python!")
result = {"test": "value", "number": 42}
print(f"Result: {result}")
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
                if 'result' in result_data:
                    print(f"Result: {result_data['result']}")
                if 'stdout' in result_data:
                    print(f"Stdout: {result_data['stdout']}")
                if 'stderr' in result_data:
                    print(f"Stderr: {result_data['stderr']}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_basic()
#!/usr/bin/env python3
import requests

def test_seaborn_import():
    """Test just the seaborn import part"""
    code = '''
# Install seaborn if not available
try:
    import seaborn as sns
    print("Seaborn already available")
except ImportError:
    import micropip
    print("Installing seaborn...")
    await micropip.install(['seaborn', 'matplotlib'])
    import seaborn as sns
    print("Seaborn installed and imported")

print("Seaborn import test completed successfully")
'''

    try:
        response = requests.post(
            "http://localhost:3000/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=60
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
                if 'result' in result_data:
                    print(f"Result: {result_data['result']}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_seaborn_import()
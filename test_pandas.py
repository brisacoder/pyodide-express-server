#!/usr/bin/env python3
import requests

def test_just_pandas():
    """Test just pandas operations"""
    code = '''
import pandas as pd
import numpy as np
import json

print("Testing pandas operations...")

# Create simple dataset
np.random.seed(42)
data = {
    'a': [1, 2, 3, 4, 5],
    'b': [2, 4, 6, 8, 10]
}

df = pd.DataFrame(data)
print(f"DataFrame created: {df}")

# Calculate correlation
corr = df.corr()
print(f"Correlation matrix: {corr}")

result = {"correlation_computed": True, "shape": list(df.shape)}
print(f"Result: {result}")
json.dumps(result)
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
                if 'result' in result_data:
                    print(f"Result: {result_data['result']}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_just_pandas()
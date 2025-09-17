#!/usr/bin/env python3
import requests

def test_gradual_imports():
    """Test imports one by one to find the problematic one"""
    tests = [
        ("numpy", "import numpy as np\nprint('numpy imported successfully')"),
        ("pandas", "import pandas as pd\nprint('pandas imported successfully')"),
        ("matplotlib", "import matplotlib\nprint('matplotlib imported successfully')"),
        ("matplotlib.pyplot", "import matplotlib.pyplot as plt\nprint('matplotlib.pyplot imported successfully')"),
        ("seaborn", "import seaborn as sns\nprint('seaborn imported successfully')")
    ]
    
    for test_name, code in tests:
        print(f"\n=== Testing {test_name} ===")
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
                if data.get('success'):
                    result_data = data.get('data', {})
                    print(f"Stdout: {result_data.get('stdout', '')}")
                else:
                    print(f"Error: {data.get('error')}")
            else:
                print(f"HTTP Error: {response.text}")
                
        except Exception as e:
            print(f"Exception: {e}")
            break  # Stop on first exception

if __name__ == "__main__":
    test_gradual_imports()
import requests
import json

BASE_URL = "http://localhost:3000"

# Test basic operations
test_cases = [
    # Test basic int
    ("2 + 3", 5),
    
    # Test basic float
    ("3.14 * 2", 6.28),
    
    # Test basic string
    ("'hello world'", "hello world"),
    
    # Test basic list
    ("[1, 2, 3]", [1, 2, 3]),
    
    # Test basic dict
    ("{'a': 1, 'b': 2}", {"a": 1, "b": 2}),
    
    # Test numpy operations
    ("import numpy as np; np.array([1, 2, 3]).sum()", 6),
    
    # Test pandas operations
    ("import pandas as pd; pd.Series([10, 20, 30]).sum()", 60),
]

for code, expected in test_cases:
    print(f"Testing: {code}")
    r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=30)
    if r.status_code == 200:
        result = r.json()
        if result.get("success"):
            actual = result.get("result")
            if actual == expected:
                print(f"  ✅ PASS: {actual}")
            else:
                print(f"  ❌ FAIL: Expected {expected}, got {actual}")
        else:
            print(f"  ❌ ERROR: {result.get('error')}")
    else:
        print(f"  ❌ HTTP ERROR: {r.status_code}")
    print()

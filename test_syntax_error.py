#!/usr/bin/env python3
import requests

def test_syntax_error():
    """Test with a syntax error to see what error we get"""
    code = '''
print("test"
# Missing closing parenthesis - this should give a clear syntax error
'''

    try:
        response = requests.post(
            "http://localhost:3000/api/execute-raw",
            data=code,
            headers={"Content-Type": "text/plain"},
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        print(f"Full response: {response.text}")
        
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_syntax_error()
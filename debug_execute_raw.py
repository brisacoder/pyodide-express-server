#!/usr/bin/env python3
import requests
import json

code = "import json\nprint('Hello from Python setup test')"

try:
    response = requests.post(
        'http://localhost:3000/api/execute-raw',
        data=code,
        headers={'Content-Type': 'text/plain'},
        timeout=30
    )
    print('Status Code:', response.status_code)
    print('Response Text:', response.text)
    
    if response.status_code == 200:
        try:
            json_data = response.json()
            print('Success:', json_data.get('success'))
            print('Data:', json_data.get('data'))
            print('Error:', json_data.get('error'))
        except Exception as e:
            print('JSON Parse Error:', e)
    else:
        print(f'HTTP Error {response.status_code}')
        
except Exception as e:
    print('Request Error:', e)
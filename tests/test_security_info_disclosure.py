import requests

# Security test: Information disclosure attack vector
code = '''import os, sys, platform

print("=== SYSTEM INFORMATION DISCLOSURE TEST ===")

# Basic system info
print("OS:", platform.system())
print("Python version:", sys.version)
print("Platform:", platform.platform())

# Environment variables (potentially sensitive)
print("Environment variables count:", len(os.environ))
print("Sample environment vars:", list(os.environ.keys())[:10])

# Python path and modules
print("Python executable:", sys.executable)
print("Python path length:", len(sys.path))
print("Current working directory:", os.getcwd())

# File system access attempts
try:
    root_contents = os.listdir('/')
    print("Root directory contents:", root_contents[:10])
except Exception as e:
    print("Root directory access failed:", str(e))

try:
    current_contents = os.listdir('.')
    print("Current directory contents:", current_contents[:10])
except Exception as e:
    print("Current directory access failed:", str(e))

# Process information
try:
    print("Process ID:", os.getpid())
except Exception as e:
    print("Process info access failed:", str(e))
'''

print("🔍 Testing information disclosure vulnerability...")
response = requests.post('http://localhost:3000/api/execute-raw', 
                        data=code, 
                        headers={'Content-Type': 'text/plain'})

result = response.json()
print(f"\\n{'='*60}")
print("SECURITY TEST RESULTS")
print(f"{'='*60}")
print(f"Success: {result['success']}")

if result['success']:
    print("\\n🚨 CRITICAL: Information disclosure successful!")
    print("--- EXPOSED INFORMATION ---")
    # Follow API contract: data should be under result['data']
    if result.get('data') and result['data'].get('stdout'):
        print(result['data']['stdout'])
    elif result.get('data') and result['data'].get('result'):
        print(result['data']['result'])
    
    # Check stderr in data section
    if result.get('data') and result['data'].get('stderr'):
        print("\\n--- STDERR ---")
        print(result['data']['stderr'])
else:
    print(f"\\n❌ Execution failed: {result.get('error')}")
    if result.get('data') and result['data'].get('stderr'):
        print(f"Stderr: {result['data']['stderr']}")

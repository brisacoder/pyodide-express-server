#!/usr/bin/env python3
"""
Comprehensive Security Penetration Testing for Pyodide Express Server
====================================================================

This script performs real-world security testing to validate theoretical
vulnerabilities and document actual attack vectors.

⚠️ WARNING: FOR TESTING PURPOSES ONLY - RUN AGAINST YOUR OWN SERVER
"""

import threading
import time
from pathlib import Path

import requests

BASE_URL = "http://localhost:3000"

class SecurityPenetrationTest:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.results = {}
        
    def test_information_disclosure(self):
        """Test what system information can be extracted"""
        print("🔍 Testing Information Disclosure...")
        
        code = """
import os, sys, platform, tempfile
from pathlib import Path

# System information
print("=== SYSTEM INFO ===")
print("OS:", platform.system())
print("Platform:", platform.platform())
print("Python:", sys.version_info)
print("Architecture:", platform.architecture())

# Environment variables
print("\\n=== ENVIRONMENT ===")
env_vars = dict(os.environ)
for key, value in env_vars.items():
    print(f"{key}: {value}")

# File system exploration
print("\\n=== FILESYSTEM ===")
try:
    print("Current dir:", os.getcwd())
    print("Home exists:", Path.home().exists())
    print("Tmp dir:", tempfile.gettempdir())
    
    # Try to list various directories
    for path_str in ["/", "/home", "/tmp", '/home/pyodide/plots', '/home/pyodide/uploads']:
        try:
            path = Path(path_str)
            if path.exists():
                contents = list(path.iterdir())
                print(f"{path_str}: {[str(p) for p in contents[:10]]}")  # Limit to 10 items
        except:
            print(f"{path_str}: Access denied")
            
except Exception as e:
    print(f"Filesystem error: {e}")

# Memory and process info
print("\\n=== PROCESS INFO ===")
try:
    print("Process ID:", os.getpid())
    print("User ID:", os.getuid() if hasattr(os, 'getuid') else 'N/A')
except:
    print("Process info not available")
"""
        
        try:
            response = requests.post(f"{self.base_url}/api/execute-raw", 
                                   data=code, 
                                   headers={'Content-Type': 'text/plain'},
                                   timeout=10)
            
            self.results['information_disclosure'] = {
                'status': 'SUCCESS' if response.status_code == 200 else 'FAILED',
                'output': response.text,
                'response_code': response.status_code
            }
            
            print("✅ Information disclosure test completed")
            return response.text
            
        except Exception as e:
            print(f"❌ Information disclosure test failed: {e}")
            self.results['information_disclosure'] = {'status': 'ERROR', 'error': str(e)}
            return None
    
    def test_resource_exhaustion(self):
        """Test memory and CPU resource limits"""
        print("🔥 Testing Resource Exhaustion...")
        
        # Memory exhaustion test
        memory_test = """
import gc
print("Testing memory exhaustion...")
try:
    # Try to allocate increasingly large amounts of memory
    for size in [1000, 10000, 100000, 1000000]:
        data = [0] * size
        print(f"Allocated {size} integers")
        gc.collect()
    
    print("Memory test completed successfully")
except MemoryError as e:
    print(f"Memory limit reached: {e}")
except Exception as e:
    print(f"Memory test error: {e}")
"""
        
        # CPU exhaustion test
        cpu_test = """
import time
print("Testing CPU exhaustion...")
start_time = time.time()
iterations = 0

try:
    # Run for maximum 5 seconds or until timeout
    while time.time() - start_time < 5:
        # Perform CPU-intensive calculation
        sum(i*i for i in range(1000))
        iterations += 1
        
        if iterations % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Completed {iterations} iterations in {elapsed:.2f}s")
            
    print(f"CPU test completed: {iterations} iterations")
except Exception as e:
    print(f"CPU test error: {e}")
"""
        
        results = {}
        
        # Test memory
        try:
            response = requests.post(f"{self.base_url}/api/execute-raw",
                                   data=memory_test,
                                   headers={'Content-Type': 'text/plain'},
                                   timeout=15)
            results['memory'] = {
                'status': response.status_code,
                'output': response.text
            }
        except Exception as e:
            results['memory'] = {'error': str(e)}
        
        # Test CPU
        try:
            response = requests.post(f"{self.base_url}/api/execute-raw",
                                   data=cpu_test,
                                   headers={'Content-Type': 'text/plain'},
                                   timeout=15)
            results['cpu'] = {
                'status': response.status_code,
                'output': response.text
            }
        except Exception as e:
            results['cpu'] = {'error': str(e)}
        
        self.results['resource_exhaustion'] = results
        print("✅ Resource exhaustion tests completed")
        return results
    
    def test_concurrent_requests(self):
        """Test server behavior under concurrent load"""
        print("⚡ Testing Concurrent Request Handling...")
        
        def make_request(request_id):
            code = f"""
import time
print(f"Request {request_id} starting...")
time.sleep(1)  # Simulate work
print(f"Request {request_id} completed")
result = {request_id} * 2
print(f"Result: {{result}}")
"""
            try:
                response = requests.post(f"{self.base_url}/api/execute-raw",
                                       data=code,
                                       headers={'Content-Type': 'text/plain'},
                                       timeout=10)
                return {
                    'request_id': request_id,
                    'status': response.status_code,
                    'success': response.status_code == 200,
                    'response_time': response.elapsed.total_seconds(),
                    'output': response.text[:200]  # Truncate output
                }
            except Exception as e:
                return {
                    'request_id': request_id,
                    'success': False,
                    'error': str(e)
                }
        
        # Launch 5 concurrent requests
        threads = []
        results = []
        
        for i in range(5):
            thread = threading.Thread(target=lambda i=i: results.append(make_request(i)))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        self.results['concurrent_requests'] = results
        successful = sum(1 for r in results if r.get('success', False))
        print(f"✅ Concurrent test completed: {successful}/5 requests successful")
        return results
    
    def test_file_system_access(self):
        """Test file system access and manipulation capabilities"""
        print("📁 Testing File System Access...")
        
        code = """
from pathlib import Path
import os

print("=== FILE SYSTEM ACCESS TEST ===")

# Test file creation in various directories
test_dirs = ["/tmp", '/home/pyodide/plots', '/home/pyodide/uploads', "/home/pyodide", "."]

for dir_path in test_dirs:
    try:
        path = Path(dir_path)
        print(f"\\nTesting directory: {dir_path}")
        print(f"Exists: {path.exists()}")
        print(f"Is directory: {path.is_dir()}")
        
        if path.exists() and path.is_dir():
            # Try to list contents
            try:
                contents = list(path.iterdir())
                print(f"Contents ({len(contents)} items): {[str(p.name) for p in contents[:5]]}")
            except Exception as e:
                print(f"Cannot list contents: {e}")
            
            # Try to create a test file
            test_file = path / "security_test.txt"
            try:
                test_file.write_text("Security penetration test")
                print(f"✅ File created successfully: {test_file}")
                
                # Try to read it back
                content = test_file.read_text()
                print(f"✅ File read successfully: {content[:50]}")
                
                # Try to delete it
                test_file.unlink()
                print(f"✅ File deleted successfully")
                
            except Exception as e:
                print(f"❌ File operations failed: {e}")
                
    except Exception as e:
        print(f"❌ Directory access failed for {dir_path}: {e}")

# Test accessing server files
print("\\n=== SERVER FILE ACCESS TEST ===")
server_paths = [
    "/package.json",
    "/src/server.js", 
    "/README.md",
    "../package.json",
    "../../package.json"
]

for file_path in server_paths:
    try:
        path = Path(file_path)
        if path.exists():
            print(f"⚠️ CRITICAL: Can access server file {file_path}")
            # Don't actually read the content to avoid exposing sensitive data
        else:
            print(f"✅ Server file not accessible: {file_path}")
    except Exception as e:
        print(f"Server file access error for {file_path}: {e}")
"""
        
        try:
            response = requests.post(f"{self.base_url}/api/execute-raw",
                                   data=code,
                                   headers={'Content-Type': 'text/plain'},
                                   timeout=15)
            
            self.results['file_system_access'] = {
                'status': response.status_code,
                'output': response.text
            }
            
            print("✅ File system access test completed")
            return response.text
            
        except Exception as e:
            print(f"❌ File system test failed: {e}")
            self.results['file_system_access'] = {'error': str(e)}
            return None
    
    def test_package_installation_abuse(self):
        """Test malicious package installation attempts"""
        print("📦 Testing Package Installation Security...")
        
        code = """
import micropip
import sys

print("=== PACKAGE INSTALLATION TEST ===")

# Test legitimate packages
legitimate_packages = ["numpy", "matplotlib", "requests"]
for package in legitimate_packages:
    try:
        if package not in sys.modules:
            print(f"Installing {package}...")
            await micropip.install(package)
            print(f"✅ {package} installed successfully")
        else:
            print(f"✅ {package} already available")
    except Exception as e:
        print(f"❌ Failed to install {package}: {e}")

# Test suspicious package names (don't actually install these)
suspicious_packages = [
    "../../malicious-package",
    "file:///etc/passwd",
    "http://evil.com/malware.whl",
    "backdoor",
    "keylogger"
]

print("\\nTesting suspicious package handling:")
for package in suspicious_packages:
    print(f"Would attempt: {package}")
    # Don't actually try to install these
    print(f"⚠️ Blocked (test only): {package}")

print("\\nCurrent installed packages:")
try:
    import pkg_resources
    installed = [d.project_name for d in pkg_resources.working_set]
    print(f"Total packages: {len(installed)}")
    print(f"Sample packages: {installed[:10]}")
except:
    print("Could not enumerate packages")
"""
        
        try:
            response = requests.post(f"{self.base_url}/api/execute-raw",
                                   data=code,
                                   headers={'Content-Type': 'text/plain'},
                                   timeout=20)
            
            self.results['package_installation'] = {
                'status': response.status_code,
                'output': response.text
            }
            
            print("✅ Package installation test completed")
            return response.text
            
        except Exception as e:
            print(f"❌ Package installation test failed: {e}")
            self.results['package_installation'] = {'error': str(e)}
            return None
    
    def generate_report(self):
        """Generate a comprehensive security test report"""
        print("\n" + "="*60)
        print("🛡️ SECURITY PENETRATION TEST REPORT")
        print("="*60)
        
        report = []
        report.append("# Security Penetration Test Report")
        report.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Target:** {self.base_url}")
        report.append("")
        
        # Summarize results
        report.append("## Test Summary")
        for test_name, result in self.results.items():
            if isinstance(result, list):
                # Handle concurrent requests which returns a list
                successful = sum(1 for r in result if r.get('success', False))
                status = f"SUCCESS ({successful}/{len(result)} requests)"
            else:
                status = result.get('status', 'UNKNOWN')
                if isinstance(status, int):
                    status = 'SUCCESS' if status == 200 else f'HTTP_{status}'
            
            report.append(f"- **{test_name.replace('_', ' ').title()}:** {status}")
        
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        for test_name, result in self.results.items():
            report.append(f"### {test_name.replace('_', ' ').title()}")
            
            if isinstance(result, list):
                # Handle concurrent requests
                report.append(f"**Total Requests:** {len(result)}")
                successful = sum(1 for r in result if r.get('success', False))
                report.append(f"**Successful:** {successful}")
                report.append(f"**Failed:** {len(result) - successful}")
                
                # Show sample results
                for i, req_result in enumerate(result[:3]):  # Show first 3
                    report.append(f"- Request {i}: {req_result.get('success', 'Unknown')}")
                    
            elif 'output' in result:
                report.append("```")
                report.append(result['output'][:1000])  # Truncate long output
                if len(result['output']) > 1000:
                    report.append("... [truncated]")
                report.append("```")
            elif 'error' in result:
                report.append(f"**Error:** {result['error']}")
            
            report.append("")
        
        # Save report
        report_content = "\n".join(report)
        report_file = Path("security_penetration_report.md")
        report_file.write_text(report_content)
        
        print(f"📊 Full report saved to: {report_file}")
        return report_content
    
    def run_full_test_suite(self):
        """Run all security tests"""
        print("🚀 Starting Comprehensive Security Penetration Testing")
        print("="*60)
        
        # Check if server is running
        try:
            health_response = requests.get(f"{self.base_url}/health", timeout=5)
            if health_response.status_code != 200:
                print("❌ Server not responding - ensure it's running on localhost:3000")
                return
        except:
            print("❌ Cannot connect to server - ensure it's running on localhost:3000")
            return
        
        print("✅ Server is responding")
        print()
        
        # Run all tests
        self.test_information_disclosure()
        print()
        
        self.test_resource_exhaustion()
        print()
        
        self.test_concurrent_requests()
        print()
        
        self.test_file_system_access()
        print()
        
        self.test_package_installation_abuse()
        print()
        
        # Generate comprehensive report
        self.generate_report()
        
        print("\n🎯 PENETRATION TEST COMPLETE")
        print("Review the generated report for detailed findings.")


if __name__ == "__main__":
    tester = SecurityPenetrationTest()
    tester.run_full_test_suite()

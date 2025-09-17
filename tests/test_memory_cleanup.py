"""
Test memory cleanup after code execution.
This test verifies that memory is properly cleaned up after running Python code,
preventing the memory leaks that were causing system crashes.
"""

import pytest
import psutil
import os
import gc
import time
import requests


class TestMemoryCleanup:
    """Test suite for memory cleanup verification"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_server(self, server_ready):
        """Ensure server is running"""
        pass
    
    def get_memory_usage_mb(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @pytest.mark.slow
    def test_single_execution_cleanup(self, base_url):
        """Test that memory is cleaned up after a single code execution"""
        # Given: Initial memory state
        gc.collect()
        time.sleep(0.1)
        initial_memory = self.get_memory_usage_mb()
        
        # When: Execute code that creates large objects
        code = """
import numpy as np

# Create a large array (100MB)
large_array = np.random.rand(13107200)  # ~100MB of float64
result = f"Array size: {large_array.nbytes / 1024 / 1024:.1f} MB"
result
"""
        response = requests.post(
            f"{base_url}/api/execute-raw", 
            data=code, 
            headers={'Content-Type': 'text/plain'}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "100" in data["data"]["result"]
        
        # Allow time for cleanup
        time.sleep(0.5)
        
        # Then: Memory should not increase significantly (allowing 50MB variance)
        final_memory = self.get_memory_usage_mb()
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 50, f"Memory increased by {memory_increase:.1f} MB"
    
    @pytest.mark.slow
    def test_multiple_executions_cleanup(self, base_url):
        """Test that memory is cleaned up after multiple executions"""
        # Given: Initial memory state
        gc.collect()
        time.sleep(0.1)
        initial_memory = self.get_memory_usage_mb()
        
        # When: Execute multiple times with large objects
        for i in range(5):
            code = f"""
import numpy as np

# Create a unique large array each time
large_array_{i} = np.random.rand(6553600)  # ~50MB each
globals_count = len([k for k in globals() if k.startswith('large_array_')])
f"Iteration {i}: Array created, globals count: {{globals_count}}"
"""
            response = requests.post(
                f"{base_url}/api/execute-raw", 
                data=code, 
                headers={'Content-Type': 'text/plain'}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            
            # The globals count should be 1 (only current array) if cleanup is working
            assert "globals count: 1" in data["data"]["result"]
        
        # Allow time for cleanup
        time.sleep(1.0)
        
        # Then: Memory should not accumulate (allowing 100MB variance for 5 executions)
        final_memory = self.get_memory_usage_mb()
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f} MB after 5 executions"
    
    def test_global_variables_cleanup(self, base_url):
        """Test that global variables are cleaned up between executions"""
        # Given: Create global variables in first execution
        code1 = """
# Create various global objects
my_list = list(range(1000000))
my_dict = {str(i): i for i in range(100000)}
my_string = "x" * 10000000
my_set = set(range(500000))

f"Created {len(globals())} globals"
"""
        response1 = requests.post(
            f"{base_url}/api/execute-raw", 
            data=code1, 
            headers={'Content-Type': 'text/plain'}
        )
        assert response1.status_code == 200
        
        # When: Check globals in second execution
        code2 = """
# Check if previous globals exist
missing_vars = []
for var in ['my_list', 'my_dict', 'my_string', 'my_set']:
    if var not in globals():
        missing_vars.append(var)

f"Missing vars: {missing_vars}, Total globals: {len(globals())}"
"""
        response2 = requests.post(
            f"{base_url}/api/execute-raw", 
            data=code2, 
            headers={'Content-Type': 'text/plain'}
        )
        assert response2.status_code == 200
        data = response2.json()
        
        # Then: Previous globals should be cleaned up
        assert "Missing vars: ['my_list', 'my_dict', 'my_string', 'my_set']" in data["data"]["result"]
    
    def test_module_cleanup(self, base_url):
        """Test that imported modules are cleaned appropriately"""
        # Given: Import and use a module
        code1 = """
import random
import string
import itertools

# Use the modules
random_data = [random.choice(string.ascii_letters) for _ in range(1000)]
combinations = list(itertools.combinations(range(10), 2))

"Modules imported and used"
"""
        response1 = requests.post(
            f"{base_url}/api/execute-raw", 
            data=code1, 
            headers={'Content-Type': 'text/plain'}
        )
        assert response1.status_code == 200
        
        # When: Check if user-created data is cleaned but core modules remain
        code2 = """
import sys

# Check if modules are still available
modules_available = all(mod in sys.modules for mod in ['random', 'string', 'itertools'])

# Check if user data is cleaned
user_data_cleaned = 'random_data' not in globals() and 'combinations' not in globals()

f"Modules available: {modules_available}, User data cleaned: {user_data_cleaned}"
"""
        response2 = requests.post(
            f"{base_url}/api/execute-raw", 
            data=code2, 
            headers={'Content-Type': 'text/plain'}
        )
        assert response2.status_code == 200
        data = response2.json()
        
        # Then: User data should be cleaned
        # Note: User-imported modules (random, string, itertools) are also cleaned 
        # for complete isolation between executions
        assert "User data cleaned: True" in data["data"]["result"]
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_stress_memory_cleanup(self, base_url):
        """Stress test: Execute many times to ensure no memory leak"""
        # Given: Initial memory state
        gc.collect()
        time.sleep(0.1)
        initial_memory = self.get_memory_usage_mb()
        
        # When: Execute 20 times with memory-intensive operations
        for i in range(20):
            code = f"""
import numpy as np

# Create and process data
data = np.random.rand(3276800)  # ~25MB
processed = data * 2 + 1
result = processed.mean()

f"Iteration {i}: Mean = {{result:.6f}}"
"""
            response = requests.post(
                f"{base_url}/api/execute-raw", 
                data=code, 
                headers={'Content-Type': 'text/plain'}
            )
            assert response.status_code == 200
            
            # Small delay between executions
            time.sleep(0.1)
        
        # Allow time for final cleanup
        time.sleep(2.0)
        gc.collect()
        
        # Then: Memory should not grow significantly (allowing 200MB for 20 executions)
        final_memory = self.get_memory_usage_mb()
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 200, f"Memory increased by {memory_increase:.1f} MB after 20 executions"
        
        # Log memory statistics
        print(f"\nMemory statistics:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        print(f"  Average per execution: {memory_increase/20:.1f} MB")
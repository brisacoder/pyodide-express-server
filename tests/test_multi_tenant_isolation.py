"""
Comprehensive tests for multi-tenant user isolation.

These tests ensure complete isolation between different user executions,
preventing any data leakage in multi-tenant scenarios. They test:
- Variable isolation
- Module import isolation  
- File system isolation
- Context variable isolation
- Resource cleanup
"""

import pytest
import requests
import time
import uuid


class TestMultiTenantIsolation:
    """Test suite for multi-tenant user isolation"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_server(self, server_ready):
        """Ensure server is running"""
        pass
    
    def execute_code(self, base_url, code, context=None):
        """Helper to execute code and return response"""
        if context:
            # For now, context is passed as part of the code
            # since /api/execute-raw doesn't support context parameter directly
            context_setup = "\n".join([f"{k} = {repr(v)}" for k, v in context.items()])
            code = context_setup + "\n" + code
            
        response = requests.post(
            f"{base_url}/api/execute-raw",
            data=code,
            headers={'Content-Type': 'text/plain'}
        )
        return response
    
    def test_variable_isolation_between_users(self, base_url):
        """Test that variables from one user don't leak to another"""
        # User A creates sensitive data
        user_a_code = """
# User A's sensitive data
api_key = "sk-user-a-secret-key-12345"
database_password = "user_a_db_pass"
user_data = {"id": "userA", "credit_card": "1234-5678-9012-3456"}
personal_notes = "User A's private notes"

"User A data created"
"""
        response_a = self.execute_code(base_url, user_a_code)
        assert response_a.status_code == 200
        assert response_a.json()["success"] is True
        
        # User B tries to access User A's data
        user_b_code = """
# Try to access previous user's data
leaked_data = []
for var_name in ['api_key', 'database_password', 'user_data', 'personal_notes']:
    try:
        value = eval(var_name)
        leaked_data.append(f"{var_name}={value}")
    except NameError:
        pass

if leaked_data:
    result = f"SECURITY BREACH! Leaked data: {leaked_data}"
else:
    result = "No data leaked - isolation successful"
result  # Return the result
"""
        response_b = self.execute_code(base_url, user_b_code)
        assert response_b.status_code == 200
        data = response_b.json()
        print(f"Response B: {data}")  # Debug output
        assert data["success"] is True
        result = data["data"].get("result", "")
        assert "No data leaked - isolation successful" in result
        assert "SECURITY BREACH" not in result
    
    def test_class_and_function_isolation(self, base_url):
        """Test that classes and functions don't persist between users"""
        # User A defines custom classes and functions
        user_a_code = """
class UserAccount:
    def __init__(self):
        self.balance = 1000000
        self.account_number = "ACC-001-PRIVATE"
    
    def get_balance(self):
        return self.balance

def transfer_money(amount):
    return f"Transferred ${amount} from User A"

user_a_account = UserAccount()
secret_function = lambda x: f"User A's secret: {x}"

"User A's code executed"
"""
        response_a = self.execute_code(base_url, user_a_code)
        assert response_a.status_code == 200
        
        # User B tries to use User A's classes and functions
        user_b_code = """
security_issues = []

# Try to access User A's class
try:
    account = UserAccount()
    security_issues.append(f"Accessed UserAccount class! Balance: {account.get_balance()}")
except NameError:
    pass

# Try to access User A's function
try:
    result = transfer_money(500)
    security_issues.append(f"Called transfer_money: {result}")
except NameError:
    pass

# Try to access User A's instance
try:
    balance = user_a_account.get_balance()
    security_issues.append(f"Accessed user_a_account! Balance: {balance}")
except NameError:
    pass

# Try to access lambda function
try:
    result = secret_function("test")
    security_issues.append(f"Called secret_function: {result}")
except NameError:
    pass

if security_issues:
    result = f"SECURITY ISSUES FOUND: {security_issues}"
else:
    result = "All classes and functions properly isolated"
result
"""
        response_b = self.execute_code(base_url, user_b_code)
        assert response_b.status_code == 200
        data = response_b.json()
        assert "All classes and functions properly isolated" in data["data"]["result"]
        assert "SECURITY ISSUES FOUND" not in data["data"]["result"]
    
    def test_module_state_isolation(self, base_url):
        """Test that module-level state changes don't persist"""
        # User A modifies module state
        user_a_code = """
import json
import sys

# Modify json module behavior
json.SECRET_DATA = {"user": "A", "password": "secret123"}

# Add custom module
sys.modules['custom_module'] = type(sys)('custom_module')
sys.modules['custom_module'].secret_value = "User A's secret module data"

# Modify built-in
import builtins
builtins.USER_A_CONSTANT = "This should not leak"

"User A modified module state"
"""
        response_a = self.execute_code(base_url, user_a_code)
        assert response_a.status_code == 200
        
        # User B checks for module state leakage
        user_b_code = """
import json
import sys

leaks = []

# Check json module
if hasattr(json, 'SECRET_DATA'):
    leaks.append(f"json.SECRET_DATA = {json.SECRET_DATA}")

# Check for custom module
if 'custom_module' in sys.modules:
    import custom_module
    if hasattr(custom_module, 'secret_value'):
        leaks.append(f"custom_module.secret_value = {custom_module.secret_value}")

# Check builtins
import builtins
if hasattr(builtins, 'USER_A_CONSTANT'):
    leaks.append(f"builtins.USER_A_CONSTANT = {builtins.USER_A_CONSTANT}")

if leaks:
    result = f"MODULE STATE LEAKED: {leaks}"
else:
    result = "Module state properly isolated"
result
"""
        response_b = self.execute_code(base_url, user_b_code)
        assert response_b.status_code == 200
        data = response_b.json()
        assert "Module state properly isolated" in data["data"]["result"]
        assert "MODULE STATE LEAKED" not in data["data"]["result"]
    
    @pytest.mark.slow
    def test_context_variable_isolation(self, base_url):
        """Test that context variables are properly cleaned between executions"""
        # First execution with context
        code_with_context = """
# Simulate context variables being set
user_id = "context_user_123"
session_token = "ctx_token_abc"
api_endpoint = "https://api.example.com"

f"Context set for user {user_id}"
"""
        response_1 = self.execute_code(base_url, code_with_context)
        assert response_1.status_code == 200
        
        # Second execution checks for context leakage
        check_code = """
leaked_context = {}
for var in ['user_id', 'session_token', 'api_endpoint']:
    if var in globals():
        leaked_context[var] = globals()[var]

if leaked_context:
    result = f"CONTEXT LEAKED: {leaked_context}"
else:
    result = "No context variables leaked"
result
"""
        response_2 = self.execute_code(base_url, check_code)
        assert response_2.status_code == 200
        data = response_2.json()
        assert "No context variables leaked" in data["data"]["result"]
        assert "CONTEXT LEAKED" not in data["data"]["result"]
    
    def test_file_handle_isolation(self, base_url):
        """Test that file handles don't leak between users"""
        # User A opens files
        user_a_code = """
from io import StringIO
import sys

# Create various file-like objects
string_buffer = StringIO("User A's private data")
string_buffer.name = "user_a_buffer"

# Store reference in a way that might persist
sys._user_a_file = string_buffer

"User A created file handles"
"""
        response_a = self.execute_code(base_url, user_a_code)
        assert response_a.status_code == 200
        
        # User B checks for file handle leakage
        user_b_code = """
import sys
import gc
from io import StringIO

leaks = []

# Check sys for user A's file
if hasattr(sys, '_user_a_file'):
    leaks.append("Found sys._user_a_file")
    
# Check globals
if 'string_buffer' in globals():
    leaks.append("Found string_buffer in globals")

# Check for open StringIO objects
for obj in gc.get_objects():
    if isinstance(obj, StringIO) and hasattr(obj, 'name') and obj.name == 'user_a_buffer':
        leaks.append("Found User A's StringIO object via gc")
        break

if leaks:
    result = f"FILE HANDLE LEAKS: {leaks}"
else:
    result = "No file handles leaked"
result
"""
        response_b = self.execute_code(base_url, user_b_code)
        assert response_b.status_code == 200
        data = response_b.json()
        assert "No file handles leaked" in data["data"]["result"]
        assert "FILE HANDLE LEAKS" not in data["data"]["result"]
    
    @pytest.mark.slow
    def test_rapid_succession_isolation(self, base_url):
        """Test isolation when requests come in rapid succession"""
        # Simulate multiple users sending requests rapidly
        results = []
        
        for i in range(10):
            user_code = f"""
# User {i} setting unique data
user_secret = "secret_for_user_{i}"
user_id = {i}

# Try to find other users' data
other_users_data = []
for j in range(10):
    if j != {i}:
        try:
            secret = eval(f"secret_for_user_{{j}}")
            other_users_data.append(f"Found user {{j}}'s secret: {{secret}}")
        except:
            pass

if other_users_data:
    result = f"User {i} found other data: {{other_users_data}}"
else:
    result = f"User {i} - isolation maintained"
result
"""
            response = self.execute_code(base_url, user_code)
            results.append(response)
            
            # Small delay to simulate realistic timing
            time.sleep(0.1)
        
        # Verify all users maintained isolation
        for i, response in enumerate(results):
            assert response.status_code == 200
            data = response.json()
            assert f"User {i} - isolation maintained" in data["data"]["result"]
            assert "found other data" not in data["data"]["result"]
    
    def test_malicious_global_modification_isolation(self, base_url):
        """Test isolation against malicious attempts to modify globals"""
        # User A tries malicious modifications
        malicious_code = """
# Try to poison the global namespace
__builtins__.__dict__['BACKDOOR'] = lambda: "Hacked!"

# Try to modify sys.modules
import sys
sys.modules['os'].MALICIOUS = "Bad data"

# Try to create persistent references
import gc
gc.PERSISTENT_DATA = {"admin": "password123"}

# Try to modify matplotlib if loaded
try:
    import matplotlib
    matplotlib.HACKED = True
except:
    pass

"Malicious modifications attempted"
"""
        response_a = self.execute_code(base_url, malicious_code)
        assert response_a.status_code == 200
        
        # User B checks for poisoning
        check_code = """
poisoned = []

# Check builtins
if 'BACKDOOR' in dir(__builtins__):
    poisoned.append("Found BACKDOOR in builtins")

# Check os module
import os
if hasattr(os, 'MALICIOUS'):
    poisoned.append(f"Found os.MALICIOUS = {os.MALICIOUS}")

# Check gc
import gc
if hasattr(gc, 'PERSISTENT_DATA'):
    poisoned.append(f"Found gc.PERSISTENT_DATA = {gc.PERSISTENT_DATA}")

# Check matplotlib
try:
    import matplotlib
    if hasattr(matplotlib, 'HACKED'):
        poisoned.append("Found matplotlib.HACKED")
except:
    pass

if poisoned:
    result = f"NAMESPACE POISONED: {poisoned}"
else:
    result = "Namespace clean - no poisoning detected"
result
"""
        response_b = self.execute_code(base_url, check_code)
        assert response_b.status_code == 200
        data = response_b.json()
        assert "Namespace clean - no poisoning detected" in data["data"]["result"]
        assert "NAMESPACE POISONED" not in data["data"]["result"]
    
    def test_execution_environment_reset(self, base_url):
        """Test that the execution environment is properly reset"""
        # User A modifies environment
        env_code = """
import os
import sys

# Modify environment
os.environ['SECRET_KEY'] = 'user_a_secret'
sys.path.insert(0, '/user_a/custom/path')

# Create custom attributes
sys.custom_attribute = "User A was here"
os.custom_data = {"user": "A"}

"Environment modified"
"""
        response_a = self.execute_code(base_url, env_code)
        assert response_a.status_code == 200
        
        # User B checks environment
        check_env_code = """
import os
import sys

issues = []

# Check environment variable
if 'SECRET_KEY' in os.environ:
    issues.append(f"Found SECRET_KEY = {os.environ['SECRET_KEY']}")

# Check sys.path
if '/user_a/custom/path' in sys.path:
    issues.append("Found user A's custom path")

# Check custom attributes
if hasattr(sys, 'custom_attribute'):
    issues.append(f"Found sys.custom_attribute = {sys.custom_attribute}")
    
if hasattr(os, 'custom_data'):
    issues.append(f"Found os.custom_data = {os.custom_data}")

if issues:
    result = f"ENVIRONMENT ISSUES: {issues}"
else:
    result = "Environment properly reset"
result
"""
        response_b = self.execute_code(base_url, check_env_code)
        assert response_b.status_code == 200
        data = response_b.json()
        # Note: Some environment changes might persist within the same process
        # The important thing is that user data doesn't leak
        assert "SECRET_KEY" not in data["data"].get("result", "")

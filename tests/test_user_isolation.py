import os
import unittest
import subprocess
import time
import requests

BASE_URL = "http://localhost:3000"


class UserIsolationTestCase(unittest.TestCase):
    """Test user isolation between execution requests."""

    @classmethod
    def setUpClass(cls):
        # Start the server in a subprocess
        cls.server = subprocess.Popen(["node", "src/server.js"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for server to be ready
        start = time.time()
        while time.time() - start < 120:
            try:
                r = requests.get(f"{BASE_URL}/health", timeout=10)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Server did not start in time")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.server.kill()

    def test_variable_isolation(self):
        """Test that variables from one execution don't leak to another"""
        # User A sets a variable
        r1 = requests.post(f"{BASE_URL}/api/execute", json={"code": "user_secret = 'User A secret data'\nuser_secret"}, timeout=10)
        self.assertEqual(r1.status_code, 200)
        response1 = r1.json()
        self.assertTrue(response1.get("success"))
        self.assertEqual(response1.get("result"), "User A secret data")
        
        # User B tries to access User A's variable (should fail)
        r2 = requests.post(f"{BASE_URL}/api/execute", json={"code": "user_secret"}, timeout=10)
        self.assertEqual(r2.status_code, 200)
        response2 = r2.json()
        self.assertFalse(response2.get("success"))
        self.assertIn("name 'user_secret' is not defined", response2.get("error", ""))
        
        # User B sets their own variable with same name
        r3 = requests.post(f"{BASE_URL}/api/execute", json={"code": "user_secret = 'User B different secret'\nuser_secret"}, timeout=10)
        self.assertEqual(r3.status_code, 200)
        response3 = r3.json()
        self.assertTrue(response3.get("success"))
        self.assertEqual(response3.get("result"), "User B different secret")
        
        # User A tries to access their variable again (should still fail - no persistence)
        r4 = requests.post(f"{BASE_URL}/api/execute", json={"code": "user_secret"}, timeout=10)
        self.assertEqual(r4.status_code, 200)
        response4 = r4.json()
        self.assertFalse(response4.get("success"))
        self.assertIn("name 'user_secret' is not defined", response4.get("error", ""))

    def test_function_isolation(self):
        """Test that function definitions don't leak between executions"""
        # User A defines a function
        code_a = """def user_a_function():
    return "User A's function"

user_a_function()"""
        
        r1 = requests.post(f"{BASE_URL}/api/execute", json={"code": code_a}, timeout=10)
        self.assertEqual(r1.status_code, 200)
        response1 = r1.json()
        self.assertTrue(response1.get("success"))
        self.assertEqual(response1.get("result"), "User A's function")
        
        # User B tries to call User A's function (should fail)
        r2 = requests.post(f"{BASE_URL}/api/execute", json={"code": "user_a_function()"}, timeout=10)
        self.assertEqual(r2.status_code, 200)
        response2 = r2.json()
        self.assertFalse(response2.get("success"))
        self.assertIn("name 'user_a_function' is not defined", response2.get("error", ""))

    def test_class_isolation(self):
        """Test that class definitions don't leak between executions"""
        # User A defines a class
        code_a = """class UserAClass:
    def __init__(self):
        self.data = "User A data"
    
    def get_data(self):
        return self.data

obj = UserAClass()
obj.get_data()"""
        
        r1 = requests.post(f"{BASE_URL}/api/execute", json={"code": code_a}, timeout=10)
        self.assertEqual(r1.status_code, 200)
        response1 = r1.json()
        self.assertTrue(response1.get("success"))
        self.assertEqual(response1.get("result"), "User A data")
        
        # User B tries to use User A's class (should fail)
        r2 = requests.post(f"{BASE_URL}/api/execute", json={"code": "UserAClass()"}, timeout=10)
        self.assertEqual(r2.status_code, 200)
        response2 = r2.json()
        self.assertFalse(response2.get("success"))
        self.assertIn("name 'UserAClass' is not defined", response2.get("error", ""))

    def test_import_isolation(self):
        """Test that imports in one execution don't affect another"""
        # User A imports a module and uses it
        code_a = """import math
math.pi"""
        
        r1 = requests.post(f"{BASE_URL}/api/execute", json={"code": code_a}, timeout=10)
        self.assertEqual(r1.status_code, 200)
        response1 = r1.json()
        self.assertTrue(response1.get("success"))
        self.assertAlmostEqual(response1.get("result"), 3.141592653589793)
        
        # User B tries to use math without importing (should fail)
        r2 = requests.post(f"{BASE_URL}/api/execute", json={"code": "math.pi"}, timeout=10)
        self.assertEqual(r2.status_code, 200)
        response2 = r2.json()
        self.assertFalse(response2.get("success"))
        self.assertIn("name 'math' is not defined", response2.get("error", ""))

    def test_global_state_isolation(self):
        """Test that global state modifications don't persist"""
        # User A sets a global counter
        code_a = """global_counter = 100
def increment():
    global global_counter
    global_counter += 1
    return global_counter

increment()"""
        
        r1 = requests.post(f"{BASE_URL}/api/execute", json={"code": code_a}, timeout=10)
        self.assertEqual(r1.status_code, 200)
        response1 = r1.json()
        self.assertTrue(response1.get("success"))
        self.assertEqual(response1.get("result"), 101)
        
        # User B tries to access the global counter (should fail)
        r2 = requests.post(f"{BASE_URL}/api/execute", json={"code": "global_counter"}, timeout=10)
        self.assertEqual(r2.status_code, 200)
        response2 = r2.json()
        self.assertFalse(response2.get("success"))
        self.assertIn("name 'global_counter' is not defined", response2.get("error", ""))
        
        # User B tries to call the increment function (should fail)
        r3 = requests.post(f"{BASE_URL}/api/execute", json={"code": "increment()"}, timeout=10)
        self.assertEqual(r3.status_code, 200)
        response3 = r3.json()
        self.assertFalse(response3.get("success"))
        self.assertIn("name 'increment' is not defined", response3.get("error", ""))

    def test_concurrent_isolation(self):
        """Test isolation between simultaneous requests"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def execute_code(user_id, code):
            try:
                r = requests.post(f"{BASE_URL}/api/execute", json={"code": code}, timeout=10)
                results.put((user_id, r.json() if r.status_code == 200 else None))
            except Exception as e:
                results.put((user_id, {"error": str(e)}))
        
        # Start multiple threads with different code
        threads = []
        codes = [
            ("user1", "user_var = 'user1_data'\nuser_var"),
            ("user2", "user_var = 'user2_data'\nuser_var"),
            ("user3", "user_var = 'user3_data'\nuser_var"),
        ]
        
        for user_id, code in codes:
            thread = threading.Thread(target=execute_code, args=(user_id, code))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)
        
        # Collect results
        user_results = {}
        while not results.empty():
            user_id, result = results.get()
            user_results[user_id] = result
        
        # Verify each user got their own data
        for user_id in ["user1", "user2", "user3"]:
            self.assertIn(user_id, user_results)
            result = user_results[user_id]
            self.assertIsNotNone(result)
            self.assertTrue(result.get("success"), f"User {user_id} execution failed: {result}")
            self.assertEqual(result.get("result"), f"{user_id}_data")


if __name__ == "__main__":
    unittest.main()

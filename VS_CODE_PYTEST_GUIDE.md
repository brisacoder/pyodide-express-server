# 🧪 Pytest Quick Reference for VS Code

## ✅ VS Code Configuration Complete!

Your VS Code is now configured for pytest with the following features:

### 🔧 **VS Code Settings (.vscode/settings.json)**
- ✅ Pytest enabled as primary test framework
- ✅ Unittest disabled
- ✅ Auto-discovery of tests on save
- ✅ Virtual environment configured
- ✅ Flake8 linting enabled
- ✅ Black formatting configured

### 🐛 **Debug Configurations (.vscode/launch.json)**
- ✅ Debug current test file
- ✅ Debug all tests
- ✅ Debug API tests specifically
- ✅ Debug any Python file

### ⚙️ **Pytest Configuration (pyproject.toml)**
- ✅ Test discovery settings
- ✅ Default command-line options
- ✅ Test markers for categorization
- ✅ Warning filters

---

## 🚀 How to Run Single Tests

### **Method 1: Command Line (Recommended)**
```bash
# Run a specific test method
uv run pytest tests/test_api.py::TestAPI::test_01_health -v

# Run tests matching a pattern
uv run pytest tests/test_api.py -k "health" -v

# Run tests with specific marker (when added)
uv run pytest tests/test_api.py -m "api" -v
```

### **Method 2: VS Code Test Explorer**
1. Open **Test Explorer** panel (🧪 icon in sidebar)
2. Click the **refresh/discover tests** button
3. Expand `test_api.py` → `TestAPI`
4. Click the **▶️ play button** next to any test name

### **Method 3: VS Code Command Palette**
1. Press `Ctrl+Shift+P`
2. Type "Python: Run Tests"
3. Select the test scope you want

### **Method 4: Code Lens (in test file)**
VS Code will show **"Run Test"** links above each test function when you open the test file.

---

## 🔍 How to Debug Tests

### **Method 1: Debug Configurations**
1. Press `F5` or go to **Run and Debug** panel
2. Select debug configuration:
   - **"Debug Current Test File"** - debugs the currently open test file
   - **"Debug API Tests"** - debugs just the API tests
   - **"Debug All Tests"** - debugs all tests

### **Method 2: Breakpoints + Test Runner**
1. Set breakpoints in your test code (`F9`)
2. Right-click on test in Test Explorer → **"Debug Test"**

### **Method 3: Command Line Debugging**
```bash
# Debug a specific test
uv run python -m debugpy --listen 5678 --wait-for-client -m pytest tests/test_api.py::TestAPI::test_01_health -v
```

---

## 📊 Useful Pytest Commands

```bash
# Run tests with different verbosity levels
uv run pytest tests/test_api.py -v          # Verbose
uv run pytest tests/test_api.py -vv         # Very verbose
uv run pytest tests/test_api.py -q          # Quiet

# Run tests and show output
uv run pytest tests/test_api.py -s          # Don't capture output

# Run tests with coverage
uv run pytest tests/test_api.py --cov=src --cov-report=html

# Run only fast tests (exclude slow ones)
uv run pytest tests/test_api.py -m "not slow"

# Run tests in parallel (if pytest-xdist installed)
uv run pytest tests/test_api.py -n auto

# Run tests and stop on first failure
uv run pytest tests/test_api.py -x

# Show local variables in tracebacks
uv run pytest tests/test_api.py --tb=long
```

---

## 🏷️ Test Markers (Available)

You can add these markers to your tests:

```python
import pytest

@pytest.mark.slow
def test_something_slow():
    """This test will be marked as slow."""
    pass

@pytest.mark.integration  
def test_integration_workflow():
    """This test will be marked as integration."""
    pass

@pytest.mark.api
def test_api_endpoint():
    """This test will be marked as API test."""
    pass
```

Then run:
```bash
# Run only API tests
uv run pytest -m "api"

# Run everything except slow tests
uv run pytest -m "not slow"

# Run integration and API tests
uv run pytest -m "integration or api"
```

---

## 🎯 VS Code Test Discovery

For VS Code to properly discover your tests:

1. **Ensure Python interpreter is set correctly:**
   - `Ctrl+Shift+P` → "Python: Select Interpreter"
   - Choose `./.venv/Scripts/python.exe`

2. **Refresh test discovery:**
   - Open Test Explorer panel
   - Click the refresh button
   - Or `Ctrl+Shift+P` → "Test: Refresh Tests"

3. **Check test discovery settings:**
   - Your tests should automatically appear in Test Explorer
   - Green ✅ for passing tests
   - Red ❌ for failing tests
   - Click any test to run/debug it

---

## 🐛 Troubleshooting

### Tests not appearing in VS Code?
1. Check Python interpreter selection
2. Ensure pytest is installed: `uv run pytest --version`
3. Refresh test discovery
4. Check VS Code Python extension is installed and updated

### Single test not running?
1. Make sure syntax is correct: `::ClassName::test_method_name`
2. Check that the test file and class/method names match exactly
3. Ensure you're in the correct directory

### Debugging not working?
1. Check that debugpy is installed: `uv add debugpy --dev`
2. Ensure breakpoints are set on executable lines
3. Try the VS Code debugger instead of command-line debugging

**Everything is now configured for optimal pytest workflow in VS Code! 🎉**
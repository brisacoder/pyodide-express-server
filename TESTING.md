# Testing Guide

This project has been simplified to use just two test runners for clarity:

## Test Runners

### 🚀 `run_comprehensive_tests.py` - Main Test Runner
**Use this for:** Complete validation of the entire system

**Features:**
- Automatically starts and stops the server
- Runs all test categories (10 categories total)
- Detailed reporting with timing and error tracking
- Server crash recovery (especially after security tests)
- Command-line options for selective testing

**Usage:**
```bash
# Run all tests
python run_comprehensive_tests.py

# Run specific categories
python run_comprehensive_tests.py --categories basic integration security

# Verbose output
python run_comprehensive_tests.py --verbose

# Quiet output  
python run_comprehensive_tests.py --quiet
```

**Test Categories:**
- `basic` - Basic API Tests
- `error` - Error Handling Tests
- `integration` - Integration Tests
- `security` - Security Tests
- `performance` - Performance Tests
- `reset` - Reset Tests
- `extra` - Extra Non-Happy Paths
- `sklearn` - Scikit-Learn Tests
- `matplotlib` - Matplotlib Plotting Tests
- `seaborn` - Seaborn Plotting Tests

### ⚡ `run_simple_tests.py` - Quick Smoke Tests
**Use this for:** Quick validation that basic functionality works

**Features:**
- Runs only basic API tests
- Assumes server is already running
- Fast execution for quick checks
- Good for development workflow

**Usage:**
```bash
# Start server first
npm start

# In another terminal, run quick tests
python run_simple_tests.py
```

## When to Use Which

- **Development:** Use `run_simple_tests.py` for quick checks while coding
- **CI/CD:** Use `run_comprehensive_tests.py` for complete validation
- **Before commits:** Use `run_comprehensive_tests.py` to ensure nothing is broken
- **Debugging:** Use `run_comprehensive_tests.py --categories <specific>` to focus on problem areas

## Notes

- The comprehensive runner handles all server management automatically
- Security tests may cause server crashes (this is expected and handled)
- Performance tests can take a long time - use selective testing during development
- All tests require the Python virtual environment to be activated

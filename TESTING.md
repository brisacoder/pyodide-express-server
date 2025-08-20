# Testing Guide

This project includes comprehensive testing with **three specialized test runners** designed for different use cases, plus extensive coverage of the new **Enhanced Security Logging System**.

## Test Runners

### 🚀 `run_comprehensive_tests.py` - Main Test Runner
**Use this for:** Complete validation of the entire system

**Features:**
- Automatically starts and stops the server
- Runs all test categories (12 categories total including new security logging)
- Detailed reporting with timing and error tracking
- Server crash recovery (especially after security tests)
- Command-line options for selective testing

**Usage:**
```bash
# Run all tests (recommended for CI/CD)
python run_comprehensive_tests.py

# Run specific categories
python run_comprehensive_tests.py --categories basic integration security_logging

# Run new security features
python run_comprehensive_tests.py --categories security security_logging

# Verbose output
python run_comprehensive_tests.py --verbose

# Quiet output  
python run_comprehensive_tests.py --quiet
```

### ⚡ `run_simple_tests.py` - Quick Development Tests
**Use this for:** Fast validation during development

**Features:**
- Runs core test categories: Basic API + Dynamic Modules + Security Logging
- Assumes server is already running
- Fast execution for quick checks
- Perfect for development workflow

**Usage:**
```bash
# Start server first
npm start

# In another terminal, run quick tests (39 tests)
python run_simple_tests.py
```

### 🔐 Security Logging Test Suite
**New Feature!** Comprehensive testing for the enhanced security logging system:

```bash
# Run just the security logging tests
python -m unittest tests.test_security_logging -v

# Or use the comprehensive runner
python run_comprehensive_tests.py --categories security_logging
```

## Test Categories (12 Total)

**Core Functionality:**
- `basic` - Basic API Tests (16 tests)
- `error` - Error Handling Tests
- `integration` - Integration Tests
- `performance` - Performance Tests
- `reset` - Reset Tests

**Security & Monitoring:**
- `security` - Security Tests
- `security_logging` - **NEW!** Enhanced Security Logging Tests (10 tests)

**Data Science & Visualization:**
- `sklearn` - Scikit-Learn Tests
- `matplotlib_base64` - Matplotlib Base64 Plotting Tests
- `matplotlib_vfs` - Matplotlib VFS Plotting Tests
- `seaborn_base64` - Seaborn Base64 Plotting Tests
- `seaborn_vfs` - Seaborn VFS Plotting Tests

**Advanced Features:**
- `extra` - Extra Non-Happy Paths
- `dynamic` - Dynamic Modules & Execution Robustness Tests (13 tests)

## When to Use Which

- **Development:** Use `run_simple_tests.py` for quick checks while coding (39 tests in ~1 second)
- **CI/CD:** Use `run_comprehensive_tests.py` for complete validation (all categories)
- **Before commits:** Use `run_comprehensive_tests.py` to ensure nothing is broken
- **Debugging:** Use `run_comprehensive_tests.py --categories <specific>` to focus on problem areas
- **Security Focus:** Use `--categories security security_logging` for security-related testing

## 🔐 Security Logging Test Details

The new security logging test suite (`tests/test_security_logging.py`) includes:

1. **Code Execution Tracking** - Verify SHA-256 hashing and execution logging
2. **Error Categorization** - Test error tracking and classification
3. **File Upload Monitoring** - Validate file upload event logging
4. **Statistics Accuracy** - Ensure metrics are calculated correctly over time
5. **Hourly Trend Tracking** - Test time-based analytics
6. **Dashboard Functionality** - Verify all dashboard endpoints work
7. **Legacy Compatibility** - Ensure backward compatibility maintained
8. **IP & User-Agent Tracking** - Test request metadata collection
9. **Execution Time Analysis** - Verify timing metrics accuracy
10. **Statistics Reset** - Test clearing/resetting functionality

## Test Environment Setup

### Python Virtual Environment (Required)
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install test dependencies
pip install -r requirements.txt
```

### Alternative: UV Package Manager (Faster)
```bash
# Install UV if not already installed
pip install uv

# Use UV for faster dependency installation
uv pip install -r requirements.txt
```

## Performance Benchmarks

**Simple Test Runner** (~1 second):
- 16 Basic API tests
- 13 Dynamic Modules tests  
- 10 Security Logging tests
- **Total: 39 tests**

**Comprehensive Test Runner** (~30-60 seconds):
- All 12 test categories
- **Total: 100+ tests**
- Server startup/shutdown included

## Notes

- The comprehensive runner handles all server management automatically
- Security tests may cause server crashes (this is expected and handled)
- Performance tests can take a long time - use selective testing during development
- All tests require the Python virtual environment to be activated
- **New security logging tests run in both simple and comprehensive modes**

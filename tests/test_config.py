"""Global test configuration and constants."""

# API Configuration
BASE_URL = "http://localhost:3000"

# Timeout Configuration (in seconds)
DEFAULT_TIMEOUT = 30
SERVER_START_TIMEOUT = 120
HEALTH_CHECK_TIMEOUT = 10
EXECUTE_TIMEOUT = 30
UPLOAD_TIMEOUT = 30
FILE_OPERATION_TIMEOUT = 10
PACKAGE_INSTALL_TIMEOUT = 120
RESET_TIMEOUT = 60

# API Endpoints
ENDPOINTS = {
    "health": "/health",
    "execute_raw": "/api/execute-raw",
    "upload_csv": "/api/upload-csv",
    "uploaded_files": "/api/uploaded-files",
    "file_info": "/api/file-info",
    "install_package": "/api/install-package",
    "reset": "/api/reset",
}

# Test Data
TEST_CSV_CONTENT = {
    "simple": "name,value,category\nitem1,1,A\nitem2,2,B\n",
    "quotes": 'name,description,value\n"Smith, John","A person named ""John""",42\n',
    "unicode": "name,value\nCafé,123\nNaïve,456\n",
    "empty_fields": "name,value,category\nitem1,,A\n,2,\n,,\n",
    "long_lines": "name,value\n" + "x" * 1000 + ",123\n",
}

# Test File Names
TEST_FILES = {
    "simple": "test_simple.csv",
    "quotes": "quotes.csv",
    "unicode": "unicode.csv",
    "empty_fields": "empty_fields.csv",
    "long_lines": "long_lines.csv",
    "concurrent": "concurrent.csv",
    "data1": "data1.csv",
    "data2": "data2.csv",
}

# Package Management
PACKAGES = {
    "beautifulsoup4": "beautifulsoup4",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
}

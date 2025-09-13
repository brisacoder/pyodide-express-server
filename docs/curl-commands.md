# API Commands for Pyodide Express Server

These examples show how to test the available endpoints using different methods:
- **Unix/Linux/macOS**: `curl` commands
- **PowerShell**: Both simplified `curl` and native `Invoke-RestMethod` commands
- Replace `sample.csv` with the path to your own file when needed

> The server is assumed to be running on `http://localhost:3000`

## PowerShell Tips 💡
- **Use `Invoke-RestMethod`** for cleaner JSON handling (recommended)
- **Use here-strings `@"..."@`** for multi-line content
- **Use single quotes** to avoid escaping in simple cases

## 1. Server health

**Unix/Linux/macOS**
```bash
curl http://localhost:3000/health
```

**PowerShell**
```powershell
# Both methods work the same for simple GET requests
curl http://localhost:3000/health
# OR
Invoke-RestMethod -Uri "http://localhost:3000/health"
```

## 2. Pyodide status

**Unix/Linux/macOS**
```bash
curl http://localhost:3000/api/status
```

**PowerShell**
```powershell
curl http://localhost:3000/api/status
# OR
Invoke-RestMethod -Uri "http://localhost:3000/api/status"
```

## 3. Execute Python code (`/execute`)

**Unix/Linux/macOS**
```bash
curl -X POST http://localhost:3000/api/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "print(\"Hello from Unix!\")\nresult = 2 + 2\nprint(f\"Result: {result}\")"}'
```

**PowerShell (curl)**
```powershell
$code = 'print("Hello from PowerShell!")\nresult = 2 + 2\nprint(f"Result: {result}")'
curl -X POST "http://localhost:3000/api/execute" `
  -H "Content-Type: application/json" `
  -d (@{code = $code} | ConvertTo-Json)
```

**PowerShell (Invoke-RestMethod - Recommended)**
```powershell
$body = @{
    code = @"
print("Hello from PowerShell!")
result = 2 + 2
print(f"Result: {result}")
"@
}

Invoke-RestMethod -Uri "http://localhost:3000/api/execute" `
  -Method POST -ContentType "application/json" `
  -Body ($body | ConvertTo-Json)
```

## 4. Execute raw Python code (`/execute-raw`)

**Unix/Linux/macOS**
```bash
curl -X POST http://localhost:3000/api/execute-raw \
  -H "Content-Type: text/plain" \
  --data 'print("Raw execution from Unix")
import math
print(f"Pi = {math.pi:.4f}")'
```

**PowerShell (curl)**
```powershell
$pythonCode = @"
print("Raw execution from PowerShell")
import math
print(f"Pi = {math.pi:.4f}")
"@

curl -X POST "http://localhost:3000/api/execute-raw" `
  -H "Content-Type: text/plain" `
  --data $pythonCode
```

**PowerShell (Invoke-RestMethod - Recommended)**
```powershell
$pythonCode = @"
print("Raw execution from PowerShell")
import math
print(f"Pi = {math.pi:.4f}")
"@

Invoke-RestMethod -Uri "http://localhost:3000/api/execute-raw" `
  -Method POST -ContentType "text/plain" `
  -Body $pythonCode
```

## 5. Upload CSV (`/upload`)

**Unix/Linux/macOS**
```bash
curl -X POST http://localhost:3000/api/upload \
  -F "csvFile=@sample.csv"
```

**PowerShell (curl)**
```powershell
curl -X POST "http://localhost:3000/api/upload" `
  -F "csvFile=@sample.csv"
```

**PowerShell (Invoke-RestMethod - Recommended)**
```powershell
$form = @{
    csvFile = Get-Item "sample.csv"
}

Invoke-RestMethod -Uri "http://localhost:3000/api/upload" `
  -Method POST -Form $form
```

## 6. List uploaded files

**Unix/Linux/macOS**
```bash
curl http://localhost:3000/api/uploaded-files
```

**PowerShell**
```powershell
# Both methods work the same for GET requests
curl http://localhost:3000/api/uploaded-files
# OR
Invoke-RestMethod -Uri "http://localhost:3000/api/uploaded-files"
```

## 7. File information

Replace `FILENAME` with the name returned by the upload endpoint.

**Unix/Linux/macOS**
```bash
curl http://localhost:3000/api/file-info/FILENAME
```

**PowerShell**
```powershell
curl "http://localhost:3000/api/file-info/FILENAME"
# OR
Invoke-RestMethod -Uri "http://localhost:3000/api/file-info/FILENAME"
```

## 8. List files in Pyodide filesystem

**Unix/Linux/macOS**
```bash
curl http://localhost:3000/api/pyodide-files
```

**PowerShell**
```powershell
curl http://localhost:3000/api/pyodide-files
# OR
Invoke-RestMethod -Uri "http://localhost:3000/api/pyodide-files"
```

## 9. Delete file from Pyodide filesystem

**Unix/Linux/macOS**
```bash
curl -X DELETE http://localhost:3000/api/pyodide-files/FILENAME
```

**PowerShell**
```powershell
curl -X DELETE "http://localhost:3000/api/pyodide-files/FILENAME"
# OR
Invoke-RestMethod -Uri "http://localhost:3000/api/pyodide-files/FILENAME" -Method DELETE
```

## 10. Delete uploaded file

**Unix/Linux/macOS**
```bash
curl -X DELETE http://localhost:3000/api/uploaded-files/FILENAME
```

**PowerShell**
```powershell
curl -X DELETE "http://localhost:3000/api/uploaded-files/FILENAME"
# OR
Invoke-RestMethod -Uri "http://localhost:3000/api/uploaded-files/FILENAME" -Method DELETE
```

## 11. Server statistics (Legacy)

**Unix/Linux/macOS**
```bash
curl http://localhost:3000/api/stats
```

**PowerShell**
```powershell
curl http://localhost:3000/api/stats
# OR
Invoke-RestMethod -Uri "http://localhost:3000/api/stats"
```

## 🔐 Security Dashboard Endpoints (NEW!)

### 12. Get security statistics (Enhanced)

**Unix/Linux/macOS**
```bash
curl http://localhost:3000/api/dashboard/stats
```

**PowerShell**
```powershell
curl http://localhost:3000/api/dashboard/stats
# OR
Invoke-RestMethod -Uri "http://localhost:3000/api/dashboard/stats"
```

### 13. View interactive dashboard (Web Interface)

**Unix/Linux/macOS**
```bash
# Opens the dashboard in your default browser
curl http://localhost:3000/api/dashboard/stats/dashboard
```

**PowerShell**
```powershell
# View dashboard HTML in browser
Start-Process "http://localhost:3000/api/dashboard/stats/dashboard"

# Or get the HTML content
Invoke-RestMethod -Uri "http://localhost:3000/api/dashboard/stats/dashboard"
```

### 14. Clear/Reset security statistics

**Unix/Linux/macOS**
```bash
curl -X DELETE http://localhost:3000/api/dashboard/stats/clear
```

**PowerShell**
```powershell
curl -X DELETE "http://localhost:3000/api/dashboard/stats/clear"
# OR
Invoke-RestMethod -Uri "http://localhost:3000/api/dashboard/stats/clear" -Method DELETE
```

## 15. Install a Python package

**Unix/Linux/macOS**
```bash
curl -X POST http://localhost:3000/api/install-package \
  -H "Content-Type: application/json" \
  -d '{"package": "beautifulsoup4"}'
```

**PowerShell (curl)**
```powershell
curl -X POST "http://localhost:3000/api/install-package" `
  -H "Content-Type: application/json" `
  -d '{"package": "beautifulsoup4"}'
```

**PowerShell (Invoke-RestMethod - Recommended)**
```powershell
$body = @{ package = "beautifulsoup4" }

Invoke-RestMethod -Uri "http://localhost:3000/api/install-package" `
  -Method POST -ContentType "application/json" `
  -Body ($body | ConvertTo-Json)
```

## 16. List installed packages

**Unix/Linux/macOS**
```bash
curl http://localhost:3000/api/packages
```

**PowerShell**
```powershell
curl http://localhost:3000/api/packages
# OR
Invoke-RestMethod -Uri "http://localhost:3000/api/packages"
```

## 17. Reset Pyodide environment

**Unix/Linux/macOS**
```bash
curl -X POST http://localhost:3000/api/reset
```

**PowerShell**
```powershell
curl -X POST "http://localhost:3000/api/reset"
# OR
Invoke-RestMethod -Uri "http://localhost:3000/api/reset" -Method POST
```

These commands provide a quick way to verify every API route.

## Practical PowerShell Examples 🚀

### 🔐 Security Monitoring Workflow
```powershell
# 1. Execute some Python code to generate security events
$pythonCode = @"
import matplotlib.pyplot as plt
import pandas as pd

# Create sample data
data = {'x': [1, 2, 3, 4], 'y': [2, 4, 6, 8]}
df = pd.DataFrame(data)

# Create plot
plt.figure(figsize=(8, 6))
plt.plot(df['x'], df['y'], 'bo-')
plt.title('Security Test Plot')
plt.savefig('/plots/matplotlib/security_test.png')
print("Security logging test completed")
"@

$result = Invoke-RestMethod -Uri "http://localhost:3000/api/execute" `
  -Method POST -ContentType "application/json" `
  -Body (@{code = $pythonCode} | ConvertTo-Json)

# 2. Check security statistics
$stats = Invoke-RestMethod -Uri "http://localhost:3000/api/dashboard/stats"
Write-Output "Current Security Stats:"
Write-Output $stats

# 3. Open the interactive dashboard
Start-Process "http://localhost:3000/api/dashboard/stats/dashboard"
```

### Data Science Workflow
```powershell
# 1. Execute Python code with matplotlib
$pythonCode = @"
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.title('Sample Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.savefig('/plots/matplotlib/sine_wave.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot saved successfully!")
"@

$result = Invoke-RestMethod -Uri "http://localhost:3000/api/execute" `
  -Method POST -ContentType "application/json" `
  -Body (@{code = $pythonCode} | ConvertTo-Json)

Write-Output $result
```

### Package Management
```powershell
# Install a package and use it
$installResult = Invoke-RestMethod -Uri "http://localhost:3000/api/install-package" `
  -Method POST -ContentType "application/json" `
  -Body (@{package = "requests"} | ConvertTo-Json)

# Use the installed package
$testCode = @"
import requests
print("Testing requests package...")
print(f"requests version: {requests.__version__}")
"@

$testResult = Invoke-RestMethod -Uri "http://localhost:3000/api/execute" `
  -Method POST -ContentType "application/json" `
  -Body (@{code = $testCode} | ConvertTo-Json)

Write-Output $testResult
```

## Running the Python API tests

The repository includes `tests/test_api.py`, which programmatically exercises
all endpoints using the `requests` library. Install the dependency and run:

```bash
pip install requests
python -m unittest tests.test_api
```

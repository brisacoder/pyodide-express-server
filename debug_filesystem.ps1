# Test filesystem mounting via API
$body = @{
    code = @"
from pathlib import Path

# Test file creation in mounted directory
plots_file = Path("/plots/matplotlib/debug_test.txt")
plots_file.parent.mkdir(parents=True, exist_ok=True)
plots_file.write_text("Debug test content from Pyodide")

print("File created:", plots_file)
print("File exists:", plots_file.exists())
print("File content:", plots_file.read_text())
print("Parent directory:", plots_file.parent)
print("Parent exists:", plots_file.parent.exists())

# List files in parent directory
import os
files = os.listdir("/plots/matplotlib")
print("Files in /plots/matplotlib:", files)
"@
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri 'http://localhost:3000/api/execute-raw' -Method POST -Body $body -ContentType 'application/json'
Write-Output $response
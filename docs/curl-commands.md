# Curl commands for Pyodide Express Server

These examples show how to test the available endpoints with `curl`.
For each endpoint there is a version that works on Unix-like shells
(macOS/Linux) and a Windows PowerShell variant. Replace `sample.csv`
with the path to your own file when needed.

> The server is assumed to be running on `http://localhost:3000`.

## 1. Server health

**Unix**
```sh
curl http://localhost:3000/health
```

**Windows**
```powershell
curl http://localhost:3000/health
```

## 2. Pyodide status

**Unix**
```sh
curl http://localhost:3000/api/status
```

**Windows**
```powershell
curl http://localhost:3000/api/status
```

## 3. Execute Python code (`/execute`)

**Unix**
```sh
curl -X POST http://localhost:3000/api/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "\"\"\"curl demo\"\"\"\nname=\"Unix\"\nf\"Hello {name}\""}'
```

**Windows**
```powershell
curl -X POST "http://localhost:3000/api/execute" \
  -H "Content-Type: application/json" \
  -d "{\"code\": \"\\\"\\\"\\\"curl demo\\\"\\\"\\\"\\nname=\\\"Windows\\\"\\nf\\\"Hello {name}\\\"\"}"
```

## 4. Execute raw Python code (`/execute-raw`)

**Unix**
```sh
curl -X POST http://localhost:3000/api/execute-raw \
  -H "Content-Type: text/plain" \
  --data '"""raw demo"""\nname="Unix"\nprint(f"""Hi {name}""")'
```

**Windows**
```powershell
curl -X POST "http://localhost:3000/api/execute-raw" \
  -H "Content-Type: text/plain" \
  --data "\"\"\"raw demo\"\"\"\\nname=\\\"Windows\\\"\\nprint(f\"\"\"Hi {name}\"\"\")"
```

## 5. Upload CSV (`/upload-csv`)

**Unix**
```sh
curl -X POST http://localhost:3000/api/upload-csv \
  -F "csvFile=@sample.csv"
```

**Windows**
```powershell
curl -X POST "http://localhost:3000/api/upload-csv" \
  -F "csvFile=@sample.csv"
```

## 6. List uploaded files

**Unix**
```sh
curl http://localhost:3000/api/uploaded-files
```

**Windows**
```powershell
curl http://localhost:3000/api/uploaded-files
```

## 7. File information

Replace `FILENAME` with the name returned by the upload endpoint.

**Unix**
```sh
curl http://localhost:3000/api/file-info/FILENAME
```

**Windows**
```powershell
curl "http://localhost:3000/api/file-info/FILENAME"
```

## 8. List files in Pyodide filesystem

**Unix**
```sh
curl http://localhost:3000/api/pyodide-files
```

**Windows**
```powershell
curl http://localhost:3000/api/pyodide-files
```

## 9. Delete file from Pyodide filesystem

**Unix**
```sh
curl -X DELETE http://localhost:3000/api/pyodide-files/FILENAME
```

**Windows**
```powershell
curl -X DELETE "http://localhost:3000/api/pyodide-files/FILENAME"
```

## 10. Delete uploaded file

**Unix**
```sh
curl -X DELETE http://localhost:3000/api/uploaded-files/FILENAME
```

**Windows**
```powershell
curl -X DELETE "http://localhost:3000/api/uploaded-files/FILENAME"
```

## 11. Install a Python package

**Unix**
```sh
curl -X POST http://localhost:3000/api/install-package \
  -H "Content-Type: application/json" \
  -d '{"package": "beautifulsoup4"}'
```

**Windows**
```powershell
curl -X POST "http://localhost:3000/api/install-package" \
  -H "Content-Type: application/json" \
  -d "{\"package\": \"beautifulsoup4\"}"
```

## 12. List installed packages

**Unix**
```sh
curl http://localhost:3000/api/packages
```

**Windows**
```powershell
curl http://localhost:3000/api/packages
```

## 13. Reset Pyodide environment

**Unix**
```sh
curl -X POST http://localhost:3000/api/reset
```

**Windows**
```powershell
curl -X POST "http://localhost:3000/api/reset"
```

## 14. Server statistics

**Unix**
```sh
curl http://localhost:3000/api/stats
```

**Windows**
```powershell
curl http://localhost:3000/api/stats
```

These commands provide a quick way to verify every API route.

## Running the Python API tests

The repository includes `tests/test_api.py`, which programmatically exercises
all endpoints using the `requests` library. Install the dependency and run:

```bash
pip install requests
python -m unittest tests.test_api
```


# Testing Rules that MUST be followed

 - Convert to Pytest if not already
 - Just back up the original file and work on a file with the same original name. 
 - Make sure any globals such as timeout is not hardcoded. Parameterize everything with constants or fixtures.
 - These tests should not use internal REST APIs, those that have 'pyodide' in them 
 - Tests should use BDD
 - Only use endpoint /api/execute-raw for pyodide code execution. No need for insane character escaping, it takes plain text
 - Do not use internal REST APIs. Internal API are those with 'pyodide' in the URL
 - Make sure coverage is comprehensive
 - Add full docstrings, include description, input, outputs and examples
 - Pyodide code must always be portable across windows and Linux. For example, always use pathlib for portability
 - Make sure JavaScript returns information following the API contract as text. This should be fixed on JavaScript Server side always! 

{
  "success": true | false,           // Indicates if the operation was successful
  "data": <object|null>,             // Main result data (object or null if error)
  "error": <string|null>,            // Error message (string or null if success)
  "meta": { "timestamp": <string> } // Metadata, always includes ISO timestamp
}


stdout, stderr and result should be under data. 

 - Pyodide can return whatever it wants, JavaScript should get the result and add under data.result. 
 - Don't hack the tests to fit a broken API - fix the server to follow the contract
 - DO not under any circumstances change the tests to fit a broken returned contract from server. Fix the server
 - UV is used for everything Python
 - Activate venv always
 - if the chat is iterative means I am paying attention and server is ALWAYS running with nodemon



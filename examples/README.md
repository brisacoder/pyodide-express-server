# Examples

This folder contains example clients showing how to interact with the Pyodide Express Server API.

## Available Examples

- **`basic-client.js`** – Simple example using `fetch` to send Python code to `/api/execute` endpoint
- **`data-science-client.js`** – Comprehensive data science workflow with pandas, matplotlib, and seaborn
- **`file-upload-client.js`** – File upload and processing example with CSV analysis
- **`execute-raw-client.js`** – Demonstrates `/api/execute-raw` endpoint for complex Python code

## Running Examples

1. **Start the server first:**
   ```bash
   npm start
   ```

2. **Run any example:**
   ```bash
   node examples/basic-client.js
   node examples/data-science-client.js
   node examples/file-upload-client.js
   node examples/execute-raw-client.js
   ```

## Example Descriptions

### Basic Client
The simplest possible example showing how to execute Python code and get results.

### Data Science Client
Demonstrates:
- Package installation (seaborn)
- Data analysis with pandas and numpy
- Statistical visualization with matplotlib/seaborn
- Direct filesystem plotting (plots saved to local directories)
- Server health checking

### File Upload Client  
Shows complete file upload workflow:
- Creating and uploading CSV files
- Processing uploaded data with pandas
- Creating visualizations from uploaded data
- File management (list, info, delete)
- Error handling and cleanup

### Execute Raw Client
Compares `/api/execute` vs `/api/execute-raw` endpoints:
- Simple vs complex Python code examples
- String escaping and JSON issues
- F-strings, regex patterns, and file paths
- When to use each endpoint

## Requirements

- Node.js 18+ (for built-in fetch support)
- Running Pyodide Express Server on localhost:3000

## Tips

- Use `/api/execute-raw` for complex Python code with quotes and escape sequences
- Use `/api/execute` for simple code when you need JSON structure and timeout control
- Check the `plots/` directory for generated visualizations
- Examples include comprehensive error handling patterns

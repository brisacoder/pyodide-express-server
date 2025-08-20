/**
 * Execute Raw Example Client
 * 
 * Demonstrates the difference between /api/execute and /api/execute-raw endpoints.
 * The execute-raw endpoint is better for complex Python code that contains quotes,
 * escape sequences, or other characters that might cause JSON escaping issues.
 *
 * Run the server in another terminal with `npm start` and then execute:
 *   node examples/execute-raw-client.js
 */

async function main() {
  const baseUrl = 'http://localhost:3000';

  try {
    console.log('🚀 Comparing /api/execute vs /api/execute-raw endpoints...\n');

    // Example 1: Simple code that works with both endpoints
    console.log('1. Simple code example (both endpoints work):');
    const simpleCode = `print("Hello from Pyodide!")
result = 2 + 2
print(f"2 + 2 = {result}")`;

    // Using /api/execute (JSON wrapped)
    const jsonResponse = await fetch(`${baseUrl}/api/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code: simpleCode }),
    });
    const jsonResult = await jsonResponse.json();
    console.log('JSON endpoint result:', jsonResult.success ? 'SUCCESS' : 'FAILED');

    // Using /api/execute-raw (raw text)
    const rawResponse = await fetch(`${baseUrl}/api/execute-raw`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' },
      body: simpleCode,
    });
    const rawResult = await rawResponse.json();
    console.log('RAW endpoint result:', rawResult.success ? 'SUCCESS' : 'FAILED');

    // Example 2: Complex code with quotes and escape sequences
    console.log('\n2. Complex code with quotes and escape sequences:');
    const complexCode = `# This code contains various quote types and escape sequences
import json

# String with mixed quotes
message = 'This is a "complex" string with \\'single\\' and "double" quotes'
print(message)

# JSON data with nested quotes
data = {
    "name": "John's Data",
    "description": "Contains \\"nested\\" quotes and \\\\backslashes",
    "path": "C:\\\\Users\\\\Documents\\\\file.txt",
    "regex": "\\\\d+\\\\.\\\\d+"
}

print("JSON data:")
print(json.dumps(data, indent=2))

# Raw string with backslashes
regex_pattern = r"\\d+\\.\\d+\\s+[A-Z]+"
print(f"Regex pattern: {regex_pattern}")

# Triple-quoted string
multiline = """
This is a multiline string
with "quotes" and 'apostrophes'
and even \\backslashes\\
"""
print("Multiline string:", multiline.strip())

print("✅ Complex string handling complete!")`;

    console.log('\nTesting complex code with /api/execute-raw:');
    const complexRawResponse = await fetch(`${baseUrl}/api/execute-raw`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' },
      body: complexCode,
    });
    const complexRawResult = await complexRawResponse.json();
    
    if (complexRawResult.success) {
      console.log('✅ RAW endpoint handled complex code successfully!');
      const output = complexRawResult.stdout || '';
      console.log('Output preview:', output.substring(0, 200) + '...');
    } else {
      console.log('❌ RAW endpoint failed:', complexRawResult.error);
    }

    // Example 3: Code that would be problematic with JSON escaping
    console.log('\n3. Code that benefits from raw execution:');
    const problematicCode = `# F-strings with complex formatting
import datetime
import math

now = datetime.datetime.now()
pi = math.pi

# Complex f-string formatting
formatted_text = f"""
Current time: {now:%Y-%m-%d %H:%M:%S}
Pi value: {pi:.6f}
Expression: {2**8} = {2**8}
Percentage: {0.1234:.2%}
JSON-like: {{"key": "value with \\"quotes\\"", "number": {42}}}
"""

print(formatted_text)

# File path operations (common source of escaping issues)
import os
paths = [
    "C:\\\\Program Files\\\\App\\\\config.json",
    "/home/user/.config/app/settings.txt",
    "..\\\\..\\\\data\\\\file.csv"
]

for path in paths:
    print(f"Path: {path}")
    print(f"Basename: {os.path.basename(path)}")

print("🎯 Raw execution handles all edge cases perfectly!")`;

    console.log('Testing problematic code with /api/execute-raw:');
    const problematicRawResponse = await fetch(`${baseUrl}/api/execute-raw`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' },
      body: problematicCode,
    });
    const problematicRawResult = await problematicRawResponse.json();
    
    if (problematicRawResult.success) {
      console.log('✅ RAW endpoint handled problematic code successfully!');
      console.log('Full output:');
      console.log(problematicRawResult.stdout);
    } else {
      console.log('❌ RAW endpoint failed:', problematicRawResult.error);
    }

    console.log('\n📊 Summary:');
    console.log('- Use /api/execute for simple code with JSON structure and timeout control');
    console.log('- Use /api/execute-raw for complex code with quotes, escapes, and formatting');
    console.log('- Raw endpoint avoids JSON escaping issues entirely');
    console.log('- Raw endpoint is perfect for f-strings, regex, file paths, and multiline code');

    console.log('\n🎉 Execute Raw Example completed successfully!');

  } catch (error) {
    console.error('❌ Example failed:', error.message);
  }
}

main();

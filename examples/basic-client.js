/**
 * Minimal client demonstrating how to call the Pyodide Express Server.
 *
 * Run the server in another terminal with `npm start` and then execute:
 *   node examples/basic-client.js
 */

async function main() {
  const code = "print('Hello from Pyodide')";

  const response = await fetch('http://localhost:3000/api/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code }),
  });

  const result = await response.json();
  console.log(result);
}

main().catch((err) => {
  console.error('Request failed:', err);
});

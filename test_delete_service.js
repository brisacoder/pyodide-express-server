const pyodideService = require('./src/services/pyodide-service');

async function testDeleteNonexistent() {
  await pyodideService.initialize();
  
  const result = await pyodideService.deletePyodideFile('nonexistent.csv');
  console.log('Result:', JSON.stringify(result, null, 2));
  process.exit(0);
}

testDeleteNonexistent().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});

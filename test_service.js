const pyodideService = require('./src/services/pyodide-service');

async function testSyntaxError() {
  await pyodideService.initialize();
  
  const code = `if True
    print("missing colon")`;
    
  const result = await pyodideService.executeCode(code);
  console.log('Result:', JSON.stringify(result, null, 2));
  process.exit(0);
}

testSyntaxError().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});

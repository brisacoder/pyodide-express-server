const pyodideService = require('./src/services/pyodide-service');

async function testDeleteCode() {
  await pyodideService.initialize();
  
  const code = `
import os
import json
try:
    if os.path.exists('nonexistent.csv'):
        os.remove('nonexistent.csv')
        json.dumps({"success": True, "message": f"File nonexistent.csv deleted successfully"})
    else:
        json.dumps({"success": False, "error": f"File nonexistent.csv not found"})
except Exception as e:
    json.dumps({"success": False, "error": f"Error deleting nonexistent.csv: {str(e)}"})
  `;
  
  const result = await pyodideService.executeCode(code);
  console.log('Execute result:', JSON.stringify(result, null, 2));
  process.exit(0);
}

testDeleteCode().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});

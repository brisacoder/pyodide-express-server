/**
 * Simple Pyodide Test
 * 
 * This script tests just the Pyodide loading to isolate the issue
 * Run with: node test-pyodide-simple.js
 */

async function testPyodideLoading() {
  console.log('🧪 Testing Pyodide Loading in Node.js');
  console.log('=' .repeat(50));
  
  try {
    console.log('1. Importing Pyodide...');
    const { loadPyodide } = await import('pyodide');
    console.log('   ✅ Import successful');
    
    console.log('2. Loading Pyodide (no config - uses bundled files)...');
    const pyodide = await Promise.race([
      loadPyodide(),
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Timeout after 30 seconds')), 30000)
      )
    ]);
    
    console.log('   ✅ Pyodide loaded successfully!');
    console.log('   Version:', pyodide.version);
    
    console.log('3. Testing basic Python execution...');
    const result = pyodide.runPython(`
import sys
print(f"Python version: {sys.version}")
print("Hello from Pyodide!")
2 + 2
    `);
    
    console.log('   ✅ Python execution successful');
    console.log('   Result: 2 + 2 =', result);
    
    console.log('4. Testing package loading...');
    try {
      await pyodide.loadPackage(['numpy']);
      console.log('   ✅ NumPy loaded successfully');
      
      const numpyTest = pyodide.runPython(`
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"NumPy array: {arr}")
arr.sum()
      `);
      
      console.log('   ✅ NumPy test successful, sum =', numpyTest);
    } catch (packageError) {
      console.log('   ⚠️  Package loading failed (this is okay for basic functionality)');
      console.log('   Error:', packageError.message);
    }
    
    console.log();
    console.log('🎉 SUCCESS! Pyodide is working correctly in your Node.js environment.');
    console.log('   You can now start your server with: npm start');
    
  } catch (error) {
    console.log('❌ FAILED:', error.message);
    console.log();
    console.log('💡 Possible solutions:');
    console.log('1. Try different Node.js version (16, 18, or 20)');
    console.log('2. Clear cache: npm cache clean --force');
    console.log('3. Reinstall: npm uninstall pyodide && npm install pyodide');
    console.log('4. Check antivirus/firewall settings');
    console.log('5. Try running as Administrator (Windows)');
  }
}

// Run the test
testPyodideLoading().catch(console.error);
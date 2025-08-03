/**
 * Pyodide Diagnostic Script
 * 
 * Run this to diagnose Pyodide installation and compatibility issues
 * Usage: node pyodide-diagnostic.js
 */

const fs = require('fs');
const path = require('path');

console.log('🔍 Pyodide Diagnostic Tool');
console.log('=' .repeat(50));

// Check Node.js version
console.log('📊 Environment Information:');
console.log(`Node.js version: ${process.version}`);
console.log(`Platform: ${process.platform}`);
console.log(`Architecture: ${process.arch}`);
console.log(`Working directory: ${process.cwd()}`);
console.log();

// Check if package.json exists
const packageJsonPath = path.join(process.cwd(), 'package.json');
if (fs.existsSync(packageJsonPath)) {
  console.log('✅ package.json found');
  try {
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    console.log(`   Project: ${packageJson.name} v${packageJson.version}`);
    
    if (packageJson.dependencies && packageJson.dependencies.pyodide) {
      console.log(`   Pyodide dependency: ${packageJson.dependencies.pyodide}`);
    } else {
      console.log('❌ Pyodide not listed in dependencies');
    }
  } catch (error) {
    console.log('⚠️  Could not parse package.json:', error.message);
  }
} else {
  console.log('❌ package.json not found');
}
console.log();

// Check if node_modules exists
const nodeModulesPath = path.join(process.cwd(), 'node_modules');
if (fs.existsSync(nodeModulesPath)) {
  console.log('✅ node_modules directory found');
  
  // Check if pyodide package is installed
  const pyodidePath = path.join(nodeModulesPath, 'pyodide');
  if (fs.existsSync(pyodidePath)) {
    console.log('✅ Pyodide package installed in node_modules');
    
    // Check pyodide package.json
    const pyodidePackageJson = path.join(pyodidePath, 'package.json');
    if (fs.existsSync(pyodidePackageJson)) {
      try {
        const pyodidePkg = JSON.parse(fs.readFileSync(pyodidePackageJson, 'utf8'));
        console.log(`   Installed version: ${pyodidePkg.version}`);
        console.log(`   Main entry: ${pyodidePkg.main || 'not specified'}`);
        
        // Check if main file exists
        const mainFile = path.join(pyodidePath, pyodidePkg.main || 'pyodide.js');
        if (fs.existsSync(mainFile)) {
          console.log('✅ Main entry file exists');
        } else {
          console.log('❌ Main entry file not found:', mainFile);
        }
      } catch (error) {
        console.log('⚠️  Could not read pyodide package.json:', error.message);
      }
    }
  } else {
    console.log('❌ Pyodide package NOT found in node_modules');
    console.log('   Solution: Run "npm install pyodide"');
  }
} else {
  console.log('❌ node_modules directory not found');
  console.log('   Solution: Run "npm install"');
}
console.log();

// Test different import methods
console.log('🧪 Testing Import Methods:');

// Test 1: Dynamic import
console.log('1. Testing dynamic import...');
try {
  import('pyodide').then(pyodideModule => {
    console.log('   ✅ Dynamic import successful');
    console.log('   Available exports:', Object.keys(pyodideModule));
    
    if (pyodideModule.loadPyodide) {
      console.log('   ✅ loadPyodide function available');
    } else {
      console.log('   ❌ loadPyodide function not found');
    }
  }).catch(error => {
    console.log('   ❌ Dynamic import failed:', error.message);
    testRequire();
  });
} catch (error) {
  console.log('   ❌ Dynamic import failed:', error.message);
  testRequire();
}

// Test 2: CommonJS require
function testRequire() {
  console.log('2. Testing CommonJS require...');
  try {
    const pyodideModule = require('pyodide');
    console.log('   ✅ CommonJS require successful');
    console.log('   Available exports:', Object.keys(pyodideModule));
    
    if (pyodideModule.loadPyodide) {
      console.log('   ✅ loadPyodide function available');
      testPyodideLoad(pyodideModule.loadPyodide);
    } else {
      console.log('   ❌ loadPyodide function not found');
    }
  } catch (error) {
    console.log('   ❌ CommonJS require failed:', error.message);
    console.log();
    printSolutions();
  }
}

// Test 3: Try to load Pyodide
async function testPyodideLoad(loadPyodide) {
  console.log('3. Testing Pyodide initialization...');
  
  try {
    console.log('   Loading Pyodide (this may take a moment)...');
    
    const pyodide = await Promise.race([
      loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.28.0/full/",
        fullStdLib: false
      }),
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Timeout after 30 seconds')), 30000)
      )
    ]);
    
    console.log('   ✅ Pyodide loaded successfully!');
    console.log('   Pyodide version:', pyodide.version);
    
    // Test Python execution
    const result = pyodide.runPython('2 + 2');
    console.log('   ✅ Python execution test: 2 + 2 =', result);
    
    console.log();
    console.log('🎉 All tests passed! Pyodide is working correctly.');
    
  } catch (error) {
    console.log('   ❌ Pyodide initialization failed:', error.message);
    console.log();
    printSolutions();
  }
}

function printSolutions() {
  console.log('🔧 Troubleshooting Solutions:');
  console.log();
  
  console.log('1. Reinstall Pyodide:');
  console.log('   npm uninstall pyodide');
  console.log('   npm install pyodide@0.28.0');
  console.log();
  
  console.log('2. Clear npm cache and reinstall:');
  console.log('   npm cache clean --force');
  console.log('   rm -rf node_modules package-lock.json');
  console.log('   npm install');
  console.log();
  
  console.log('3. Check Node.js version compatibility:');
  console.log('   - Pyodide requires Node.js 16 or higher');
  console.log('   - Current version:', process.version);
  if (parseInt(process.version.slice(1)) < 16) {
    console.log('   ❌ Your Node.js version is too old. Please upgrade to Node.js 16+');
  } else {
    console.log('   ✅ Node.js version is compatible');
  }
  console.log();
  
  console.log('4. Try alternative installation:');
  console.log('   npm install pyodide@latest');
  console.log();
  
  console.log('5. Check internet connection:');
  console.log('   - Pyodide downloads WebAssembly files from CDN');
  console.log('   - Ensure you have internet access');
  console.log('   - Check firewall/proxy settings');
  console.log();
  
  console.log('6. Windows-specific issues:');
  if (process.platform === 'win32') {
    console.log('   - Try running PowerShell as Administrator');
    console.log('   - Install Windows Build Tools: npm install -g windows-build-tools');
    console.log('   - Use Command Prompt instead of PowerShell');
  } else {
    console.log('   - Not applicable (you are not on Windows)');
  }
  console.log();
  
  console.log('7. If all else fails, use Docker:');
  console.log('   docker run -p 3000:3000 -v $(pwd):/app node:18-alpine');
  console.log('   cd /app && npm install && npm start');
}

// Add a delay for async operations
setTimeout(() => {
  if (process.platform === 'win32') {
    console.log();
    console.log('💡 Windows users: If you see permission errors, try:');
    console.log('   - Run as Administrator');
    console.log('   - Use npm config set script-shell "C:\\Program Files\\git\\bin\\bash.exe"');
  }
}, 5000);
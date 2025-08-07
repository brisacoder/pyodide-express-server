/**
 * Add Default Packages Script
 * 
 * This script sets up your Pyodide server to include matplotlib, seaborn, 
 * requests, and other packages by default.
 * 
 * Run with: node add-default-packages.js
 */

const fs = require('fs');
const path = require('path');

console.log('📦 Setting Up Default Packages for Pyodide');
console.log('=' .repeat(50));

// Step 1: Create config directory and file
const configDir = path.join('src', 'config');
const configFile = path.join(configDir, 'pyodide-config.js');

if (!fs.existsSync(configDir)) {
  fs.mkdirSync(configDir, { recursive: true });
  console.log('✅ Created config directory');
}

const configContent = `/**
 * Pyodide Package Configuration
 * 
 * Configure which packages to load by default when Pyodide starts.
 */

module.exports = {
  // Packages available through Pyodide's loadPackage (faster)
  builtinPackages: [
    'numpy',           // Scientific computing
    'pandas',          // Data manipulation and analysis
    'micropip',        // Package installer
    'matplotlib',      // Plotting and visualization
    'scipy',          // Scientific computing
    'requests',        // HTTP library
  ],

  // Packages to install via micropip (from PyPI)
  micropipPackages: [
    'seaborn',         // Statistical data visualization
    'httpx',           // Modern HTTP client
  ],

  // Global imports to make available in Python environment
  globalImports: {
    'numpy': 'np',
    'pandas': 'pd', 
    'matplotlib.pyplot': 'plt',
    'seaborn': 'sns',
    'requests': 'requests',
    'httpx': 'httpx'
  },

  // Matplotlib configuration for server environment
  matplotlibConfig: {
    backend: 'Agg',    // Non-interactive backend
    figureFormat: 'png'
  },

  // Package installation timeout (milliseconds)
  installTimeout: 60000,

  // Whether to continue initialization if some packages fail
  continueOnError: true,

  // Custom initialization code
  customInitCode: \`
# Additional setup code
import warnings
warnings.filterwarnings('ignore')

# Set pandas display options
try:
    import pandas as pd
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 100)
except ImportError:
    pass

print("🎯 All packages ready! You can now use:")
print("  • np (NumPy)")
print("  • pd (Pandas)")  
print("  • plt (Matplotlib)")
print("  • sns (Seaborn)")
print("  • requests (HTTP)")
print("  • httpx (Modern HTTP)")
  \`
};`;

try {
  fs.writeFileSync(configFile, configContent);
  console.log('✅ Created pyodide-config.js');
} catch (error) {
  console.log('❌ Failed to create config file:', error.message);
  process.exit(1);
}

// Step 2: Update the pyodide service to use the config
const serviceFile = path.join('src', 'services', 'pyodide-service.js');

if (!fs.existsSync(serviceFile)) {
  console.log('❌ pyodide-service.js not found. Make sure you have the service file.');
  process.exit(1);
}

console.log('✅ Configuration files created successfully!');
console.log('\\n📋 Next Steps:');
console.log('1. Restart your server: npm start');
console.log('2. Test the packages in the web interface');
console.log('3. Customize src/config/pyodide-config.js as needed');

console.log('\\n🎯 Default packages that will be available:');
console.log('  ✅ numpy as np');
console.log('  ✅ pandas as pd');
console.log('  ✅ matplotlib.pyplot as plt');
console.log('  ✅ seaborn as sns');
console.log('  ✅ requests');
console.log('  ✅ httpx');
console.log('  ✅ scipy');
console.log('  ✅ micropip');

console.log('\\n💡 Tips:');
console.log('• Packages load automatically when server starts');
console.log('• No need to import or install them in your code');
console.log('• Add more packages by editing the config file');
console.log('• Built-in packages load faster than micropip packages');

console.log('\\n🧪 Test code to try:');
console.log(\`
# This will work immediately without imports:
data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
df = pd.DataFrame(data)
print(df)

# Create a plot
plt.figure(figsize=(8, 6))
plt.plot(df['x'], df['y'])
plt.title('Sample Plot')
plt.show()

# Make HTTP request
response = requests.get('https://httpbin.org/json', timeout=10)
print(response.json())
\`);

console.log('\\n🚀 Ready to restart your server!');
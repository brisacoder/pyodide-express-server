/**
 * Quick Logger Fix
 * 
 * Adds the missing getLogInfo method to your logger
 * Run with: node quick-logger-fix.js
 */

const fs = require('fs');
const path = require('path');

console.log('🔧 Adding Missing Logger Method');
console.log('=' .repeat(40));

const loggerFile = path.join('src', 'utils', 'logger.js');

if (!fs.existsSync(loggerFile)) {
  console.log('❌ Logger file not found');
  process.exit(1);
}

try {
  let content = fs.readFileSync(loggerFile, 'utf8');
  
  // Check if getLogInfo method already exists
  if (content.includes('getLogInfo()')) {
    console.log('✅ getLogInfo method already exists');
    process.exit(0);
  }
  
  // Add the missing methods before the final module.exports
  const methodsToAdd = `
  /**
   * Get log file path
   */
  getLogFile() {
    return this.logFile;
  }

  /**
   * Get log info for status checks
   */
  getLogInfo() {
    return {
      level: this.getLevel(),
      logFile: this.logFile,
      isFileLoggingEnabled: !!this.logFile,
      logDirectory: this.logFile ? require('path').dirname(this.logFile) : null
    };
  }
`;

  // Find the last method and add our methods before module.exports
  const beforeExport = content.lastIndexOf('module.exports');
  
  if (beforeExport === -1) {
    console.log('❌ Could not find module.exports in logger file');
    process.exit(1);
  }
  
  // Insert the new methods before module.exports
  const newContent = content.slice(0, beforeExport) + methodsToAdd + '\n' + content.slice(beforeExport);
  
  fs.writeFileSync(loggerFile, newContent);
  console.log('✅ Added getLogInfo method to logger');
  console.log('🚀 Now restart your server: npm start');
  
} catch (error) {
  console.log('❌ Failed to fix logger:', error.message);
  process.exit(1);
}
#!/usr/bin/env node
/**
 * Crash Report Analyzer CLI
 * 
 * Simple command-line tool to view and analyze crash reports
 * Usage: node crash-analyzer.js [command] [options]
 */

const fs = require('fs').promises;
const path = require('path');

const CRASH_DIR = path.join(process.cwd(), 'crash-reports');

async function listCrashes() {
    try {
        const files = await fs.readdir(CRASH_DIR);
        const crashFiles = files.filter(f => f.endsWith('.json') && f.startsWith('crash-'));
        
        if (crashFiles.length === 0) {
            console.log('📁 No crash reports found');
            return;
        }

        console.log(`\n💥 Found ${crashFiles.length} crash reports:\n`);
        
        const crashes = await Promise.all(crashFiles.map(async (file) => {
            try {
                const content = await fs.readFile(path.join(CRASH_DIR, file), 'utf8');
                const crash = JSON.parse(content);
                return {
                    id: crash.crashId,
                    timestamp: crash.timestamp,
                    type: crash.type,
                    error: `${crash.error.name}: ${crash.error.message.substring(0, 80)}...`
                };
            } catch (error) {
                return null;
            }
        }));

        crashes.filter(c => c !== null)
               .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
               .forEach((crash, index) => {
                   const time = new Date(crash.timestamp).toLocaleString();
                   console.log(`${index + 1}. ${crash.id}`);
                   console.log(`   Time: ${time}`);
                   console.log(`   Type: ${crash.type}`);
                   console.log(`   Error: ${crash.error}`);
                   console.log('');
               });
    } catch (error) {
        console.error('❌ Error listing crashes:', error.message);
    }
}

async function showCrash(crashId) {
    try {
        const crashFile = path.join(CRASH_DIR, `${crashId}.json`);
        const summaryFile = path.join(CRASH_DIR, `${crashId}-summary.txt`);
        
        // Try to show summary first (more readable)
        try {
            const summary = await fs.readFile(summaryFile, 'utf8');
            console.log(summary);
            return;
        } catch (error) {
            // Fall back to JSON
        }
        
        const content = await fs.readFile(crashFile, 'utf8');
        const crash = JSON.parse(content);
        
        console.log(`\n💥 CRASH REPORT: ${crash.crashId}`);
        console.log(`🕐 Time: ${crash.timestamp}`);
        console.log(`🏷  Type: ${crash.type}`);
        console.log(`\n📋 ERROR:`);
        console.log(`${crash.error.name}: ${crash.error.message}`);
        console.log(`\n📍 STACK TRACE:`);
        console.log(crash.error.stack);
        
        if (crash.memory) {
            console.log(`\n💾 MEMORY:`);
            console.log(`Process: ${crash.memory.heapUsedMB} MB / ${crash.memory.heapTotalMB} MB`);
            console.log(`System: ${crash.memory.systemFreeMB} MB free`);
        }
        
        if (crash.pyodide) {
            console.log(`\n🐍 PYODIDE STATE:`);
            console.log(JSON.stringify(crash.pyodide, null, 2));
        }
        
    } catch (error) {
        console.error(`❌ Error reading crash ${crashId}:`, error.message);
    }
}

async function crashStats() {
    try {
        const files = await fs.readdir(CRASH_DIR);
        const crashFiles = files.filter(f => f.endsWith('.json') && f.startsWith('crash-'));
        
        if (crashFiles.length === 0) {
            console.log('📊 No crash data available');
            return;
        }

        const crashes = await Promise.all(crashFiles.map(async (file) => {
            try {
                const content = await fs.readFile(path.join(CRASH_DIR, file), 'utf8');
                return JSON.parse(content);
            } catch (error) {
                return null;
            }
        }));

        const validCrashes = crashes.filter(c => c !== null);
        const now = new Date();
        const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        
        const recentCrashes = validCrashes.filter(c => new Date(c.timestamp) > oneDayAgo);
        const weekCrashes = validCrashes.filter(c => new Date(c.timestamp) > oneWeekAgo);
        
        const crashTypes = {};
        validCrashes.forEach(crash => {
            const type = crash.type || 'unknown';
            crashTypes[type] = (crashTypes[type] || 0) + 1;
        });

        console.log('\n📊 CRASH STATISTICS');
        console.log('===================');
        console.log(`Total crashes: ${validCrashes.length}`);
        console.log(`Last 24 hours: ${recentCrashes.length}`);
        console.log(`Last 7 days: ${weekCrashes.length}`);
        console.log('\nCrash types:');
        Object.entries(crashTypes).forEach(([type, count]) => {
            console.log(`  ${type}: ${count}`);
        });
        
        if (validCrashes.length > 0) {
            const latest = validCrashes.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];
            console.log(`\nLatest crash: ${latest.crashId}`);
            console.log(`  Time: ${new Date(latest.timestamp).toLocaleString()}`);
            console.log(`  Error: ${latest.error.name}: ${latest.error.message}`);
        }
        
    } catch (error) {
        console.error('❌ Error generating stats:', error.message);
    }
}

async function cleanup(days = 7) {
    try {
        const files = await fs.readdir(CRASH_DIR);
        const crashFiles = files.filter(f => (f.endsWith('.json') || f.endsWith('-summary.txt')) && f.includes('crash-'));
        
        const cutoffDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
        let deletedCount = 0;
        
        for (const file of crashFiles) {
            const filePath = path.join(CRASH_DIR, file);
            const stat = await fs.stat(filePath);
            
            if (stat.mtime < cutoffDate) {
                await fs.unlink(filePath);
                deletedCount++;
            }
        }
        
        console.log(`🧹 Cleaned up ${deletedCount} old crash files (older than ${days} days)`);
    } catch (error) {
        console.error('❌ Error during cleanup:', error.message);
    }
}

function showHelp() {
    console.log(`
🔧 Crash Report Analyzer

Usage: node crash-analyzer.js <command> [options]

Commands:
  list                 List all crash reports
  show <crash-id>      Show detailed crash report
  stats                Show crash statistics
  cleanup [days]       Clean up old crash files (default: 7 days)
  help                 Show this help

Examples:
  node crash-analyzer.js list
  node crash-analyzer.js show crash-1640995200000-abc123
  node crash-analyzer.js stats
  node crash-analyzer.js cleanup 14
`);
}

async function main() {
    const command = process.argv[2];
    const arg = process.argv[3];
    
    // Ensure crash directory exists
    try {
        await fs.access(CRASH_DIR);
    } catch (error) {
        console.log('📁 No crash reports directory found. No crashes yet!');
        return;
    }
    
    switch (command) {
        case 'list':
            await listCrashes();
            break;
        case 'show':
            if (!arg) {
                console.error('❌ Please provide a crash ID');
                return;
            }
            await showCrash(arg);
            break;
        case 'stats':
            await crashStats();
            break;
        case 'cleanup':
            const days = arg ? parseInt(arg) : 7;
            await cleanup(days);
            break;
        case 'help':
        default:
            showHelp();
            break;
    }
}

if (require.main === module) {
    main().catch(error => {
        console.error('❌ Fatal error:', error.message);
        process.exit(1);
    });
}

module.exports = { listCrashes, showCrash, crashStats, cleanup };

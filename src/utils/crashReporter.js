const fs = require('fs').promises;
const path = require('path');
const os = require('os');

class CrashReporter {
    constructor(options = {}) {
        this.crashDir = options.crashDir || path.join(process.cwd(), 'crash-reports');
        this.maxCrashFiles = options.maxCrashFiles || 50;
        this.includeEnvironment = options.includeEnvironment !== false;
        this.includeMemoryInfo = options.includeMemoryInfo !== false;
        this.includePyodideState = options.includePyodideState !== false;
        
        this.ensureCrashDirectory();
        this.setupCrashHandlers();
    }

    async ensureCrashDirectory() {
        try {
            await fs.mkdir(this.crashDir, { recursive: true });
        } catch (error) {
            console.error('Failed to create crash reports directory:', error);
        }
    }

    setupCrashHandlers() {
        // Handle uncaught exceptions
        process.on('uncaughtException', async (error, origin) => {
            await this.reportCrash('uncaughtException', error, { origin });
            process.exit(1);
        });

        // Handle unhandled promise rejections
        process.on('unhandledRejection', async (reason, promise) => {
            const error = reason instanceof Error ? reason : new Error(String(reason));
            await this.reportCrash('unhandledRejection', error, { promise: promise.toString() });
            process.exit(1);
        });

        // Handle SIGTERM gracefully
        process.on('SIGTERM', async () => {
            await this.reportGracefulShutdown('SIGTERM');
            process.exit(0);
        });

        // Handle SIGINT gracefully
        process.on('SIGINT', async () => {
            await this.reportGracefulShutdown('SIGINT');
            process.exit(0);
        });
    }

    async reportCrash(type, error, metadata = {}) {
        const timestamp = new Date().toISOString();
        const crashId = `crash-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        
        const crashReport = {
            crashId,
            timestamp,
            type,
            error: {
                name: error.name,
                message: error.message,
                stack: error.stack,
                code: error.code,
                errno: error.errno,
                syscall: error.syscall,
                path: error.path
            },
            metadata,
            process: await this.getProcessInfo(),
            system: this.getSystemInfo(),
            environment: this.includeEnvironment ? this.getEnvironmentInfo() : null,
            memory: this.includeMemoryInfo ? this.getMemoryInfo() : null,
            pyodide: this.includePyodideState ? await this.getPyodideState() : null,
            requestContext: this.getCurrentRequestContext(),
            recentLogs: await this.getRecentLogs()
        };

        // Save detailed crash report
        await this.saveCrashReport(crashId, crashReport);
        
        // Save human-readable summary
        await this.saveCrashSummary(crashId, crashReport);
        
        // Cleanup old crash files
        await this.cleanupOldCrashFiles();
        
        console.error(`💥 CRASH REPORTED: ${crashId}`);
        console.error(`📁 Report saved to: ${path.join(this.crashDir, crashId)}.json`);
        console.error(`📋 Summary saved to: ${path.join(this.crashDir, crashId)}-summary.txt`);
        
        return crashId;
    }

    async reportGracefulShutdown(signal) {
        const timestamp = new Date().toISOString();
        const shutdownId = `shutdown-${Date.now()}`;
        
        const shutdownReport = {
            shutdownId,
            timestamp,
            signal,
            type: 'graceful_shutdown',
            uptime: process.uptime(),
            memory: this.getMemoryInfo(),
            process: await this.getProcessInfo()
        };

        await this.saveShutdownReport(shutdownId, shutdownReport);
    }

    async getProcessInfo() {
        return {
            pid: process.pid,
            ppid: process.ppid,
            platform: process.platform,
            arch: process.arch,
            nodeVersion: process.version,
            uptime: process.uptime(),
            cwd: process.cwd(),
            execPath: process.execPath,
            argv: process.argv,
            execArgv: process.execArgv,
            versions: process.versions
        };
    }

    getSystemInfo() {
        return {
            hostname: os.hostname(),
            type: os.type(),
            platform: os.platform(),
            arch: os.arch(),
            release: os.release(),
            cpus: os.cpus().length,
            totalMemory: os.totalmem(),
            freeMemory: os.freemem(),
            loadAverage: os.loadavg(),
            uptime: os.uptime()
        };
    }

    getEnvironmentInfo() {
        // Filter sensitive environment variables
        const sensitiveKeys = ['password', 'secret', 'key', 'token', 'auth'];
        const env = {};
        
        for (const [key, value] of Object.entries(process.env)) {
            if (sensitiveKeys.some(sensitive => key.toLowerCase().includes(sensitive))) {
                env[key] = '[REDACTED]';
            } else {
                env[key] = value;
            }
        }
        
        return env;
    }

    getMemoryInfo() {
        const memUsage = process.memoryUsage();
        return {
            rss: memUsage.rss,
            heapTotal: memUsage.heapTotal,
            heapUsed: memUsage.heapUsed,
            external: memUsage.external,
            arrayBuffers: memUsage.arrayBuffers,
            rssMB: Math.round(memUsage.rss / 1024 / 1024),
            heapTotalMB: Math.round(memUsage.heapTotal / 1024 / 1024),
            heapUsedMB: Math.round(memUsage.heapUsed / 1024 / 1024),
            systemFreeMB: Math.round(os.freemem() / 1024 / 1024),
            systemTotalMB: Math.round(os.totalmem() / 1024 / 1024)
        };
    }

    async getPyodideState() {
        try {
            // This would require access to your Pyodide service
            // You'll need to inject this dependency or make it globally accessible
            const pyodideService = global.pyodideService;
            if (!pyodideService) return { error: 'Pyodide service not available' };

            return {
                initialized: pyodideService.isInitialized(),
                packagesLoaded: pyodideService.getLoadedPackages(),
                lastExecutionTime: pyodideService.getLastExecutionTime(),
                errorCount: pyodideService.getErrorCount(),
                memoryUsage: pyodideService.getMemoryUsage()
            };
        } catch (error) {
            return { error: `Failed to get Pyodide state: ${error.message}` };
        }
    }

    getCurrentRequestContext() {
        try {
            // Get current request context if available
            const requestContext = global.currentRequestContext;
            if (!requestContext) return null;

            return {
                method: requestContext.method,
                url: requestContext.url,
                userAgent: requestContext.userAgent,
                ip: requestContext.ip,
                timestamp: requestContext.timestamp,
                requestId: requestContext.requestId
            };
        } catch (error) {
            return { error: `Failed to get request context: ${error.message}` };
        }
    }

    async getRecentLogs() {
        try {
            const logFile = path.join(process.cwd(), 'logs', 'pm2-error.log');
            const logContent = await fs.readFile(logFile, 'utf8');
            const lines = logContent.split('\n');
            
            // Get last 50 lines
            return lines.slice(-50).filter(line => line.trim());
        } catch (error) {
            return [`Failed to read recent logs: ${error.message}`];
        }
    }

    async saveCrashReport(crashId, report) {
        const filename = path.join(this.crashDir, `${crashId}.json`);
        const content = JSON.stringify(report, null, 2);
        
        try {
            await fs.writeFile(filename, content, 'utf8');
        } catch (error) {
            console.error('Failed to save crash report:', error);
        }
    }

    async saveCrashSummary(crashId, report) {
        const filename = path.join(this.crashDir, `${crashId}-summary.txt`);
        
        const summary = `
CRASH REPORT SUMMARY
====================
Crash ID: ${crashId}
Timestamp: ${report.timestamp}
Type: ${report.type}

ERROR DETAILS
=============
Name: ${report.error.name}
Message: ${report.error.message}

STACK TRACE
===========
${report.error.stack}

SYSTEM INFO
===========
Platform: ${report.system.platform} ${report.system.arch}
Node.js: ${report.process.nodeVersion}
Hostname: ${report.system.hostname}
Uptime: ${Math.round(report.process.uptime)} seconds

MEMORY INFO
===========
Process Memory: ${report.memory?.heapUsedMB || 'N/A'} MB / ${report.memory?.heapTotalMB || 'N/A'} MB
System Memory: ${report.memory?.systemFreeMB || 'N/A'} MB free / ${report.memory?.systemTotalMB || 'N/A'} MB total

PYODIDE STATE
=============
${report.pyodide ? JSON.stringify(report.pyodide, null, 2) : 'Not available'}

REQUEST CONTEXT
===============
${report.requestContext ? `
Method: ${report.requestContext.method}
URL: ${report.requestContext.url}
IP: ${report.requestContext.ip}
User-Agent: ${report.requestContext.userAgent}
Request ID: ${report.requestContext.requestId}
` : 'No active request'}

RECENT LOGS (Last 10 lines)
============================
${report.recentLogs.slice(-10).join('\n')}

METADATA
========
${JSON.stringify(report.metadata, null, 2)}
`;

        try {
            await fs.writeFile(filename, summary, 'utf8');
        } catch (error) {
            console.error('Failed to save crash summary:', error);
        }
    }

    async saveShutdownReport(shutdownId, report) {
        const filename = path.join(this.crashDir, `${shutdownId}.json`);
        const content = JSON.stringify(report, null, 2);
        
        try {
            await fs.writeFile(filename, content, 'utf8');
        } catch (error) {
            console.error('Failed to save shutdown report:', error);
        }
    }

    async cleanupOldCrashFiles() {
        try {
            const files = await fs.readdir(this.crashDir);
            const crashFiles = files.filter(f => f.startsWith('crash-') && f.endsWith('.json'));
            
            if (crashFiles.length > this.maxCrashFiles) {
                // Sort by modification time and remove oldest
                const fileStats = await Promise.all(
                    crashFiles.map(async file => {
                        const stat = await fs.stat(path.join(this.crashDir, file));
                        return { file, mtime: stat.mtime };
                    })
                );
                
                fileStats.sort((a, b) => a.mtime - b.mtime);
                
                const filesToDelete = fileStats.slice(0, crashFiles.length - this.maxCrashFiles);
                
                for (const { file } of filesToDelete) {
                    await fs.unlink(path.join(this.crashDir, file));
                    // Also delete corresponding summary file
                    const summaryFile = file.replace('.json', '-summary.txt');
                    try {
                        await fs.unlink(path.join(this.crashDir, summaryFile));
                    } catch (error) {
                        // Summary file might not exist
                    }
                }
            }
        } catch (error) {
            console.error('Failed to cleanup old crash files:', error);
        }
    }

    // Method to manually report application errors
    async reportError(error, context = {}) {
        return this.reportCrash('application_error', error, context);
    }
}

module.exports = CrashReporter;

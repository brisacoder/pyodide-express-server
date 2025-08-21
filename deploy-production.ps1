# Production deployment script for Pyodide Express Server with PM2 (PowerShell)
# Usage: .\deploy-production.ps1

param(
    [switch]$SkipTests,
    [switch]$Force
)

Write-Host "🚀 Deploying Pyodide Express Server to Production with PM2" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# Check if PM2 is installed
try {
    pm2 --version | Out-Null
    Write-Host "✅ PM2 is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ PM2 is not installed. Installing globally..." -ForegroundColor Red
    npm install -g pm2
}

# Check Node.js version
$nodeVersion = node -v
$requiredVersion = "v18.0.0"
Write-Host "✅ Node.js version: $nodeVersion" -ForegroundColor Green

# Create necessary directories
Write-Host "📁 Creating required directories..." -ForegroundColor Yellow
$directories = @(
    "uploads", 
    "logs", 
    "plots/matplotlib", 
    "plots/seaborn", 
    "plots/base64/matplotlib", 
    "plots/base64/seaborn"
)
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Install dependencies
Write-Host "📦 Installing production dependencies..." -ForegroundColor Yellow
npm ci --production

# Test application (if not skipped)
if (!$SkipTests) {
    Write-Host "🧪 Running health check..." -ForegroundColor Yellow
    
    # Start a temporary server for health check
    $healthCheckScript = @"
const app = require('./src/app');
const server = app.listen(3001, () => {
    console.log('Health check server started');
    setTimeout(() => {
        require('http').get('http://localhost:3001/health', (res) => {
            if (res.statusCode === 200) {
                console.log('✅ Health check passed');
                server.close();
                process.exit(0);
            } else {
                console.log('❌ Health check failed');
                server.close();
                process.exit(1);
            }
        }).on('error', (err) => {
            console.log('❌ Health check error:', err.message);
            server.close();
            process.exit(1);
        });
    }, 2000);
});
"@

    $healthCheckScript | Out-File -FilePath "temp-health-check.js" -Encoding UTF8
    
    try {
        $result = Start-Process -FilePath "node" -ArgumentList "temp-health-check.js" -Wait -PassThru -NoNewWindow
        Remove-Item "temp-health-check.js" -Force
        
        if ($result.ExitCode -ne 0) {
            Write-Host "❌ Application health check failed" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "❌ Health check error: $($_.Exception.Message)" -ForegroundColor Red
        Remove-Item "temp-health-check.js" -Force -ErrorAction SilentlyContinue
        exit 1
    }
}

# Stop existing PM2 process if running
try {
    $pm2List = pm2 list 2>$null
    if ($pm2List -match "pyodide-express-server") {
        Write-Host "🔄 Stopping existing PM2 process..." -ForegroundColor Yellow
        pm2 stop pyodide-express-server 2>$null
        pm2 delete pyodide-express-server 2>$null
    }
} catch {
    # Process might not exist, continue
}

# Start with PM2
Write-Host "🚀 Starting application with PM2..." -ForegroundColor Yellow
pm2 start ecosystem.config.js --env production

# Save PM2 configuration
pm2 save

# Setup PM2 startup (Windows service)
Write-Host "⚙️ Setting up PM2 auto-startup..." -ForegroundColor Yellow
pm2-installer

# Wait for application to be ready
Write-Host "⏳ Waiting for application to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Final health check
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000/health" -UseBasicParsing -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "🎉 Deployment successful!" -ForegroundColor Green
        Write-Host "📊 Application status:" -ForegroundColor Green
        pm2 status
        Write-Host ""
        Write-Host "🔗 Application URLs:" -ForegroundColor Green
        Write-Host "   Health Check: http://localhost:3000/health"
        Write-Host "   API Docs: http://localhost:3000/api-docs"
        Write-Host "   Status: http://localhost:3000/api/status"
        Write-Host ""
        Write-Host "📋 Management Commands:" -ForegroundColor Green
        Write-Host "   pm2 status                      - View status"
        Write-Host "   pm2 logs pyodide-express-server - View logs"
        Write-Host "   pm2 monit                       - Real-time monitoring"
        Write-Host "   pm2 restart pyodide-express-server - Restart app"
        Write-Host "   pm2 stop pyodide-express-server    - Stop app"
    } else {
        throw "Health check returned status code: $($response.StatusCode)"
    }
} catch {
    Write-Host "❌ Deployment failed - health check not responding: $($_.Exception.Message)" -ForegroundColor Red
    pm2 logs pyodide-express-server --lines 20
    exit 1
}

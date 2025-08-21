#!/bin/bash
# Production deployment script for Pyodide Express Server with PM2
# Usage: ./deploy-production.sh

set -e  # Exit on any error

echo "🚀 Deploying Pyodide Express Server to Production with PM2"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo -e "${RED}❌ PM2 is not installed. Installing globally...${NC}"
    npm install -g pm2
fi

# Check if Node.js version is compatible
NODE_VERSION=$(node -v | cut -d'v' -f2)
REQUIRED_VERSION="18.0.0"
if ! node -e "process.exit(process.version.slice(1).localeCompare('$REQUIRED_VERSION', undefined, {numeric: true}) >= 0 ? 0 : 1)"; then
    echo -e "${RED}❌ Node.js version $NODE_VERSION is too old. Required: $REQUIRED_VERSION+${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Node.js version: $NODE_VERSION${NC}"
echo -e "${GREEN}✅ PM2 installed${NC}"

# Create necessary directories
echo -e "${YELLOW}📁 Creating required directories...${NC}"
mkdir -p uploads logs plots/matplotlib plots/seaborn plots/base64/matplotlib plots/base64/seaborn

# Install dependencies
echo -e "${YELLOW}📦 Installing production dependencies...${NC}"
npm ci --production

# Test application
echo -e "${YELLOW}🧪 Running health check...${NC}"
timeout 30s node -e "
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
" || {
    echo -e "${RED}❌ Application health check failed${NC}"
    exit 1
}

# Stop existing PM2 process if running
if pm2 list | grep -q "pyodide-express-server"; then
    echo -e "${YELLOW}🔄 Stopping existing PM2 process...${NC}"
    pm2 stop pyodide-express-server || true
    pm2 delete pyodide-express-server || true
fi

# Start with PM2
echo -e "${YELLOW}🚀 Starting application with PM2...${NC}"
pm2 start ecosystem.config.js --env production

# Save PM2 configuration
pm2 save

# Setup PM2 startup (only run once per server)
if ! pm2 startup | grep -q "already"; then
    echo -e "${YELLOW}⚙️  Setting up PM2 auto-startup...${NC}"
    pm2 startup
fi

# Wait for application to be ready
echo -e "${YELLOW}⏳ Waiting for application to be ready...${NC}"
sleep 5

# Final health check
if curl -f http://localhost:3000/health > /dev/null 2>&1; then
    echo -e "${GREEN}🎉 Deployment successful!${NC}"
    echo -e "${GREEN}📊 Application status:${NC}"
    pm2 status
    echo ""
    echo -e "${GREEN}🔗 Application URLs:${NC}"
    echo "   Health Check: http://localhost:3000/health"
    echo "   API Docs: http://localhost:3000/api-docs"
    echo "   Status: http://localhost:3000/api/status"
    echo ""
    echo -e "${GREEN}📋 Management Commands:${NC}"
    echo "   pm2 status               - View status"
    echo "   pm2 logs pyodide-express-server  - View logs"
    echo "   pm2 monit               - Real-time monitoring"
    echo "   pm2 restart pyodide-express-server  - Restart app"
    echo "   pm2 stop pyodide-express-server     - Stop app"
else
    echo -e "${RED}❌ Deployment failed - health check not responding${NC}"
    pm2 logs pyodide-express-server --lines 20
    exit 1
fi

# Package Installation Guide for Pyodide Express Server

## 🚀 Quick Start Installation

### 1. Fresh Project Setup
```bash
# Clone or create your project
git clone https://github.com/yourusername/pyodide-express-server.git
cd pyodide-express-server

# Install all dependencies
npm install

# Alternative: Clean install (recommended)
npm ci
```

### 2. Check Installation Success
```bash
# Verify Node.js and npm versions
node --version    # Should show v16+ 
npm --version     # Should show 8+

# List installed packages
npm list          # Shows dependency tree
npm list --depth=0  # Shows only top-level packages

# Check for issues
npm doctor        # Diagnoses npm setup
```

## 📋 **Understanding package.json Dependencies**

Your `package.json` has two types of dependencies:

```json
{
  "dependencies": {
    "express": "^4.18.2",
    "multer": "^1.4.5-lts.1",
    "pyodide": "^0.28.0"
  },
  "devDependencies": {
    "nodemon": "^3.0.1",
    "jest": "^29.5.0",
    "eslint": "^8.42.0"
  }
}
```

**Dependencies:**
- Runtime packages needed to run your app
- Installed in production
- Examples: `express`, `pyodide`, `multer`

**DevDependencies:**
- Development tools only
- NOT installed in production
- Examples: `nodemon`, `jest`, `eslint`

## 🛠️ **Installation Commands Reference**

### Basic Installation
```bash
# Install all dependencies (production + development)
npm install

# Install from lock file (exact versions)
npm ci

# Install and update package-lock.json
npm install --save
```

### Production Installation
```bash
# Install only production dependencies
npm install --production
npm ci --production

# Set production environment
NODE_ENV=production npm install
```

### Development Installation
```bash
# Install only development dependencies
npm install --only=dev
npm install --only=development
```

### Individual Package Installation
```bash
# Add new production dependency
npm install express
npm install --save express    # Same as above

# Add new development dependency
npm install --save-dev nodemon
npm install -D nodemon        # Short form

# Install specific version
npm install express@4.18.2

# Install globally (for CLI tools)
npm install -g nodemon
```

## 🎯 **Step-by-Step Setup for Your Project**

### Step 1: Environment Setup
```bash
# 1. Ensure you have Node.js installed
node --version   # Should be 16+

# 2. If using nvm, set Node version
nvm use 18.18.0  # Or whatever version you prefer

# 3. Navigate to your project
cd pyodide-express-server
```

### Step 2: Install Dependencies
```bash
# Option A: Standard install (for development)
npm install

# Option B: Clean install (recommended)
npm ci

# Option C: Production install (for servers)
npm ci --production
```

### Step 3: Verify Installation
```bash
# Check if key packages are installed
npm list express pyodide multer

# Should show something like:
# ├── express@4.18.2
# ├── multer@1.4.5-lts.1
# └── pyodide@0.28.0

# Test if server can start
npm start
```

## 🔧 **Common Installation Issues & Solutions**

### Issue 1: Permission Errors
```bash
# Fix npm permissions (Unix/Linux/macOS)
sudo chown -R $(whoami) ~/.npm
sudo chown -R $(whoami) /usr/local/lib/node_modules

# Or use a Node version manager like nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
```

### Issue 2: Network/Proxy Issues
```bash
# Check npm configuration
npm config list

# Set registry if needed
npm config set registry https://registry.npmjs.org/

# Clear cache
npm cache clean --force
```

### Issue 3: Version Conflicts
```bash
# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check for outdated packages
npm outdated

# Update packages
npm update
```

### Issue 4: Platform-Specific Issues
```bash
# For Windows users with build tools issues
npm install --global windows-build-tools

# For Python/C++ build dependencies (if needed)
npm config set python /usr/bin/python3
```

## 📊 **Different Package Managers**

### npm (Default)
```bash
npm install          # Standard
npm ci              # Clean install
```

### Yarn (Alternative)
```bash
# Install Yarn
npm install -g yarn

# Use Yarn commands
yarn install        # Equivalent to npm install
yarn               # Short form
```

### pnpm (Faster, More Efficient)
```bash
# Install pnpm
npm install -g pnpm

# Use pnpm commands
pnpm install       # Faster than npm
pnpm i            # Short form
```

## 🐳 **Docker Installation**

If you prefer using Docker:

```dockerfile
# Dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

```bash
# Build and run with Docker
docker build -t pyodide-server .
docker run -p 3000:3000 pyodide-server
```

## ✅ **Installation Checklist**

Before running your Pyodide server:

- [ ] **Node.js 16+** installed (`node --version`)
- [ ] **npm 8+** installed (`npm --version`)
- [ ] **Dependencies installed** (`npm ci`)
- [ ] **No security vulnerabilities** (`npm audit`)
- [ ] **Environment variables set** (copy `.env.example` to `.env`)
- [ ] **Server starts successfully** (`npm start`)
- [ ] **API responds** (visit `http://localhost:3000`)

## 🚦 **Quick Commands Summary**

```bash
# Fresh setup
npm ci                    # Install exact versions

# Development
npm install              # Install with updates
npm run dev             # Start development server

# Production
npm ci --production     # Install production only
npm start              # Start production server

# Maintenance
npm outdated           # Check for updates
npm audit              # Security check
npm update             # Update packages
```

## 🎯 **For Your Pyodide Project**

**Recommended installation flow:**

1. **Clone repository:**
```bash
git clone https://github.com/yourusername/pyodide-express-server.git
cd pyodide-express-server
```

2. **Install dependencies:**
```bash
npm ci
```

3. **Set up environment:**
```bash
cp .env.example .env
# Edit .env file as needed
```

4. **Start development:**
```bash
npm run dev
```

5. **Verify it works:**
```bash
# Visit http://localhost:3000
# Should see "Pyodide ready" message
```

That's it! The `npm ci` command will install all the packages listed in your `package.json` file.
module.exports = {
  apps: [{
    name: 'pyodide-express-server',
    script: 'src/server.js',
    
    // Production settings optimized for Pyodide
    instances: 1, // Single instance to avoid Pyodide memory conflicts
    exec_mode: 'fork',
    
    // Environment
    env: {
      NODE_ENV: 'production',
      PORT: 3000
    },
    
    // Auto-restart configuration (aggressive for Pyodide stability)
    autorestart: true,
    watch: false,
    max_memory_restart: '2G', // Restart if Pyodide memory grows too large
    restart_delay: 3000, // Quick restart for faster recovery
    max_restarts: 15, // More restarts allowed (Pyodide can be finicky)
    min_uptime: '5s', // Minimum uptime before considering stable
    
    // Error handling optimized for WebAssembly
    kill_timeout: 8000, // Give Pyodide time to cleanup WebAssembly
    listen_timeout: 5000, // WebAssembly compilation can be slow
    
    // Enhanced logging for debugging Pyodide issues and crash analysis
    log_file: './logs/pm2-combined.log',
    out_file: './logs/pm2-out.log',
    error_file: './logs/pm2-error.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    log_type: 'json', // Structured logs for better debugging
    
    // Crash reporting integration
    crash_restart_delay: 5000, // Wait 5s before restart after crash
    max_crash_restart: 20, // Allow more crash restarts for debugging
    min_crash_interval: 10000, // Minimum time between crashes (10s)
    
    // Health monitoring with crash detection
    health_check_url: 'http://localhost:3000/health',
    health_check_grace_period: 3000,
    
    // Performance optimizations
    node_args: [
      '--max-old-space-size=4096', // 4GB heap for Pyodide
      '--gc-interval=100', // More frequent GC for WebAssembly cleanup
      '--optimize-for-size' // Optimize for memory usage
    ],
    
    // Advanced monitoring
    pmx: false,
    automation: false,
    
    // Custom environment variables for Pyodide optimization
    env_production: {
      NODE_ENV: 'production',
      PORT: 3000,
      PYODIDE_CACHE_SIZE: '512', // Cache size in MB
      PYODIDE_MEMORY_LIMIT: '1024', // Memory limit in MB
      LOG_LEVEL: 'info'
    }
  }],
  
  // Deployment configuration for AWS/production
  deploy: {
    production: {
      user: 'nodejs', // Non-root user for security
      host: ['your-aws-server.com'],
      ref: 'origin/main',
      repo: 'git@github.com:your-username/pyodide-express-server.git',
      path: '/opt/pyodide-express-server',
      'pre-deploy-local': 'echo "Starting deployment"',
      'post-deploy': 'npm ci --production && pm2 reload ecosystem.config.js --env production && pm2 save',
      'post-setup': 'ls -la && npm ci --production'
    }
  }
};

const express = require('express');
const router = express.Router();
const logger = require('../utils/logger');

/**
 * @swagger
 * /api/dashboard/stats:
 *   get:
 *     summary: Get detailed execution statistics and metrics
 *     description: Returns comprehensive statistics about code executions, security events, and system performance for the monitoring dashboard
 *     tags:
 *       - Statistics Dashboard
 *     responses:
 *       200:
 *         description: Statistics retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/StatisticsResponse'
 *       500:
 *         description: Error retrieving statistics
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
router.get('/stats', (req, res) => {
  try {
    const stats = logger.getStats();
    
    logger.info('Statistics retrieved', {
      component: 'stats-endpoint',
      statsOverview: {
        totalExecutions: stats.overview.totalExecutions,
        successRate: stats.overview.successRate,
        uptimeHuman: stats.overview.uptimeHuman
      }
    });

    res.json({
      success: true,
      stats,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Failed to retrieve statistics', {
      component: 'stats-endpoint',
      error: error.message
    });

    res.status(500).json({
      success: false,
      error: 'Failed to retrieve statistics'
    });
  }
});

/**
 * @swagger
 * /api/dashboard/stats/clear:
 *   post:
 *     summary: Clear all execution statistics
 *     description: Resets all statistics counters to zero (useful for testing or fresh start)
 *     tags:
 *       - Statistics Dashboard
 *     responses:
 *       200:
 *         description: Statistics cleared successfully
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/StatsClearResponse'
 *       500:
 *         description: Error clearing statistics
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */
router.post('/stats/clear', (req, res) => {
  try {
    const oldStats = logger.getStats();
    logger.clearStats();
    
    logger.info('Statistics cleared', {
      component: 'stats-endpoint',
      action: 'clear',
      previousStats: {
        totalExecutions: oldStats.overview.totalExecutions,
        successRate: oldStats.overview.successRate
      }
    });

    res.json({
      success: true,
      message: 'Statistics cleared successfully',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Failed to clear statistics', {
      component: 'stats-endpoint',
      error: error.message
    });

    res.status(500).json({
      success: false,
      error: 'Failed to clear statistics'
    });
  }
});

/**
 * @swagger
 * /api/dashboard/stats/dashboard:
 *   get:
 *     summary: Get statistics dashboard HTML page
 *     description: Returns a formatted HTML dashboard with interactive charts and visualizations using Chart.js. Provides real-time statistics monitoring with professional UI design.
 *     tags:
 *       - Statistics Dashboard
 *     responses:
 *       200:
 *         description: Dashboard HTML page with interactive charts
 *         content:
 *           text/html:
 *             schema:
 *               type: string
 *               description: Complete HTML page with embedded Chart.js visualizations
 *               example: '<!DOCTYPE html><html>...'
 *       500:
 *         description: Error generating dashboard
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/ErrorResponse'
 */

// Simple Chart.js test route
router.get('/stats/test', (req, res) => {
  const testHTML = `
<!DOCTYPE html>
<html>
<head>
    <title>Chart.js Test</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
</head>
<body>
    <h1>Chart.js Test</h1>
    <div style="width: 400px; height: 400px;">
        <canvas id="testChart"></canvas>
    </div>
    <script>
        console.log('Test page loaded');
        console.log('Chart available:', typeof Chart !== 'undefined');
        
        if (typeof Chart !== 'undefined') {
            console.log('Chart.js version:', Chart.version);
            
            const ctx = document.getElementById('testChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['A', 'B', 'C', 'D'],
                    datasets: [{
                        label: 'Test Data',
                        data: [12, 19, 3, 5],
                        backgroundColor: ['red', 'green', 'blue', 'yellow']
                    }]
                },
                options: {
                    responsive: true
                }
            });
            console.log('Test chart created!');
        } else {
            document.body.innerHTML += '<p style="color: red;">Chart.js not loaded!</p>';
        }
    </script>
</body>
</html>`;
  
  res.setHeader('Content-Type', 'text/html');
  res.send(testHTML);
});

router.get('/stats/dashboard', (req, res) => {
  try {
    const stats = logger.getStats();
    
    const dashboardHTML = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pyodide Express Server - Statistics Dashboard</title>
    
    <!-- Primary Chart.js CDN with integrity check -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js" 
            integrity="sha512-ElRFoEQdI5Ht6kZvyzXhYG9NqjtkmlkfYk0wr6wHxU9JEHakS7UJZNeml5ALk+8IKlU6jDgMabC3vkumRokgJA==" 
            crossorigin="anonymous" 
            referrerpolicy="no-referrer"></script>
    
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #f5f5f5; 
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 30px; 
            border-radius: 12px; 
            margin-bottom: 30px; 
            text-align: center;
        }
        .stats-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px; 
        }
        .stat-card { 
            background: white; 
            padding: 25px; 
            border-radius: 12px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
            border-left: 4px solid #667eea;
        }
        .stat-value { 
            font-size: 2.5em; 
            font-weight: bold; 
            color: #333; 
            margin-bottom: 5px; 
        }
        .stat-label { 
            color: #666; 
            font-size: 0.9em; 
            text-transform: uppercase; 
            letter-spacing: 1px; 
        }
        .chart-container { 
            background: white; 
            padding: 30px; 
            border-radius: 12px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
            margin-bottom: 20px;
            position: relative;
            height: 400px;
        }
        .chart-title { 
            font-size: 1.3em; 
            margin-bottom: 20px; 
            color: #333; 
            border-bottom: 2px solid #eee; 
            padding-bottom: 10px; 
        }
        .table-container { 
            background: white; 
            border-radius: 12px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
            overflow: hidden; 
        }
        .table { 
            width: 100%; 
            border-collapse: collapse; 
        }
        .table th, .table td { 
            padding: 15px; 
            text-align: left; 
            border-bottom: 1px solid #eee; 
        }
        .table th { 
            background: #f8f9fa; 
            font-weight: 600; 
            color: #555; 
        }
        .refresh-btn { 
            background: #667eea; 
            color: white; 
            border: none; 
            padding: 12px 24px; 
            border-radius: 6px; 
            cursor: pointer; 
            font-size: 16px; 
            margin-bottom: 20px; 
        }
        .refresh-btn:hover { 
            background: #5a6fd8; 
        }
        .grid-2 { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
        }
        .success { color: #28a745; }
        .error { color: #dc3545; }
        .warning { color: #ffc107; }
        @media (max-width: 768px) { 
            .grid-2 { grid-template-columns: 1fr; }
            .stats-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐍 Pyodide Express Server Dashboard</h1>
            <p>Real-time statistics and security monitoring</p>
            <p><strong>Uptime:</strong> ${stats.overview.uptimeHuman} | <strong>Last Updated:</strong> ${new Date().toLocaleString()}</p>
        </div>

        <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh Data</button>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value success">${stats.overview.totalExecutions}</div>
                <div class="stat-label">Total Executions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value ${parseFloat(stats.overview.successRate) > 90 ? 'success' : 'warning'}">${stats.overview.successRate}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.overview.averageExecutionTime}ms</div>
                <div class="stat-label">Avg Execution Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.recent.lastHourExecutions}</div>
                <div class="stat-label">Last Hour Executions</div>
            </div>
        </div>

        <div class="grid-2">
            <div class="chart-container">
                <div class="chart-title">📊 24-Hour Execution Trend</div>
                <canvas id="hourlyChart" width="400" height="200"></canvas>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">🔢 Success vs Error Rate</div>
                <canvas id="successChart" width="400" height="200"></canvas>
            </div>
        </div>

        <div class="grid-2">
            <div class="table-container">
                <div class="chart-title" style="padding: 20px 20px 0;">🌐 Top IP Addresses</div>
                <table class="table">
                    <thead>
                        <tr><th>IP Address</th><th>Count</th></tr>
                    </thead>
                    <tbody>
                        ${stats.topIPs.slice(0, 5).map(ip => 
                            `<tr><td>${ip.ip}</td><td>${ip.count}</td></tr>`
                        ).join('')}
                    </tbody>
                </table>
            </div>

            <div class="table-container">
                <div class="chart-title" style="padding: 20px 20px 0;">❌ Top Error Types</div>
                <table class="table">
                    <thead>
                        <tr><th>Error Type</th><th>Count</th></tr>
                    </thead>
                    <tbody>
                        ${stats.topErrors.length > 0 ? 
                            stats.topErrors.slice(0, 5).map(error => 
                                `<tr><td>${error.error}</td><td class="error">${error.count}</td></tr>`
                            ).join('') :
                            '<tr><td colspan="2" style="text-align: center; color: #28a745;">✅ No errors recorded</td></tr>'
                        }
                    </tbody>
                </table>
            </div>
        </div>

        <div class="table-container">
            <div class="chart-title" style="padding: 20px 20px 0;">🌍 User Agents</div>
            <table class="table">
                <thead>
                    <tr><th>User Agent</th><th>Request Count</th></tr>
                </thead>
                <tbody>
                    ${stats.userAgents.slice(0, 8).map(agent => 
                        `<tr><td>${agent.agent}</td><td>${agent.count}</td></tr>`
                    ).join('')}
                </tbody>
            </table>
        </div>

        <div class="table-container">
            <div class="chart-title" style="padding: 20px 20px 0;">⚡ Recent Executions (CPU & Memory)</div>
            <table class="table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>IP</th>
                        <th>Memory (MB)</th>
                        <th>Code Hash</th>
                        <th>Error</th>
                    </tr>
                </thead>
                <tbody>
                    ${stats.recentExecutions.slice(0, 15).map(exec => 
                        `<tr>
                            <td style="font-size: 0.85em;">${exec.timestamp}</td>
                            <td><span class="${exec.success ? 'success' : 'error'}">${exec.success ? '✅' : '❌'}</span></td>
                            <td>${exec.executionTime}ms</td>
                            <td style="font-family: monospace; font-size: 0.85em;">${exec.ip}</td>
                            <td>${exec.memoryMB}/${exec.memoryTotalMB}</td>
                            <td style="font-family: monospace; font-size: 0.8em;">${exec.codeHash}</td>
                            <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; font-size: 0.8em;" title="${exec.error || ''}">${exec.error ? exec.error.substring(0, 50) + '...' : ''}</td>
                        </tr>`
                    ).join('')}
                </tbody>
            </table>
        </div>
    </div>
    
    <!-- Debug Information -->
    <div style="background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #dee2e6;">
        <h3>Debug Information</h3>
        <div id="debugInfo">Loading debug info...</div>
    </div>

    <script>
        console.log('Dashboard script starting...');
        
        function addDebugInfo(message) {
            const debugDiv = document.getElementById('debugInfo');
            if (debugDiv) {
                debugDiv.innerHTML += '<p>' + new Date().toLocaleTimeString() + ': ' + message + '</p>';
            }
            console.log(message);
        }
        
        // Check Chart.js availability immediately
        addDebugInfo('Initial Chart.js check: ' + (typeof Chart !== 'undefined'));
        
        // Function to initialize charts
        function initializeCharts() {
            addDebugInfo('initializeCharts called');
            
            if (typeof Chart === 'undefined') {
                addDebugInfo('Chart.js not available, loading fallback...');
                
                // Load fallback Chart.js
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js';
                script.onload = function() {
                    addDebugInfo('Fallback Chart.js loaded successfully');
                    setTimeout(function() {
                        if (typeof Chart !== 'undefined') {
                            addDebugInfo('Chart.js now available after fallback');
                            startDashboard();
                        } else {
                            addDebugInfo('ERROR: Chart.js still not available after fallback');
                        }
                    }, 100);
                };
                script.onerror = function() {
                    addDebugInfo('ERROR: Fallback Chart.js also failed to load');
                };
                document.head.appendChild(script);
            } else {
                addDebugInfo('Chart.js available, starting dashboard');
                startDashboard();
            }
        }
        
        function startDashboard() {
            addDebugInfo('Starting dashboard with Chart.js version: ' + Chart.version);
            
            // Get stats data
            const stats = ${JSON.stringify(stats)};
            addDebugInfo('Stats data loaded: ' + stats.overview.totalExecutions + ' executions');
            
            // Refresh function
            window.refreshDashboard = function() {
                location.reload();
            };
            
            // Create charts
            createCharts(stats);
        }
        
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initializeCharts);
        } else {
            // DOM already loaded
            setTimeout(initializeCharts, 100);
        }
        
        function createCharts(stats) {
            addDebugInfo('Starting chart creation...');
            
            // Hourly trend chart
            try {
                const hourlyCanvas = document.getElementById('hourlyChart');
                addDebugInfo('Hourly canvas found: ' + !!hourlyCanvas);
                
                if (hourlyCanvas) {
                    addDebugInfo('Creating hourly chart...');
                    const ctx = hourlyCanvas.getContext('2d');
                    
                    const chart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', 
                                   '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
                            datasets: [{
                                label: 'Executions',
                                data: stats.hourlyTrend,
                                borderColor: '#667eea',
                                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                                fill: true,
                                pointBackgroundColor: '#667eea',
                                pointBorderColor: '#fff',
                                pointBorderWidth: 2,
                                pointRadius: 4
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: '24-Hour Execution Trend'
                                }
                            },
                            scales: {
                                y: { 
                                    beginAtZero: true,
                                    ticks: { stepSize: 1 }
                                }
                            }
                        }
                    });
                    addDebugInfo('✅ Hourly chart created successfully! Chart ID: ' + chart.id);
                } else {
                    addDebugInfo('❌ ERROR: Hourly chart canvas not found');
                }
            } catch (error) {
                addDebugInfo('❌ ERROR creating hourly chart: ' + error.message);
            }
            
            // Success rate chart
            try {
                const successCanvas = document.getElementById('successChart');
                addDebugInfo('Success canvas found: ' + !!successCanvas);
                
                if (successCanvas) {
                    addDebugInfo('Creating success chart...');
                    const ctx = successCanvas.getContext('2d');
                    
                    const successRate = parseFloat(stats.overview.successRate) || 0;
                    const errorRate = 100 - successRate;
                    addDebugInfo('Success rate: ' + successRate + '%, Error rate: ' + errorRate + '%');
                    
                    const chart = new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: ['Success (' + successRate + '%)', 'Errors (' + errorRate + '%)'],
                            datasets: [{
                                data: [successRate, errorRate],
                                backgroundColor: ['#28a745', '#dc3545'],
                                borderWidth: 2,
                                borderColor: '#fff'
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Success vs Error Rate'
                                },
                                legend: {
                                    position: 'bottom'
                                }
                            }
                        }
                    });
                    addDebugInfo('✅ Success chart created successfully! Chart ID: ' + chart.id);
                } else {
                    addDebugInfo('❌ ERROR: Success chart canvas not found');
                }
            } catch (error) {
                addDebugInfo('❌ ERROR creating success chart: ' + error.message);
            }
            
            addDebugInfo('🎉 Chart creation process completed');
        }
    </script>
</body>
</html>`;

    logger.info('Statistics dashboard accessed', {
      component: 'stats-dashboard',
      userAgent: req.get('User-Agent'),
      ip: req.ip
    });

    res.setHeader('Content-Type', 'text/html');
    res.send(dashboardHTML);
  } catch (error) {
    logger.error('Failed to generate statistics dashboard', {
      component: 'stats-dashboard',
      error: error.message
    });

    res.status(500).json({
      success: false,
      error: 'Failed to generate statistics dashboard'
    });
  }
});

module.exports = router;

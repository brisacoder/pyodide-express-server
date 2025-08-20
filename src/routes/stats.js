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
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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

        <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Data</button>

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
    </div>

    <script>
        // Hourly trend chart
        const hourlyCtx = document.getElementById('hourlyChart').getContext('2d');
        const hourLabels = Array.from({length: 24}, (_, i) => {
            const hour = new Date();
            hour.setHours(hour.getHours() - (23 - i));
            return hour.getHours() + ':00';
        });
        
        new Chart(hourlyCtx, {
            type: 'line',
            data: {
                labels: hourLabels,
                datasets: [{
                    label: 'Executions',
                    data: ${JSON.stringify(stats.hourlyTrend)},
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });

        // Success rate doughnut chart
        const successCtx = document.getElementById('successChart').getContext('2d');
        new Chart(successCtx, {
            type: 'doughnut',
            data: {
                labels: ['Success', 'Errors'],
                datasets: [{
                    data: [${stats.overview.successRate}, ${100 - parseFloat(stats.overview.successRate)}],
                    backgroundColor: ['#28a745', '#dc3545'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
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

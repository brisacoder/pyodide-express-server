const express = require('express');
const router = express.Router();
const { DASHBOARD } = require('../config/constants');
const logger = require('../utils/logger');

// Dashboard route - serves HTML dashboard directly
router.get('/dashboard', (req, res) => {
  try {
    const logger = require('./utils/logger');
    const stats = logger.getStats();
    const dashboardHTML = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pyodide Express Server - Dashboard</title>
    <!-- Local Chart.js for reliable loading -->
    <script src="${DASHBOARD.CHARTJS.LOCAL_PATH}"></script>
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
        .charts-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
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
        .refresh-btn:hover { background: #5a6fd8; }
        .debug-section {
            background: #f8f9fa;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        .debug-section h3 { margin-top: 0; }
        .recent-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .recent-table th, .recent-table td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
        .recent-table th { background: #f8f9fa; font-weight: 600; }
        .success { color: #28a745; font-weight: bold; }
        .error { color: #dc3545; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Pyodide Express Dashboard</h1>
            <p>Real-time execution monitoring and analytics</p>
            <button class="refresh-btn" id="refreshButton">🔄 Refresh Data</button>
        </div>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${stats.overview.totalExecutions}</div>
                <div class="stat-label">Total Executions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.overview.successRate}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.overview.averageExecutionTime}ms</div>
                <div class="stat-label">Avg Execution Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.overview.uptimeHuman}</div>
                <div class="stat-label">Server Uptime</div>
            </div>
        </div>
        <div class="charts-grid">
            <div class="chart-container">
                <h3>24-Hour Execution Trend</h3>
                <canvas id="hourlyChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Success vs Error Rate</h3>
                <canvas id="successChart"></canvas>
            </div>
        </div>
        <!-- Recent Executions Table -->
        <div style="background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3>Recent Executions</h3>
            <table class="recent-table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Memory (MB)</th>
                        <th>Error</th>
                    </tr>
                </thead>
                <tbody>
                    ${stats.recentExecutions
                      .map(
                        (exec) => `
                        <tr>
                            <td>${exec.timestamp}</td>
                            <td><span class="${exec.success ? 'success' : 'error'}">${exec.success ? '✅ Success' : '❌ Error'}</span></td>
                            <td>${exec.executionTime}ms</td>
                            <td>${exec.memoryMB || 'N/A'}</td>
                            <td>${exec.error || '-'}</td>
                        </tr>
                    `
                      )
                      .join('')}
                </tbody>
            </table>
        </div>
        <!-- Debug Information -->
        <div class="debug-section">
            <h3>Debug Information</h3>
            <div id="debugInfo">Initializing...</div>
        </div>
    </div>
    <script>
        // Dashboard configuration from server constants
        const DASHBOARD_CONFIG = ${JSON.stringify(DASHBOARD)};
        console.log('Dashboard script starting...');
        // Global chart instances for updating
        let hourlyChart = null;
        let successChart = null;

        /**
         * Adds debug information to the dashboard debug panel
         * @param {string} message - Debug message to display
         */
        function addDebugInfo(message) {
            const debugDiv = document.getElementById('debugInfo');
            if (debugDiv) {
                debugDiv.innerHTML += '<p>' + new Date().toLocaleTimeString() + ': ' + message + '</p>';
            }
            console.log(message);
        }

        /**
         * Refreshes dashboard data without page reload
         * Fetches fresh statistics and updates charts
         */
        function refreshDashboard() {
            console.log('🔄 refreshDashboard function called!');
            addDebugInfo('🔄 Refreshing dashboard data...');

            // Fetch fresh stats data
            console.log('🔄 About to fetch: /api/dashboard/stats');
            fetch('/api/dashboard/stats')
                .then(response => {
                    console.log('📡 Fetch response received:', response.status);
                    return response.json();
                })
                .then(data => {
                    console.log('📊 Data received:', data);
                    addDebugInfo('✅ Fresh data loaded, updating charts...');
                    // Extract stats from the response (API returns {success: true, stats: {...}})
                    const statsData = data.stats || data;
                    // Update the charts with new data
                    updateCharts(statsData);
                    // Update the recent executions table
                    updateRecentExecutionsTable(statsData.recentExecutions || []);
                    addDebugInfo('✅ Dashboard refreshed successfully!');
                })
                .catch(error => {
                    addDebugInfo('❌ Failed to refresh data: ' + error.message);
                    console.error('Refresh error:', error);
                });
        }

        // Check Chart.js availability immediately
        addDebugInfo('Chart.js check: ' + (typeof Chart !== 'undefined'));

        /**
         * Initializes Chart.js charts for the dashboard
         * Checks for Chart.js availability and starts dashboard
         */
        function initializeCharts() {
            addDebugInfo('initializeCharts called');

            // Since we're using local Chart.js, it should be available immediately
            if (typeof Chart !== 'undefined') {
                addDebugInfo('Chart.js available, starting dashboard');
                startDashboard();
            } else {
                addDebugInfo('❌ Local Chart.js failed to load - check console for errors');
                setTimeout(function() {
                    if (typeof Chart !== 'undefined') {
                        addDebugInfo('Chart.js loaded after delay, starting dashboard');
                        startDashboard();
                    } else {
                        addDebugInfo('❌ Chart.js still not available after delay');
                    }
                }, 1000);
            }
        }

        /**
         * Starts the dashboard by fetching data and creating charts
         * Main initialization function for dashboard functionality
         */
        function startDashboard() {
            addDebugInfo('Starting dashboard with Chart.js version: ' + Chart.version);

            // Get stats data
            const stats = ${JSON.stringify(stats)};
            addDebugInfo('Stats data loaded: ' + stats.overview.totalExecutions + ' executions');
            // Create charts
            createCharts(stats);
        }
        /**
         * Sets up event listeners for dashboard interactions
         * Adds refresh button click handler
         */
        function setupEventListeners() {
            addDebugInfo('Setting up event listeners...');
            // Add refresh button event listener
            const refreshButton = document.getElementById('refreshButton');
            if (refreshButton) {
                refreshButton.addEventListener('click', function() {
                    console.log('🔄 Refresh button clicked!');
                    refreshDashboard();
                });
                addDebugInfo('✅ Refresh button event listener added');
            } else {
                addDebugInfo('❌ Refresh button not found');
            }
        }
        /**
         * Combined initialization function for the dashboard
         * Sets up event listeners and initializes charts
         */
        function initializeDashboard() {
            setupEventListeners();
            initializeCharts();
        }
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initializeDashboard);
        } else {
            // DOM already loaded
            setTimeout(initializeDashboard, 100);
        }
        /**
         * Creates charts using Chart.js with provided statistics data
         * @param {Object} stats - Statistics data for chart creation
         */
        function createCharts(stats) {
            addDebugInfo('Starting chart creation...');
            // Hourly trend chart
            try {
                const hourlyCanvas = document.getElementById('hourlyChart');
                addDebugInfo('Hourly canvas found: ' + !!hourlyCanvas);
                if (hourlyCanvas) {
                    addDebugInfo('Creating hourly chart...');
                    const ctx = hourlyCanvas.getContext('2d');
                    hourlyChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
                                   '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
                            datasets: [{
                                label: 'Executions',
                                data: stats.hourlyTrend,
                                borderColor: DASHBOARD_CONFIG.CHARTS.COLORS.PRIMARY,
                                backgroundColor: DASHBOARD_CONFIG.CHARTS.COLORS.BACKGROUND,
                                fill: true,
                                pointBackgroundColor: DASHBOARD_CONFIG.CHARTS.COLORS.PRIMARY,
                                pointBorderColor: DASHBOARD_CONFIG.CHARTS.COLORS.BORDER,
                                pointBorderWidth: DASHBOARD_CONFIG.CHARTS.BORDER_WIDTH,
                                pointRadius: DASHBOARD_CONFIG.CHARTS.POINT_RADIUS
                            }]
                        },
                        options: {
                            responsive: DASHBOARD_CONFIG.CHARTS.RESPONSIVE,
                            maintainAspectRatio: DASHBOARD_CONFIG.CHARTS.MAINTAIN_ASPECT_RATIO,
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
                    addDebugInfo('✅ Hourly chart created successfully! Chart ID: ' + hourlyChart.id);
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
                    successChart = new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: ['Success (' + successRate + '%)', 'Errors (' + errorRate + '%)'],
                            datasets: [{
                                data: [successRate, errorRate],
                                backgroundColor: [DASHBOARD_CONFIG.CHARTS.COLORS.SUCCESS, DASHBOARD_CONFIG.CHARTS.COLORS.ERROR],
                                borderWidth: DASHBOARD_CONFIG.CHARTS.BORDER_WIDTH,
                                borderColor: DASHBOARD_CONFIG.CHARTS.COLORS.BORDER
                            }]
                        },
                        options: {
                            responsive: DASHBOARD_CONFIG.CHARTS.RESPONSIVE,
                            maintainAspectRatio: DASHBOARD_CONFIG.CHARTS.MAINTAIN_ASPECT_RATIO,
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
                    addDebugInfo('✅ Success chart created successfully! Chart ID: ' + successChart.id);
                } else {
                    addDebugInfo('❌ ERROR: Success chart canvas not found');
                }
            } catch (error) {
                addDebugInfo('❌ ERROR creating success chart: ' + error.message);
            }
            addDebugInfo('🎉 Chart creation process completed');
        }
        /**
         * Updates existing charts with new data
         * Destroys old charts and recreates them to avoid canvas reuse errors
         * @param {Object} newStats - Fresh statistics data for chart updates
         */
        function updateCharts(newStats) {
            addDebugInfo('Updating charts with fresh data...');
            try {
                // Destroy existing charts first to avoid Canvas reuse error
                if (hourlyChart) {
                    hourlyChart.destroy();
                    hourlyChart = null;
                    addDebugInfo('🗑️ Destroyed existing hourly chart');
                }
                if (successChart) {
                    successChart.destroy();
                    successChart = null;
                    addDebugInfo('🗑️ Destroyed existing success chart');
                }
                // Recreate charts with new data
                addDebugInfo('🔄 Recreating charts with fresh data...');
                createCharts(newStats);
                addDebugInfo('🎉 All charts updated successfully!');
            } catch (error) {
                addDebugInfo('❌ Error updating charts: ' + error.message);
                console.error('Chart update error:', error);
            }
        }
        /**
         * Updates the recent executions table with fresh data
         * @param {Array} recentExecutions - Array of recent execution data
         */
        function updateRecentExecutionsTable(recentExecutions) {
            addDebugInfo('Updating recent executions table...');
            try {
                const tableBody = document.querySelector('.recent-table tbody');
                if (!tableBody) {
                    addDebugInfo('⚠️ Recent executions table body not found');
                    return;
                }
                // Clear existing rows
                tableBody.innerHTML = '';
                // Add new rows
                recentExecutions.forEach(execution => {
                    const row = document.createElement('tr');
                    const timestamp = new Date(execution.timestamp).toLocaleTimeString();
                    const statusClass = execution.success ? 'success' : 'error';
                    const statusText = execution.success ? '✅ Success' : '❌ Error';
                    const duration = execution.executionTime || execution.duration || 'N/A';
                    const memory = execution.memoryMB || execution.memoryUsed || 'N/A';
                    const error = execution.error || '-';
                    row.innerHTML =
                        '<td>' + timestamp + '</td>' +
                        '<td><span class="' + statusClass + '">' + statusText + '</span></td>' +
                        '<td>' + duration + 'ms</td>' +
                        '<td>' + memory + '</td>' +
                        '<td>' + error + '</td>';
                    tableBody.appendChild(row);
                });
                addDebugInfo('✅ Recent executions table updated with ' + recentExecutions.length + ' entries');
            } catch (error) {
                addDebugInfo('❌ Error updating table: ' + error.message);
            }
        }
        // Fallback function if Chart.js completely fails
        function createSimpleCharts() {
            addDebugInfo('Creating simple fallback charts without Chart.js');
            try {
                const stats = ${JSON.stringify(stats)};
                // Simple hourly chart fallback
                const hourlyCanvas = document.getElementById('hourlyChart');
                if (hourlyCanvas) {
                    const ctx = hourlyCanvas.getContext('2d');
                    const width = hourlyCanvas.width = hourlyCanvas.offsetWidth;
                    const height = hourlyCanvas.height = hourlyCanvas.offsetHeight;
                    // Clear canvas
                    ctx.clearRect(0, 0, width, height);
                    // Draw simple bar chart
                    ctx.fillStyle = '#667eea';
                    const maxValue = Math.max(...stats.hourlyTrend) || 1;
                    const barWidth = width / 24;
                    for (let i = 0; i < 24; i++) {
                        const barHeight = (stats.hourlyTrend[i] / maxValue) * (height - 40);
                        ctx.fillRect(i * barWidth + 2, height - barHeight - 20, barWidth - 4, barHeight);
                    }
                    // Add title
                    ctx.fillStyle = '#333';
                    ctx.font = '16px Arial';
                    ctx.fillText('24-Hour Execution Trend (Simple View)', 10, 20);
                    addDebugInfo('✅ Simple hourly chart created');
                }
                // Simple success chart fallback
                const successCanvas = document.getElementById('successChart');
                if (successCanvas) {
                    const ctx = successCanvas.getContext('2d');
                    const width = successCanvas.width = successCanvas.offsetWidth;
                    const height = successCanvas.height = successCanvas.offsetHeight;
                    // Clear canvas
                    ctx.clearRect(0, 0, width, height);
                    const centerX = width / 2;
                    const centerY = height / 2 + 10;
                    const radius = Math.min(width, height) / 3;
                    const successRate = parseFloat(stats.overview.successRate) || 0;
                    const successAngle = (successRate / 100) * 2 * Math.PI;
                    // Draw success portion (green)
                    ctx.beginPath();
                    ctx.moveTo(centerX, centerY);
                    ctx.arc(centerX, centerY, radius, -Math.PI/2, -Math.PI/2 + successAngle);
                    ctx.closePath();
                    ctx.fillStyle = '#28a745';
                    ctx.fill();
                    // Draw error portion (red)
                    ctx.beginPath();
                    ctx.moveTo(centerX, centerY);
                    ctx.arc(centerX, centerY, radius, -Math.PI/2 + successAngle, 3*Math.PI/2);
                    ctx.closePath();
                    ctx.fillStyle = '#dc3545';
                    ctx.fill();
                    // Add title and labels
                    ctx.fillStyle = '#333';
                    ctx.font = '16px Arial';
                    ctx.fillText('Success vs Error Rate (Simple View)', 10, 20);
                    ctx.font = '12px Arial';
                    ctx.fillText('Success: ' + successRate + '%', 10, height - 40);
                    ctx.fillText('Errors: ' + (100 - successRate) + '%', 10, height - 25);
                    addDebugInfo('✅ Simple success chart created');
                }
                addDebugInfo('🎉 Simple fallback charts completed');
            } catch (error) {
                addDebugInfo('❌ ERROR creating simple charts: ' + error.message);
            }
        }
    </script>
</body>
</html>`;
    res.setHeader('Content-Type', 'text/html');
    res.send(dashboardHTML);
  } catch (error) {
    logger.error('Failed to generate dashboard', { error: error.message });
    res.status(500).json({
      success: false,
      error: 'Failed to generate dashboard',
    });
  }
});

module.exports = router;
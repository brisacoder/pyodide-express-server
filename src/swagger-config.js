/**
 * Swagger/OpenAPI Configuration for Pyodide Express Server
 *
 * This file configures the API documentation using OpenAPI 3.0 specification
 * and provides interactive testing interface similar to FastAPI's docs.
 * The documented endpoints target Pyodide 0.28.0 (Python 3.13) running inside
 * a Node.js environment.
 */
const swaggerJsdoc = require('swagger-jsdoc');
const swaggerUi = require('swagger-ui-express');
// Basic API information
const swaggerDefinition = {
  openapi: '3.0.0',
  info: {
    title: 'Pyodide Express Server API',
    version: '1.0.0',
    description: `
    A powerful REST API for executing Python code using Pyodide (WebAssembly).
    ## Features
    - Execute Python code with full pandas, numpy, matplotlib support
    - Install Python packages via micropip
    - Upload and process CSV files
    - Environment management and health monitoring
    - Real-time execution with output capture
    ## Getting Started
    1. Check if Pyodide is ready using \`/api/status\`
    2. Execute Python code using \`/api/execute\`
    3. Upload CSV files using \`/api/upload-csv\`
    4. Install packages using \`/api/install-package\`
    **Note**: All Python code executes in a sandboxed WebAssembly environment.
    `,
    contact: {
      name: 'API Support',
      email: 'support@example.com',
    },
    license: {
      name: 'MIT',
      url: 'https://opensource.org/licenses/MIT',
    },
  },
  servers: [
    {
      url: 'http://localhost:3000',
      description: 'Development server',
    },
    {
      url: 'https://your-domain.com',
      description: 'Production server',
    },
  ],
  tags: [
    {
      name: 'Python Execution',
      description: 'Execute Python code and manage the environment',
    },
    {
      name: 'Package Management',
      description: 'Install and list Python packages',
    },
    {
      name: 'File Operations',
      description: 'Upload and process files',
    },
    {
      name: 'System',
      description: 'Health checks and system information',
    },
    {
      name: 'Statistics Dashboard',
      description: 'Security logging, execution statistics, and monitoring dashboard',
    },
  ],
  components: {
    schemas: {
      ExecuteRequest: {
        type: 'object',
        required: ['code'],
        properties: {
          code: {
            type: 'string',
            description: 'Python code to execute',
            example: `import pandas as pd
import numpy as np
# Create sample data
data = {'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]}
df = pd.DataFrame(data)
print("DataFrame created:")
print(df)
# Return correlation
df.corr().to_dict()`,
          },
          context: {
            type: 'object',
            description: 'Variables to make available in Python execution context',
            additionalProperties: true,
            example: {
              user_name: 'Alice',
              multiplier: 2,
            },
          },
          timeout: {
            type: 'integer',
            description: 'Execution timeout in milliseconds',
            example: 30000,
            minimum: 1000,
            maximum: 300000,
          },
        },
      },
      ExecuteResponse: {
        type: 'object',
        properties: {
          success: {
            type: 'boolean',
            description: 'Whether execution was successful',
          },
          result: {
            description: 'The result of the Python expression (if any)',
            example: { x: { 0: 1.0, 1: 1.0 }, y: { 0: 1.0, 1: 1.0 } },
          },
          stdout: {
            type: 'string',
            description: 'Standard output from the Python code',
            example:
              'DataFrame created:\n   x   y\n0  1   2\n1  2   4\n2  3   6\n3  4   8\n4  5  10',
          },
          stderr: {
            type: 'string',
            description: 'Standard error output',
          },
          error: {
            type: 'string',
            description: 'Error message if execution failed',
          },
          timestamp: {
            type: 'string',
            format: 'date-time',
            description: 'Execution timestamp',
          },
        },
      },
      PackageRequest: {
        type: 'object',
        required: ['package'],
        properties: {
          package: {
            type: 'string',
            description: 'Name of the Python package to install',
            example: 'scikit-learn',
          },
        },
      },
      StatusResponse: {
        type: 'object',
        properties: {
          isReady: {
            type: 'boolean',
            description: 'Whether Pyodide is ready for code execution',
          },
          initialized: {
            type: 'boolean',
            description: 'Whether Pyodide has been initialized',
          },
          executionTimeout: {
            type: 'integer',
            description: 'Default execution timeout in milliseconds',
          },
          version: {
            type: 'string',
            description: 'Pyodide version',
          },
          timestamp: {
            type: 'string',
            format: 'date-time',
          },
        },
      },
      HealthResponse: {
        type: 'object',
        properties: {
          status: {
            type: 'string',
            enum: ['ok', 'error'],
          },
          server: {
            type: 'string',
            example: 'running',
          },
          pyodide: {
            $ref: '#/components/schemas/StatusResponse',
          },
          logging: {
            type: 'object',
            properties: {
              isFileLoggingEnabled: { type: 'boolean' },
              logFile: { type: 'string' },
              logDirectory: { type: 'string' },
            },
          },
          timestamp: {
            type: 'string',
            format: 'date-time',
          },
          uptime: {
            type: 'number',
            description: 'Server uptime in seconds',
          },
          memory: {
            type: 'object',
            properties: {
              rss: { type: 'number' },
              heapTotal: { type: 'number' },
              heapUsed: { type: 'number' },
              external: { type: 'number' },
              arrayBuffers: { type: 'number' },
            },
          },
        },
      },
      UploadResponse: {
        type: 'object',
        properties: {
          success: {
            type: 'boolean',
          },
          file: {
            type: 'object',
            properties: {
              originalName: {
                type: 'string',
                example: 'data.csv',
              },
              size: {
                type: 'number',
                example: 3820915,
              },
              pyodideFilename: {
                type: 'string',
                example: 'data.csv',
              },
              tempPath: {
                type: 'string',
                example: 'uploads/csvFile-1691234567890-123456789.csv',
              },
              keepFile: {
                type: 'boolean',
                example: true,
              },
            },
          },
          analysis: {
            type: 'object',
            properties: {
              success: { type: 'boolean' },
              filename: { type: 'string' },
              shape: {
                type: 'array',
                items: { type: 'number' },
                example: [100, 5],
              },
              columns: {
                type: 'array',
                items: { type: 'string' },
                example: ['name', 'age', 'city', 'salary', 'department'],
              },
              dtypes: {
                type: 'object',
                additionalProperties: { type: 'string' },
                example: {
                  name: 'object',
                  age: 'int64',
                  salary: 'float64',
                },
              },
              memory_usage: {
                type: 'number',
                example: 2048,
              },
              null_counts: {
                type: 'object',
                additionalProperties: { type: 'number' },
                example: {
                  name: 0,
                  age: 2,
                  salary: 1,
                },
              },
              sample_data: {
                type: 'array',
                items: { type: 'object' },
                example: [
                  { name: 'Alice', age: 25, salary: 50000 },
                  { name: 'Bob', age: 30, salary: 60000 },
                ],
              },
              numeric_columns: {
                type: 'array',
                items: { type: 'string' },
                example: ['age', 'salary'],
              },
              statistics: {
                type: 'object',
                description: 'Statistical summary for numeric columns',
                additionalProperties: {
                  type: 'object',
                  properties: {
                    count: { type: 'number' },
                    mean: { type: 'number' },
                    std: { type: 'number' },
                    min: { type: 'number' },
                    '25%': { type: 'number' },
                    '50%': { type: 'number' },
                    '75%': { type: 'number' },
                    max: { type: 'number' },
                  },
                },
              },
            },
          },
          timestamp: {
            type: 'string',
            format: 'date-time',
          },
        },
      },
      FileListResponse: {
        type: 'object',
        properties: {
          success: {
            type: 'boolean',
          },
          files: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                filename: {
                  type: 'string',
                  example: 'data.csv',
                },
                size: {
                  type: 'number',
                  example: 3820915,
                },
                uploaded: {
                  type: 'string',
                  format: 'date-time',
                },
                modified: {
                  type: 'string',
                  format: 'date-time',
                },
              },
            },
          },
          count: {
            type: 'number',
            example: 3,
          },
          uploadDir: {
            type: 'string',
            example: 'uploads',
          },
        },
      },
      FileDeleteResponse: {
        type: 'object',
        properties: {
          success: {
            type: 'boolean',
            example: true,
          },
          message: {
            type: 'string',
            example: 'File data.csv deleted successfully',
          },
        },
      },
      ErrorResponse: {
        type: 'object',
        properties: {
          success: {
            type: 'boolean',
            example: false,
          },
          error: {
            type: 'string',
            description: 'Error message',
          },
          timestamp: {
            type: 'string',
            format: 'date-time',
          },
        },
      },
      StatisticsOverview: {
        type: 'object',
        properties: {
          totalExecutions: {
            type: 'integer',
            description: 'Total number of code executions',
            example: 150,
          },
          successRate: {
            type: 'string',
            description: 'Success rate as a percentage',
            example: '94.7',
          },
          averageExecutionTime: {
            type: 'integer',
            description: 'Average execution time in milliseconds',
            example: 1250,
          },
          uptimeSeconds: {
            type: 'integer',
            description: 'Server uptime in seconds',
            example: 3600,
          },
          uptimeHuman: {
            type: 'string',
            description: 'Human-readable uptime',
            example: '1h 0m 0s',
          },
        },
      },
      StatisticsRecent: {
        type: 'object',
        properties: {
          lastHourExecutions: {
            type: 'integer',
            description: 'Number of executions in the last hour',
            example: 25,
          },
          recentSuccessRate: {
            type: 'string',
            description: 'Success rate for recent executions',
            example: '96.0',
          },
          packagesInstalled: {
            type: 'integer',
            description: 'Total packages installed',
            example: 8,
          },
          filesUploaded: {
            type: 'integer',
            description: 'Total files uploaded',
            example: 3,
          },
        },
      },
      IPStatistic: {
        type: 'object',
        properties: {
          ip: {
            type: 'string',
            description: 'IP address',
            example: '127.0.0.1',
          },
          count: {
            type: 'integer',
            description: 'Number of requests from this IP',
            example: 45,
          },
        },
      },
      ErrorStatistic: {
        type: 'object',
        properties: {
          error: {
            type: 'string',
            description: 'Error type',
            example: 'SyntaxError',
          },
          count: {
            type: 'integer',
            description: 'Number of occurrences',
            example: 5,
          },
        },
      },
      UserAgentStatistic: {
        type: 'object',
        properties: {
          agent: {
            type: 'string',
            description: 'User agent string',
            example: 'Mozilla/5.0',
          },
          count: {
            type: 'integer',
            description: 'Number of requests from this agent',
            example: 120,
          },
        },
      },
      StatisticsResponse: {
        type: 'object',
        properties: {
          success: {
            type: 'boolean',
            example: true,
          },
          stats: {
            type: 'object',
            properties: {
              overview: {
                $ref: '#/components/schemas/StatisticsOverview',
              },
              recent: {
                $ref: '#/components/schemas/StatisticsRecent',
              },
              topIPs: {
                type: 'array',
                items: {
                  $ref: '#/components/schemas/IPStatistic',
                },
                description: 'Top IP addresses by request count',
              },
              topErrors: {
                type: 'array',
                items: {
                  $ref: '#/components/schemas/ErrorStatistic',
                },
                description: 'Most common error types',
              },
              userAgents: {
                type: 'array',
                items: {
                  $ref: '#/components/schemas/UserAgentStatistic',
                },
                description: 'User agents by request count',
              },
              hourlyTrend: {
                type: 'array',
                items: {
                  type: 'integer',
                },
                description: 'Hourly execution counts for the last 24 hours',
                example: [
                  0, 0, 1, 3, 5, 8, 12, 15, 20, 25, 30, 28, 25, 22, 18, 15, 12, 8, 5, 3, 2, 1, 0, 0,
                ],
              },
            },
          },
          timestamp: {
            type: 'string',
            format: 'date-time',
            example: '2025-01-26T10:30:00.000Z',
          },
        },
      },
      StatsClearResponse: {
        type: 'object',
        properties: {
          success: {
            type: 'boolean',
            example: true,
          },
          message: {
            type: 'string',
            example: 'Statistics cleared successfully',
          },
          timestamp: {
            type: 'string',
            format: 'date-time',
            example: '2025-01-26T10:30:00.000Z',
          },
        },
      },
    },
    responses: {
      BadRequest: {
        description: 'Bad request - invalid input',
        content: {
          'application/json': {
            schema: {
              $ref: '#/components/schemas/ErrorResponse',
            },
          },
        },
      },
      InternalError: {
        description: 'Internal server error',
        content: {
          'application/json': {
            schema: {
              $ref: '#/components/schemas/ErrorResponse',
            },
          },
        },
      },
      ServiceUnavailable: {
        description: 'Service unavailable - Pyodide not ready',
        content: {
          'application/json': {
            schema: {
              $ref: '#/components/schemas/ErrorResponse',
            },
          },
        },
      },
    },
  },
};
// Options for swagger-jsdoc
const swaggerOptions = {
  definition: swaggerDefinition,
  apis: [
    './src/routes/execute.js',
    './src/routes/files.js', // ← File upload/management routes
    './src/routes/stats.js', // ← NEW: Statistics dashboard routes
    './src/routes/crashReports.js', // ← NEW: Crash reporting and debugging routes
    './src/server.js',
  ],
};
// Generate OpenAPI specification
const swaggerSpec = swaggerJsdoc(swaggerOptions);
// Swagger UI options
const swaggerUiOptions = {
  customCss: `
    .swagger-ui .topbar { display: none; }
    .swagger-ui .info { margin: 20px 0; }
    .swagger-ui .info .title { color: #3b4151; }
    .swagger-ui .scheme-container { background: #f7f7f7; padding: 15px; margin: 20px 0; }
    .swagger-ui .btn.try-out__btn { background: #4990e2; color: white; }
    .swagger-ui .btn.execute { background: #61affe; color: white; }
    .swagger-ui .response-col_status { color: #61affe; }
    .swagger-ui .opblock.opblock-post { border-color: #49cc90; background: rgba(73, 204, 144, 0.1); }
    .swagger-ui .opblock.opblock-get { border-color: #61affe; background: rgba(97, 175, 254, 0.1); }
    .swagger-ui .opblock.opblock-delete { border-color: #f93e3e; background: rgba(249, 62, 62, 0.1); }
  `,
  customSiteTitle: 'Pyodide Express API Documentation',
  customfavIcon: '/favicon.ico',
  swaggerOptions: {
    persistAuthorization: true,
    displayRequestDuration: true,
    filter: true,
    tryItOutEnabled: true,
    defaultModelsExpandDepth: 2,
    defaultModelExpandDepth: 2,
    docExpansion: 'list',
    operationsSorter: 'alpha',
    tagsSorter: 'alpha',
  },
};
module.exports = {
  swaggerSpec,
  swaggerUi,
  swaggerUiOptions,
};

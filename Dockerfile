# Dockerfile for Pyodide Express Server
# Optimized for AWS deployment with multi-stage build

# Build stage
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files first (for better Docker layer caching)
COPY package*.json ./
COPY pyproject.toml uv.lock ./

# Install system dependencies for Python and uv
RUN apk add --no-cache \
    python3 \
    py3-pip \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && /root/.local/bin/uv --version

# Update npm to latest version
RUN npm install -g npm@latest

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Install Node.js dependencies (production only)
RUN npm ci --omit=dev

# Install Python dependencies using uv
RUN /root/.local/bin/uv sync --frozen

# Production stage
FROM node:20-alpine AS production

# Install runtime dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    curl \
    dumb-init \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && npm install -g npm@latest

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Create app user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S appuser -u 1001

# Set working directory
WORKDIR /app

# Copy package files and install production dependencies
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy Python environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY pyproject.toml uv.lock ./

# Copy application source
COPY src/ ./src/
COPY public/ ./public/

# Create required directories
RUN mkdir -p uploads logs plots/matplotlib plots/seaborn plots/base64/matplotlib plots/base64/seaborn && \
    chown -R appuser:nodejs /app

# Switch to non-root user
USER appuser

# Environment variables
ENV NODE_ENV=production \
    PORT=3000 \
    PATH="/root/.local/bin:/app/.venv/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Expose port
EXPOSE 3000

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start the application
CMD ["node", "src/server.js"]

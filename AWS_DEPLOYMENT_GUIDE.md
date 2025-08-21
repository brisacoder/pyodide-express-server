# AWS Deployment Guide for Pyodide Express Server

## 🚀 Overview

This guide provides comprehensive instructions for deploying the Pyodide Express Server to AWS using modern containerization and infrastructure best practices.

## 📋 Prerequisites

### Required Tools
- **Docker** (for containerization)
- **AWS CLI** (for AWS operations)
- **Node.js 18+** (for local development)
- **Git** (for source control)

### AWS Services Used
- **ECS Fargate** (container orchestration)
- **Application Load Balancer** (traffic distribution)
- **VPC** (network isolation)
- **CloudWatch** (logging and monitoring)
- **ECR** (container registry)
- **EFS** (persistent file storage)

## 🏗️ Deployment Options

### Option 1: Manual ECS Deployment (Recommended for Testing)

#### Step 1: Build and Push Docker Image

```bash
# Build the production Docker image
docker build -t pyodide-express-server:latest .

# Tag for ECR (replace YOUR_ACCOUNT and REGION)
docker tag pyodide-express-server:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/pyodide-express-server:latest

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Push to ECR
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/pyodide-express-server:latest
```

#### Step 2: Create ECS Resources

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name pyodide-express-server-cluster

# Register task definition (update the image URI in aws/ecs-task-definition.json first)
aws ecs register-task-definition --cli-input-json file://aws/ecs-task-definition.json

# Create ECS service
aws ecs create-service \
  --cluster pyodide-express-server-cluster \
  --service-name pyodide-express-server \
  --task-definition pyodide-express-server:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

### Option 2: Terraform Infrastructure as Code (Recommended for Production)

#### Step 1: Initialize Terraform

```bash
cd aws/
terraform init
terraform plan
terraform apply
```

#### Step 2: Deploy Application

```bash
# Build and push image (same as Option 1)
# Update task definition with new image URI
# Update ECS service to use new task definition
```

### Option 3: Docker Compose (Local Testing)

```bash
# Start complete environment locally
docker-compose up --build

# Access application
curl http://localhost:3000/health
```

## 🔧 Configuration for AWS

### Environment Variables

Update `.env.production` with AWS-specific settings:

```env
NODE_ENV=production
PORT=3000
LOG_LEVEL=info

# AWS specific
TRUST_PROXY=true
X_FORWARDED_PROTO=true
ENABLE_METRICS=true

# File storage (if using EFS)
UPLOAD_DIRECTORY=/app/uploads
PLOTS_DIRECTORY=/app/plots
```

### ECS Task Definition Updates

Before deploying, update `aws/ecs-task-definition.json`:

1. **Replace YOUR_ACCOUNT** with your AWS account ID
2. **Update fileSystemId** with your EFS file system ID  
3. **Configure log group** in CloudWatch
4. **Set resource limits** (CPU/memory) based on your needs

### Security Groups Configuration

Required security group rules:

```bash
# ALB Security Group
# Inbound: Port 80 (HTTP) from 0.0.0.0/0
# Inbound: Port 443 (HTTPS) from 0.0.0.0/0
# Outbound: All traffic

# ECS Security Group  
# Inbound: Port 3000 from ALB security group
# Outbound: All traffic (for Pyodide package downloads)
```

## 🎯 Production Considerations

### Cross-Platform Compatibility ✅

The application is already configured for Windows → Linux compatibility:

- **Path handling**: Uses `path.join()` throughout codebase
- **File operations**: Proper Unix-style paths in containers
- **Environment variables**: Cross-platform environment configuration
- **Dependencies**: All packages support Linux containers

### Performance Optimization

```javascript
// PM2 Cluster Mode (ecosystem.config.json)
{
  "instances": "max",           // Use all CPU cores
  "exec_mode": "cluster",       // Cluster mode for scalability
  "max_memory_restart": "1G"    // Restart if memory exceeds 1GB
}
```

### Health Checks

The application provides three health check endpoints:

- **`/health`** - Basic health check for load balancers
- **`/healthcheck`** - Detailed system status for monitoring
- **`/ready`** - Kubernetes-style readiness probe

### Logging Strategy

```javascript
// CloudWatch Integration
{
  "logConfiguration": {
    "logDriver": "awslogs",
    "options": {
      "awslogs-group": "/ecs/pyodide-express-server",
      "awslogs-region": "us-east-1",
      "awslogs-stream-prefix": "ecs"
    }
  }
}
```

### Resource Requirements

**Minimum Resources:**
- **CPU**: 512 vCPU (0.5 cores)
- **Memory**: 1024 MB (1 GB)
- **Storage**: 20 GB container storage + EFS for persistent data

**Recommended for Production:**
- **CPU**: 1024 vCPU (1 core)
- **Memory**: 2048 MB (2 GB)
- **Storage**: 50 GB + EFS

### Persistent Storage

Use EFS (Elastic File System) for persistent data:

```json
{
  "volumes": [
    {
      "name": "uploads",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-XXXXXXXXX",
        "rootDirectory": "/uploads"
      }
    }
  ]
}
```

## 🔍 Monitoring and Troubleshooting

### CloudWatch Metrics

Monitor these key metrics:

- **CPU Utilization** (target: <70%)
- **Memory Utilization** (target: <80%)
- **Request Count** (track traffic patterns)
- **Response Time** (target: <2 seconds)
- **Error Rate** (target: <1%)

### Common Issues and Solutions

#### Issue: Container fails to start
```bash
# Check ECS service events
aws ecs describe-services --cluster pyodide-express-server-cluster --services pyodide-express-server

# Check CloudWatch logs
aws logs describe-log-streams --log-group-name /ecs/pyodide-express-server
```

#### Issue: Health checks failing
```bash
# Test health endpoint directly
curl http://LOAD_BALANCER_DNS/health

# Check container logs
aws logs get-log-events --log-group-name /ecs/pyodide-express-server --log-stream-name STREAM_NAME
```

#### Issue: Pyodide initialization problems
```bash
# Check memory limits in task definition
# Ensure sufficient startup time (60+ seconds)
# Verify Python package availability
```

### Performance Monitoring

```bash
# Monitor PM2 processes
npm run pm2:monit

# Check application logs
npm run pm2:logs

# View detailed health status
curl http://localhost:3000/healthcheck
```

## 🚀 CI/CD Pipeline

The included GitHub Actions workflow (`.github/workflows/deploy.yml`) provides:

1. **Automated Testing** - Run comprehensive test suite
2. **Docker Build** - Build and test container images
3. **Security Scanning** - Check for vulnerabilities
4. **Deployment** - Deploy to staging and production environments

### GitHub Secrets Required

```env
DOCKER_USERNAME=your_docker_username
DOCKER_PASSWORD=your_docker_password
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

## 📚 Quick Reference Commands

### Development
```bash
npm run dev                    # Local development
npm run docker:compose         # Local containerized testing
```

### Production
```bash
npm run prod:start            # Start with PM2
npm run pm2:logs              # View logs
npm run pm2:monit             # Monitor processes
```

### Docker
```bash
docker build -t pyodide-express-server .
docker run -p 3000:3000 pyodide-express-server
```

### AWS
```bash
aws ecs list-clusters
aws ecs describe-services --cluster CLUSTER_NAME --services SERVICE_NAME
aws logs describe-log-groups
```

## ✅ Deployment Checklist

### Pre-Deployment
- [ ] Docker image builds successfully
- [ ] All tests pass locally
- [ ] Environment variables configured
- [ ] AWS resources provisioned
- [ ] Security groups configured
- [ ] EFS file system created (if needed)

### Post-Deployment
- [ ] Health checks passing
- [ ] Load balancer routing correctly
- [ ] Logs appearing in CloudWatch
- [ ] File uploads working (if using EFS)
- [ ] Pyodide execution functional
- [ ] Monitoring alerts configured

## 🔒 Security Considerations

Since this is for internal testing with "security is of no concern":

- Basic Helmet.js security headers enabled
- Rate limiting configured but generous (1000 req/15min)
- No authentication/authorization implemented
- CORS allows all origins (`*`)
- File uploads allowed with size limits
- Container runs as non-root user
- Network isolation via VPC

For production environments, consider adding:
- Authentication (OAuth, JWT, etc.)
- API rate limiting per user/IP
- Input validation and sanitization
- SSL/TLS termination
- Web Application Firewall (WAF)
- Regular security scanning

---

**Ready for AWS deployment!** 🚀 The application now includes all necessary infrastructure components and is fully cross-platform compatible.

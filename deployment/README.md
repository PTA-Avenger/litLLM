# Deployment Guide

This directory contains deployment configurations for the Stylistic Poetry LLM Framework.

## Quick Start

### Local Development

```bash
# Start basic development environment
docker-compose up -d

# View logs
docker-compose logs -f poetry-llm

# Stop services
docker-compose down
```

### GPU-Enabled Deployment

```bash
# Start with GPU support (requires NVIDIA Docker runtime)
docker-compose --profile gpu up -d poetry-llm-gpu

# Check GPU availability
docker-compose exec poetry-llm-gpu python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Production Deployment

```bash
# Navigate to deployment directory
cd deployment/

# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# Start with monitoring
docker-compose -f docker-compose.prod.yml --profile monitoring up -d
```

## Configuration Files

### docker-compose.yml
Main development configuration with:
- Basic poetry-llm service
- GPU-enabled service (with profile)
- Redis cache (with profile)
- Nginx reverse proxy (with profile)

### docker-compose.prod.yml
Production configuration with:
- Resource limits and health checks
- Persistent volumes
- Monitoring stack (Prometheus + Grafana)
- Production-grade security settings

### nginx.conf
Nginx reverse proxy configuration with:
- Rate limiting
- SSL/TLS termination
- Security headers
- Request routing
- Static file serving

## Environment Profiles

### Development Profile (default)
```bash
docker-compose up -d
```
- Single poetry-llm container
- Debug logging enabled
- Hot reload support
- Local file mounts

### GPU Profile
```bash
docker-compose --profile gpu up -d
```
- NVIDIA GPU support
- CUDA-enabled PyTorch
- GPU memory management
- Performance optimization

### Cache Profile
```bash
docker-compose --profile cache up -d
```
- Redis caching layer
- Session management
- Result caching
- Performance improvement

### Production Profile
```bash
docker-compose --profile production up -d
```
- Nginx load balancer
- SSL termination
- Security hardening
- Production logging

### Monitoring Profile
```bash
docker-compose -f docker-compose.prod.yml --profile monitoring up -d
```
- Prometheus metrics collection
- Grafana dashboards
- System monitoring
- Performance analytics

## Deployment Scenarios

### 1. Local Development

**Requirements:**
- Docker and Docker Compose
- 8GB RAM minimum
- 10GB disk space

**Setup:**
```bash
# Clone repository
git clone <repository-url>
cd stylistic-poetry-llm

# Create local configuration
cp config/default.yaml config/local.yaml

# Start development environment
docker-compose up -d

# Test the service
curl http://localhost:5000/health
```

### 2. GPU-Accelerated Development

**Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime
- 16GB RAM recommended
- 20GB disk space

**Setup:**
```bash
# Install NVIDIA Docker runtime
# (See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

# Start GPU-enabled service
docker-compose --profile gpu up -d poetry-llm-gpu

# Verify GPU access
docker-compose exec poetry-llm-gpu nvidia-smi
```

### 3. Production Deployment

**Requirements:**
- Production server with Docker
- SSL certificates
- Domain name
- 32GB RAM recommended
- 100GB disk space

**Setup:**
```bash
# Prepare production configuration
cp config/default.yaml config/production.yaml
# Edit production.yaml with production settings

# Generate SSL certificates
mkdir -p deployment/ssl
# Copy your SSL certificates to deployment/ssl/

# Start production services
cd deployment/
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
curl https://your-domain.com/health
```

### 4. Kubernetes Deployment

**Requirements:**
- Kubernetes cluster
- kubectl configured
- Helm (optional)

**Setup:**
```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/k8s/

# Or use Helm chart
helm install poetry-llm deployment/helm/poetry-llm/
```

## Configuration Management

### Environment Variables

Set these environment variables for different deployments:

```bash
# Basic configuration
export POETRY_LLM_LOG_LEVEL=INFO
export POETRY_LLM_DATA_DIR=/app/data
export POETRY_LLM_OUTPUT_DIR=/app/output

# Performance tuning
export POETRY_LLM_MAX_CONCURRENT_REQUESTS=50
export POETRY_LLM_ENABLE_GPU=true
export POETRY_LLM_CACHE_SIZE=5000

# Security settings
export POETRY_LLM_API_KEY=your-api-key
export POETRY_LLM_ALLOWED_ORIGINS=https://your-domain.com
```

### Volume Mounts

Configure persistent storage:

```yaml
volumes:
  # Application data
  - ./data:/app/data:ro                    # Read-only corpus data
  - poetry_output:/app/output              # Generated poetry output
  - poetry_logs:/app/logs                  # Application logs
  - poetry_models:/app/models              # Trained models
  
  # Configuration
  - ./config/production.yaml:/app/config/local.yaml:ro
  
  # Cache and temporary files
  - poetry_cache:/app/cache
```

## Monitoring and Logging

### Application Logs

```bash
# View real-time logs
docker-compose logs -f poetry-llm

# View specific service logs
docker-compose logs nginx

# Export logs
docker-compose logs poetry-llm > poetry-llm.log
```

### Health Checks

```bash
# Check service health
curl http://localhost:5000/health

# Detailed system status
curl http://localhost:5000/status

# Metrics endpoint
curl http://localhost:5000/metrics
```

### Prometheus Metrics

Available at `http://localhost:9090` when monitoring profile is enabled:

- `poetry_generation_requests_total`
- `poetry_generation_duration_seconds`
- `poetry_analysis_requests_total`
- `poetry_model_load_duration_seconds`
- `system_memory_usage_bytes`
- `system_cpu_usage_percent`

### Grafana Dashboards

Available at `http://localhost:3000` (admin/admin123):

- System Overview Dashboard
- Poetry Generation Metrics
- Performance Analytics
- Error Rate Monitoring

## Security Considerations

### Network Security

```yaml
# Use custom networks
networks:
  poetry_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Container Security

```dockerfile
# Run as non-root user
RUN groupadd -r poetry && useradd -r -g poetry poetry
USER poetry

# Read-only root filesystem
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp
  - /var/tmp
```

### SSL/TLS Configuration

```nginx
# Strong SSL configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
ssl_prefer_server_ciphers off;

# Security headers
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
```

## Scaling and Performance

### Horizontal Scaling

```yaml
# Scale poetry-llm service
deploy:
  replicas: 3
  update_config:
    parallelism: 1
    delay: 10s
  restart_policy:
    condition: on-failure
```

### Load Balancing

```nginx
upstream poetry_llm {
    least_conn;
    server poetry-llm-1:5000;
    server poetry-llm-2:5000;
    server poetry-llm-3:5000;
}
```

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
    reservations:
      memory: 2G
      cpus: '1.0'
```

## Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check logs
docker-compose logs poetry-llm

# Check resource usage
docker stats

# Verify configuration
docker-compose config
```

**GPU not detected:**
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Check Docker daemon configuration
cat /etc/docker/daemon.json
```

**High memory usage:**
```bash
# Monitor memory usage
docker stats poetry-llm

# Adjust model parameters
# Edit config/local.yaml:
model:
  default_model: "gpt2"  # Use smaller model
  batch_size: 4          # Reduce batch size
```

**Slow response times:**
```bash
# Enable caching
docker-compose --profile cache up -d

# Use GPU acceleration
docker-compose --profile gpu up -d

# Scale horizontally
docker-compose up -d --scale poetry-llm=3
```

### Performance Tuning

**CPU Optimization:**
```yaml
environment:
  - OMP_NUM_THREADS=4
  - MKL_NUM_THREADS=4
  - NUMEXPR_NUM_THREADS=4
```

**Memory Optimization:**
```yaml
environment:
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  - TRANSFORMERS_CACHE=/app/cache/transformers
```

**Network Optimization:**
```nginx
# Enable HTTP/2
listen 443 ssl http2;

# Optimize keepalive
keepalive_timeout 65;
keepalive_requests 100;
```

## Backup and Recovery

### Data Backup

```bash
# Backup volumes
docker run --rm -v poetry_models:/data -v $(pwd):/backup alpine tar czf /backup/models-backup.tar.gz -C /data .

# Backup configuration
cp -r config/ backups/config-$(date +%Y%m%d)/
```

### Disaster Recovery

```bash
# Restore from backup
docker run --rm -v poetry_models:/data -v $(pwd):/backup alpine tar xzf /backup/models-backup.tar.gz -C /data

# Restart services
docker-compose down
docker-compose up -d
```

## Maintenance

### Updates

```bash
# Pull latest images
docker-compose pull

# Restart with new images
docker-compose up -d

# Clean up old images
docker image prune -f
```

### Log Rotation

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### Health Monitoring

```bash
# Automated health checks
#!/bin/bash
if ! curl -f http://localhost:5000/health; then
    docker-compose restart poetry-llm
fi
```
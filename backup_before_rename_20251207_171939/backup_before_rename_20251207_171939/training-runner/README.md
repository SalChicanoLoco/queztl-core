# Queztl Distributed Training System 🐝

## Overview

A sandboxed, scalable training orchestration system that dynamically manages ML model training across multiple isolated runners. Think **Kubernetes for ML training** or **apt/yum for model management**.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Orchestrator                     │
│              (Manages queue & scales runners)                │
│                    Port: 9000 (API)                          │
└────────────┬──────────────────────────────┬─────────────────┘
             │                              │
    ┌────────▼────────┐           ┌────────▼────────┐
    │  Training       │           │  Training       │
    │  Runner 1       │    ...    │  Runner N       │
    │  (Sandboxed)    │           │  (Sandboxed)    │
    │  2 CPU / 4GB    │           │  2 CPU / 4GB    │
    └─────────────────┘           └─────────────────┘
            │                              │
            └──────────┬───────────────────┘
                       │
              ┌────────▼────────┐
              │  Shared Storage │
              │  - Models       │
              │  - Logs         │
              │  - Data         │
              └─────────────────┘
```

## Key Features

### 🔒 Sandboxed Execution
- Each runner is an isolated Docker container
- Minimal dependencies (Python + PyTorch + NumPy + SciPy)
- Resource limits (CPU, memory)
- No interference between training jobs

### 📈 Dynamic Scaling
- Auto-scale based on queue depth
- Manual scaling: `./hive-control scale 4`
- Spawn/kill runners on-demand
- Scale from 0 to MAX_RUNNERS (default: 4)

### 🎯 Priority Queue
- High/medium/low priority jobs
- Automatic job distribution
- Track completed, active, and queued jobs

### 📊 Monitoring
- Real-time status dashboard
- Runner health checks
- System resource monitoring
- Training progress logs

### 🔄 Fault Tolerance
- Automatic runner restart on failure
- Job retry mechanism
- State persistence
- Graceful shutdown

## Quick Start

### 1. Initialize the Hive

```bash
./hive-control init
```

This builds the orchestrator and runner images.

### 2. Start the System

```bash
./hive-control start
```

Starts the orchestrator and 1 initial runner.

### 3. Check Status

```bash
./hive-control status
```

Output:
```
📊 Hive Status:

✅ Orchestrator: Running

🎯 System:
  CPU: 15.3%
  Memory: 42.1%
  Disk: 68.5%

🏃 Runners:
  Active: 1
  Available: 3
  Max: 4

📋 Queue:
  Pending: 0
  Completed: 0
```

### 4. Submit Training Jobs

```bash
# Submit a high-priority job
./hive-control submit image-to-3d high

# Submit multiple jobs
./hive-control submit gis-lidar high
./hive-control submit geophysics-magnetic high
./hive-control submit enhanced-3d medium
```

### 5. Scale Up for Parallel Training

```bash
# Scale to 4 runners for parallel training
./hive-control scale 4
```

### 6. Monitor Progress

```bash
# Real-time monitoring
./hive-control monitor

# Or check specific logs
./hive-control logs image-to-3d
./hive-control logs queztl-runner-1
```

## Available Commands

### Setup
```bash
./hive-control init       # Initialize hive
./hive-control start      # Start orchestrator + runners
./hive-control stop       # Stop all
./hive-control restart    # Restart everything
```

### Monitoring
```bash
./hive-control status     # Current status
./hive-control runners    # List all runners
./hive-control jobs       # List all jobs
./hive-control logs <id>  # View logs
./hive-control monitor    # Real-time monitoring
```

### Scaling
```bash
./hive-control scale 4    # Scale to 4 runners
./hive-control spawn 2    # Spawn 2 more runners
./hive-control kill <id>  # Kill specific runner
```

### Job Management
```bash
./hive-control submit <module> [priority]  # Submit job
```

### Maintenance
```bash
./hive-control clean      # Clean old logs
./hive-control web        # Open web dashboard
```

## Available Training Modules

### 3D Generation
- **image-to-3d** - Photo to 3D model conversion (60 min)
- **enhanced-3d** - High-quality 3D generation (40 min)
- **text-to-3d** - Text description to 3D (80 min)

### GIS
- **gis-lidar** - Point cloud classification (30 min)
- **gis-buildings** - Building extraction (25 min)

### Geophysics
- **geophysics-magnetic** - Magnetic interpretation (35 min)
- **geophysics-resistivity** - Resistivity inversion (30 min)
- **geophysics-seismic** - Seismic analysis (25 min)

## Example Workflows

### Train All High-Priority Modules

```bash
# Scale up
./hive-control scale 4

# Submit all high-priority jobs
./hive-control submit image-to-3d high
./hive-control submit enhanced-3d high
./hive-control submit gis-lidar high
./hive-control submit geophysics-magnetic high

# Monitor progress
./hive-control monitor
```

### Incremental Training

```bash
# Start with 1 runner
./hive-control start

# Submit first job
./hive-control submit image-to-3d high

# As queue builds, auto-scale kicks in
# Or manually scale: ./hive-control scale 2
```

### Development/Testing

```bash
# Single runner for testing
./hive-control start

# Submit test job
./hive-control submit gis-lidar low

# Watch logs
./hive-control logs gis-lidar
```

## API Endpoints

The orchestrator exposes a REST API on port 9000:

### Status
- `GET /health` - Health check
- `GET /status` - Detailed status
- `GET /runners` - List runners
- `GET /jobs` - List jobs

### Runners
- `POST /runners/spawn?count=N` - Spawn N runners
- `DELETE /runners/{id}` - Terminate runner

### Jobs
- `POST /jobs/submit` - Submit training job
- `POST /jobs/{id}/cancel` - Cancel job

### Scaling
- `POST /scale?target_runners=N` - Scale to N runners

### Web Dashboard
```bash
# Open API docs
./hive-control web

# Or visit: http://localhost:9000/docs
```

## Resource Configuration

### Per-Runner Limits
- **CPU**: 2 cores
- **Memory**: 4GB
- **Storage**: Shared volumes

### Orchestrator Limits
- **CPU**: 1 core
- **Memory**: 2GB

### Scaling Limits
- **Max Runners**: 4 (configurable)
- **Auto-scale**: Enabled by default

### Modify Limits

Edit `docker-compose.training.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'      # Change CPU limit
      memory: 4G       # Change memory limit
```

## Advantages Over Manual Training

### Before (Manual)
```bash
# One job at a time
python train_image_to_3d.py     # Wait 60 min
python train_gis_lidar.py       # Wait 30 min
python train_geophysics.py      # Wait 35 min
# Total: 125 minutes sequential
```

### After (Hive)
```bash
# All jobs in parallel
./hive-control scale 4
./hive-control submit image-to-3d high
./hive-control submit gis-lidar high
./hive-control submit geophysics-magnetic high
# Total: ~60 minutes parallel
```

**Time savings: 52%** ⚡

## Comparison to Other Systems

| Feature | Queztl Hive | Kubernetes | AWS Batch | Manual |
|---------|-------------|------------|-----------|--------|
| Setup Time | 2 min | 30+ min | 15+ min | 0 min |
| Auto-scaling | ✅ | ✅ | ✅ | ❌ |
| Resource Isolation | ✅ | ✅ | ✅ | ❌ |
| Local Execution | ✅ | ⚠️ | ❌ | ✅ |
| Cost | Free | $0-$$$ | $$$ | Free |
| Complexity | Low | High | Medium | Low |
| ML-Optimized | ✅ | ⚠️ | ⚠️ | ✅ |

## Troubleshooting

### Orchestrator Won't Start
```bash
# Check logs
docker logs queztl-orchestrator

# Restart
./hive-control restart
```

### Runner Stuck
```bash
# Check runner logs
./hive-control logs queztl-runner-1

# Kill and respawn
./hive-control kill queztl-runner-1
./hive-control spawn 1
```

### Jobs Not Starting
```bash
# Check queue
./hive-control jobs

# Check runner availability
./hive-control runners

# Scale up if needed
./hive-control scale 4
```

### Out of Resources
```bash
# Check system resources
./hive-control status

# Scale down
./hive-control scale 1

# Clean old artifacts
./hive-control clean
```

## Advanced Configuration

### Environment Variables

Edit `docker-compose.training.yml`:

```yaml
environment:
  - MAX_RUNNERS=8           # Allow up to 8 runners
  - AUTO_SCALE=true         # Enable auto-scaling
  - GPU_ENABLED=true        # Enable GPU support
  - MAX_MEMORY=8G           # Increase memory limit
```

### Custom Training Scripts

Add to `training-runner/`:
1. Create `train_custom.py`
2. Update `TRAINING_MODULES` in `training_manager.py`
3. Rebuild: `./hive-control init`

### Persistent Storage

Volumes are mapped to:
- `./models` - Trained models
- `./training_logs` - Training logs
- `./data` - Training data

## Next Steps

1. **Initialize**: `./hive-control init`
2. **Start**: `./hive-control start`
3. **Submit Jobs**: `./hive-control submit <module> high`
4. **Monitor**: `./hive-control monitor`
5. **Scale as Needed**: `./hive-control scale <n>`

## Support

For issues or questions:
- Check logs: `./hive-control logs <target>`
- View status: `./hive-control status`
- Clean and restart: `./hive-control clean && ./hive-control restart`

---

**Built for Queztl-Core** | Distributed ML Training Made Easy 🐝

#!/usr/bin/env python3
"""
Queztl Training Orchestrator
Manages sandboxed training runners and dynamically scales the hive
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import docker
import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Queztl Training Orchestrator")

# Docker client
docker_client = docker.from_env()

# Configuration
MAX_RUNNERS = int(os.getenv("MAX_RUNNERS", "4"))
AUTO_SCALE = os.getenv("AUTO_SCALE", "true").lower() == "true"
RUNNER_IMAGE = os.getenv("RUNNER_IMAGE", "queztl-training-runner:latest")

# State
training_queue: List[Dict] = []
active_runners: Dict[str, Dict] = {}
completed_jobs: List[Dict] = []


class TrainingJob(BaseModel):
    module_id: str
    priority: str = "medium"
    gpu_required: bool = False
    memory_gb: int = 4


class RunnerStatus(BaseModel):
    runner_id: str
    status: str
    current_job: Optional[str] = None
    jobs_completed: int = 0
    uptime_seconds: float = 0


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_runners": len(active_runners),
        "queued_jobs": len(training_queue),
        "completed_jobs": len(completed_jobs)
    }


@app.get("/status")
async def get_status():
    """Get orchestrator status"""
    system_stats = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
    
    return {
        "orchestrator": {
            "max_runners": MAX_RUNNERS,
            "auto_scale": AUTO_SCALE,
            "system": system_stats
        },
        "runners": {
            "active": len(active_runners),
            "available": MAX_RUNNERS - len(active_runners),
            "details": list(active_runners.values())
        },
        "queue": {
            "pending": len(training_queue),
            "completed": len(completed_jobs),
            "jobs": training_queue[:5]  # Show first 5
        }
    }


@app.get("/runners")
async def list_runners():
    """List all training runners"""
    runners = []
    
    try:
        containers = docker_client.containers.list(
            all=True,
            filters={"name": "queztl-runner"}
        )
        
        for container in containers:
            stats = container.stats(stream=False)
            
            runners.append({
                "id": container.name,
                "status": container.status,
                "image": container.image.tags[0] if container.image.tags else "unknown",
                "created": container.attrs['Created'],
                "cpu_usage": calculate_cpu_percent(stats),
                "memory_mb": stats['memory_stats'].get('usage', 0) / 1024 / 1024
            })
    except Exception as e:
        logger.error(f"Error listing runners: {e}")
    
    return {"runners": runners, "total": len(runners)}


@app.post("/runners/spawn")
async def spawn_runner(count: int = 1):
    """Spawn new training runners"""
    if len(active_runners) + count > MAX_RUNNERS:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot spawn {count} runners. Max: {MAX_RUNNERS}, Active: {len(active_runners)}"
        )
    
    spawned = []
    
    for i in range(count):
        runner_id = f"queztl-runner-{len(active_runners) + i + 1}"
        
        try:
            container = docker_client.containers.run(
                RUNNER_IMAGE,
                name=runner_id,
                detach=True,
                network="training-network",
                volumes={
                    '/workspace/models': {'bind': '/models', 'mode': 'rw'},
                    '/workspace/training_logs': {'bind': '/training_logs', 'mode': 'rw'},
                },
                environment={
                    'RUNNER_ID': str(len(active_runners) + i + 1),
                    'ORCHESTRATOR_URL': 'http://training-orchestrator:9000'
                },
                mem_limit='4g',
                cpu_period=100000,
                cpu_quota=200000  # 2 CPUs
            )
            
            active_runners[runner_id] = {
                "runner_id": runner_id,
                "container_id": container.id,
                "status": "idle",
                "spawned_at": datetime.now().isoformat(),
                "jobs_completed": 0
            }
            
            spawned.append(runner_id)
            logger.info(f"Spawned runner: {runner_id}")
            
        except Exception as e:
            logger.error(f"Failed to spawn runner: {e}")
    
    return {
        "spawned": spawned,
        "total_active": len(active_runners)
    }


@app.delete("/runners/{runner_id}")
async def terminate_runner(runner_id: str):
    """Terminate a training runner"""
    try:
        container = docker_client.containers.get(runner_id)
        container.stop(timeout=10)
        container.remove()
        
        if runner_id in active_runners:
            del active_runners[runner_id]
        
        logger.info(f"Terminated runner: {runner_id}")
        return {"status": "terminated", "runner_id": runner_id}
        
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Runner {runner_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jobs/submit")
async def submit_job(job: TrainingJob):
    """Submit a training job to the queue"""
    job_data = {
        "job_id": f"job-{len(training_queue) + 1}",
        "module_id": job.module_id,
        "priority": job.priority,
        "gpu_required": job.gpu_required,
        "memory_gb": job.memory_gb,
        "submitted_at": datetime.now().isoformat(),
        "status": "queued"
    }
    
    # Insert based on priority
    if job.priority == "high":
        training_queue.insert(0, job_data)
    else:
        training_queue.append(job_data)
    
    logger.info(f"Job submitted: {job_data['job_id']} - {job.module_id}")
    
    # Try to auto-scale if needed
    if AUTO_SCALE and len(active_runners) < MAX_RUNNERS:
        await auto_scale()
    
    return job_data


@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {
        "queued": training_queue,
        "active": [r for r in active_runners.values() if r.get("current_job")],
        "completed": completed_jobs[-10:]  # Last 10
    }


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a queued job"""
    for i, job in enumerate(training_queue):
        if job["job_id"] == job_id:
            cancelled_job = training_queue.pop(i)
            cancelled_job["status"] = "cancelled"
            cancelled_job["cancelled_at"] = datetime.now().isoformat()
            completed_jobs.append(cancelled_job)
            return cancelled_job
    
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found in queue")


@app.post("/scale")
async def manual_scale(target_runners: int):
    """Manually scale runners"""
    if target_runners > MAX_RUNNERS:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot scale to {target_runners}. Max: {MAX_RUNNERS}"
        )
    
    current = len(active_runners)
    
    if target_runners > current:
        # Scale up
        await spawn_runner(target_runners - current)
        return {"action": "scaled_up", "from": current, "to": target_runners}
    
    elif target_runners < current:
        # Scale down
        to_remove = current - target_runners
        idle_runners = [
            r_id for r_id, r in active_runners.items() 
            if r["status"] == "idle"
        ]
        
        removed = []
        for runner_id in idle_runners[:to_remove]:
            await terminate_runner(runner_id)
            removed.append(runner_id)
        
        return {
            "action": "scaled_down",
            "from": current,
            "to": len(active_runners),
            "removed": removed
        }
    
    return {"action": "no_change", "current": current}


async def auto_scale():
    """Auto-scale runners based on queue"""
    queue_length = len(training_queue)
    active = len(active_runners)
    
    # Scale up if queue is building
    if queue_length > active * 2 and active < MAX_RUNNERS:
        needed = min(queue_length // 2, MAX_RUNNERS - active)
        logger.info(f"Auto-scaling up: +{needed} runners")
        await spawn_runner(needed)
    
    # Scale down if idle
    elif queue_length == 0 and active > 1:
        idle_count = sum(1 for r in active_runners.values() if r["status"] == "idle")
        if idle_count > 1:
            logger.info(f"Auto-scaling down: -{idle_count - 1} runners")
            await manual_scale(active - (idle_count - 1))


def calculate_cpu_percent(stats: Dict) -> float:
    """Calculate CPU percentage from Docker stats"""
    try:
        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                    stats['precpu_stats']['cpu_usage']['total_usage']
        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                       stats['precpu_stats']['system_cpu_usage']
        cpu_count = stats['cpu_stats'].get('online_cpus', 1)
        
        if system_delta > 0:
            return (cpu_delta / system_delta) * cpu_count * 100.0
    except (KeyError, ZeroDivisionError):
        pass
    return 0.0


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 Queztl Training Orchestrator starting...")
    logger.info(f"   Max runners: {MAX_RUNNERS}")
    logger.info(f"   Auto-scale: {AUTO_SCALE}")
    
    # Check for existing runners
    try:
        containers = docker_client.containers.list(filters={"name": "queztl-runner"})
        logger.info(f"   Found {len(containers)} existing runners")
        
        for container in containers:
            active_runners[container.name] = {
                "runner_id": container.name,
                "container_id": container.id,
                "status": "idle",
                "jobs_completed": 0
            }
    except Exception as e:
        logger.error(f"Error checking existing runners: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Orchestrator shutting down...")
    
    # Save state
    state = {
        "training_queue": training_queue,
        "completed_jobs": completed_jobs,
        "shutdown_at": datetime.now().isoformat()
    }
    
    with open("/workspace/orchestrator_state.json", "w") as f:
        json.dump(state, f, indent=2)
    
    logger.info("   State saved")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)

#!/usr/bin/env python3
"""
Queztl Training Runner
Sandboxed minimal environment for executing training jobs
"""

import os
import sys
import json
import time
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Optional, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Runner-%(runner_id)s - %(levelname)s - %(message)s'
)

RUNNER_ID = os.getenv("RUNNER_ID", "unknown")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://training-orchestrator:9000")
WORKSPACE = Path("/workspace")
MODELS_DIR = Path("/models")
LOGS_DIR = Path("/training_logs")

logger = logging.getLogger(__name__)
logger.info(f"🚀 Runner {RUNNER_ID} initializing...")

# Runner state
current_job: Optional[Dict] = None
jobs_completed: int = 0
start_time = time.time()
status = "idle"


class HealthHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for health checks"""
    
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            response = {
                "runner_id": RUNNER_ID,
                "status": status,
                "uptime": time.time() - start_time,
                "current_job": current_job.get("module_id") if current_job else None,
                "jobs_completed": jobs_completed
            }
            
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress HTTP logs
        pass


def start_health_server():
    """Start health check HTTP server"""
    server = HTTPServer(("0.0.0.0", 8080), HealthHandler)
    logger.info(f"Health server started on :8080")
    server.serve_forever()


def poll_orchestrator():
    """Poll orchestrator for new jobs"""
    global current_job, status
    
    while True:
        try:
            if status == "idle":
                # Request work from orchestrator
                # In real implementation, use requests library
                # For now, simulate with file-based queue
                logger.debug("Polling for work...")
            
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error polling orchestrator: {e}")
            time.sleep(10)


def execute_training_job(job: Dict):
    """Execute a training job"""
    global current_job, jobs_completed, status
    
    current_job = job
    status = "training"
    
    module_id = job["module_id"]
    logger.info(f"📦 Starting training: {module_id}")
    
    try:
        # Determine training script and function
        script_map = {
            "image-to-3d": "train_image_to_3d.py",
            "enhanced-3d": "enhanced_training.py",
            "gis-lidar": "train_gis_geophysics.py:train_lidar_classifier",
            "gis-buildings": "train_gis_geophysics.py:train_building_extractor",
            "geophysics-magnetic": "train_gis_geophysics.py:train_magnetic_interpreter",
            "geophysics-resistivity": "train_gis_geophysics.py:train_resistivity_inverter",
            "geophysics-seismic": "train_gis_geophysics.py:train_seismic_analyzer",
        }
        
        if module_id not in script_map:
            raise ValueError(f"Unknown module: {module_id}")
        
        script_info = script_map[module_id]
        log_file = LOGS_DIR / f"{module_id}_runner{RUNNER_ID}_{int(time.time())}.log"
        
        # Build command
        if ":" in script_info:
            script, function = script_info.split(":")
            cmd = [
                "python3", "-c",
                f"from {script.replace('.py', '')} import {function}; {function}()"
            ]
        else:
            cmd = ["python3", script_info]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        # Execute training
        start = time.time()
        
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                cmd,
                cwd=WORKSPACE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Stream output
            for line in process.stdout:
                log.write(line)
                log.flush()
                if "Epoch" in line or "Loss" in line:
                    logger.info(line.strip())
            
            process.wait()
        
        duration = time.time() - start
        
        if process.returncode == 0:
            logger.info(f"✅ Training completed: {module_id} ({duration:.1f}s)")
            jobs_completed += 1
            
            # Update job status
            job["status"] = "completed"
            job["completed_at"] = datetime.now().isoformat()
            job["duration_seconds"] = duration
            
        else:
            logger.error(f"❌ Training failed: {module_id}")
            job["status"] = "failed"
            job["failed_at"] = datetime.now().isoformat()
        
        # Save result
        result_file = LOGS_DIR / f"{module_id}_result.json"
        with open(result_file, "w") as f:
            json.dump(job, f, indent=2)
        
    except Exception as e:
        logger.error(f"Error executing job: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
    
    finally:
        current_job = None
        status = "idle"
        logger.info(f"Runner back to idle (completed: {jobs_completed})")


def main():
    """Main runner loop"""
    logger.info(f"✅ Runner {RUNNER_ID} ready")
    logger.info(f"   Orchestrator: {ORCHESTRATOR_URL}")
    logger.info(f"   Models dir: {MODELS_DIR}")
    logger.info(f"   Logs dir: {LOGS_DIR}")
    
    # Start health server in background
    health_thread = Thread(target=start_health_server, daemon=True)
    health_thread.start()
    
    # Main loop - poll for work
    logger.info("🔄 Starting work loop...")
    poll_orchestrator()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 Runner shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

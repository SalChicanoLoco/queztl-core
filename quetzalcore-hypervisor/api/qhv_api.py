#!/usr/bin/env python3
"""
QuetzalCore Hypervisor Control API
Provides REST and WebSocket interfaces for VM management
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json

app = FastAPI(title="QuetzalCore Hypervisor API", version="0.1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VMConfig(BaseModel):
    name: str
    vcpus: int = 2
    memory_mb: int = 2048
    disk_gb: int = 20
    kernel: Optional[str] = None

class VMStatus(BaseModel):
    name: str
    status: str
    vcpus: int
    memory_mb: int
    uptime_seconds: int

# In-memory VM database (replace with real DB)
vms = {}

@app.get("/")
async def root():
    return {
        "service": "QuetzalCore Hypervisor API",
        "version": "0.1.0",
        "status": "running"
    }

@app.post("/api/vm/create")
async def create_vm(config: VMConfig):
    """Create a new virtual machine"""
    if config.name in vms:
        return {"error": f"VM '{config.name}' already exists"}
    
    vms[config.name] = {
        "name": config.name,
        "vcpus": config.vcpus,
        "memory_mb": config.memory_mb,
        "disk_gb": config.disk_gb,
        "status": "stopped",
        "uptime_seconds": 0
    }
    
    return {"success": True, "vm": vms[config.name]}

@app.get("/api/vm/list")
async def list_vms() -> List[VMStatus]:
    """List all virtual machines"""
    return [VMStatus(**vm) for vm in vms.values()]

@app.post("/api/vm/{name}/start")
async def start_vm(name: str):
    """Start a virtual machine"""
    if name not in vms:
        return {"error": f"VM '{name}' not found"}
    
    vms[name]["status"] = "running"
    return {"success": True, "vm": vms[name]}

@app.post("/api/vm/{name}/stop")
async def stop_vm(name: str):
    """Stop a virtual machine"""
    if name not in vms:
        return {"error": f"VM '{name}' not found"}
    
    vms[name]["status"] = "stopped"
    return {"success": True, "vm": vms[name]}

@app.get("/api/vm/{name}/status")
async def vm_status(name: str):
    """Get VM status"""
    if name not in vms:
        return {"error": f"VM '{name}' not found"}
    
    return vms[name]

@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """Real-time VM monitoring via WebSocket"""
    await websocket.accept()
    
    try:
        while True:
            # Send VM metrics every second
            metrics = {
                "timestamp": asyncio.get_event_loop().time(),
                "vms": list(vms.values())
            }
            await websocket.send_json(metrics)
            await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

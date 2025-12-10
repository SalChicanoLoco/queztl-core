# 🦅 QuetzalCore Hypervisor

Type-1 bare-metal hypervisor for running custom Linux instances.

## Quick Start

### 1. Build Hypervisor Core
```bash
cd core
cargo build --release
```

### 2. Start API Server
```bash
cd api
pip install fastapi uvicorn websockets
python qhv_api.py
```

### 3. Build Custom Kernel
```bash
cd kernel
./build-custom-kernel.sh
```

### 4. Create VM Image
```bash
cd tools
./build-vm-image.sh my-linux-vm 2G alpine
```

### 5. Run Hypervisor
```bash
cd core
./target/release/quetzalcore-hypervisor start
```

## Architecture

See [../QUETZALCORE_HYPERVISOR_ARCHITECTURE.md](../QUETZALCORE_HYPERVISOR_ARCHITECTURE.md)

## Commands

```bash
# Create VM
quetzalcore-hv create --name my-vm --vcpus 4 --memory 4096

# List VMs
quetzalcore-hv list

# Start VM
quetzalcore-hv run my-vm

# Stop VM
quetzalcore-hv stop my-vm
```

## API Endpoints

- `POST /api/vm/create` - Create new VM
- `GET /api/vm/list` - List all VMs
- `POST /api/vm/{name}/start` - Start VM
- `POST /api/vm/{name}/stop` - Stop VM
- `GET /api/vm/{name}/status` - Get VM status
- `WS /ws/monitor` - Real-time monitoring

## Requirements

- Linux with KVM support
- Rust 1.70+
- Python 3.8+
- QEMU/KVM
- 8GB+ RAM recommended


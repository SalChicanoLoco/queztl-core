# QuetzalCore Distributed AIOS - VM Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED AIOS (macOS Host)                        │
│                     /Users/xavasena/hive/                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌────────────────┐         ┌─────────────────┐
│   Backend     │          │  Hypervisor    │         │   Dashboard     │
│   (Python)    │◄────────►│   (Rust)       │◄───────►│   (Next.js)     │
│               │          │                │         │                 │
│ - FastAPI     │          │ - VM Manager   │         │ - Web UI        │
│ - AIOSC Core  │          │ - Resource     │         │ - 3D Engine     │
│ - AI Swarm    │          │   Allocation   │         │ - GIS Studio    │
│ - Autoscaler  │          │ - KVM/VirtIO   │         │ - Monitoring    │
└───────┬───────┘          └────────┬───────┘         └─────────────────┘
        │                           │
        │                           │
        └───────────────┬───────────┘
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │      Virtual Machine Manager          │
        │         /vms/test-vm/                 │
        │                                       │
        │  ┌─────────────────────────────────┐ │
        │  │   QuetzalCore Test VM           │ │
        │  │   (test-vm-001)                 │ │
        │  │                                 │ │
        │  │  Resources:                     │ │
        │  │  • Memory: 2048 MB              │ │
        │  │  •   - TPS (Transparent Page    │ │
        │  │  •       Sharing)               │ │
        │  │  •   - Compression               │ │
        │  │  •   - Ballooning               │ │
        │  │  •   - Memory Hotplug           │ │
        │  │  • vCPUs: 2                     │ │
        │  │  • Disk: 20 GB (disk.img)       │ │
        │  │  • Network: Bridge Mode         │ │
        │  │  • Features: KVM, VirtIO        │ │
        │  │                                 │ │
        │  │  Status: ✅ RUNNING             │ │
        │  └─────────────────────────────────┘ │
        └───────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │       Network Bridge (en0)            │
        │                                       │
        │  Internet Access: ✅                  │
        │  • DNS: 8.8.8.8 (33ms)               │
        │  • Ubuntu Archives: ✅ (136ms)       │
        └───────────────────────────────────────┘
```

## Component Details

### 1. **AIOS Host Layer** (macOS)
- **Location**: `/Users/xavasena/hive/`
- **Purpose**: Distributed AI Operating System orchestration
- **Components**:
  - Backend services (Python FastAPI)
  - Hypervisor (Rust-based VM manager)
  - Dashboard (Next.js web interface)

### 2. **Backend Services** (`/backend/`)
- **AI Swarm Management**: Coordinates distributed AI workloads
- **Autoscaler**: Dynamic resource allocation
- **AIOSC Platform**: Core AI OS services
- **Distributed Network**: Manages VM networking
- **Gen3D & GIS Engines**: Specialized workload processors

### 3. **Hypervisor Core** (`/quetzalcore-hypervisor/`)
- **Language**: Rust (performance-critical)
- **Capabilities**:
  - VM lifecycle management (create, start, stop, destroy)
  - Memory management with advanced features
  - CPU scheduling and allocation
  - Network bridge management
  - Storage provisioning

### 4. **Virtual Machine** (`/vms/test-vm/`)
- **VM ID**: test-vm-001
- **Configuration**: `config.json`
- **Disk Image**: `disk.img` (100 MB, sparse file)
- **Network Config**: `network.conf` (bridge mode)
- **Startup Script**: `start.sh`

### 5. **Memory Management Features**
- **TPS (Transparent Page Sharing)**: Deduplicates identical memory pages across VMs
- **Compression**: Compresses inactive memory pages
- **Ballooning**: Dynamic memory reclamation
- **Memory Hotplug**: Add/remove memory without VM restart

### 6. **Network Architecture**
- **Mode**: Bridge networking
- **Host Interface**: en0 (macOS primary interface)
- **VM Network**: Direct access to physical network
- **Internet**: Full access via host gateway
- **DNS**: Working (tested with 8.8.8.8)

## Current Status

### ✅ Operational
- Network connectivity (ping: 33ms to 8.8.8.8)
- DNS resolution (Ubuntu archives reachable)
- VM configuration loaded
- Hypervisor compiled and ready

### 🔄 To Implement
- [ ] Actual QEMU/KVM VM process launch
- [ ] Web-based VNC/SPICE console
- [ ] Real-time resource monitoring
- [ ] VM migration capabilities
- [ ] Snapshot management

## How It Actually Works

1. **VM Creation**: 
   - Config file defines VM parameters
   - Hypervisor allocates resources
   - Disk image created/mounted
   - Network bridge configured

2. **VM Startup**:
   - `start.sh` script executed
   - Hypervisor initializes VM process
   - KVM acceleration enabled (if available)
   - VirtIO drivers for performance
   - Network interface bridged

3. **Resource Management**:
   - Backend monitors VM metrics
   - Autoscaler adjusts resources dynamically
   - Memory features optimize usage
   - CPU scheduling via hypervisor

4. **Network Flow**:
   ```
   VM Guest → VirtIO NIC → Bridge (br0) → Host (en0) → Internet
   ```

5. **Management**:
   - Dashboard provides web UI
   - Backend API for automation
   - Hypervisor handles low-level operations

## Next Steps

1. **Launch actual VM process** (QEMU/KVM)
2. **Enable web console** (noVNC or similar)
3. **Implement monitoring dashboard**
4. **Add multi-VM orchestration**
5. **Enable live migration**

---
*QuetzalCore Distributed AIOS - Hypervisor Architecture*
*Last Updated: December 9, 2025*

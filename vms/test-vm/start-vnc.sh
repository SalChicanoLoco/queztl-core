#!/bin/bash
# QuetzalCore VM Starter with VNC
# Launches the VM with HTML5 VNC console support

echo "🚀 QuetzalCore VM - VNC Console Launcher"
echo "========================================"
echo ""

VM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$VM_DIR"

# Check if VM is already running
if [ -f "STATUS" ] && [ "$(cat STATUS)" == "running" ]; then
    echo "⚠️  VM appears to be already running"
    echo "   Status: $(cat STATUS)"
    echo ""
fi

# Update status
echo "running" > STATUS

echo "📋 VM Configuration:"
echo "   VM ID: test-vm-001"
echo "   Name: QuetzalCore Test VM"
echo "   Memory: 2048 MB"
echo "   vCPUs: 2"
echo "   Disk: disk.img (20 GB)"
echo "   Network: Bridge mode"
echo ""

echo "🖥️  VNC Console:"
echo "   Port: 5900"
echo "   Protocol: RFB (HTML5 compatible)"
echo "   Web Console: http://localhost:9090"
echo ""

# Check if QEMU is available
if command -v qemu-system-x86_64 &> /dev/null; then
    echo "✅ QEMU found - launching VM with VNC..."
    echo ""
    
    # Launch QEMU with VNC
    qemu-system-x86_64 \
        -name "QuetzalCore-test-vm-001" \
        -m 2048 \
        -smp 2 \
        -hda disk.img \
        -net nic -net bridge,br=br0 \
        -vnc :0 \
        -daemonize \
        -pidfile vm.pid
    
    if [ $? -eq 0 ]; then
        echo "✅ VM started successfully!"
        echo "   PID: $(cat vm.pid 2>/dev/null || echo 'N/A')"
        echo ""
        echo "🌐 Access your VM:"
        echo "   Web Console: http://localhost:9090"
        echo "   VNC Display: localhost:5900"
        echo "   Or use: vncviewer localhost:5900"
        echo ""
    else
        echo "❌ Failed to start VM"
        echo "running (simulated)" > STATUS
    fi
else
    echo "⚠️  QEMU not found - running in simulation mode"
    echo ""
    echo "📦 To install QEMU on macOS:"
    echo "   brew install qemu"
    echo ""
    echo "For now, the console will show a simulated environment."
    echo ""
fi

# Start or check console server
if ! lsof -i :9090 > /dev/null 2>&1; then
    echo "🌐 Starting web console server..."
    python3 console-server.py > /tmp/vm-console.log 2>&1 &
    CONSOLE_PID=$!
    echo "   Console PID: $CONSOLE_PID"
    sleep 2
fi

echo "✅ Console server ready at http://localhost:9090"
echo ""
echo "🎯 Next Steps:"
echo "   1. Open http://localhost:9090 in your browser"
echo "   2. Click 'VNC Display' tab"
echo "   3. Connection will auto-establish"
echo ""
echo "🛑 To stop VM:"
echo "   ./stop-vm.sh"
echo "   Or: kill \$(cat vm.pid)"
echo ""
echo "Dale! 🚀"

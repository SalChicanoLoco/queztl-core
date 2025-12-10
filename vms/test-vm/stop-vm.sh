#!/bin/bash
# Stop QuetzalCore VM

echo "🛑 Stopping QuetzalCore VM..."

VM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$VM_DIR"

# Stop QEMU if running
if [ -f "vm.pid" ]; then
    PID=$(cat vm.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "   Stopping VM (PID: $PID)..."
        kill $PID
        sleep 2
        if kill -0 $PID 2>/dev/null; then
            echo "   Force stopping..."
            kill -9 $PID
        fi
        rm -f vm.pid
        echo "   ✅ VM stopped"
    else
        echo "   VM process not running"
        rm -f vm.pid
    fi
else
    echo "   No PID file found"
fi

# Update status
echo "stopped" > STATUS

echo "✅ VM stopped successfully"

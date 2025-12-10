#!/bin/bash
# Build a minimal Linux VM image for QuetzalCore Hypervisor

set -e

VM_NAME="${1:-quetzalcore-vm}"
VM_SIZE="${2:-2G}"
DISTRO="${3:-alpine}"

echo "📦 Building VM Image: $VM_NAME"
echo "===================================="
echo "   Distribution: $DISTRO"
echo "   Size: $VM_SIZE"
echo ""

# Create disk image
echo "💾 Creating disk image..."
qemu-img create -f qcow2 "../vms/images/${VM_NAME}.qcow2" "$VM_SIZE"
echo "✅ Disk image created"
echo ""

# TODO: Bootstrap Linux distribution
# TODO: Install custom kernel
# TODO: Configure auto-login
# TODO: Install QuetzalCore drivers

echo "✅ VM image built: ../vms/images/${VM_NAME}.qcow2"
echo ""


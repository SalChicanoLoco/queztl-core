#!/bin/bash
# Build custom Linux kernel for QuetzalCore VMs

set -e

KERNEL_VERSION="6.6.7"
KERNEL_DIR="linux-${KERNEL_VERSION}"
KERNEL_URL="https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-${KERNEL_VERSION}.tar.xz"

echo "🐧 Building Custom Linux Kernel for QuetzalCore"
echo "============================================"
echo ""

# Download kernel source
if [ ! -d "$KERNEL_DIR" ]; then
    echo "📥 Downloading Linux kernel ${KERNEL_VERSION}..."
    wget "$KERNEL_URL"
    tar xf "linux-${KERNEL_VERSION}.tar.xz"
    echo "✅ Kernel source downloaded"
else
    echo "✅ Kernel source already exists"
fi

cd "$KERNEL_DIR"

# Create QuetzalCore kernel config
echo "📝 Creating QuetzalCore kernel configuration..."
make defconfig

# Enable KVM guest support
scripts/config --enable CONFIG_HYPERVISOR_GUEST
scripts/config --enable CONFIG_KVM_GUEST
scripts/config --enable CONFIG_PARAVIRT
scripts/config --enable CONFIG_PARAVIRT_SPINLOCKS

# Enable Virtio drivers
scripts/config --enable CONFIG_VIRTIO_PCI
scripts/config --enable CONFIG_VIRTIO_NET
scripts/config --enable CONFIG_VIRTIO_BLK
scripts/config --enable CONFIG_VIRTIO_CONSOLE

# Enable real-time features
scripts/config --enable CONFIG_PREEMPT
scripts/config --enable CONFIG_HIGH_RES_TIMERS

# Minimize size
scripts/config --disable CONFIG_DEBUG_KERNEL
scripts/config --disable CONFIG_WIRELESS
scripts/config --disable CONFIG_WLAN

echo "✅ Configuration complete"
echo ""

# Build kernel
echo "🔨 Building kernel (this may take a while)..."
make -j$(nproc) bzImage modules

echo ""
echo "✅ Kernel built successfully!"
echo "📦 Kernel image: arch/x86/boot/bzImage"
echo ""


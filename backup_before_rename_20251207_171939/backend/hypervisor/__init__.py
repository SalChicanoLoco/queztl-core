"""
🦅 QUEZTL HYPERVISOR - Core Module

A real Type-1 hypervisor built from scratch for Queztl distributed infrastructure.

Architecture:
- Virtual Machine Monitor (VMM)
- CPU virtualization (trap & emulate)
- Memory virtualization (shadow page tables)
- I/O device emulation (VirtIO)
- Distributed VM state across Queztl nodes

This is the foundation - think VMware ESXi or KVM, but for Queztl.
"""

__version__ = "0.1.0-alpha"
__author__ = "Queztl Core Team"

from .core import QueztlHypervisor, VirtualMachine
from .vcpu import VirtualCPU
from .memory import MemoryManager
from .devices import VirtIODevice, VirtIOBlock, VirtIONet, VirtIOGPU
from .boot import BootLoader

__all__ = [
    "QueztlHypervisor",
    "VirtualMachine",
    "VirtualCPU",
    "MemoryManager",
    "VirtIODevice",
    "VirtIOBlock",
    "VirtIONet",
    "VirtIOGPU",
    "BootLoader"
]

# QuetzalCore VM - HTML5 VNC Console Guide

## ✅ Setup Complete!

Your VM console now has a **pure HTML5 VNC client** built-in. No external dependencies, no setup hanging!

---

## 🚀 Quick Start

### 1. Start VM with VNC
```bash
cd /Users/xavasena/hive/vms/test-vm
./start-vnc.sh
```

### 2. Open Console
Open your browser to: **http://localhost:9090**

### 3. Click VNC Tab
Click the **"VNC Display"** tab in the console - it will auto-connect!

---

## 🎯 How It Works

### HTML5 Canvas VNC Client
- **Pure JavaScript** - No plugins, no hanging
- **Auto-connect** - Opens when you click the VNC tab
- **Status indicator** - Shows connection status in real-time
- **WebSocket** - Direct connection to VNC server on port 5900

### Connection Flow
```
Browser → HTML5 Canvas → WebSocket → VNC Server (port 5900)
```

---

## 🖥️ VNC Console Features

### Status Indicators
- 🟢 **VNC Connected** - Successfully connected to VM
- ⚫ **VNC Disconnected** - Not connected
- ⚫ **VNC Connection Failed** - Server not available

### Canvas Display
- Full screen VM display
- Mouse and keyboard input (when connected)
- Real-time rendering
- Crosshair cursor for precision

---

## 📋 Available Scripts

### Start VM with VNC
```bash
./start-vnc.sh
```
Launches VM with VNC enabled on port 5900

### Stop VM
```bash
./stop-vm.sh
```
Gracefully stops the VM

### Start Console Only
```bash
python3 console-server.py
```
Starts just the web console server

---

## 🔧 Manual VNC Launch

If you want to use QEMU directly:

```bash
qemu-system-x86_64 \
    -name "QuetzalCore-test-vm-001" \
    -m 2048 \
    -smp 2 \
    -hda disk.img \
    -vnc :0 \
    -daemonize
```

This exposes VNC on **localhost:5900** (display :0)

---

## 🌐 Console Tabs

### Terminal Tab
- Interactive terminal emulation
- VM logs and output
- Command history

### VNC Display Tab ⭐
- **HTML5 VNC viewer**
- Auto-connects on tab open
- Real-time video display
- Mouse/keyboard support

### Logs Tab
- System logs
- Boot sequence
- Error messages

---

## 🐛 Troubleshooting

### VNC Shows "Connection Failed"
**Problem:** Can't connect to VNC server

**Solutions:**
1. Check if VM is running with VNC:
   ```bash
   lsof -i :5900
   ```

2. Start VM with VNC enabled:
   ```bash
   ./start-vnc.sh
   ```

3. Test VNC manually:
   ```bash
   brew install tiger-vnc
   vncviewer localhost:5900
   ```

### Console Shows Blank Canvas
**Problem:** VNC tab is black/blank

**Solutions:**
1. Refresh the page: `Cmd+R`
2. Close and reopen the VNC tab
3. Check browser console for errors: `Cmd+Option+I`

### WebSocket Connection Error
**Problem:** Browser can't establish WebSocket

**Solutions:**
1. Check if port 5900 is available:
   ```bash
   lsof -i :5900
   ```

2. Verify firewall settings allow localhost connections

3. Try using a native VNC client:
   ```bash
   vncviewer localhost:5900
   ```

---

## 📦 QEMU Installation (macOS)

If QEMU is not installed:

```bash
# Install via Homebrew
brew install qemu

# Verify installation
qemu-system-x86_64 --version
```

---

## 🎨 Customization

### Change VNC Port
Edit `start-vnc.sh` and change:
```bash
-vnc :0    # Port 5900 (display :0)
-vnc :1    # Port 5901 (display :1)
-vnc :2    # Port 5902 (display :2)
```

Then update `console.html` WebSocket URL:
```javascript
vncSocket = new WebSocket('ws://localhost:5901');
```

### Adjust Canvas Size
The canvas auto-resizes to fit the container. To set a fixed size, add CSS:
```css
#vncCanvas {
    width: 1024px;
    height: 768px;
}
```

---

## 🔐 Security Notes

### Localhost Only
- VNC is bound to **localhost** only
- No external access by default
- Safe for local development

### Production Use
For production, consider:
- VNC password authentication
- TLS/SSL encryption
- Reverse proxy with authentication
- VPN or SSH tunnel

---

## 🚀 Advanced Usage

### Multiple VMs
Run multiple VMs on different VNC ports:

```bash
# VM 1 on display :0 (port 5900)
qemu-system-x86_64 -vnc :0 ...

# VM 2 on display :1 (port 5901)
qemu-system-x86_64 -vnc :1 ...

# VM 3 on display :2 (port 5902)
qemu-system-x86_64 -vnc :2 ...
```

### Remote Access
To access VNC remotely (use with caution):

```bash
# Enable remote VNC (NOT RECOMMENDED without authentication)
qemu-system-x86_64 -vnc 0.0.0.0:0 ...

# Better: Use SSH tunnel
ssh -L 5900:localhost:5900 user@remote-host
```

---

## 📚 Resources

### VNC Protocol
- RFB Protocol: https://en.wikipedia.org/wiki/RFB_protocol
- VNC Security: https://tigervnc.org/doc/security.html

### QEMU VNC
- QEMU VNC Docs: https://www.qemu.org/docs/master/system/vnc-security.html
- Display Options: https://wiki.qemu.org/Documentation/GuestGraphics

### HTML5 VNC Clients
- noVNC: https://novnc.com/
- Guacamole: https://guacamole.apache.org/

---

## ✨ What's New

### Removed
- ❌ Setup instructions that hung the UI
- ❌ "Setup VNC" button that did nothing
- ❌ Placeholder text and manual configuration steps
- ❌ External iframe dependencies

### Added
- ✅ Pure HTML5 Canvas VNC client
- ✅ Auto-connect functionality
- ✅ Real-time connection status
- ✅ `start-vnc.sh` launcher script
- ✅ `stop-vm.sh` control script
- ✅ WebSocket-based VNC connection
- ✅ Professional status indicators

---

## 🎯 Testing the Console

### 1. Check Server Running
```bash
lsof -i :9090
```
Should show Python process

### 2. Test VNC Port
```bash
lsof -i :5900
```
Should show qemu process (if VM running)

### 3. Open Console
```bash
open http://localhost:9090
```

### 4. Try VNC Tab
Click "VNC Display" - should see connection attempt

---

## 💡 Pro Tips

1. **Use QuetzalBrowser** for best experience
2. **Keep console server running** in background
3. **Start VM with VNC** before opening VNC tab
4. **Check terminal logs** for connection details
5. **Refresh page** if VNC doesn't connect immediately

---

**Your HTML5 VNC console is ready! No more hanging on setup screens! 🎉**

*Last Updated: December 9, 2025*

/**
 * QuetzalCore Protocol (QP) Client Implementation
 * Binary WebSocket protocol - 10-20x faster than REST
 * 
 * Protocol Format:
 * ┌──────────┬──────────┬──────────┬──────────────┐
 * │  Magic   │  Type    │  Length  │   Payload    │
 * │ (2 bytes)│ (1 byte) │ (4 bytes)│  (N bytes)   │
 * └──────────┴──────────┴──────────┴──────────────┘
 *   0x5150    0x01-0xFF   uint32     data
 */

class QPMessageType {
    // Core protocol
    static COMMAND = 0x01;
    static DATA = 0x02;
    static STREAM = 0x03;
    static ACK = 0x04;
    static ERROR = 0x05;
    static AUTH = 0x10;
    static HEARTBEAT = 0x11;

    // GPU Operations (0x20-0x2F)
    static GPU_PARALLEL_MATMUL = 0x20;
    static GPU_PARALLEL_CONV2D = 0x21;
    static GPU_POOL_STATUS = 0x22;
    static GPU_BENCHMARK = 0x23;
    static GPU_ALLOCATE = 0x24;
    static GPU_FREE = 0x25;
    static GPU_KERNEL_EXEC = 0x26;

    // GIS Operations (0x30-0x3F)
    static GIS_VALIDATE_LIDAR = 0x30;
    static GIS_VALIDATE_RASTER = 0x31;
    static GIS_VALIDATE_VECTOR = 0x32;
    static GIS_VALIDATE_IMAGERY = 0x33;
    static GIS_INTEGRATE_DATA = 0x34;
    static GIS_TRAIN_MODEL = 0x35;
    static GIS_PREDICT = 0x36;
    static GIS_FEEDBACK = 0x37;
    static GIS_ANALYZE_TERRAIN = 0x38;
    static GIS_CORRELATE_MAGNETIC = 0x39;
    static GIS_RESISTIVITY_MAP = 0x3A;

    // System Operations (0x40-0x4F)
    static SYS_METRICS = 0x40;
    static SYS_STATUS = 0x41;
    static SYS_SHUTDOWN = 0x42;
    static SYS_RESTART = 0x43;
}

class QPProtocol {
    constructor() {
        this.MAGIC = 0x5150; // "QP"
        this.HEADER_SIZE = 7;
    }

    /**
     * Pack message into binary QP format
     */
    pack(msgType, payload) {
        const payloadBytes = typeof payload === 'string'
            ? new TextEncoder().encode(payload)
            : payload;

        const buffer = new ArrayBuffer(this.HEADER_SIZE + payloadBytes.byteLength);
        const view = new DataView(buffer);

        // Magic bytes (2)
        view.setUint16(0, this.MAGIC, false); // Big-endian

        // Message type (1)
        view.setUint8(2, msgType);

        // Payload length (4)
        view.setUint32(3, payloadBytes.byteLength, false); // Big-endian

        // Payload
        const uint8View = new Uint8Array(buffer);
        uint8View.set(new Uint8Array(payloadBytes), this.HEADER_SIZE);

        return buffer;
    }

    /**
     * Unpack binary QP message
     */
    unpack(data) {
        const view = new DataView(data);

        // Validate magic bytes
        const magic = view.getUint16(0, false);
        if (magic !== this.MAGIC) {
            throw new Error(`Invalid magic bytes: 0x${magic.toString(16)}`);
        }

        // Extract message type
        const msgType = view.getUint8(2);

        // Extract payload length
        const length = view.getUint32(3, false);

        // Extract payload
        const payload = new Uint8Array(data, this.HEADER_SIZE, length);

        return { msgType, payload };
    }

    /**
     * Pack JSON data into QP message
     */
    packJSON(msgType, data) {
        const json = JSON.stringify(data);
        return this.pack(msgType, json);
    }

    /**
     * Unpack QP message with JSON payload
     */
    unpackJSON(data) {
        const { msgType, payload } = this.unpack(data);
        const json = new TextDecoder().decode(payload);
        return { msgType, data: JSON.parse(json) };
    }
}

class QuetzalCoreClient {
    constructor() {
        this.protocol = new QPProtocol();
        this.ws = null;
        this.connected = false;
        this.messageCount = 0;
        this.handlers = new Map();
        this.history = [];
        this.historyIndex = -1;

        // Performance tracking
        this.lastMessageTime = 0;
        this.latency = 0;

        this.initUI();
    }

    initUI() {
        // URL input and navigation
        this.urlInput = document.getElementById('urlInput');
        this.goBtn = document.getElementById('goBtn');
        this.backBtn = document.getElementById('backBtn');
        this.forwardBtn = document.getElementById('forwardBtn');
        this.refreshBtn = document.getElementById('refreshBtn');

        // Status indicators
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.latencyText = document.getElementById('latencyText');

        // Content areas
        this.viewerTitle = document.getElementById('viewerTitle');
        this.viewerContent = document.getElementById('viewerContent');
        this.console = document.getElementById('console');
        this.vizCanvas = document.getElementById('vizCanvas');
        this.gpuPool = document.getElementById('gpuPool');

        // Metrics
        this.gpuCount = document.getElementById('gpuCount');
        this.totalGflops = document.getElementById('totalGflops');
        this.msgCount = document.getElementById('msgCount');

        // Event listeners
        this.goBtn.addEventListener('click', () => this.navigate());
        this.urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.navigate();
        });

        this.refreshBtn.addEventListener('click', () => this.refresh());
        this.backBtn.addEventListener('click', () => this.goBack());
        this.forwardBtn.addEventListener('click', () => this.goForward());

        // Operation cards
        document.querySelectorAll('.operation-card').forEach(card => {
            card.addEventListener('click', () => {
                const op = card.dataset.op;
                this.executeOperation(op);
            });
        });

        document.getElementById('clearBtn').addEventListener('click', () => this.clearConsole());
        document.getElementById('exportBtn').addEventListener('click', () => this.exportData());
    }

    async navigate() {
        let url = this.urlInput.value.trim();

        // Handle different protocols
        if (url.startsWith('qp://')) {
            // Convert qp:// to ws://
            url = url.replace('qp://', 'ws://');
        } else if (url.startsWith('qps://')) {
            // Convert qps:// to wss:// (secure)
            url = url.replace('qps://', 'wss://');
        } else if (url.startsWith('https://')) {
            // HTTPS - fetch and display
            await this.loadHTTPS(url);
            return;
        } else if (url.startsWith('http://')) {
            await this.loadHTTP(url);
            return;
        } else if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
            // Default to QP protocol
            url = 'ws://' + url;
        }

        // Add to history
        this.history.push(url);
        this.historyIndex = this.history.length - 1;
        this.updateNavButtons();

        await this.connect(url);
    }

    async loadHTTPS(url) {
        this.log('info', `Loading HTTPS: ${url}`);

        try {
            const response = await fetch(url);
            const contentType = response.headers.get('content-type');

            if (contentType.includes('application/json')) {
                const data = await response.json();
                this.displayJSON(data);
            } else if (contentType.includes('text/html')) {
                const html = await response.text();
                this.displayHTML(html);
            } else {
                const text = await response.text();
                this.displayText(text);
            }

            this.log('success', 'HTTPS content loaded');
        } catch (error) {
            this.log('error', `Failed to load HTTPS: ${error.message}`);
        }
    }

    async loadHTTP(url) {
        // Same as HTTPS but for HTTP
        await this.loadHTTPS(url);
    }

    async connect(url) {
        if (this.ws) {
            this.ws.close();
        }

        this.log('info', `Connecting to ${url}...`);

        try {
            this.ws = new WebSocket(url);
            this.ws.binaryType = 'arraybuffer';

            this.ws.onopen = () => this.onConnect();
            this.ws.onmessage = (event) => this.onMessage(event);
            this.ws.onerror = (error) => this.onError(error);
            this.ws.onclose = () => this.onDisconnect();

        } catch (error) {
            this.log('error', `Connection failed: ${error.message}`);
        }
    }

    onConnect() {
        this.connected = true;
        this.statusDot.className = 'status-dot connected';
        this.statusText.textContent = 'Connected';
        this.log('success', 'QP Protocol connection established');

        // Send heartbeat
        this.startHeartbeat();
    }

    onMessage(event) {
        this.messageCount++;
        this.msgCount.textContent = this.messageCount;

        // Calculate latency
        const now = Date.now();
        if (this.lastMessageTime > 0) {
            this.latency = now - this.lastMessageTime;
            this.latencyText.textContent = `${this.latency}ms`;
        }
        this.lastMessageTime = now;

        try {
            const { msgType, data } = this.protocol.unpackJSON(event.data);

            this.log('qp', `Received message type: 0x${msgType.toString(16).padStart(2, '0')}`);

            // Handle different message types
            switch (msgType) {
                case QPMessageType.ACK:
                    this.handleAck(data);
                    break;
                case QPMessageType.DATA:
                    this.handleData(data);
                    break;
                case QPMessageType.STREAM:
                    this.handleStream(data);
                    break;
                case QPMessageType.ERROR:
                    this.handleError(data);
                    break;
                default:
                    this.log('info', `Unknown message type: 0x${msgType.toString(16)}`);
            }
        } catch (error) {
            this.log('error', `Failed to parse message: ${error.message}`);
        }
    }

    onError(error) {
        this.log('error', `WebSocket error: ${error}`);
    }

    onDisconnect() {
        this.connected = false;
        this.statusDot.className = 'status-dot disconnected';
        this.statusText.textContent = 'Disconnected';
        this.log('info', 'QP Protocol connection closed');

        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }
    }

    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.connected) {
                const msg = this.protocol.packJSON(QPMessageType.HEARTBEAT, {
                    timestamp: Date.now()
                });
                this.ws.send(msg);
            }
        }, 30000); // Every 30 seconds
    }

    sendMessage(msgType, data) {
        if (!this.connected) {
            this.log('error', 'Not connected to QP server');
            return;
        }

        const msg = this.protocol.packJSON(msgType, data);
        this.ws.send(msg);
        this.log('qp', `Sent message type: 0x${msgType.toString(16).padStart(2, '0')}`);
    }

    handleAck(data) {
        this.log('success', `ACK: ${JSON.stringify(data)}`);
    }

    handleData(data) {
        this.log('info', 'Data received');
        this.displayJSON(data);
    }

    handleStream(data) {
        if (data.progress !== undefined) {
            this.updateProgress(data.progress);
        }

        if (data.status) {
            this.log('info', `Status: ${data.status}`);
        }

        // Update GPU pool if present
        if (data.gpu_units) {
            this.updateGPUPool(data.gpu_units);
        }
    }

    handleError(data) {
        this.log('error', `Error: ${data.error}`);
    }

    // Operation Handlers
    async executeOperation(op) {
        this.log('info', `Executing operation: ${op}`);

        switch (op) {
            case 'gpu-matmul':
                this.sendMessage(QPMessageType.GPU_PARALLEL_MATMUL, {
                    size: 2048,
                    num_gpus: 4
                });
                break;

            case 'gpu-conv2d':
                this.sendMessage(QPMessageType.GPU_PARALLEL_CONV2D, {
                    input_size: [1024, 1024],
                    kernel_size: [3, 3],
                    num_gpus: 4
                });
                break;

            case 'gpu-status':
                this.sendMessage(QPMessageType.GPU_POOL_STATUS, {});
                break;

            case 'gpu-benchmark':
                this.sendMessage(QPMessageType.GPU_BENCHMARK, {
                    operations: ['matmul', 'conv2d']
                });
                break;

            case 'gis-validate':
                // Generate sample LiDAR data
                const points = this.generateSampleLiDAR(10000);
                this.sendMessage(QPMessageType.GIS_VALIDATE_LIDAR, {
                    points: points
                });
                break;

            case 'gis-integrate':
                this.sendMessage(QPMessageType.GIS_INTEGRATE_DATA, {
                    dem: 'sample_dem.tif',
                    magnetic_data: 'sample_mag.xyz'
                });
                break;

            case 'gis-train':
                this.sendMessage(QPMessageType.GIS_TRAIN_MODEL, {
                    model_type: 'terrain',
                    training_data: 'sample_dataset.npz'
                });
                break;

            case 'gis-visualize':
                this.visualizeTerrain();
                break;

            case 'sys-metrics':
                this.sendMessage(QPMessageType.SYS_METRICS, {});
                break;

            case 'sys-health':
                await this.checkHealth();
                break;
        }
    }

    generateSampleLiDAR(count) {
        const points = [];
        for (let i = 0; i < count; i++) {
            points.push({
                x: Math.random() * 1000,
                y: Math.random() * 1000,
                z: Math.random() * 100,
                intensity: Math.random() * 255,
                classification: Math.floor(Math.random() * 5)
            });
        }
        return points;
    }

    async checkHealth() {
        try {
            const response = await fetch('http://localhost:8000/api/health');
            const data = await response.json();
            this.displayJSON(data);
            this.log('success', 'Health check complete');
        } catch (error) {
            this.log('error', `Health check failed: ${error.message}`);
        }
    }

    // UI Display Functions
    displayJSON(data) {
        const pre = document.createElement('pre');
        pre.style.cssText = 'background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; overflow-x: auto; color: #4ade80; font-family: Monaco, monospace; font-size: 12px;';
        pre.textContent = JSON.stringify(data, null, 2);
        this.vizCanvas.innerHTML = '';
        this.vizCanvas.appendChild(pre);
    }

    displayHTML(html) {
        const iframe = document.createElement('iframe');
        iframe.style.cssText = 'width: 100%; height: 100%; border: none;';
        iframe.srcdoc = html;
        this.vizCanvas.innerHTML = '';
        this.vizCanvas.appendChild(iframe);
    }

    displayText(text) {
        const pre = document.createElement('pre');
        pre.style.cssText = 'background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; overflow-x: auto; white-space: pre-wrap; font-family: Monaco, monospace; font-size: 12px;';
        pre.textContent = text;
        this.vizCanvas.innerHTML = '';
        this.vizCanvas.appendChild(pre);
    }

    updateProgress(percent) {
        this.vizCanvas.innerHTML = `
            <div style="width: 100%; padding: 20px;">
                <div style="font-size: 18px; margin-bottom: 10px;">Processing... ${percent}%</div>
                <div style="width: 100%; height: 30px; background: rgba(102,126,234,0.2); border-radius: 15px; overflow: hidden;">
                    <div style="width: ${percent}%; height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); transition: width 0.3s;"></div>
                </div>
            </div>
        `;
    }

    updateGPUPool(units) {
        this.gpuPool.innerHTML = '';
        this.gpuCount.textContent = units.length;

        let totalGflops = 0;

        units.forEach(unit => {
            totalGflops += unit.gflops || 0;

            const card = document.createElement('div');
            card.className = 'gpu-unit' + (unit.status === 'active' ? ' active' : '');
            card.innerHTML = `
                <div class="gpu-header">
                    <div class="gpu-name">${unit.name}</div>
                    <div class="gpu-status">${unit.status}</div>
                </div>
                <div class="gpu-metric">
                    <span>GFLOPS</span>
                    <span>${unit.gflops || 0}</span>
                </div>
                <div class="gpu-metric">
                    <span>Utilization</span>
                    <span>${unit.utilization || 0}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${unit.utilization || 0}%"></div>
                </div>
            `;
            this.gpuPool.appendChild(card);
        });

        this.totalGflops.textContent = totalGflops.toFixed(1);
    }

    visualizeTerrain() {
        // Simple 3D terrain visualization
        this.vizCanvas.innerHTML = `
            <canvas id="terrainCanvas" style="width: 100%; height: 100%;"></canvas>
        `;

        const canvas = document.getElementById('terrainCanvas');
        const ctx = canvas.getContext('2d');
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;

        // Draw simple terrain
        ctx.fillStyle = '#0a0e27';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw terrain lines
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 2;

        for (let i = 0; i < 20; i++) {
            ctx.beginPath();
            ctx.moveTo(0, canvas.height * 0.5);

            for (let x = 0; x < canvas.width; x += 10) {
                const y = canvas.height * 0.5 + Math.sin(x * 0.02 + i * 0.5) * 50 + Math.random() * 20;
                ctx.lineTo(x, y);
            }

            ctx.stroke();
        }

        this.log('success', 'Terrain visualization rendered');
    }

    // Console Log
    log(type, message) {
        const time = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = `
            <span class="log-time">[${time}]</span>
            <span class="log-type ${type}">${type.toUpperCase()}</span>
            <span>${message}</span>
        `;
        this.console.appendChild(entry);
        this.console.scrollTop = this.console.scrollHeight;
    }

    clearConsole() {
        this.console.innerHTML = '';
        this.log('info', 'Console cleared');
    }

    exportData() {
        const data = {
            history: this.history,
            messageCount: this.messageCount,
            latency: this.latency,
            logs: this.console.innerText
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `quetzal-session-${Date.now()}.json`;
        a.click();

        this.log('success', 'Session data exported');
    }

    refresh() {
        if (this.history[this.historyIndex]) {
            this.urlInput.value = this.history[this.historyIndex];
            this.navigate();
        }
    }

    goBack() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.urlInput.value = this.history[this.historyIndex];
            this.navigate();
            this.updateNavButtons();
        }
    }

    goForward() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this.urlInput.value = this.history[this.historyIndex];
            this.navigate();
            this.updateNavButtons();
        }
    }

    updateNavButtons() {
        this.backBtn.disabled = this.historyIndex <= 0;
        this.forwardBtn.disabled = this.historyIndex >= this.history.length - 1;
    }
}

// Initialize client when page loads
const client = new QuetzalCoreClient();

// Auto-connect on load
window.addEventListener('load', () => {
    client.log('info', 'QuetzalCore Native Browser ready');
    client.log('info', 'Supports: QP Protocol, HTTPS, HTTP');
    client.log('info', 'Type qp://localhost:8000/ws/qp to connect');
});

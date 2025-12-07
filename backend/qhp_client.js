/**
 * Queztl Protocol Client (JavaScript/Browser)
 * Binary WebSocket protocol - 10-20x faster than REST
 * 
 * Usage:
 *   const qp = new QueztlProtocol('wss://api.queztl.com');
 *   await qp.connect();
 *   await qp.execute('text-to-3d', { prompt: 'dragon' });
 */

class QueztlProtocol {
    // Message types
    static COMMAND = 0x01;
    static DATA = 0x02;
    static STREAM = 0x03;
    static ACK = 0x04;
    static ERROR = 0x05;
    static AUTH = 0x10;
    static HEARTBEAT = 0x11;

    constructor(url) {
        this.url = url;
        this.ws = null;
        this.handlers = {};
        this.pending = {};
        this.jobId = 0;
        this.stats = {
            messagesSent: 0,
            messagesReceived: 0,
            bytesSent: 0,
            bytesReceived: 0
        };
    }

    /**
     * Pack binary message
     * Format: [Magic(2) | Type(1) | Length(4) | Payload(N)]
     */
    pack(type, payload) {
        const magic = new Uint8Array([0x51, 0x50]); // "QP"
        const typeByte = new Uint8Array([type]);

        // Convert payload to Uint8Array if needed
        let payloadBytes;
        if (payload instanceof Uint8Array) {
            payloadBytes = payload;
        } else if (typeof payload === 'string') {
            payloadBytes = new TextEncoder().encode(payload);
        } else {
            payloadBytes = new Uint8Array(payload);
        }

        // Create length field (4 bytes, big-endian)
        const length = new DataView(new ArrayBuffer(4));
        length.setUint32(0, payloadBytes.byteLength, false); // Big-endian

        // Combine all parts
        const message = new Uint8Array(7 + payloadBytes.byteLength);
        message.set(magic, 0);
        message.set(typeByte, 2);
        message.set(new Uint8Array(length.buffer), 3);
        message.set(payloadBytes, 7);

        return message.buffer;
    }

    /**
     * Unpack binary message
     */
    unpack(data) {
        const view = new DataView(data);

        // Check magic bytes
        const magic = view.getUint16(0, false); // Big-endian
        if (magic !== 0x5150) { // "QP"
            throw new Error(`Invalid magic bytes: 0x${magic.toString(16)}`);
        }

        const type = view.getUint8(2);
        const length = view.getUint32(3, false); // Big-endian
        const payload = data.slice(7, 7 + length);

        return { type, payload };
    }

    /**
     * Pack JSON into message
     */
    packJSON(type, data) {
        const json = JSON.stringify(data);
        const payload = new TextEncoder().encode(json);
        return this.pack(type, payload);
    }

    /**
     * Unpack message and decode JSON
     */
    unpackJSON(data) {
        const { type, payload } = this.unpack(data);
        const json = new TextDecoder().decode(payload);
        const data_parsed = JSON.parse(json);
        return { type, data: data_parsed };
    }

    /**
     * Connect to server
     */
    async connect() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.url);
            this.ws.binaryType = 'arraybuffer';

            this.ws.onopen = () => {
                console.log('Queztl Protocol: Connected');
                this.startHeartbeat();
                resolve();
            };

            this.ws.onerror = (error) => {
                console.error('Queztl Protocol: Error', error);
                reject(error);
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(event.data);
            };

            this.ws.onclose = () => {
                console.log('Queztl Protocol: Disconnected');
                this.stopHeartbeat();
            };
        });
    }

    /**
     * Disconnect from server
     */
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    /**
     * Send message
     */
    send(type, payload) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            throw new Error('Not connected');
        }

        const message = this.pack(type, payload);
        this.ws.send(message);

        this.stats.messagesSent++;
        this.stats.bytesSent += message.byteLength;
    }

    /**
     * Send JSON message
     */
    sendJSON(type, data) {
        const message = this.packJSON(type, data);
        this.ws.send(message);

        this.stats.messagesSent++;
        this.stats.bytesSent += message.byteLength;
    }

    /**
     * Handle incoming message
     */
    handleMessage(data) {
        this.stats.messagesReceived++;
        this.stats.bytesReceived += data.byteLength;

        try {
            const { type, data: msgData } = this.unpackJSON(data);

            // Call registered handlers
            const handlers = this.handlers[type] || [];
            handlers.forEach(handler => handler(msgData));

            // Handle specific message types
            switch (type) {
                case QueztlProtocol.ACK:
                    console.log('ACK:', msgData);
                    break;

                case QueztlProtocol.ERROR:
                    console.error('Error:', msgData.error);
                    break;

                case QueztlProtocol.STREAM:
                    // Handled by registered handlers
                    break;

                case QueztlProtocol.DATA:
                    // Handled by registered handlers
                    break;
            }
        } catch (error) {
            console.error('Error handling message:', error);
        }
    }

    /**
     * Register message handler
     */
    on(type, handler) {
        if (!this.handlers[type]) {
            this.handlers[type] = [];
        }
        this.handlers[type].push(handler);
    }

    /**
     * Remove message handler
     */
    off(type, handler) {
        if (this.handlers[type]) {
            this.handlers[type] = this.handlers[type].filter(h => h !== handler);
        }
    }

    /**
     * Authenticate with server
     */
    async auth(token) {
        this.sendJSON(QueztlProtocol.AUTH, { token });
    }

    /**
     * Execute capability
     */
    async execute(capability, params = {}) {
        const jobId = ++this.jobId;

        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Request timeout'));
            }, 30000); // 30s timeout

            // Handler for progress updates
            const progressHandler = (data) => {
                if (this.onProgress) {
                    this.onProgress(data.progress, data);
                }
            };

            // Handler for completion
            const dataHandler = (data) => {
                clearTimeout(timeout);
                this.off(QueztlProtocol.STREAM, progressHandler);
                this.off(QueztlProtocol.DATA, dataHandler);
                this.off(QueztlProtocol.ERROR, errorHandler);
                resolve(data);
            };

            // Handler for errors
            const errorHandler = (data) => {
                clearTimeout(timeout);
                this.off(QueztlProtocol.STREAM, progressHandler);
                this.off(QueztlProtocol.DATA, dataHandler);
                this.off(QueztlProtocol.ERROR, errorHandler);
                reject(new Error(data.error));
            };

            // Register handlers
            this.on(QueztlProtocol.STREAM, progressHandler);
            this.on(QueztlProtocol.DATA, dataHandler);
            this.on(QueztlProtocol.ERROR, errorHandler);

            // Send command
            this.sendJSON(QueztlProtocol.COMMAND, {
                cap: capability,
                params,
                job_id: jobId
            });
        });
    }

    /**
     * Start heartbeat to keep connection alive
     */
    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.sendJSON(QueztlProtocol.HEARTBEAT, {
                    timestamp: Date.now()
                });
            }
        }, 30000); // Every 30s
    }

    /**
     * Stop heartbeat
     */
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    /**
     * Get connection stats
     */
    getStats() {
        return {
            ...this.stats,
            connected: this.ws && this.ws.readyState === WebSocket.OPEN,
            url: this.url
        };
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = QueztlProtocol;
}

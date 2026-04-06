// ws-client.js — WebSocket connection manager.
// Handles connect/reconnect, binary/text dispatch, and connection state.

export class WsClient {
    constructor() {
        this.ws = null;
        this.url = null;
        this.onOpen = null;
        this.onClose = null;
        this.onText = null;
        this.onBinary = null;
        this._reconnectTimer = null;
    }

    connect(url) {
        this.url = url;
        this._doConnect();
    }

    disconnect() {
        clearTimeout(this._reconnectTimer);
        this._reconnectTimer = null;
        if (this.ws) {
            this.ws.onclose = null;   // prevent reconnect
            this.ws.close();
            this.ws = null;
        }
    }

    sendBinary(data) {
        if (this.ws?.readyState === WebSocket.OPEN)
            this.ws.send(data);
    }

    sendText(msg) {
        if (this.ws?.readyState === WebSocket.OPEN)
            this.ws.send(msg);
    }

    get connected() {
        return this.ws?.readyState === WebSocket.OPEN;
    }

    _doConnect() {
        if (this.ws) return;
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
            this.onOpen?.();
        };

        this.ws.onclose = () => {
            this.ws = null;
            this.onClose?.();
            // Auto-reconnect after 2 s.
            this._reconnectTimer = setTimeout(() => this._doConnect(), 2000);
        };

        this.ws.onerror = () => {
            this.ws?.close();
        };

        this.ws.onmessage = (ev) => {
            if (ev.data instanceof ArrayBuffer)
                this.onBinary?.(ev.data);
            else
                this.onText?.(ev.data);
        };
    }
}

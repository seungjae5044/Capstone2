import type { WebSocketMessage } from '../types';

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private messageHandler: ((data: WebSocketMessage) => void) | null = null;

  connect(onMessage: (data: WebSocketMessage) => void) {
    this.messageHandler = onMessage;
    this.ws = new WebSocket('ws://localhost:8000/ws');

    this.ws.onopen = () => console.log('WebSocket connected');
    this.ws.onclose = () => this.scheduleReconnect();
    this.ws.onerror = (err) => console.error('WebSocket error:', err);
    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.messageHandler?.(data);
      } catch (err) {
        console.error('Failed to parse message:', err);
      }
    };
  }

  send(command: string, payload: any = {}) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ command, ...payload }));
    }
  }

  disconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }

  private scheduleReconnect() {
    this.reconnectTimer = setTimeout(() => {
      if (this.messageHandler) this.connect(this.messageHandler);
    }, 3000);
  }
}

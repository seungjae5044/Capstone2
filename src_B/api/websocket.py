"""
WebSocket connection management for real-time updates.

This module handles:
- WebSocket connection lifecycle
- Broadcasting messages to all connected clients
- Background worker for event queue processing
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts messages to clients."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)
        logger.info("WebSocket connected: %d active connections", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket connection."""
        async with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected: %d active connections", len(self.active_connections))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected WebSocket clients."""
        async with self.lock:
            connections = list(self.active_connections)

        if not connections:
            return

        disconnect_list: List[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as exc:  # noqa: BLE001
                logger.error("WebSocket send failed: %s", exc)
                disconnect_list.append(connection)

        if disconnect_list:
            async with self.lock:
                for connection in disconnect_list:
                    if connection in self.active_connections:
                        self.active_connections.remove(connection)


# Global connection manager instance
manager = ConnectionManager()


async def broadcast_worker(event_queue: asyncio.Queue) -> None:
    """
    Background worker that reads from event queue and broadcasts to WebSocket clients.

    Args:
        event_queue: Async queue containing messages to broadcast
    """
    while True:
        message = await event_queue.get()
        try:
            await manager.broadcast(message)
        finally:
            event_queue.task_done()


async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint handler.

    Accepts connection, keeps it alive, and handles disconnection.

    Args:
        websocket: The WebSocket connection
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive by receiving (and ignoring) client messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)

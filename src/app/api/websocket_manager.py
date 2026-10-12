"""
EchoLocate — Real-Time WebSocket Streaming Manager
===================================================
Broadcasts live localization packets, heatmaps, and latency metrics
to connected frontend web dashboards.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """Manages active browser WebSocket connections for live localization streams."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New WebSocket client connected. Total clients: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket client disconnected. Remaining clients: %d", len(self.active_connections))

    async def broadcast_json(self, data: dict[str, Any]) -> None:
        """Broadcast payload to all active clients."""
        for connection in list(self.active_connections):
            try:
                await connection.send_text(json.dumps(data))
            except Exception as e:
                logger.warning("Failed to send WebSocket message: %s. Disconnecting client.", e)
                self.disconnect(connection)

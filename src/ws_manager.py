"""Gestor de conexiones WebSocket para broadcast de alertas en tiempo real."""
import asyncio
import logging
from typing import List
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Gestiona conexiones WebSocket activas y difunde mensajes a todos los clientes."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._loop = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Cliente WebSocket conectado. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Cliente WebSocket desconectado. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Envía mensaje a todos los clientes conectados. Elimina conexiones muertas."""
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d)

    def broadcast_sync(self, message: dict):
        """Broadcast thread-safe desde código síncrono (ej: poll() en daemon thread).
        Usa run_coroutine_threadsafe para enviar al event loop de uvicorn.
        """
        if not self.active_connections:
            return
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)
            try:
                future.result(timeout=5)
            except Exception as e:
                logger.warning(f"Error en broadcast WebSocket: {e}")


# Instancia singleton
ws_manager = ConnectionManager()

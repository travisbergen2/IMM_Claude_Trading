"""
IMM AGI Hub — WebSocket Server
FastAPI + ngrok bridge. EAs connect as WebSocket clients.
Hub pushes commands; EAs push tick updates and heartbeats.
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Optional, Callable, Awaitable
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

log = logging.getLogger("hub_server")


class HubServer:
    """
    Manages WebSocket connections to EAs and exposes REST status endpoints.
    """

    def __init__(self, hub_callback: Callable):
        """
        hub_callback: async function called with (ea_id, message_dict)
                      when an EA sends a message.
        """
        self.hub_callback = hub_callback
        self.app = FastAPI(title="IMM AGI Hub", version="1.0")
        self._connections: Dict[str, WebSocket] = {}
        self._last_seen:   Dict[str, float]     = {}
        self._hub_state: dict = {}   # set externally by IMMHub

        self._register_routes()

    # ── Routes ────────────────────────────────────────────────────────

    def _register_routes(self):
        app = self.app

        @app.websocket("/ea/{ea_id}")
        async def ea_endpoint(websocket: WebSocket, ea_id: str):
            await websocket.accept()
            self._connections[ea_id] = websocket
            self._last_seen[ea_id]   = time.time()
            log.info(f"EA connected: {ea_id}")
            try:
                while True:
                    raw = await websocket.receive_text()
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        log.warning(f"EA {ea_id} sent non-JSON: {raw[:80]}")
                        continue
                    self._last_seen[ea_id] = time.time()
                    await self.hub_callback(ea_id, data)
            except WebSocketDisconnect:
                log.warning(f"EA disconnected: {ea_id}")
                self._connections.pop(ea_id, None)
                self._last_seen.pop(ea_id, None)
            except Exception as e:
                log.error(f"EA {ea_id} error: {e}")
                self._connections.pop(ea_id, None)

        @app.get("/status")
        async def status():
            return JSONResponse({
                "timestamp": datetime.utcnow().isoformat(),
                "connected_eas": list(self._connections.keys()),
                "hub": self._hub_state,
            })

        @app.get("/ping")
        async def ping():
            return {"pong": True}

        @app.post("/command/{ea_id}")
        async def send_command_http(ea_id: str, body: dict):
            """REST fallback for sending a command to an EA."""
            ok = await self.send_to_ea(ea_id, body)
            if not ok:
                raise HTTPException(404, f"EA {ea_id} not connected")
            return {"status": "sent"}

    # ── Public API ────────────────────────────────────────────────────

    async def send_to_ea(self, ea_id: str, message: dict) -> bool:
        ws = self._connections.get(ea_id)
        if ws is None:
            log.warning(f"send_to_ea({ea_id}): not connected")
            return False
        try:
            await ws.send_text(json.dumps(message))
            return True
        except Exception as e:
            log.error(f"send_to_ea({ea_id}) failed: {e}")
            self._connections.pop(ea_id, None)
            return False

    async def broadcast(self, message: dict):
        """Send to all connected EAs."""
        for ea_id in list(self._connections.keys()):
            await self.send_to_ea(ea_id, message)

    def is_connected(self, ea_id: str) -> bool:
        return ea_id in self._connections

    def connected_eas(self) -> list:
        return list(self._connections.keys())

    def set_hub_state(self, state: dict):
        self._hub_state = state

    # ── Server lifecycle ──────────────────────────────────────────────

    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start uvicorn in background (non-blocking)."""
        config = uvicorn.Config(
            self.app, host=host, port=port,
            log_level="warning", loop="asyncio"
        )
        server = uvicorn.Server(config)
        log.info(f"Hub server starting on {host}:{port}")
        await server.serve()


def start_ngrok(port: int, auth_token: Optional[str] = None) -> str:
    """
    Start ngrok tunnel and return the public URL.
    Returns empty string if pyngrok is not available.
    """
    try:
        from pyngrok import ngrok, conf
        if auth_token:
            conf.get_default().auth_token = auth_token
        tunnel = ngrok.connect(port, "http")
        url = tunnel.public_url
        # Convert http to wss for WebSocket
        ws_url = url.replace("http://", "ws://").replace("https://", "wss://")
        log.info(f"ngrok tunnel active: {url}")
        log.info(f"EA WebSocket endpoint: {ws_url}/ea/{{ea_id}}")
        log.info(f"Status dashboard: {url}/status")
        return url
    except ImportError:
        log.warning("pyngrok not installed — ngrok tunnel unavailable")
        return ""
    except Exception as e:
        log.warning(f"ngrok startup failed: {e}")
        return ""

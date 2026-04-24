"""
IMM AGI Hub — HTTP Polling Server
Replaces WebSocket with HTTP request/response for full Windows
TradeLocker / MT5 compatibility using WebRequest().

Architecture:
  EA → POST /ea/tick/{ea_id}     (push tick data each OnTick)
  EA → GET  /ea/poll/{ea_id}     (pull pending commands each poll cycle)
  EA → POST /ea/event/{ea_id}    (trade opened, closed, error events)
  Hub → GET  /status             (dashboard)
  Hub → GET  /cipolla            (Cipolla population snapshot)

No WebSocket dependency. Works with MT5 WebRequest() on any Windows build.

MT5 Setup:
  Tools → Options → Expert Advisors
  → Enable "Allow WebRequest for listed URL"
  → Add your ngrok URL (e.g. https://xxxx.ngrok-free.app)
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime
from typing import Dict, Deque, Optional, Callable, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

log = logging.getLogger("hub_server_http")

# Max commands held in queue per EA before oldest are dropped
COMMAND_QUEUE_MAX = 50
# Seconds before an EA is considered disconnected
EA_TIMEOUT_SECS   = 30


class HubServerHTTP:
    """
    HTTP polling bridge between the Python hub and TradeLocker EAs.

    EAs poll GET /ea/poll/{ea_id} at their OnTimer interval.
    Hub queues commands via queue_command(ea_id, cmd).
    Commands are returned as a JSON array and cleared on receipt.
    """

    def __init__(self, hub_callback: Callable):
        self.hub_callback = hub_callback
        self.app          = FastAPI(title="IMM AGI Hub HTTP", version="2.0")

        # Per-EA command queues (hub → EA direction)
        self._cmd_queues:  Dict[str, Deque[dict]] = {}
        # Per-EA last seen timestamp
        self._last_seen:   Dict[str, float]       = {}
        # Last tick received per EA (EA → hub direction)
        self._last_ticks:  Dict[str, dict]        = {}
        # Hub state snapshot for status endpoint
        self._hub_state:   dict                   = {}
        # Cipolla state snapshot
        self._cipolla_state: dict                 = {}

        self._register_routes()

    # ── Routes ────────────────────────────────────────────────────────

    def _register_routes(self):
        app = self.app

        # ── EA: Push tick data ────────────────────────────────────────
        @app.post("/ea/tick/{ea_id}")
        async def receive_tick(ea_id: str, request: Request):
            """
            EA calls this on every OnTick (or OnTimer).
            Body: JSON with tick data.
            Returns: {"status": "ok", "queue_depth": N}
            """
            try:
                body = await request.json()
            except Exception:
                body = {}

            self._last_seen[ea_id]  = time.time()
            self._last_ticks[ea_id] = body
            self._ensure_queue(ea_id)

            await self.hub_callback(ea_id, {**body, "type": "tick_update"})

            return {
                "status":      "ok",
                "queue_depth": len(self._cmd_queues[ea_id]),
                "timestamp":   datetime.utcnow().isoformat(),
            }

        # ── EA: Poll for commands ─────────────────────────────────────
        @app.get("/ea/poll/{ea_id}")
        async def poll_commands(ea_id: str):
            """
            EA calls this every N seconds (OnTimer).
            Returns all pending commands as a JSON array.
            Clears the queue after returning.

            MT5 WebRequest example:
              string url = HubURL + "/ea/poll/remora";
              WebRequest("GET", url, headers, timeout, empty, result, rheaders);
            """
            self._last_seen[ea_id] = time.time()
            self._ensure_queue(ea_id)

            queue    = self._cmd_queues[ea_id]
            commands = list(queue)
            queue.clear()

            return {
                "commands":  commands,
                "count":     len(commands),
                "timestamp": datetime.utcnow().isoformat(),
            }

        # ── EA: Push events (trade open/close/error) ──────────────────
        @app.post("/ea/event/{ea_id}")
        async def receive_event(ea_id: str, request: Request):
            """EA reports trade events back to hub."""
            try:
                body = await request.json()
            except Exception:
                body = {}

            self._last_seen[ea_id] = time.time()
            await self.hub_callback(ea_id, body)
            return {"status": "ok"}

        # ── EA: Heartbeat ─────────────────────────────────────────────
        @app.get("/ea/heartbeat/{ea_id}")
        async def heartbeat(ea_id: str):
            self._last_seen[ea_id] = time.time()
            self._ensure_queue(ea_id)
            return {
                "status": "alive",
                "queue_depth": len(self._cmd_queues[ea_id]),
            }

        # ── Hub: Status dashboard ─────────────────────────────────────
        @app.get("/status")
        async def status():
            connected = [
                ea for ea, t in self._last_seen.items()
                if time.time() - t < EA_TIMEOUT_SECS
            ]
            return JSONResponse({
                "timestamp":     datetime.utcnow().isoformat(),
                "connected_eas": connected,
                "hub":           self._hub_state,
            })

        # ── Hub: Cipolla snapshot ─────────────────────────────────────
        @app.get("/cipolla")
        async def cipolla():
            return JSONResponse(self._cipolla_state)

        # ── Hub: Inject command via REST (for testing) ────────────────
        @app.post("/hub/command/{ea_id}")
        async def inject_command(ea_id: str, request: Request):
            """Send a command directly to an EA queue (test/admin endpoint)."""
            try:
                cmd = await request.json()
            except Exception:
                raise HTTPException(400, "Invalid JSON")
            self.queue_command(ea_id, cmd)
            return {"status": "queued", "ea_id": ea_id}

        @app.get("/ping")
        async def ping():
            return {"pong": True, "ts": datetime.utcnow().isoformat()}

    # ── Public API ────────────────────────────────────────────────────

    def queue_command(self, ea_id: str, command: dict):
        """
        Queue a command for delivery to an EA on its next poll.
        Thread-safe for asyncio context.
        """
        self._ensure_queue(ea_id)
        queue = self._cmd_queues[ea_id]
        if len(queue) >= COMMAND_QUEUE_MAX:
            queue.popleft()  # drop oldest if overflow
        queue.append(command)
        log.debug(f"Queued cmd for {ea_id}: {command.get('type', '?')} "
                  f"(queue depth {len(queue)})")

    async def send_to_ea(self, ea_id: str, command: dict) -> bool:
        """
        Async wrapper — queues command for next poll.
        Always returns True (fire-and-forget; EA picks up on next poll).
        """
        self.queue_command(ea_id, command)
        return True

    async def broadcast(self, command: dict):
        """Queue command to all known EAs."""
        for ea_id in list(self._cmd_queues.keys()):
            self.queue_command(ea_id, command)

    def is_connected(self, ea_id: str) -> bool:
        last = self._last_seen.get(ea_id, 0)
        return time.time() - last < EA_TIMEOUT_SECS

    def connected_eas(self) -> List[str]:
        return [
            ea for ea, t in self._last_seen.items()
            if time.time() - t < EA_TIMEOUT_SECS
        ]

    def set_hub_state(self, state: dict):
        self._hub_state = state

    def set_cipolla_state(self, state: dict):
        self._cipolla_state = state

    def get_last_tick(self, ea_id: str) -> dict:
        return self._last_ticks.get(ea_id, {})

    # ── Server lifecycle ──────────────────────────────────────────────

    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        config = uvicorn.Config(
            self.app, host=host, port=port,
            log_level="warning", loop="asyncio"
        )
        server = uvicorn.Server(config)
        log.info(f"HTTP polling hub starting on {host}:{port}")
        await server.serve()

    # ── Helpers ───────────────────────────────────────────────────────

    def _ensure_queue(self, ea_id: str):
        if ea_id not in self._cmd_queues:
            self._cmd_queues[ea_id]  = deque(maxlen=COMMAND_QUEUE_MAX)
            self._last_seen[ea_id]   = time.time()
            log.info(f"Registered EA: {ea_id}")


_NGROK_PLACEHOLDERS = {"YOUR_NGROK_AUTH_TOKEN_HERE", "PASTE_YOUR_NGROK_TOKEN_HERE", ""}


def start_ngrok(port: int, auth_token: Optional[str] = None) -> str:
    """Start ngrok tunnel and return the public HTTPS URL."""
    import os
    # Allow env var override so the token is never stored in the config file
    token = os.environ.get("NGROK_AUTH_TOKEN") or auth_token or ""
    if token in _NGROK_PLACEHOLDERS:
        log.warning(
            "ngrok auth token not set — hub running on localhost only.\n"
            "  To expose publicly:\n"
            "    1. Get your token from https://dashboard.ngrok.com/get-started/your-authtoken\n"
            "    2. Set it in imm_config.yaml  →  ngrok.auth_token: \"<your_token>\"\n"
            "       OR export NGROK_AUTH_TOKEN=<your_token>  before running deploy.py"
        )
        return ""
    try:
        from pyngrok import ngrok, conf
        if token:
            conf.get_default().auth_token = token
        tunnel   = ngrok.connect(port, "http")
        url      = tunnel.public_url
        https    = url.replace("http://", "https://")
        log.info(f"ngrok tunnel: {https}")
        log.info(f"EA poll URL:  {https}/ea/poll/{{ea_id}}")
        log.info(f"EA tick URL:  {https}/ea/tick/{{ea_id}}")
        log.info(f"Dashboard:    {https}/status")
        print(f"\n{'='*60}")
        print(f"  ngrok URL:   {https}")
        print(f"  EA POLL:     {https}/ea/poll/remora")
        print(f"  EA TICK:     {https}/ea/tick/remora")
        print(f"  In MT5: Tools→Options→Expert Advisors")
        print(f"  Add URL: {https}")
        print(f"{'='*60}\n")
        return https
    except ImportError:
        log.warning("pyngrok not installed")
        return ""
    except Exception as e:
        log.warning(f"ngrok failed: {e}")
        return ""

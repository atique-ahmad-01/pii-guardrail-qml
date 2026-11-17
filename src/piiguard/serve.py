"""A tiny local HTTP service around ``PiiGuardrail`` (standard library only).

    POST /mask   {"text": "...", "mode": "document"}  ->  {"masked": "...", "spans": [...], ...}
    GET  /health ->  {"status": "ok"}

Binds to 127.0.0.1 by default and has no authentication: the text you send is personal data, so do
not expose this port to a network without putting authentication and TLS in front of it.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from piiguard.inference import MODES, PiiGuardrail

MAX_BODY = 1_000_000


def make_server(
    guard: PiiGuardrail, host: str = "127.0.0.1", port: int = 8000
) -> ThreadingHTTPServer:
    """Build (but do not start) the server; ``port=0`` picks a free port."""
    lock = threading.Lock()  # one model, one request at a time

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok", "gate_loaded": guard.gate is not None})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/mask":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > MAX_BODY:
                return self._send(400, {"error": "body must be 1 byte to 1 MB"})
            try:
                req = json.loads(self.rfile.read(n))
                text, mode = req["text"], req.get("mode", "document")
                if not isinstance(text, str) or mode not in MODES:
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                return self._send(400, {"error": f'send {{"text": str, "mode": one of {MODES}}}'})
            try:
                with lock:
                    r = guard.mask(text, mode)
            except ValueError as exc:
                return self._send(400, {"error": str(exc)})
            self._send(
                200,
                {"masked": r.masked, "spans": r.spans, "mode": r.mode, "masker_ran": r.masker_ran},
            )

        def log_message(self, *args: Any) -> None:  # keep personal data out of server logs
            pass

    return ThreadingHTTPServer((host, port), Handler)

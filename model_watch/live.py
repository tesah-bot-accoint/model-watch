"""Watch replies form live in the full replay viewer.

LiveServer runs a small web server (Python standard library only) next to a
loaded ModelWatcher. It serves the replay viewer in live mode: type a
message, and the reply fills in word by word while you click around.

The page asks for new words with long polling (each request waits until
there is something new, up to about 15 seconds). That works through
proxies that buffer streaming responses, such as Colab's port forwarding.

    live = LiveServer(watcher).start(port=8765)
    print(live.url)        # open it in a browser; live.stop() when done

API (all JSON, relative to the page):
    GET  api/state?run=&since=&v=   new words since `since` in run `run`, waiting while version <= v
    POST api/ask    {"message": str}   start a reply (409 while one is being written)
    POST api/stop                       end the current reply early
    POST api/reset                      start a new conversation
    GET  api/trace.json, api/replay.html   the last finished reply
POSTs must send Content-Type: application/json, so other websites can't
start replies from your browser. It only observes, like the rest of Model Watch.
"""
from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .core import feature_key
from .export import TEMPLATE, trace_to_html

LIVE_PLACEHOLDER = "/*__LIVE__*/null"


class LiveSession:
    """One conversation, its current reply as it grows, and the thread writing it."""

    def __init__(self, watcher, max_new_tokens: int = 200, labels: bool = True, label_lookups: int = 400,
                 delay: float = 0.0, log=print):
        self.watcher = watcher
        self.max_new_tokens = max_new_tokens
        self.labels = labels
        self.label_lookups = label_lookups
        self.delay = delay
        self.log = log
        self.cond = threading.Condition()
        self.messages: list[dict] = []
        self.run = 0  # 0 = nothing asked yet; each reply gets the next number, even after a reset
        self._runs = 0
        self.version = 0  # bumps on every change; pollers wait for it to move
        self.header: dict | None = None
        self.steps: list[dict] = []
        self.final: dict | None = None
        self.final_version = 0
        self.busy = False
        self.status = "Ask a question to start."
        self.error: str | None = None
        self._stop = False

    # ---- changes -------------------------------------------------------------

    def _changed(self, **fields) -> None:
        with self.cond:
            for name, value in fields.items():
                setattr(self, name, value)
            self.version += 1
            self.cond.notify_all()

    def ask(self, message: str) -> bool:
        """Start a reply in the background. False if one is already being written."""
        message = (message or "").strip()
        if not message:
            raise ValueError("Type a message first.")
        with self.cond:
            if self.busy:
                return False
            self.busy, self._stop = True, False
            prompt = (self.messages + [{"role": "user", "content": message}]) if self.watcher.chat else message
        threading.Thread(target=self._write, args=(prompt,), daemon=True, name="model-watch-live").start()
        return True

    def stop(self) -> None:
        self._stop = True

    def reset(self) -> bool:
        with self.cond:
            if self.busy:
                return False
        self._changed(messages=[], run=0, header=None, steps=[], final=None, error=None,
                      status="New conversation. Ask a question to start.")
        return True

    def _write(self, prompt) -> None:
        try:
            header = self.watcher.trace_header(prompt)
            self._runs += 1
            self._changed(run=self._runs, header=header, steps=[], final=None, error=None, status="Writing…")

            def on_step(record, steps):
                with self.cond:
                    self.steps.append(record)
                    self.status = f"Writing… {len(steps)} words"
                    self.version += 1
                    self.cond.notify_all()
                if self.delay:
                    time.sleep(self.delay)

            trace = self.watcher.trace(prompt, self.max_new_tokens, on_step=on_step, should_stop=lambda: self._stop)
            if self.labels and self.watcher.labelers:
                self._changed(status="Looking up concept labels…")
                self.watcher.add_labels(trace, max_lookups=self.label_lookups, log=self.log)
            if self.watcher.chat:
                messages = list(prompt) + [{"role": "assistant", "content": trace["reply"]}]
            else:
                messages = []
            ended = "Stopped early" if self._stop else "Done"
            with self.cond:
                self.messages = messages
                self.final = trace
                self.busy = False
                self.status = f"{ended}. {len(trace['steps'])} words."
                self.version += 1
                self.final_version = self.version
                self.cond.notify_all()
        except Exception as err:  # keep the server alive and tell the page what went wrong
            self._changed(busy=False, error=f"{type(err).__name__}: {err}", status="Something went wrong.")
            self.log(f"[live] {type(err).__name__}: {err}")

    # ---- reading ---------------------------------------------------------------

    def state(self, run: int = -1, since: int = 0, v: int = -1, timeout: float = 15.0) -> dict:
        """What changed since the caller's last look. Waits up to `timeout` while nothing has."""
        with self.cond:
            self.cond.wait_for(lambda: self.version > v, timeout=timeout)
            if run != self.run:
                since = 0
            since = max(0, min(since, len(self.steps)))
            new = self.steps[since:]
            info = {}
            for step in new:
                for layer, rows in step["features"].items():
                    for f in rows:
                        key = feature_key(int(layer), f["index"])
                        if key in self.watcher.feature_info:
                            info[key] = self.watcher.feature_info[key]
            return {
                "v": self.version,
                "run": self.run,
                "busy": self.busy,
                "status": self.status,
                "error": self.error,
                "header": self.header if since == 0 else None,
                "since": since,
                "steps": new,
                "feature_info": info,
                "final": self.final if (self.final is not None and v < self.final_version) else None,
            }


class _Handler(BaseHTTPRequestHandler):
    session: LiveSession  # set on the subclass made by LiveServer
    page: bytes

    def log_message(self, *_):  # keep the notebook quiet
        pass

    def _send(self, status: int, body: bytes, kind: str, extra: dict | None = None) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (extra or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True  # the page was closed or reloaded while it waited; nothing to do

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path.rstrip("/").rsplit("/api/", 1)
        if len(path) == 1:  # the page itself, at whatever prefix a proxy adds
            if url.path.endswith(("/", "/index.html")) or url.path == "":
                return self._send(200, self.page, "text/html; charset=utf-8")
            return self._json({"error": "not found"}, 404)
        route, q = path[1], parse_qs(url.query)
        num = lambda name, default: int(q.get(name, [default])[0] or default)  # noqa: E731
        if route == "state":
            return self._json(self.session.state(num("run", -1), num("since", 0), num("v", -1)))
        final = self.session.final
        if route in ("trace.json", "replay.html"):
            if final is None:
                return self._json({"error": "No finished reply yet."}, 404)
            name = f"model-watch-{self.session.run}"
            if route == "trace.json":
                body = json.dumps(final, ensure_ascii=False, indent=1).encode("utf-8")
                return self._send(200, body, "application/json; charset=utf-8",
                                  {"Content-Disposition": f'attachment; filename="{name}.json"'})
            return self._send(200, trace_to_html(final).encode("utf-8"), "text/html; charset=utf-8",
                              {"Content-Disposition": f'attachment; filename="{name}.html"'})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            return self._json({"error": "Send JSON."}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        route = urlparse(self.path).path.rsplit("/api/", 1)[-1]
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
        except ValueError:
            return self._json({"error": "Send JSON."}, 400)
        if route == "ask":
            try:
                started = self.session.ask(str(body.get("message", "")))
            except ValueError as err:
                return self._json({"error": str(err)}, 400)
            return self._json({"ok": started} if started else {"error": "Still writing the last reply."},
                              200 if started else 409)
        if route == "stop":
            self.session.stop()
            return self._json({"ok": True})
        if route == "reset":
            ok = self.session.reset()
            return self._json({"ok": ok} if ok else {"error": "Wait for the reply to finish, or stop it."},
                              200 if ok else 409)
        return self._json({"error": "not found"}, 404)


def live_page() -> str:
    """The viewer in live mode: it talks to api/ next to itself."""
    page = TEMPLATE.read_text(encoding="utf-8")
    if LIVE_PLACEHOLDER not in page:
        raise RuntimeError("viewer.html is missing the live placeholder")
    return page.replace(LIVE_PLACEHOLDER, json.dumps({"api": "api/"}), 1)


class LiveServer:
    """Serve the live viewer for one watcher. start() returns at once; the server runs in a thread."""

    def __init__(self, watcher, max_new_tokens: int = 200, labels: bool = True, label_lookups: int = 400,
                 delay: float = 0.0, log=print):
        self.session = LiveSession(watcher, max_new_tokens, labels, label_lookups, delay, log)
        self.log = log
        self.httpd: ThreadingHTTPServer | None = None

    def start(self, port: int = 8765, host: str = "127.0.0.1") -> "LiveServer":
        handler = type("LiveHandler", (_Handler,), {"session": self.session, "page": live_page().encode("utf-8")})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True, name="model-watch-server").start()
        return self

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    @property
    def url(self) -> str:
        host = self.httpd.server_address[0]
        return f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{self.port}/"

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


__all__ = ["LiveServer", "LiveSession", "live_page"]

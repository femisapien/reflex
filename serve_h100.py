"""Supervise the pinned Reflex and SGLang servers in one CentML GPU container."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx

MODEL = "Qwen/Qwen3.8-27B-FP8"
REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
SOURCE = "d08a8c6cd63a81164c27f4c0f1f10b991eb2c32d+admission:0c5074951d67+startup:970d17dce298"
STARTUP_LOG = Path("/tmp/reflex-app/startup.log")


class StartupState:
    def __init__(self):
        STARTUP_LOG.parent.mkdir(parents=True, exist_ok=True)
        STARTUP_LOG.touch(mode=0o600)
        self.lock = threading.Lock()
        self.started_at = time.monotonic()
        self.phase = "booting"
        self.error = None
        self.engine_pid = None
        self.front_pid = None
        self.set_phase("booting")

    def set_phase(self, phase, *, error=None, engine_pid=None, front_pid=None):
        with self.lock:
            self.phase = phase
            self.error = error
            if engine_pid is not None:
                self.engine_pid = engine_pid
            if front_pid is not None:
                self.front_pid = front_pid
            event = {"at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "phase": phase}
            if error is not None:
                event["error"] = error
        with STARTUP_LOG.open("a", encoding="utf-8") as log:
            log.write("REFLEX_STARTUP " + json.dumps(event, separators=(",", ":")) + "\n")
        print("REFLEX_STARTUP " + json.dumps(event, separators=(",", ":")), flush=True)

    def snapshot(self):
        with self.lock:
            return {
                "phase": self.phase,
                "elapsed_seconds": round(time.monotonic() - self.started_at, 1),
                "error": self.error,
                "engine_pid": self.engine_pid,
                "front_pid": self.front_pid,
                "model": MODEL,
                "revision": REVISION,
            }

    @staticmethod
    def log_tail(max_bytes=65536):
        if not STARTUP_LOG.is_file():
            return b""
        with STARTUP_LOG.open("rb") as log:
            log.seek(max(0, STARTUP_LOG.stat().st_size - max_bytes))
            return log.read(max_bytes)


class BootstrapServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, state, port=8008):
        super().__init__(("0.0.0.0", port), BootstrapHandler)
        self.state = state
        self.thread = threading.Thread(target=self.serve_forever, name="startup-status", daemon=True)
        self.active = False

    def start(self):
        self.thread.start()
        self.active = True

    def stop(self):
        if self.active:
            self.shutdown()
            self.server_close()
            self.thread.join(timeout=5)
            self.active = False


class BootstrapHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *args):
        return

    def _send(self, status, body, content_type):
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urlsplit(self.path).path
        state = self.server.state
        if path == "/_control/log":
            self._send(200, state.log_tail(), "text/plain; charset=utf-8")
            return
        snapshot = state.snapshot()
        body = json.dumps(snapshot, separators=(",", ":")).encode("utf-8")
        if path in ("/healthz", "/status"):
            # CentML must route to this private status service before the model
            # is ready. The benchmark client waits for the direct Reflex server.
            self._send(200, body, "application/json")
        else:
            self._send(503, body, "application/json")


def commands(snapshot, *, permutations=2, mamba_slots=160, ssm_dtype=None):
    # Historical harnesses import this helper; preserve their original defaults.
    engine = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        snapshot,
        "--served-model-name",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
        "--dtype",
        "bfloat16",
        "--mem-fraction-static",
        "0.85",
        "--context-length",
        "65536",
        "--max-running-requests",
        "32",
        "--max-total-tokens",
        "65536",
        "--max-mamba-cache-size",
        str(mamba_slots),
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--cuda-graph-max-bs-decode",
        "32",
        "--chunked-prefill-size",
        "8192",
        "--attention-backend",
        "fa3",
        "--enable-metrics",
    ]
    front = [
        sys.executable,
        "-m",
        "reflex.server",
        "--backend",
        "sglang",
        "--sglang-url",
        "http://127.0.0.1:30000",
        "--model",
        snapshot,
        "--permutations",
        str(permutations),
        "--served-name",
        "reflex-27b",
        "--host",
        "0.0.0.0",
        "--port",
        "8008",
    ]
    if ssm_dtype is not None:
        engine += ["--mamba-ssm-dtype", ssm_dtype]
    return engine, front


def deployment_commands(snapshot):
    return commands(snapshot, permutations=2, mamba_slots=320, ssm_dtype="bfloat16")


def wait_ready(url, child, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"Server exited during startup: {child.returncode}")
        try:
            if httpx.get(url, timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise TimeoutError(f"Server was not ready within {timeout}s")


def supervise(engine_command, front_command, state, bootstrap):
    children = []
    try:
        state.set_phase("starting_sglang")
        with STARTUP_LOG.open("ab", buffering=0) as log:
            engine = subprocess.Popen(
                engine_command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        children.append(engine)
        state.set_phase("waiting_for_sglang", engine_pid=engine.pid)
        wait_ready("http://127.0.0.1:30000/health", engine, 1200)
        state.set_phase("sglang_ready")
        bootstrap.stop()
        state.set_phase("starting_reflex")
        with STARTUP_LOG.open("ab", buffering=0) as log:
            front = subprocess.Popen(
                front_command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        children.append(front)
        state.set_phase("waiting_for_reflex", front_pid=front.pid)
        wait_ready("http://127.0.0.1:8008/healthz", front, 120)
        state.set_phase("serving")
        print("REFLEX_SERVICE_READY", flush=True)
        while True:
            for child in children:
                if child.poll() is not None:
                    raise RuntimeError(f"Serving process exited: {child.returncode}")
            time.sleep(2)
    finally:
        # Both process groups include any SGLang workers. Do not leave the API
        # healthy while its inference process has exited.
        for child in reversed(children):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for child in children:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()


def main():
    from huggingface_hub import snapshot_download

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    os.umask(0o077)
    os.environ["PYTHONPATH"] = str(Path(__file__).parent / "src")
    print(json.dumps({"source": SOURCE, "model": MODEL, "revision": REVISION}), flush=True)
    state = StartupState()
    bootstrap = BootstrapServer(state)
    bootstrap.start()
    try:
        state.set_phase("downloading_model")
        with STARTUP_LOG.open("a", encoding="utf-8") as log, contextlib.redirect_stderr(log):
            snapshot = snapshot_download(
                MODEL,
                revision=REVISION,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
            )
        state.set_phase("model_downloaded")
        supervise(*deployment_commands(snapshot), state, bootstrap)
    except Exception as exc:
        # Keep the authenticated diagnostics reachable until the bounded
        # client-side timeout or cleanup watchdog removes this temporary pod.
        message = type(exc).__name__
        if type(exc) in (RuntimeError, TimeoutError):
            message += ": " + str(exc)[:200]
        state.set_phase("failed", error=message)
        if not bootstrap.active:
            bootstrap = BootstrapServer(state)
            bootstrap.start()
        while True:
            time.sleep(60)


if __name__ == "__main__":
    main()

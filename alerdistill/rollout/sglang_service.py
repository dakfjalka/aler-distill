from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TextIO

import httpx


def find_free_port(host: str = "127.0.0.1") -> int:
    """Bind to an ephemeral port and return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


@dataclass
class SGLangServiceConfig:
    host: str = "127.0.0.1"
    port: int = 0
    work_dir: str = "."
    startup_timeout_s: int = 240
    shutdown_timeout_s: int = 30
    extra_args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)


class SGLangService:
    """Manage a single persistent SGLang OpenAI-compatible server.

    We keep the process alive for the whole training run and hot-update its
    weights via the `/update_weights_from_disk` endpoint.

    Note: SGLang provides an OpenAI compatible API under `/v1`.
    """

    def __init__(self, cfg: SGLangServiceConfig, *, managed: bool = True):
        self.cfg = cfg
        # managed=True: this object spawns and owns the server process.
        # managed=False: attach to an already-running server (no spawning/termination).
        self.managed = bool(managed)
        self._proc: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None

        self._stdout_f: Optional[TextIO] = None
        self._stderr_f: Optional[TextIO] = None

    @property
    def work_dir(self) -> str:
        return str(Path(self.cfg.work_dir).resolve())

    @property
    def host(self) -> str:
        return self.cfg.host

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("SGLangService not started yet")
        return self._port

    @property
    def root_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def openai_base_url(self) -> str:
        return f"{self.root_url}/v1"

    def start(self, model_path: str, cuda_visible_devices: Optional[str] = None) -> None:
        """Start SGLang server once and keep it alive.

        Logs are redirected to:
          - {work_dir}/server_stdout.log
          - {work_dir}/server_stderr.log
        """
        # External service: only wait readiness.
        if not self.managed:
            print("[SGLangService] attaching to external SGLang service...", flush=True)
            # In external mode, cfg.port must be provided.
            port = int(self.cfg.port)
            if port <= 0:
                raise ValueError("SGLangService(managed=False) requires cfg.port > 0")
            self._port = port
            print(f"[SGLangService] waiting for SGLang service to become ready at {self.root_url}...", flush=True)
            self._wait_ready()
            print(f"[SGLangService] attached to external SGLang service at {self.root_url}", flush=True)
            return

        if self._proc is not None:
            return

        # Prepare work dir
        os.makedirs(self.work_dir, exist_ok=True)

        # Decide port
        port = int(self.cfg.port)
        if port == 0:
            port = find_free_port(self.cfg.host)
        self._port = port

        # Prepare env
        env = os.environ.copy()
        env.update({k: str(v) for k, v in (self.cfg.env or {}).items()})
        if cuda_visible_devices is not None:
            # empty string means CPU-only / no visible GPU
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        # Use unbuffered so logs appear in real-time.
        env.setdefault("PYTHONUNBUFFERED", "1")

        # Redirect logs to files in work_dir
        stdout_path = os.path.join(self.work_dir, "server_stdout.log")
        stderr_path = os.path.join(self.work_dir, "server_stderr.log")
        self._stdout_f = open(stdout_path, "w", buffering=1)
        self._stderr_f = open(stderr_path, "w", buffering=1)

        cmd = [
            "python",
            "-u",
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(model_path),
            "--host",
            str(self.cfg.host),
            "--port",
            str(port),
        ]
        if self.cfg.extra_args:
            cmd.extend([str(x) for x in self.cfg.extra_args])

        try:
            print(f"[SGLangService] starting SGLang server with command: {' '.join(cmd)}")
            self._proc = subprocess.Popen(
                cmd,
                cwd=self.work_dir,
                env=env,
                stdout=self._stdout_f,
                stderr=self._stderr_f,
                text=True,
                start_new_session=True,
            )
        except Exception:
            # If Popen fails, close file handles to avoid leaking
            self._close_log_files()
            self._proc = None
            self._port = None
            raise

        self._wait_ready()

    def _wait_ready(self) -> None:
        assert self._port is not None
        deadline = time.time() + float(self.cfg.startup_timeout_s)
        last_err: Optional[Exception] = None

        while time.time() < deadline:
            # If process died early, surface a helpful message
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"SGLang server process exited early with code {self._proc.returncode}. "
                    f"Check logs under {self.work_dir}/server_stdout.log and server_stderr.log"
                )

            try:
                with httpx.Client(timeout=5.0) as client:
                    r = client.get(f"{self.openai_base_url}/models")
                    if r.status_code == 200:
                        return
                    last_err = RuntimeError(f"/models returned {r.status_code}: {r.text[:200]}")
            except Exception as e:
                last_err = e

            time.sleep(0.5)

        raise RuntimeError(f"SGLang server did not become ready in time: {last_err}")

    def hot_update_from_disk(self, model_path: str, endpoint: str = "update_weights_from_disk") -> None:
        """Hot-update model weights.

        For SGLang, the endpoint is typically:
            POST {root_url}/update_weights_from_disk
        """
        # In managed mode, _proc must exist. In external mode, we don't own a proc.
        if self._port is None:
            raise RuntimeError("SGLangService not started")

        model_path = str(Path(model_path).resolve())
        url = f"{self.root_url}/{endpoint.lstrip('/')}"
        with httpx.Client(timeout=300.0) as client:
            r = client.post(url, json={"model_path": model_path})
            r.raise_for_status()

        # Wait until /models responds again.
        self._wait_ready()

    def flush_cache(self) -> None:
        if self._port is None:
            return
        try:
            with httpx.Client(timeout=60.0) as client:
                r = client.post(f"{self.root_url}/flush_cache")
                # Some versions may not expose this; ignore.
                if r.status_code >= 400:
                    return
        except Exception:
            return

    def _close_log_files(self) -> None:
        for f in (self._stdout_f, self._stderr_f):
            if f is None:
                continue
            try:
                f.flush()
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass
        self._stdout_f = None
        self._stderr_f = None

    def terminate(self) -> None:
        # External mode: we don't own the server process.
        if not self.managed:
            self._close_log_files()
            self._port = None
            return

        if self._proc is None:
            self._close_log_files()
            self._port = None
            return

        try:
            # Prefer killing the whole process group
            try:
                os.killpg(self._proc.pid, signal.SIGTERM)
            except Exception:
                self._proc.terminate()

            try:
                self._proc.wait(timeout=float(self.cfg.shutdown_timeout_s))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self._proc.pid, signal.SIGKILL)
                except Exception:
                    self._proc.kill()
        finally:
            self._proc = None
            self._port = None
            self._close_log_files()

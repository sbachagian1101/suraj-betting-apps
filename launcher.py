"""Start Horse Race Predictor as a desktop app: no console, its own window.

Picks a free port (8501 upward), starts the Streamlit server with no console of
its own, waits until it answers, and opens it in a real window (pywebview; falls
back to the default browser).  A copy that is already running is reused and
brought to the front instead of starting a second server.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT_FILE = HERE / ".running_port"
TITLE = "Horse Race Predictor"
FIRST_PORT = 8501
PORT_TRIES = 40
STARTUP_TIMEOUT = 120


def alive(port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def free_port(start: int = FIRST_PORT, tries: int = PORT_TRIES) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port between {start} and {start + tries}")


def already_running() -> int | None:
    try:
        port = int(PORT_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return None
    return port if alive(port) else None


def start_server(port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    cmd = [sys.executable, "-m", "streamlit", "run", str(HERE / "app.py"),
           "--server.port", str(port), "--server.headless", "true",
           "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false",
           "--global.developmentMode", "false"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(cmd, cwd=str(HERE), env=env, creationflags=flags,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_until_up(port: int, proc: subprocess.Popen, timeout: int = STARTUP_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        if alive(port):
            return True
        time.sleep(0.4)
    return False


def open_window(url: str) -> None:
    try:
        import webview
    except Exception:
        import webbrowser
        webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return
        return
    icon = HERE / "HorseRacePredictor.ico"
    webview.create_window(TITLE, url, width=1500, height=940, min_size=(1000, 680),
                          confirm_close=False)
    try:
        webview.start(icon=str(icon) if icon.exists() else None)
    except TypeError:               # older pywebview without the icon argument
        webview.start()


def focus_existing() -> bool:
    try:
        import ctypes
        u = ctypes.windll.user32
        hwnd = u.FindWindowW(None, TITLE)
        if not hwnd:
            return False
        if u.IsIconic(hwnd):
            u.ShowWindow(hwnd, 9)
        u.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False


def main() -> int:
    running = already_running()
    if running:
        if focus_existing():
            return 0
        open_window(f"http://127.0.0.1:{running}")
        return 0
    port = free_port()
    proc = start_server(port)
    if not wait_until_up(port, proc):
        proc.terminate()
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                None, "Horse Race Predictor could not start.\n\nRun 'run.bat' in this folder "
                      "to see the error in full.", TITLE, 0x10)
        except Exception:
            print("Horse Race Predictor could not start. Run run.bat to see why.")
        return 1
    PORT_FILE.write_text(str(port), encoding="utf-8")
    try:
        open_window(f"http://127.0.0.1:{port}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        PORT_FILE.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

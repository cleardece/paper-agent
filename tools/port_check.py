"""Health check helper: wait for TCP ports to be ready."""
import socket
import sys
import time


def wait_for_port(host: str, port: int, timeout: int = 60) -> bool:
    """Return True if port is reachable within timeout seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(1)
    return False


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2])
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    ok = wait_for_port(host, port, timeout)
    sys.exit(0 if ok else 1)

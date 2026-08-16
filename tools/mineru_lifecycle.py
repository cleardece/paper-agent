"""管理本机 MinerU Docker 容器的按需生命周期。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
import subprocess
import threading
import time


class MinerUContainerError(RuntimeError):
    """MinerU 容器无法启动、健康检查失败或无法停止。"""


class MinerUContainerManager:
    """为同一进程中的解析任务提供可重入的 MinerU 容器租约。"""

    def __init__(
        self,
        base_url: str,
        *,
        idle_shutdown_seconds: int = 0,
        start_timeout_seconds: int = 90,
        memory_limit: str = "",
        cpu_limit: str = "",
        command_runner: Callable[[list[str]], None] | None = None,
        health_check: Callable[[], bool] | None = None,
    ) -> None:
        if idle_shutdown_seconds < 0:
            raise ValueError("idle_shutdown_seconds 不能小于 0")

        self.base_url = base_url.rstrip("/")
        self.idle_shutdown_seconds = idle_shutdown_seconds
        self.start_timeout_seconds = start_timeout_seconds
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self._command_runner = command_runner or self._run_command
        self._health_check = health_check or self._is_healthy
        self._lock = threading.Lock()
        self._active_parses = 0
        self._stop_timer: threading.Timer | None = None

    @contextmanager
    def lease(self) -> Iterator[None]:
        """确保容器存活，且在最后一个解析结束后按策略释放它。"""
        self.acquire()
        try:
            yield
        finally:
            self.release()

    def acquire(self) -> None:
        with self._lock:
            if self._stop_timer is not None:
                self._stop_timer.cancel()
                self._stop_timer = None

            if self._active_parses == 0:
                self._ensure_running()
            self._active_parses += 1

    def release(self) -> None:
        with self._lock:
            if self._active_parses <= 0:
                raise RuntimeError("MinerU 容器租约释放次数超过获取次数")

            self._active_parses -= 1
            if self._active_parses != 0:
                return

            if self.idle_shutdown_seconds == 0:
                self._stop()
                return

            self._stop_timer = threading.Timer(
                self.idle_shutdown_seconds,
                self._stop_if_idle,
            )
            self._stop_timer.daemon = True
            self._stop_timer.start()

    def _ensure_running(self) -> None:
        try:
            self._command_runner(["docker", "compose", "up", "-d", "mineru-api"])
            self._wait_until_healthy()
            self._apply_resource_limits()
        except MinerUContainerError:
            raise
        except Exception as exc:
            raise MinerUContainerError(f"无法启动 MinerU 容器: {exc}") from exc

    def _wait_until_healthy(self) -> None:
        deadline = time.monotonic() + self.start_timeout_seconds
        while time.monotonic() < deadline:
            try:
                if self._health_check():
                    return
            except Exception:
                pass
            time.sleep(1)
        raise MinerUContainerError(
            f"MinerU 在 {self.start_timeout_seconds} 秒内未通过健康检查: {self.base_url}/health"
        )

    def _apply_resource_limits(self) -> None:
        if not self.memory_limit and not self.cpu_limit:
            return

        command = ["docker", "update"]
        if self.memory_limit:
            command.extend(["--memory", self.memory_limit])
        if self.cpu_limit:
            command.extend(["--cpus", self.cpu_limit])
        command.append("mineru-api")
        self._command_runner(command)

    def _stop_if_idle(self) -> None:
        with self._lock:
            self._stop_timer = None
            if self._active_parses == 0:
                self._stop()

    def _stop(self) -> None:
        try:
            self._command_runner(["docker", "compose", "stop", "mineru-api"])
        except Exception as exc:
            raise MinerUContainerError(f"无法停止 MinerU 容器: {exc}") from exc

    @staticmethod
    def _run_command(command: list[str]) -> None:
        project_root = Path(__file__).resolve().parents[1]
        subprocess.run(
            command,
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def _is_healthy(self) -> bool:
        import httpx

        response = httpx.get(f"{self.base_url}/health", timeout=5)
        return response.is_success

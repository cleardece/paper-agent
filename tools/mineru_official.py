"""MinerU 官方精准解析 API 客户端。"""

from __future__ import annotations

import io
import logging
import re
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


logger = logging.getLogger("paper-agent")


class OfficialMinerUError(RuntimeError):
    """官方 MinerU 请求或结果不符合可用解析合同。"""


class OfficialMinerUClient:
    """上传单个本地 PDF，轮询官方任务并返回 Markdown。"""

    ACTIVE_STATES = {"waiting-file", "pending", "running", "converting"}

    def __init__(
        self,
        token: str,
        base_url: str = "https://mineru.net",
        poll_seconds: float = 5,
        timeout_seconds: float = 900,
        transport: Any = None,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        token = str(token or "").strip()
        if not token:
            raise ValueError(
                "未配置 MINERU_OFFICIAL_TOKEN，请在项目根目录 .env 中填写 MinerU 官方 API Token"
            )
        if float(poll_seconds) < 0:
            raise ValueError("MINERU_OFFICIAL_POLL_SECONDS 不能小于 0")
        if float(timeout_seconds) <= 0:
            raise ValueError("MINERU_OFFICIAL_TIMEOUT_SECONDS 必须大于 0")
        self.token = token
        self.base_url = str(base_url or "https://mineru.net").rstrip("/")
        self.model = "vlm"
        self.poll_seconds = float(poll_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.transport = transport
        self.sleep = sleep
        self.clock = clock

    def _redact(self, value: Any) -> str:
        text = str(value or "")
        if self.token:
            text = text.replace(self.token, "[已隐藏Token]")
        return re.sub(r"https?://\S+", "[已隐藏URL]", text)

    @staticmethod
    def _status_code(response: Any) -> int:
        return int(getattr(response, "status_code", 0) or 0)

    def _api_data(self, response: Any, stage: str) -> tuple[dict[str, Any], str]:
        status_code = self._status_code(response)
        if status_code < 200 or status_code >= 300:
            raise OfficialMinerUError(f"{stage}：HTTP {status_code}")
        try:
            payload = response.json()
        except Exception as exc:
            raise OfficialMinerUError(f"{stage}：返回内容不是 JSON") from exc
        if not isinstance(payload, dict):
            raise OfficialMinerUError(f"{stage}：返回结构无效")
        trace_id = self._redact(payload.get("trace_id", ""))
        code = payload.get("code")
        if code not in {0, "0"}:
            message = self._redact(payload.get("msg", "未知错误"))
            suffix = f"，trace_id={trace_id}" if trace_id else ""
            raise OfficialMinerUError(
                f"{stage}：业务码 {code}，{message}{suffix}"
            )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise OfficialMinerUError(f"{stage}：缺少 data")
        return data, trace_id

    def _new_transport(self):
        import httpx

        return httpx.Client(
            timeout=httpx.Timeout(self.timeout_seconds),
            follow_redirects=True,
        )

    def _call(self, stage: str, method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except OfficialMinerUError:
            raise
        except Exception as exc:
            raise OfficialMinerUError(
                f"{stage}：网络请求异常（{exc.__class__.__name__}）"
            ) from exc

    @staticmethod
    def _markdown_from_zip(content: bytes) -> str:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                candidates = sorted(
                    (
                        name for name in archive.namelist()
                        if PurePosixPath(name).name == "full.md"
                    ),
                    key=len,
                )
                if not candidates:
                    raise OfficialMinerUError("官方结果 ZIP 中缺少 full.md")
                markdown = archive.read(candidates[0]).decode("utf-8-sig").strip()
        except OfficialMinerUError:
            raise
        except (zipfile.BadZipFile, KeyError, UnicodeDecodeError) as exc:
            raise OfficialMinerUError("官方结果 ZIP 无效或无法读取") from exc
        if not markdown:
            raise OfficialMinerUError("官方结果 full.md 为空")
        return markdown

    def parse(self, pdf_path: str) -> dict[str, Any]:
        path = Path(pdf_path)
        if not path.is_file():
            raise OfficialMinerUError("待解析 PDF 不存在")

        started_at = self.clock()
        deadline = started_at + self.timeout_seconds
        transport = self.transport or self._new_transport()
        owns_transport = self.transport is None
        auth_headers = {"Authorization": f"Bearer {self.token}"}
        try:
            apply_response = self._call(
                "申请上传地址失败",
                transport.post,
                f"{self.base_url}/api/v4/file-urls/batch",
                headers=auth_headers,
                json={
                    "files": [{"name": path.name}],
                    "model_version": self.model,
                    "enable_formula": True,
                    "enable_table": True,
                },
            )
            apply_data, _ = self._api_data(apply_response, "申请上传地址失败")
            batch_id = str(apply_data.get("batch_id", "")).strip()
            file_urls = apply_data.get("file_urls")
            if not batch_id or not isinstance(file_urls, list) or len(file_urls) != 1:
                raise OfficialMinerUError("申请上传地址失败：缺少唯一上传地址或 batch_id")
            upload_url = str(file_urls[0])
            if not upload_url:
                raise OfficialMinerUError("申请上传地址失败：上传地址为空")

            with path.open("rb") as stream:
                upload_response = self._call(
                    "上传 PDF 失败", transport.put, upload_url,
                    content=stream, headers={},
                )
            upload_status = self._status_code(upload_response)
            if upload_status < 200 or upload_status >= 300:
                raise OfficialMinerUError(f"上传 PDF 失败：HTTP {upload_status}")
            upload_finished_at = self.clock()
            logger.info(
                "[MinerUOfficial] %s 上传完成，耗时 %.2fs",
                path.name, upload_finished_at - started_at,
            )

            full_zip_url = ""
            while self.clock() < deadline:
                poll_response = self._call(
                    "查询解析任务失败",
                    transport.get,
                    f"{self.base_url}/api/v4/extract-results/batch/{batch_id}",
                    headers=auth_headers,
                )
                poll_data, trace_id = self._api_data(
                    poll_response, "查询解析任务失败"
                )
                results = poll_data.get("extract_result")
                if not isinstance(results, list) or not results:
                    raise OfficialMinerUError("查询解析任务失败：缺少 extract_result")
                result = next(
                    (
                        item for item in results
                        if isinstance(item, dict) and item.get("file_name") == path.name
                    ),
                    results[0],
                )
                if not isinstance(result, dict):
                    raise OfficialMinerUError("查询解析任务失败：任务结果结构无效")
                state = str(result.get("state", "")).strip().lower()
                if state == "done":
                    full_zip_url = str(result.get("full_zip_url", "")).strip()
                    if not full_zip_url:
                        raise OfficialMinerUError("官方解析完成但缺少结果 ZIP")
                    break
                if state == "failed":
                    error = self._redact(result.get("err_msg", "未知错误"))
                    suffix = f"，trace_id={trace_id}" if trace_id else ""
                    raise OfficialMinerUError(f"官方解析失败：{error}{suffix}")
                if state not in self.ACTIVE_STATES:
                    raise OfficialMinerUError(f"官方解析返回未知状态：{state or '空'}")
                if self.poll_seconds:
                    self.sleep(min(self.poll_seconds, max(0, deadline - self.clock())))
            if not full_zip_url:
                raise OfficialMinerUError(
                    f"官方解析超过总超时 {self.timeout_seconds:g} 秒"
                )

            remote_finished_at = self.clock()
            download_response = self._call(
                "下载解析结果失败", transport.get, full_zip_url, headers={},
            )
            download_status = self._status_code(download_response)
            if download_status < 200 or download_status >= 300:
                raise OfficialMinerUError(
                    f"下载解析结果失败：HTTP {download_status}"
                )
            markdown = self._markdown_from_zip(
                bytes(getattr(download_response, "content", b""))
            )
            finished_at = self.clock()
            metrics = {
                "upload_seconds": round(upload_finished_at - started_at, 3),
                "remote_wait_seconds": round(remote_finished_at - upload_finished_at, 3),
                "download_seconds": round(finished_at - remote_finished_at, 3),
                "total_seconds": round(finished_at - started_at, 3),
                "provider": "official",
                "model": self.model,
            }
            logger.info(
                "[MinerUOfficial] %s 解析完成：等待 %.2fs，下载 %.2fs，总耗时 %.2fs",
                path.name, metrics["remote_wait_seconds"],
                metrics["download_seconds"], metrics["total_seconds"],
            )
            return {"markdown": markdown, "metrics": metrics}
        finally:
            if owns_transport:
                transport.close()

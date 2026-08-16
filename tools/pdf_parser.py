"""
Paper Agent - PDF解析工具 v3
支持 MinerU（Markdown 输出）和 pdfplumber（fallback）
"""

import os
import re
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.mineru_lifecycle import MinerUContainerManager

logger = logging.getLogger("paper-agent")


class MinerUParseError(RuntimeError):
    """MinerU 解析失败且当前策略不允许静默降级。"""


class PDFParser:
    """PDF论文解析器 - 支持 MinerU 和 pdfplumber"""

    def __init__(
        self,
        mineru_url: str = None,
        mineru_backend: str = None,
        mineru_manager: "MinerUContainerManager | None" = None,
        require_accurate_parse: bool = True,
    ):
        """
        Args:
            mineru_url: MinerU API 地址，如 http://localhost:8888
                       如果为 None，使用 pdfplumber fallback
            mineru_backend: MinerU 后端；CPU 环境应使用 pipeline
            mineru_manager: 本机 MinerU 的按需启动/释放管理器
            require_accurate_parse: MinerU 失败时是否阻止低质量回退结果入库
        """
        self.mineru_url = mineru_url or os.getenv("MINERU_URL")
        self.mineru_backend = mineru_backend or os.getenv("MINERU_BACKEND", "pipeline")
        self.mineru_manager = mineru_manager
        self.require_accurate_parse = require_accurate_parse
        self._pdfplumber = None

    def parse(self, pdf_path: str) -> dict:
        """
        解析PDF文件
        返回: {
            "title": str,
            "text": str,
            "markdown": str,        # MinerU 输出的 Markdown
            "sections": [{"heading": str, "content": str, "page": int, "level": int}],
            "page_count": int,
            "source": "mineru" | "pdfplumber",
        }
        """
        if self.mineru_url:
            try:
                if self.mineru_manager:
                    with self.mineru_manager.lease():
                        return self._parse_with_mineru(pdf_path)
                return self._parse_with_mineru(pdf_path)
            except Exception as e:
                if self.require_accurate_parse:
                    raise MinerUParseError(f"MinerU 解析失败: {e}") from e
                logger.warning(f"[PDFParser] MinerU 解析失败，明确降级到 pdfplumber: {e}")

        return self._parse_with_pdfplumber(pdf_path)

    def _parse_with_mineru(self, pdf_path: str) -> dict:
        """使用 MinerU API 解析 PDF"""
        import httpx

        logger.info(
            f"[PDFParser] 使用 MinerU 解析: {pdf_path} "
            f"(backend={self.mineru_backend})"
        )

        with open(pdf_path, "rb") as f:
            files = {"files": (os.path.basename(pdf_path), f, "application/pdf")}
            response = httpx.post(
                f"{self.mineru_url}/file_parse",
                files=files,
                data={
                    "backend": self.mineru_backend,
                    "parse_method": "auto",
                    "return_md": "true",
                },
                timeout=300,
            )
            response.raise_for_status()

        data = response.json()
        # MinerU 返回格式：{"results": {"filename": {"md_content": "..."}}}
        markdown = ""
        results = data.get("results", {})
        for fname, result in results.items():
            markdown = result.get("md_content", "")
            if markdown:
                break

        # 从 Markdown 解析章节
        sections = self._markdown_to_sections(markdown)

        # 提取标题
        title = self._extract_title_from_markdown(markdown) or os.path.basename(pdf_path).replace(".pdf", "")

        return {
            "title": title,
            "text": markdown,
            "markdown": markdown,
            "sections": sections,
            "page_count": len(sections),
            "source": "mineru",
        }

    def _parse_with_pdfplumber(self, pdf_path: str) -> dict:
        """使用 pdfplumber 解析 PDF（fallback）"""
        logger.info(f"[PDFParser] 使用 pdfplumber 解析: {pdf_path}")

        pdf = self._get_pdfplumber().open(pdf_path)
        all_text = []
        sections = []

        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            all_text.append(text)

            current_section = {"heading": "", "content": "", "page": page_num + 1, "level": 0}
            for line in text.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue

                heading_info = self._is_heading(stripped)
                if heading_info:
                    if current_section["content"].strip():
                        sections.append(current_section)
                    current_section = {
                        "heading": heading_info["title"],
                        "content": "",
                        "page": page_num + 1,
                        "level": heading_info["level"],
                        "is_appendix": heading_info.get("is_appendix", False),
                    }
                else:
                    current_section["content"] += stripped + " "

            if current_section["content"].strip():
                sections.append(current_section)

        pdf.close()

        full_text = "\n".join(all_text)
        title = self._extract_title(full_text, pdf_path)

        return {
            "title": title,
            "text": full_text,
            "markdown": None,
            "sections": sections,
            "page_count": len(all_text),
            "source": "pdfplumber",
        }

    def _markdown_to_sections(self, markdown: str) -> list[dict]:
        """将 Markdown 转换为 sections 列表"""
        sections = []
        current_section = {"heading": "", "content": "", "page": 0, "level": 0}

        for line in markdown.split("\n"):
            # 识别 Markdown 标题
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                # 保存之前的 section
                if current_section["content"].strip():
                    sections.append(current_section)

                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                current_section = {
                    "heading": title,
                    "content": "",
                    "page": 0,
                    "level": level,
                    "is_appendix": "appendix" in title.lower(),
                }
            else:
                current_section["content"] += line + "\n"

        # 最后一个 section
        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def _extract_title_from_markdown(self, markdown: str) -> str:
        """从 Markdown 提取标题"""
        lines = markdown.strip().split("\n")
        for line in lines[:20]:
            # 匹配 # 标题
            match = re.match(r'^#\s+(.+)$', line)
            if match:
                return match.group(1).strip()
            # 跳过空行和元数据
            if line.strip() and not line.startswith("---"):
                return line.strip()[:100]
        return ""

    def chunk(
            self,
            sections: list[dict],
            max_chunk_size: int = 2000,
            overlap: int = 300,
            min_chunk_size: int = 200,
    ) -> list[dict]:
        """
        分块策略 v3：
        - MinerU 输出：直接按 Markdown 标题分块
        - pdfplumber 输出：使用 v2 策略
        """
        chunks = []
        chunk_index = 0

        for section in sections:
            content = section["content"].strip()
            if not content:
                continue

            is_appendix = section.get("is_appendix", False)
            cur_max = 2000 if is_appendix else max_chunk_size

            if len(content) <= cur_max:
                chunks.append(self._create_chunk(chunk_index, content, section))
                chunk_index += 1
            else:
                new_chunks, chunk_index = self._split_long_section(
                    content, section, cur_max, overlap, chunk_index
                )
                chunks.extend(new_chunks)

        # 后处理：合并过短的 chunk
        chunks = self._merge_short_chunks(chunks, min_chunk_size)

        # 重新编号
        for i, chunk in enumerate(chunks):
            chunk["chunk_index"] = i

        return chunks

    def _split_long_section(self, content, section, max_size, overlap, start_index):
        """切分长章节"""
        chunks = []
        chunk_index = start_index

        sentences = re.split(r'(?<=[.!?;:])\s+', content)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_size and current_chunk.strip():
                chunks.append(self._create_chunk(
                    chunk_index, current_chunk.strip(), section
                ))
                chunk_index += 1

                overlap_text = self._take_last_sentences(current_chunk, overlap)
                current_chunk = overlap_text + sentence
            else:
                current_chunk += sentence

        if current_chunk.strip():
            chunks.append(self._create_chunk(
                chunk_index, current_chunk.strip(), section
            ))
            chunk_index += 1

        return chunks, chunk_index

    def _create_chunk(self, index, content, section):
        """创建 chunk 字典"""
        return {
            "chunk_index": index,
            "content": content,
            "metadata": {
                "section": section.get("heading", ""),
                "page": section.get("page", 0),
                "level": section.get("level", 0),
                "is_appendix": section.get("is_appendix", False),
            },
        }

    def _take_last_sentences(self, text: str, max_chars: int) -> str:
        """从文本尾部取若干完整句子"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = ""
        for sent in reversed(sentences):
            if len(result) + len(sent) > max_chars:
                break
            result = sent + " " + result
        return result.strip()

    def _merge_short_chunks(self, chunks: list[dict], min_size: int) -> list[dict]:
        """合并过短的 chunk"""
        if not chunks:
            return chunks

        merged = [chunks[0]]
        for chunk in chunks[1:]:
            prev = merged[-1]
            if (len(prev["content"]) < min_size and
                    len(prev["content"]) + len(chunk["content"]) <= 2000):
                prev["content"] = prev["content"] + "\n" + chunk["content"]
                prev["metadata"]["merged"] = True
            else:
                merged.append(chunk)

        return merged

    def _get_pdfplumber(self):
        if self._pdfplumber is None:
            import pdfplumber
            self._pdfplumber = pdfplumber
        return self._pdfplumber

    def _is_heading(self, line: str) -> dict | None:
        """增强标题识别"""
        no_number_patterns = [
            (r'^(Abstract)\s*$', 1, False),
            (r'^(Introduction)\s*$', 1, False),
            (r'^(Conclusion|Conclusions)\s*$', 1, False),
            (r'^(References|Bibliography)\s*$', 1, False),
            (r'^(Acknowledgment|Acknowledgement|Acknowledgments|Acknowledgements)\s*$', 1, False),
        ]
        for pattern, level, is_appendix in no_number_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return {"title": line.strip(), "level": level, "is_appendix": is_appendix}

        appendix_match = re.match(r'^(Appendix)\s+([A-Z])(?:\.(\d+))?\s*(.*)', line, re.IGNORECASE)
        if appendix_match:
            sub = appendix_match.group(3)
            level = 2 if sub else 1
            return {"title": line.strip(), "level": level, "is_appendix": True}

        number_match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)$', line)
        if number_match:
            num = number_match.group(1)
            level = num.count('.') + 1
            return {"title": line.strip(), "level": level, "is_appendix": False}

        appendix_sub_match = re.match(r'^([A-Z]\.\d+(?:\.\d+)*)\s+(.+)', line)
        if appendix_sub_match:
            return {"title": line.strip(), "level": 2, "is_appendix": True}

        return None

    def _extract_title(self, text: str, pdf_path: str) -> str:
        """从文本开头提取标题"""
        lines = text.strip().split("\n")
        candidates = []
        for line in lines[:15]:
            stripped = line.strip()
            if stripped and len(stripped) > 5:
                if re.match(r'^(Vol\.|ISSN|DOI|http|www\.)', stripped, re.IGNORECASE):
                    continue
                candidates.append(stripped)
        return candidates[0] if candidates else os.path.basename(pdf_path).replace(".pdf", "")

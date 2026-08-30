"""MinerU 官方精准 API 解析与论文分块。"""

import re
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.mineru_official import OfficialMinerUClient

logger = logging.getLogger("paper-agent")


class MinerUParseError(RuntimeError):
    """MinerU 解析失败且当前策略不允许静默降级。"""


class PDFParser:
    """只使用 MinerU 官方精准 API 的 PDF 论文解析器。"""

    def __init__(self, official_client: "OfficialMinerUClient"):
        if official_client is None:
            raise ValueError("PDFParser 需要 MinerU 官方 API 客户端")
        self.official_client = official_client

    @property
    def provider_label(self) -> str:
        return f"MinerU 官方 {self.official_client.model.upper()}"

    def parse(self, pdf_path: str) -> dict:
        """
        解析PDF文件
        返回: {
            "title": str,
            "text": str,
            "markdown": str,        # MinerU 输出的 Markdown
            "sections": [{"heading": str, "content": str, "page": int, "level": int}],
            "page_count": int,
            "source": "mineru",
        }
        """
        try:
            return self._parse_with_official_mineru(pdf_path)
        except Exception as exc:
            raise MinerUParseError(f"MinerU 官方 API 解析失败: {exc}") from exc

    def _result_from_markdown(
        self, pdf_path: str, markdown: str, parse_source: str,
        parse_metrics: dict,
    ) -> dict:
        sections = self._markdown_to_sections(markdown)
        title = self._extract_title_from_markdown(markdown) or Path(pdf_path).stem
        return {
            "title": title,
            "text": markdown,
            "markdown": markdown,
            "sections": sections,
            "page_count": len(sections),
            "source": "mineru",
            "parse_source": parse_source,
            "parse_metrics": parse_metrics,
        }

    def _parse_with_official_mineru(self, pdf_path: str) -> dict:
        logger.info(
            "[PDFParser] 使用 MinerU 官方精准 API 解析: %s (model=%s)",
            pdf_path, self.official_client.model,
        )
        result = self.official_client.parse(pdf_path)
        return self._result_from_markdown(
            pdf_path,
            result["markdown"],
            f"mineru_official_{self.official_client.model}",
            dict(result.get("metrics") or {}),
        )

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
        分块策略 v3：按 MinerU Markdown 标题分块。
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

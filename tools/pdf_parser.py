"""
Paper Agent - PDF解析工具
解析论文PDF，提取文本内容并分块
"""

import re
from typing import Optional


class PDFParser:
    """PDF论文解析器"""

    def __init__(self):
        self._pdfplumber = None

    def _get_pdfplumber(self):
        if self._pdfplumber is None:
            import pdfplumber
            self._pdfplumber = pdfplumber
        return self._pdfplumber

    def parse(self, pdf_path: str) -> dict:
        """
        解析PDF文件
        返回: {
            "title": str,
            "text": str,
            "sections": [{"heading": str, "content": str, "page": int}],
            "page_count": int,
        }
        """
        pdf = self._get_pdfplumber().open(pdf_path)
        all_text = []
        sections = []

        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            all_text.append(text)

            # 按段落拆分，识别章节标题
            current_section = {"heading": "", "content": "", "page": page_num + 1}
            for line in text.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                if self._is_heading(stripped):
                    if current_section["content"]:
                        sections.append(current_section)
                    current_section = {"heading": stripped, "content": "", "page": page_num + 1}
                else:
                    current_section["content"] += stripped + " "

            if current_section["content"]:
                sections.append(current_section)

        pdf.close()

        full_text = "\n".join(all_text)
        title = self._extract_title(full_text, pdf_path)

        return {
            "title": title,
            "text": full_text,
            "sections": sections,
            "page_count": len(all_text),
        }

    def chunk(
            self,
            sections: list[dict],
            max_chunk_size: int = 1024,
            overlap: int = 100,
    ) -> list[dict]:
        chunks = []
        chunk_index = 0

        for section in sections:
            content = section["content"].strip()
            if not content:
                continue

            if len(content) <= max_chunk_size:
                chunks.append({
                    "chunk_index": chunk_index,
                    "content": content,
                    "metadata": {
                        "section": section["heading"],
                        "page": section["page"],
                    },
                })
                chunk_index += 1
            else:
                # 按句子边界切分，避免截断
                sentences = re.split(r'(?<=[.!?])\s+', content)
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > max_chunk_size:
                        if current_chunk.strip():
                            chunks.append({
                                "chunk_index": chunk_index,
                                "content": current_chunk.strip(),
                                "metadata": {
                                    "section": section["heading"],
                                    "page": section["page"],
                                },
                            })
                            chunk_index += 1
                            # overlap: 保留最后一句做上下文
                            if ". " in current_chunk:
                                last_sentence = current_chunk.strip().rsplit(". ", 1)[-1]
                                current_chunk = last_sentence + " "
                            else:
                                current_chunk = ""
                        else:
                            current_chunk = ""

                    current_chunk += sentence + " "

                if current_chunk.strip():
                    chunks.append({
                        "chunk_index": chunk_index,
                        "content": current_chunk.strip(),
                        "metadata": {
                            "section": section["heading"],
                            "page": section["page"],
                        },
                    })
                    chunk_index += 1

        return chunks

    def _is_heading(self, line: str) -> bool:
        """
        简单启发式判断是否为章节标题
        """
        # 匹配 "1 Introduction", "2.1 Method", "III. Results" 等常见格式
        patterns = [
                    r"^\d+[\.\s]",           # "1. " / "2.1 "
                    r"^[IVX]+[\.\s]+",       # "III. "
                    r"^Abstract$",           # Abstract
                    r"^Introduction$",
                    r"^Conclusion",
                    r"^References$",
                    r"^(Acknowledgment|Acknowledgement)",
                ]

        return any(re.match(p, line, re.IGNORECASE) for p in patterns)


    def _extract_title(self, text: str, pdf_path: str) -> str:
        """从文本开头提取标题（简单启发式）"""
        lines = text.strip().split("\n")
        # 取前10行中最长的一行作为标题候选
        candidates = []
        for line in lines[:10]:
            stripped = line.strip()
            if stripped and len(stripped) > 5:
                candidates.append(stripped)
        return candidates[0] if candidates else pdf_path.split("/")[-1].replace(".pdf", "")
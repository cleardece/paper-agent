"""
Paper Agent - PDF解析工具 v2
增强分块策略：更好的标题识别、保护区块、后处理合并
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
            "sections": [{"heading": str, "content": str, "page": int, "level": int}],
            "page_count": int,
        }
        """
        pdf = self._get_pdfplumber().open(pdf_path)
        all_text = []
        sections = []

        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            all_text.append(text)

            # 按行识别章节标题
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
            "sections": sections,
            "page_count": len(all_text),
        }

    def chunk(
            self,
            sections: list[dict],
            max_chunk_size: int = 1500,
            overlap: int = 150,
            min_chunk_size: int = 200,
    ) -> list[dict]:
        """
        分块策略 v2：
        1. 增强标题识别
        2. 保护区块（公式、表格、列表）
        3. 动态 chunk size
        4. 后处理合并过短 chunk
        """
        chunks = []
        chunk_index = 0

        for section in sections:
            content = section["content"].strip()
            if not content:
                continue

            # 附录使用更大的 chunk size
            is_appendix = section.get("is_appendix", False)
            cur_max = 2000 if is_appendix else max_chunk_size

            # 保护区块：将公式、表格、列表替换为占位符
            protected_content, placeholders = self._protect_blocks(content)

            if len(protected_content) <= cur_max:
                # 短内容直接作为一个 chunk
                chunks.append(self._create_chunk(
                    chunk_index, content, section, placeholders
                ))
                chunk_index += 1
            else:
                # 按句子边界切分
                new_chunks, chunk_index = self._split_long_section(
                    protected_content, content, section, placeholders,
                    cur_max, overlap, chunk_index
                )
                chunks.extend(new_chunks)

        # 后处理：合并过短的 chunk
        chunks = self._merge_short_chunks(chunks, min_chunk_size)

        # 重新编号
        for i, chunk in enumerate(chunks):
            chunk["chunk_index"] = i

        return chunks

    def _split_long_section(self, protected_content, original_content, section,
                            placeholders, max_size, overlap, start_index):
        """切分长章节"""
        chunks = []
        chunk_index = start_index

        # 按句子边界切分（保护区块视为一个整体）
        sentences = self._split_into_sentences(protected_content)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_size and current_chunk.strip():
                # 保存当前 chunk
                restored = self._restore_blocks(current_chunk.strip(), placeholders)
                chunks.append(self._create_chunk(
                    chunk_index, restored, section, placeholders
                ))
                chunk_index += 1

                # 重叠处理：从尾部取完整句子
                overlap_text = self._take_last_sentences(current_chunk, overlap)
                current_chunk = overlap_text + sentence
            else:
                current_chunk += sentence

        # 最后一个 chunk
        if current_chunk.strip():
            restored = self._restore_blocks(current_chunk.strip(), placeholders)
            chunks.append(self._create_chunk(
                chunk_index, restored, section, placeholders
            ))
            chunk_index += 1

        return chunks, chunk_index

    def _create_chunk(self, index, content, section, placeholders=None):
        """创建 chunk 字典"""
        return {
            "chunk_index": index,
            "content": content,
            "metadata": {
                "section": section.get("heading", ""),
                "page": section.get("page", 0),
                "level": section.get("level", 0),
                "is_appendix": section.get("is_appendix", False),
                "has_formula": bool(placeholders) if placeholders else False,
            },
        }

    def _is_heading(self, line: str) -> dict | None:
        """
        增强标题识别
        返回: {"title": str, "level": int, "is_appendix": bool} 或 None
        """
        # 无编号标题
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

        # 附录标题: Appendix A, Appendix B.1
        appendix_match = re.match(r'^(Appendix)\s+([A-Z])(?:\.(\d+))?\s*(.*)', line, re.IGNORECASE)
        if appendix_match:
            sub = appendix_match.group(3)
            level = 2 if sub else 1
            title = line.strip()
            return {"title": title, "level": level, "is_appendix": True}

        # 编号标题: 1, 1.1, 1.1.1, 2.3 Method
        number_match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)$', line)
        if number_match:
            num = number_match.group(1)
            level = num.count('.') + 1
            return {"title": line.strip(), "level": level, "is_appendix": False}

        # 附录内小节: A.1, B.2, C.3.1
        appendix_sub_match = re.match(r'^([A-Z]\.\d+(?:\.\d+)*)\s+(.+)', line)
        if appendix_sub_match:
            return {"title": line.strip(), "level": 2, "is_appendix": True}

        return None

    def _protect_blocks(self, text: str) -> tuple[str, list[str]]:
        """
        保护区块：将公式、表格、列表替换为占位符
        返回: (处理后的文本, 占位符列表)
        """
        placeholders = []
        protected = text

        # 保护行间公式: $$ ... $$ 或 \[ ... \]
        for pattern in [r'\$\$.*?\$\$', r'\\\[.*?\\\]']:
            protected = re.sub(pattern, lambda m: self._store(m.group(0), placeholders),
                               protected, flags=re.DOTALL)

        # 保护表格：连续行以 | 开头
        lines = protected.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                table_lines = [lines[i]]
                i += 1
                while i < len(lines) and lines[i].strip().startswith('|'):
                    table_lines.append(lines[i])
                    i += 1
                block = '\n'.join(table_lines)
                new_lines.append(self._store(block, placeholders))
            else:
                new_lines.append(lines[i])
                i += 1
        protected = '\n'.join(new_lines)

        # 保护列表：连续行以 - * • 或数字. 开头
        lines = protected.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if re.match(r'^[-*•]\s', stripped) or re.match(r'^\d+\.\s', stripped):
                list_lines = [lines[i]]
                i += 1
                while i < len(lines):
                    s = lines[i].strip()
                    if re.match(r'^[-*•]\s', s) or re.match(r'^\d+\.\s', s):
                        list_lines.append(lines[i])
                        i += 1
                    else:
                        break
                block = '\n'.join(list_lines)
                new_lines.append(self._store(block, placeholders))
            else:
                new_lines.append(lines[i])
                i += 1
        protected = '\n'.join(new_lines)

        return protected, placeholders

    def _store(self, text: str, placeholders: list) -> str:
        """存储受保护文本，返回占位符"""
        placeholders.append(text)
        return f"[PROTECTED_{len(placeholders) - 1}]"

    def _restore_blocks(self, text: str, placeholders: list) -> str:
        """还原占位符为原始文本"""
        result = text
        for i, block in enumerate(placeholders):
            result = result.replace(f"[PROTECTED_{i}]", block)
        return result

    def _split_into_sentences(self, text: str) -> list[str]:
        """按句子边界切分，保护区块视为一个整体"""
        # 先保护区块
        protected = text
        placeholders = []
        protected = re.sub(
            r'\[PROTECTED_\d+\]',
            lambda m: self._store(m.group(0), placeholders),
            protected
        )

        # 按句子边界切分
        sentences = re.split(r'(?<=[.!?;:])\s+', protected)

        # 还原保护区块
        result = []
        for sent in sentences:
            result.append(self._restore_blocks(sent, placeholders))

        return result

    def _take_last_sentences(self, text: str, max_chars: int) -> str:
        """从文本尾部取若干完整句子，总长度不超过 max_chars"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = ""
        for sent in reversed(sentences):
            if len(result) + len(sent) > max_chars:
                break
            result = sent + " " + result
        return result.strip()

    def _merge_short_chunks(self, chunks: list[dict], min_size: int) -> list[dict]:
        """合并过短的 chunk 到相邻块"""
        if not chunks:
            return chunks

        merged = [chunks[0]]
        for chunk in chunks[1:]:
            prev = merged[-1]
            if (len(prev["content"]) < min_size and
                    len(prev["content"]) + len(chunk["content"]) <= 2000):
                # 合并到前一个
                prev["content"] = prev["content"] + "\n" + chunk["content"]
                prev["metadata"]["merged"] = True
            else:
                merged.append(chunk)

        return merged

    def _extract_title(self, text: str, pdf_path: str) -> str:
        """从文本开头提取标题"""
        lines = text.strip().split("\n")
        candidates = []
        for line in lines[:15]:
            stripped = line.strip()
            if stripped and len(stripped) > 5:
                # 跳过期刊信息、页码等
                if re.match(r'^(Vol\.|ISSN|DOI|http|www\.)', stripped, re.IGNORECASE):
                    continue
                candidates.append(stripped)
        return candidates[0] if candidates else pdf_path.split("/")[-1].replace(".pdf", "")

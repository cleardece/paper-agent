"""
Paper Agent - Direct Analyzer Agent
单篇论文安全分析通道：读取已解析本地论文 → 提取核心章节 → 喂 LLM
"""

import logging
import re
from langchain_core.messages import SystemMessage, HumanMessage
from state.graph_state import AgentState

logger = logging.getLogger("paper-agent")


# 章节分类规则：关键词 → 需要的章节
SECTION_KEYWORDS = {
    "background": ["背景", "动机", "introduction", "background", "motivation", "problem"],
    "method": ["方法", "算法", "模型", "框架", "method", "approach", "algorithm", "model", "framework", "architecture"],
    "experiment": ["实验", "评估", "结果", "数据集", "experiment", "evaluation", "result", "dataset", "benchmark", "accuracy"],
    "conclusion": ["结论", "总结", "未来", "conclusion", "summary", "future", "limitation"],
    "related": ["对比", "区别", "相关", "related", "comparison", "difference", "prior work"],
}

# 核心章节（几乎每次都需要）
CORE_SECTIONS = ["abstract", "introduction", "method", "approach", "conclusion"]

# 章节名匹配模式（支持中英文）
SECTION_PATTERNS = {
    "abstract": r"^(abstract|摘要)",
    "introduction": r"^(introduction|引言|简介|1\.|1\s)",
    "method": r"(method|approach|algorithm|model|framework|proposed|方法|算法|模型|框架)",
    "experiment": r"(experiment|evaluation|result|analysis|discussion|实验|评估|结果|分析|讨论)",
    "conclusion": r"(conclusion|summary|discussion|future work|结论|总结|讨论|未来)",
    "related": r"(related work|background|prior|相关工作|背景)",
}


class DirectAnalyzerAgent:
    """单篇论文快速分析 Agent - 动态章节提取"""

    def __init__(self, llm, mongodb_client, embedder, milvus_client, pdf_parser):
        self.llm = llm
        self.mongo = mongodb_client
        self.embedder = embedder
        self.milvus = milvus_client
        self.parser = pdf_parser

    def invoke(self, state: AgentState) -> dict:
        query = state["user_query"]
        turn_context = state.get("turn_context") or {}
        target_paper_id = (
            turn_context.get("primary_paper_id")
            or state.get("target_paper_id")
        )
        logger.info(f"[DirectAnalyzer] 开始分析: {query[:50]}...")

        if not target_paper_id:
            return {
                "answer": "请先选择或明确指定要分析的论文。",
                "error": "NEED_PAPER_CONTEXT",
                "primary_paper_id": None,
                "resolved_paper_ids": [],
            }

        paper_info = self.mongo.get_paper(target_paper_id)
        if not paper_info:
            logger.warning(f"[DirectAnalyzer] 论文库选择不存在: {target_paper_id}")
            return {
                "answer": "所选论文已不存在，请从论文库重新选择。",
                "error": "selected_paper_not_found",
                "primary_paper_id": None,
                "resolved_paper_ids": [],
            }
        logger.info(
            f"[DirectAnalyzer] 使用已解析论文: "
            f"{paper_info.get('arxiv_id', target_paper_id)}"
        )

        logger.info(f"[DirectAnalyzer] 知识库中找到论文: {paper_info['title'][:50]}")
        paper_id = paper_info.get("arxiv_id", target_paper_id)
        full_text = paper_info.get("full_text", "")
        chunks = list(self.mongo.get_chunks_by_paper(paper_id))
        if not full_text and chunks:
            full_text = self._chunks_to_full_text(chunks)
        if not full_text:
            return {
                "answer": "所选论文没有可用正文片段，无法分析。",
                "error": "selected_paper_has_no_content",
                "primary_paper_id": None,
                "resolved_paper_ids": [],
            }

        if chunks and self.embedder is not None and self.milvus is not None:
            self._ensure_indexed(paper_info, chunks)

        sections = self._parse_sections_from_text(full_text)
        core_text = self._extract_relevant_sections(query, sections)
        result = self._analyze(paper_info, core_text, query)
        result.setdefault("resolved_paper_id", paper_id)
        result.setdefault("primary_paper_id", paper_id)
        result.setdefault("resolved_paper_ids", [paper_id])
        return result

    @staticmethod
    def _chunks_to_full_text(chunks: list[dict]) -> str:
        """将 MongoDB 中按序保存的 chunks 组合为可供单篇分析的正文。"""
        parts = []
        for chunk in sorted(chunks, key=lambda item: item.get("chunk_index", 0)):
            content = str(chunk.get("content", "")).strip()
            if not content:
                continue
            metadata = chunk.get("metadata") or {}
            section = str(metadata.get("section") or "Content").strip()
            parts.append(f"# {section}\n{content}")
        return "\n\n".join(parts)

    def _parse_sections_from_text(self, text: str) -> list[dict]:
        """从 Markdown 文本中解析章节结构"""
        sections = []
        current_heading = "Abstract"
        current_content = []

        for line in text.split("\n"):
            # 检测 Markdown 标题行
            if re.match(r'^#{1,3}\s+', line):
                # 保存上一个章节
                if current_content:
                    sections.append({
                        "heading": current_heading,
                        "content": "\n".join(current_content).strip(),
                    })
                current_heading = re.sub(r'^#{1,3}\s+', '', line).strip()
                current_content = []
            else:
                current_content.append(line)

        # 保存最后一个章节
        if current_content:
            sections.append({
                "heading": current_heading,
                "content": "\n".join(current_content).strip(),
            })

        logger.info(f"[DirectAnalyzer] 解析出 {len(sections)} 个章节")
        for s in sections[:5]:
            logger.info(f"  - {s['heading'][:40]}: {len(s['content'])} chars")

        return sections

    def _classify_query(self, query: str) -> list[str]:
        """根据用户查询判断需要哪些章节"""
        query_lower = query.lower()
        needed = set(CORE_SECTIONS)  # 核心章节总是需要

        for section_type, keywords in SECTION_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    needed.add(section_type)
                    break

        logger.info(f"[DirectAnalyzer] 查询需要的章节: {needed}")
        return list(needed)

    def _match_section(self, heading: str, section_type: str) -> bool:
        """判断章节标题是否匹配某个类型"""
        pattern = SECTION_PATTERNS.get(section_type, "")
        return bool(re.search(pattern, heading, re.IGNORECASE))

    def _extract_relevant_sections(self, query: str, sections: list[dict]) -> str:
        """根据用户查询动态提取相关章节"""
        needed_types = self._classify_query(query)
        extracted = []
        total_chars = 0
        max_chars = 20000  # 总字符上限

        for section in sections:
            heading = section["heading"]
            content = section["content"]

            # 检查这个章节是否是需要的类型
            is_needed = False
            for section_type in needed_types:
                if self._match_section(heading, section_type):
                    is_needed = True
                    break

            # abstract 总是提取
            if re.search(r"abstract|摘要", heading, re.IGNORECASE):
                is_needed = True

            if is_needed and content.strip():
                # 单个章节最多 5000 字符
                if len(content) > 5000:
                    content = content[:5000] + "\n[...章节过长，已截断...]"

                extracted.append(f"## {heading}\n{content}")
                total_chars += len(content)

                if total_chars >= max_chars:
                    logger.info(f"[DirectAnalyzer] 已达到字符上限 ({total_chars})，停止提取")
                    break

        result = "\n\n".join(extracted)
        logger.info(f"[DirectAnalyzer] 提取了 {len(extracted)} 个章节，共 {len(result)} 字符")
        return result

    def _index_paper(self, arxiv_id: str, sections: list, existing_chunks: list = None):
        """分块 + embedding + 存 Milvus"""
        if existing_chunks:
            chunks = existing_chunks
        else:
            try:
                chunks = self.parser.chunk(sections)
                if not chunks:
                    logger.warning(f"[DirectAnalyzer] 分块结果为空: {arxiv_id}")
                    return False
                mongo_chunks = [
                    {
                        "paper_arxiv_id": arxiv_id,
                        "chunk_index": c["chunk_index"],
                        "content": c["content"],
                        "metadata": c.get("metadata", {}),
                    }
                    for c in chunks
                ]
                self.mongo.insert_chunks(mongo_chunks)
                self.mongo.update_paper_status(arxiv_id, "chunked")
            except Exception as e:
                logger.error(f"[DirectAnalyzer] 分块入库失败: {e}", exc_info=True)
                return False

        paper = self.mongo.get_paper(arxiv_id) or {"arxiv_id": arxiv_id, "title": arxiv_id}
        return self._ensure_indexed(paper, chunks)

    def _ensure_indexed(self, paper: dict, chunks: list[dict]) -> bool:
        """从已有 chunks 恢复完整索引，避免残缺或重复的 Milvus 记录。"""
        arxiv_id = paper["arxiv_id"]
        expected_chunk_count = len(chunks)
        try:
            current_chunk_count = self.milvus.count(arxiv_id)
            current_paper_count = self.milvus.count_paper_embeddings(arxiv_id)
        except Exception as e:
            logger.error(f"[DirectAnalyzer] 无法检查索引状态: {e}", exc_info=True)
            self.mongo.update_paper_status(arxiv_id, "milvus_failed")
            return False

        if (
            paper.get("status") == "indexed"
            and current_chunk_count == expected_chunk_count
            and current_paper_count == 1
        ):
            return True

        try:
            self.milvus.delete_by_paper(arxiv_id)
        except Exception as e:
            logger.error(f"[DirectAnalyzer] 清理残缺向量失败: {e}", exc_info=True)
            self.mongo.update_paper_status(arxiv_id, "milvus_failed")
            return False

        try:
            texts = [c["content"] for c in chunks]
            vectors = self.embedder.embed_texts(texts)
            paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}".strip()
            paper_embedding = self.embedder.embed_texts([paper_text])[0] if paper_text else None

            # 释放 GPU 显存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        except Exception as e:
            logger.error(f"[DirectAnalyzer] Embedding 生成失败: {e}", exc_info=True)
            self.mongo.update_paper_status(arxiv_id, "embedding_failed")
            return False

        try:
            milvus_records = [
                {
                    "paper_arxiv_id": arxiv_id,
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                    "embedding": vectors[i],
                    "section": c.get("metadata", {}).get("section", ""),
                    "page": c.get("metadata", {}).get("page", 0),
                    "heading": c.get("metadata", {}).get("heading", ""),
                }
                for i, c in enumerate(chunks)
            ]
            self.milvus.insert(milvus_records)
            if paper_embedding is not None:
                self.milvus.insert_paper_embedding(
                    arxiv_id,
                    paper.get("title", arxiv_id),
                    paper_embedding,
                )
            self.mongo.update_paper_status(arxiv_id, "indexed")
            logger.info(f"[DirectAnalyzer] 索引恢复完成: {len(chunks)} 个分块")
            return True

        except Exception as e:
            logger.error(f"[DirectAnalyzer] Milvus 写入失败: {e}", exc_info=True)
            self.mongo.update_paper_status(arxiv_id, "milvus_failed")
            return False

    def _analyze(self, paper_info: dict, core_text: str, query: str = ""):
        """核心章节喂 LLM 分析"""
        title = paper_info.get("title", "未知")
        authors = paper_info.get("authors", [])
        if isinstance(authors, list):
            authors = ", ".join(authors)

        # 根据用户问题动态生成分析要求
        analysis_requirements = self._generate_analysis_requirements(query)

        prompt = f"""你是一位资深的学术论文分析师。请基于以下论文的核心内容进行分析。

## 论文信息
标题：{title}
作者：{authors}

## 核心内容
{core_text}

## 分析要求
{analysis_requirements}

请用清晰的结构输出分析结果，引用论文中的具体内容。"""

        logger.info(f"[DirectAnalyzer] 正在分析论文: {title[:50]}... (核心内容 {len(core_text)} 字符)")
        messages = [
            SystemMessage(content="你是一位资深的学术论文分析师。"),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        analysis = response.content.strip()
        logger.info(f"[DirectAnalyzer] 分析完成，长度: {len(analysis)}")

        return {
            "analysis": analysis,
            "answer": analysis,
            "error": None,
        }

    def _generate_analysis_requirements(self, query: str) -> str:
        """根据用户问题动态生成分析要求"""
        query_lower = query.lower()

        # 检测用户关注的方向
        if any(kw in query_lower for kw in ["实验", "评估", "结果", "experiment", "result", "accuracy"]):
            return """请重点分析：
1. **实验设计**：用了什么数据集？怎么评估的？基线方法是什么？
2. **主要结果**：关键实验数据和结论是什么？
3. **方法概述**：简要说明核心方法（1-2段即可）
4. **局限性**：论文自己提到了哪些不足？"""

        if any(kw in query_lower for kw in ["方法", "算法", "模型", "怎么做的", "method", "approach", "algorithm"]):
            return """请重点分析：
1. **核心方法**：提出了什么新方法？技术创新点是什么？
2. **方法细节**：算法流程、模型架构、关键公式
3. **实验验证**：方法在哪些任务上验证了？效果如何？
4. **与现有方法对比**：相比已有方法有什么优势？"""

        if any(kw in query_lower for kw in ["对比", "区别", "compare", "difference", "vs"]):
            return """请重点分析：
1. **核心方法**：这篇论文的方法是什么？
2. **关键特点**：与同类方法相比有什么独特之处？
3. **实验结果**：在哪些指标上表现好/差？
4. **适用场景**：这个方法最适合什么类型的问题？"""

        # 默认：全面分析
        return """请从以下几个方面进行分析：
1. **研究背景与动机**：为什么要做这个研究？解决什么问题？
2. **核心方法/贡献**：提出了什么新方法？技术创新点是什么？
3. **实验设计**：用了什么数据集？怎么评估的？基线方法是什么？
4. **主要结果**：关键实验数据和结论是什么？
5. **局限性与未来工作**：论文自己提到了哪些不足？"""

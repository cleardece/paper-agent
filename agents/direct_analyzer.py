"""
Paper Agent - Direct Analyzer Agent
单篇论文快速通道：下载 → 解析 → 全文喂 LLM + 后台入库
"""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from state.graph_state import AgentState

logger = logging.getLogger("paper-agent")


ANALYSIS_PROMPT = """你是一位资深的学术论文分析师。请对以下论文进行全面分析。

## 论文信息
标题：{title}
作者：{authors}

## 论文全文
{full_text}

## 分析要求
请从以下几个方面进行分析：
1. **研究背景与动机**：为什么要做这个研究？解决什么问题？
2. **核心方法/贡献**：提出了什么新方法？技术创新点是什么？
3. **实验设计**：用了什么数据集？怎么评估的？基线方法是什么？
4. **主要结果**：关键实验数据和结论是什么？
5. **局限性与未来工作**：论文自己提到了哪些不足？

请用清晰的结构输出分析结果，引用论文中的具体内容。"""


class DirectAnalyzerAgent:
    """单篇论文快速分析 Agent"""

    def __init__(self, llm, mongodb_client, embedder, milvus_client, pdf_parser, arxiv_api):
        self.llm = llm
        self.mongo = mongodb_client
        self.embedder = embedder
        self.milvus = milvus_client
        self.parser = pdf_parser
        self.arxiv = arxiv_api

    def invoke(self, state: AgentState) -> dict:
        query = state["user_query"]
        target_paper = state.get("target_paper")
        logger.info(f"[DirectAnalyzer] 开始分析: {query[:50]}...")

        # 1. 尝试从知识库找到论文（优先用 target_paper）
        paper_info = self._find_paper(query, target_paper)

        if paper_info:
            # 库中有 → 直接读全文分析
            logger.info(f"[DirectAnalyzer] 知识库中找到论文: {paper_info['title'][:50]}")
            full_text = paper_info.get("full_text", "")
            if not full_text:
                return {"answer": "论文全文未存储，请重新入库。", "error": None}
            return self._analyze(paper_info, full_text)

        # 2. 库中没有 → 下载 + 解析 + 分析 + 后台入库
        logger.info("[DirectAnalyzer] 知识库中未找到，开始下载论文...")
        return self._fetch_and_analyze(query)

    def _find_paper(self, query: str, target_paper: str = None):
        """从知识库中查找论文"""
        try:
            papers = self.mongo.list_papers(limit=50)
            if not papers:
                return None

            # 优先用 target_paper 匹配
            if target_paper:
                target_lower = target_paper.lower()
                for paper in papers:
                    title = paper.get("title", "").lower()
                    if any(kw in title for kw in target_lower.split() if len(kw) > 3):
                        logger.info(f"[DirectAnalyzer] 通过 target_paper 匹配: {paper.get('title', '')[:50]}")
                        return paper

            # 用查询关键词匹配
            query_lower = query.lower()
            import re
            keywords = [kw for kw in re.findall(r'[a-zA-Z]{3,}', query_lower) if len(kw) > 3]
            if not keywords:
                return None

            best_match = None
            best_score = 0
            for paper in papers:
                title = paper.get("title", "").lower()
                matched = sum(1 for kw in keywords if kw in title)
                if matched > best_score:
                    best_score = matched
                    best_match = paper

            if best_score >= 1:
                return best_match
            return None
        except Exception as e:
            logger.warning(f"[DirectAnalyzer] 查找论文失败: {e}")
            return None

    def _fetch_and_analyze(self, query: str):
        """下载论文 → 解析 → 分析 + 后台入库"""
        try:
            # 搜索 arXiv
            papers = self.arxiv.search(query, max_results=1)
            if not papers:
                return {"answer": "未找到相关论文，请检查论文标题。", "error": None}

            paper_meta = papers[0]
            arxiv_id = paper_meta["arxiv_id"]
            title = paper_meta.get("title", "未知")

            # 检查是否已入库
            if self.mongo.paper_exists(arxiv_id):
                existing = self.mongo.get_paper(arxiv_id)
                if existing and existing.get("full_text"):
                    logger.info(f"[DirectAnalyzer] 论文已入库: {title[:50]}")
                    return self._analyze(existing, existing["full_text"])

            # 下载 PDF
            if not paper_meta.get("pdf_url"):
                return {"answer": f"论文 {title} 没有 PDF 链接。", "error": None}

            pdf_path = self._download_pdf(paper_meta["pdf_url"], arxiv_id)

            # MinerU 解析
            logger.info(f"[DirectAnalyzer] 正在解析 PDF: {title[:50]}...")
            parsed = self.parser.parse(pdf_path)
            full_text = parsed.get("full_text", "")
            sections = parsed.get("sections", [])

            if not full_text:
                return {"answer": f"论文 {title} 解析失败，无法提取文本。", "error": None}

            # 存 MongoDB（含全文）
            self.mongo.upsert_paper({
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": paper_meta.get("abstract", ""),
                "authors": paper_meta.get("authors", []),
                "pdf_url": paper_meta.get("pdf_url", ""),
                "full_text": full_text,
                "status": "parsed",
            })

            # 后台分块 + 入库（不阻塞分析）
            paper_info = {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": paper_meta.get("authors", []),
                "full_text": full_text,
            }

            # 同步分块入库（确保数据一致性）
            self._index_paper(arxiv_id, sections)

            # 全文喂 LLM 分析
            return self._analyze(paper_info, full_text)

        except Exception as e:
            logger.error(f"[DirectAnalyzer] 处理失败: {e}", exc_info=True)
            return {"answer": f"处理论文时出错: {str(e)}", "error": str(e)}

    def _index_paper(self, arxiv_id: str, sections: list):
        """分块 + embedding + 存 Milvus"""
        try:
            # 分块
            chunks = self.parser.chunk(sections)
            if not chunks:
                logger.warning(f"[DirectAnalyzer] 分块结果为空: {arxiv_id}")
                return

            # 存 MongoDB chunks
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

            # Embedding
            texts = [c["content"] for c in chunks]
            vectors = self.embedder.embed_texts(texts)

            # 存 Milvus
            milvus_records = [
                {
                    "paper_arxiv_id": arxiv_id,
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                    "embedding": vectors[i],
                    "metadata_json": json.dumps(c.get("metadata", {})),
                }
                for i, c in enumerate(chunks)
            ]
            self.milvus.insert(milvus_records)
            self.mongo.update_paper_status(arxiv_id, "indexed")
            logger.info(f"[DirectAnalyzer] 入库完成: {len(chunks)} 个分块")

        except Exception as e:
            logger.error(f"[DirectAnalyzer] 入库失败: {e}", exc_info=True)

    def _analyze(self, paper_info: dict, full_text: str):
        """全文喂 LLM 分析"""
        title = paper_info.get("title", "未知")
        authors = paper_info.get("authors", [])
        if isinstance(authors, list):
            authors = ", ".join(authors)

        # 截断过长的全文（保留前 30000 字符）
        if len(full_text) > 30000:
            full_text = full_text[:30000] + "\n\n[...全文过长，已截断...]"

        prompt = ANALYSIS_PROMPT.format(
            title=title,
            authors=authors,
            full_text=full_text,
        )

        logger.info(f"[DirectAnalyzer] 正在分析论文: {title[:50]}...")
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

    def _download_pdf(self, url: str, arxiv_id: str) -> str:
        """下载 PDF"""
        import os
        import httpx

        tmp_dir = os.path.join(os.getcwd(), "tmp_pdfs")
        os.makedirs(tmp_dir, exist_ok=True)
        pdf_path = os.path.join(tmp_dir, f"{arxiv_id.replace('/', '_')}.pdf")

        if not os.path.exists(pdf_path):
            logger.info(f"[DirectAnalyzer] 下载 PDF: {url[:60]}...")
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                with open(pdf_path, "wb") as f:
                    f.write(response.content)
            logger.info(f"[DirectAnalyzer] PDF 下载完成")

        return pdf_path

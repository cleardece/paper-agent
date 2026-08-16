"""
Paper Agent - Direct Analyzer Agent
单篇论文快速通道：下载 → 解析 → 动态提取核心章节 → 喂 LLM + 后台入库
"""

import json
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

        # 1. 尝试从知识库找到论文
        paper_info = self._find_paper(query, target_paper)

        if paper_info:
            logger.info(f"[DirectAnalyzer] 知识库中找到论文: {paper_info['title'][:50]}")
            full_text = paper_info.get("full_text", "")
            if not full_text:
                logger.info("[DirectAnalyzer] 论文全文缺失，尝试下载补全...")
                return self._fetch_and_analyze(query)

            # 补上 Milvus 入库
            if paper_info.get("status") != "indexed":
                logger.info("[DirectAnalyzer] 论文未完成向量化，补上入库...")
                chunks = list(self.mongo.get_chunks_by_paper(paper_info["arxiv_id"]))
                if chunks:
                    self._index_paper(paper_info["arxiv_id"], None, chunks)

            # 动态提取章节 + 分析
            sections = self._parse_sections_from_text(full_text)
            core_text = self._extract_relevant_sections(query, sections)
            return self._analyze(paper_info, core_text, query)

        # 2. 库中没有 → 下载 + 解析 + 分析 + 后台入库
        logger.info("[DirectAnalyzer] 知识库中未找到，开始下载论文...")
        return self._fetch_and_analyze(query)

    def _find_paper(self, query: str, target_paper: str = None):
        """从知识库中查找论文"""
        try:
            papers = self.mongo.list_papers(
                limit=50,
                projection={"arxiv_id": 1, "title": 1, "abstract": 1, "authors": 1, "full_text": 1, "status": 1},
            )
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
        """下载论文 → 解析 → 动态提取章节 → 分析 + 后台入库"""
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
                    sections = self._parse_sections_from_text(existing["full_text"])
                    core_text = self._extract_relevant_sections(query, sections)
                    return self._analyze(existing, core_text, query)

            # 下载 PDF
            if not paper_meta.get("pdf_url"):
                return {"answer": f"论文 {title} 没有 PDF 链接。", "error": None}

            pdf_path = self._download_pdf(paper_meta["pdf_url"], arxiv_id)

            # MinerU 解析
            logger.info(f"[DirectAnalyzer] 正在解析 PDF: {title[:50]}...")
            parsed = self.parser.parse(pdf_path)
            full_text = parsed.get("full_text") or parsed.get("text") or parsed.get("markdown") or ""
            if not full_text and parsed.get("sections"):
                full_text = "\n\n".join(
                    f"## {s.get('heading', '')}\n{s.get('content', '')}"
                    for s in parsed["sections"]
                )
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

            paper_info = {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": paper_meta.get("authors", []),
                "full_text": full_text,
            }

            # 同步分块入库
            self._index_paper(arxiv_id, sections)

            # 动态提取章节 + 分析
            parsed_sections = self._parse_sections_from_text(full_text)
            core_text = self._extract_relevant_sections(query, parsed_sections)
            return self._analyze(paper_info, core_text, query)

        except Exception as e:
            logger.error(f"[DirectAnalyzer] 处理失败: {e}", exc_info=True)
            return {"answer": f"处理论文时出错: {str(e)}", "error": str(e)}

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
        try:
            if existing_chunks:
                chunks = existing_chunks
            else:
                chunks = self.parser.chunk(sections)
                if not chunks:
                    logger.warning(f"[DirectAnalyzer] 分块结果为空: {arxiv_id}")
                    return
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

            texts = [c["content"] for c in chunks]
            vectors = self.embedder.embed_texts(texts)

            # 释放 GPU 显存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

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
            self.mongo.update_paper_status(arxiv_id, "indexed")
            logger.info(f"[DirectAnalyzer] 入库完成: {len(chunks)} 个分块")

        except Exception as e:
            logger.error(f"[DirectAnalyzer] 入库失败: {e}", exc_info=True)

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
            logger.info("[DirectAnalyzer] PDF 下载完成")

        return pdf_path

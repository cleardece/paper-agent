"""
初始化论文的 Memory 系统
从现有 MongoDB 数据生成 Paper Memory + Section Memory
"""

import os
import sys
import json
import logging

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("init_memory")


def main():
    from pymongo import MongoClient
    from storage.memory import MemoryManager
    from tools.pdf_parser import PDFParser

    client = MongoClient('mongodb://localhost:27017')
    db = client['paper_agent']
    memory = MemoryManager(client)

    # 获取所有已入库论文
    papers = list(db.papers.find({'status': {'$in': ['chunked', 'indexed']}}))
    logger.info(f"找到 {len(papers)} 篇论文需要初始化 Memory")

    for paper in papers:
        arxiv_id = paper.get('arxiv_id')
        title = paper.get('title', 'Unknown')
        logger.info(f"\n处理: {title[:60]}...")

        # 检查是否已有 Memory
        if memory.paper.get(arxiv_id):
            logger.info(f"  已存在，跳过")
            continue

        # 获取 chunks
        chunks = list(db.chunks.find({'paper_arxiv_id': arxiv_id}))
        if not chunks:
            logger.info(f"  无 chunks，跳过")
            continue

        # 按 section 分组 chunks
        sections_map = {}
        for c in chunks:
            meta = c.get('metadata', {})
            section_heading = meta.get('section', 'Unknown')
            if section_heading not in sections_map:
                sections_map[section_heading] = {
                    'heading': section_heading,
                    'content': '',
                    'chunk_count': 0,
                }
            sections_map[section_heading]['content'] += c.get('content', '') + '\n'
            sections_map[section_heading]['chunk_count'] += 1

        # 创建 sections 列表
        sections = []
        for i, (heading, data) in enumerate(sections_map.items()):
            sections.append({
                'heading': heading,
                'summary': '',  # 稍后用 LLM 生成
                'content': data['content'][:2000],  # 前2000字符作为预览
                'chunk_count': data['chunk_count'],
            })

        # 初始化 Memory
        memory.init_paper_memory(arxiv_id, {
            'title': title,
            'authors': paper.get('authors', []),
            'year': paper.get('year'),
            'abstract': paper.get('abstract', ''),
            'sections': sections,
        })

        logger.info(f"  创建了 {len(sections)} 个 Section Memory")
        logger.info(f"  完成!")

    # 统计
    paper_count = db.paper_memory.count_documents({})
    section_count = db.section_memory.count_documents({})
    logger.info(f"\nMemory 初始化完成:")
    logger.info(f"  Paper Memory: {paper_count} 篇")
    logger.info(f"  Section Memory: {section_count} 个 sections")

    client.close()


if __name__ == "__main__":
    main()

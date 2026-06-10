"""
测试本地 PDF 论文的分块和入库流程
用法: python test_local_paper.py <pdf路径>
"""

import sys
import os
import json
import logging

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("test")

def main():
    if len(sys.argv) < 2:
        print("用法: python test_local_paper.py <pdf路径>")
        print("示例: python test_local_paper.py D:\\papers\\my_paper.pdf")
        return

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"文件不存在: {pdf_path}")
        return

    print(f"=" * 60)
    print(f"测试本地 PDF 分块")
    print(f"文件: {pdf_path}")
    print(f"=" * 60)

    # 初始化组件
    print("\n[1/5] 初始化组件...")
    from config import get_llm
    from storage.mongodb import MongoDBClient
    from storage.milvus import MilvusClient
    from tools.embeddings import EmbeddingService
    from tools.pdf_parser import PDFParser

    llm = get_llm()
    mongodb = MongoDBClient()
    milvus = MilvusClient()
    embedder = EmbeddingService()
    parser = PDFParser()

    # 解析 PDF
    print("\n[2/5] 解析 PDF...")
    parsed = parser.parse(pdf_path)
    print(f"  标题: {parsed.get('title', '未知')[:50]}...")
    print(f"  章节数: {len(parsed.get('sections', []))}")
    for i, section in enumerate(parsed.get('sections', [])[:5]):
        title = section.get('title', '未知')
        content_len = len(section.get('content', ''))
        print(f"    [{i+1}] {title[:30]}... ({content_len} 字符)")

    # 分块
    print("\n[3/5] 分块...")
    chunks = parser.chunk(parsed.get('sections', []))
    print(f"  总分块数: {len(chunks)}")
    for i, chunk in enumerate(chunks[:5]):
        content_preview = chunk.get('content', '')[:80].replace('\n', ' ')
        section = chunk.get('metadata', {}).get('section', '')
        print(f"    [{i+1}] chunk_{chunk.get('chunk_index', i)} [{section}]: {content_preview}...")
    if len(chunks) > 5:
        print(f"    ... 共 {len(chunks)} 个分块")

    # 生成 arxiv_id（用文件名，限制长度）
    arxiv_id = os.path.splitext(os.path.basename(pdf_path))[0]
    arxiv_id = arxiv_id.replace(' ', '_').replace('/', '_')
    if len(arxiv_id) > 60:
        arxiv_id = arxiv_id[:60]

    # 存入 MongoDB
    print("\n[4/5] 存入 MongoDB...")
    mongodb.upsert_paper({
        "arxiv_id": arxiv_id,
        "title": parsed.get('title', os.path.basename(pdf_path)),
        "abstract": "",
        "authors": [],
        "pdf_url": f"file://{pdf_path}",
        "status": "chunked",
    })

    mongo_chunks = [
        {
            "paper_arxiv_id": arxiv_id,
            "chunk_index": c["chunk_index"],
            "content": c["content"],
            "metadata": c.get("metadata", {}),
        }
        for c in chunks
    ]
    inserted = mongodb.insert_chunks(mongo_chunks)
    print(f"  已插入 {inserted} 个分块")

    # 生成 Embedding 并存入 Milvus
    print("\n[5/5] 生成 Embedding 并存入 Milvus...")
    texts = [c["content"] for c in chunks]
    vectors = embedder.embed_texts(texts, batch_size=8)
    print(f"  生成 {len(vectors)} 个向量，维度: {len(vectors[0]) if vectors else 0}")

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
    milvus.insert(milvus_records)
    print(f"  已插入 {len(milvus_records)} 条向量到 Milvus")

    # 统计
    print(f"\n" + "=" * 60)
    print(f"完成！")
    print(f"  论文 ID: {arxiv_id}")
    print(f"  分块数: {len(chunks)}")
    print(f"  向量数: {len(vectors)}")
    print(f"  MongoDB 论文数: {mongodb.count_papers()}")
    print(f"  MongoDB 分块数: {mongodb.count_chunks()}")
    print(f"  Milvus 向量数: {milvus.count()}")
    print(f"=" * 60)

    # 测试检索
    print(f"\n测试检索: '什么是PINN'")
    query_vector = embedder.embed_query("什么是PINN")
    hits = milvus.search(query_embedding=query_vector, top_k=3)
    print(f"  找到 {len(hits)} 个相关分块:")
    for i, hit in enumerate(hits):
        content_preview = hit.get('content', '')[:80].replace('\n', ' ')
        score = hit.get('score', 0)
        print(f"    [{i+1}] score={score:.3f}: {content_preview}...")

    mongodb.close()
    milvus.close()
    print("\n测试完成！")

if __name__ == "__main__":
    main()

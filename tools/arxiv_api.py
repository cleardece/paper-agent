import arxiv

def search_papers(query: str, max_results: int = 5):
    """搜索arXiv论文"""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for r in search.results():
        results.append({
            "title": r.title,
            "url": r.entry_id,
            "abstract": r.summary,
            "authors": [a.name for a in r.authors],
            "pdf_url": r.pdf_url
        })
    return results

def fetch_paper_content(url: str) -> str:
    """下载论文PDF并提取文本"""
    import pdfplumber
    import requests
    from io import BytesIO

    # 从arXiv ID获取PDF URL
    if "arxiv.org" in url:
        pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
    else:
        pdf_url = url

    resp = requests.get(pdf_url)
    with pdfplumber.open(BytesIO(resp.content)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return text
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
import feedparser
import aiohttp
import asyncio
import json
import os
import io
import PyPDF2

load_dotenv()

# Default number of results to return
DEFAULT_RESULT_LIMIT = 5

@dataclass
class ArxivContext:
    """Context for the arXiv MCP server."""
    session: aiohttp.ClientSession

@asynccontextmanager
async def arxiv_lifespan(server: FastMCP) -> AsyncIterator[ArxivContext]:
    """
    Manages the arXiv client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        ArxivContext: The context containing the aiohttp session
    """
    # Create aiohttp session
    session = aiohttp.ClientSession()   
    
    try:
        yield ArxivContext(session=session)
    finally:
        # Close the session when done
        await session.close()

# Initialize FastMCP server with the arXiv context
mcp = FastMCP(
    "arxiv-mcp",
    instructions="MCP server for retrieving papers from arXiv based on keywords",
    lifespan=arxiv_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8050")),
    sse_path="/sse",
    streamable_http_path="/mcp",
)

async def fetch_arxiv_papers(session: aiohttp.ClientSession, query: str, limit: int = DEFAULT_RESULT_LIMIT) -> list:
    """
    Fetch papers from arXiv based on the query.

    Args:
        session: The aiohttp session
        query: The search query
        limit: Maximum number of results to return

    Returns:
        List of paper metadata
    """
    import logging
    _logger = logging.getLogger("arxiv-mcp")

    # 构建多策略搜索：先用 ti:(标题) 搜索，再用 all:(全文) 搜索
    formatted_query = query.replace(' ', '+')

    # 策略1: 标题搜索 (最精确)
    search_strategies = [
        f"ti:{formatted_query}",
        f"all:{formatted_query}",
    ]

    all_papers = []
    seen_ids = set()

    for search_query in search_strategies:
        url = f"http://export.arxiv.org/api/query?search_query={search_query}&start=0&max_results={limit}"
        _logger.info(f"[arXiv MCP] 搜索: {search_query}")

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status != 200:
                    _logger.warning(f"[arXiv MCP] HTTP {response.status} for query: {search_query}")
                    continue

                content = await response.text()
                feed = feedparser.parse(content)

                for entry in feed.entries:
                    paper_id = entry.id.split('/abs/')[-1]
                    if paper_id in seen_ids:
                        continue
                    seen_ids.add(paper_id)

                    authors = [author.name for author in entry.authors]
                    categories = [tag.term for tag in entry.tags] if hasattr(entry, 'tags') else []

                    paper = {
                        "id": paper_id,
                        "title": entry.title,
                        "authors": authors,
                        "summary": entry.summary,
                        "published": entry.published,
                        "updated": entry.updated,
                        "link": entry.link,
                        "pdf_link": f"http://arxiv.org/pdf/{paper_id}",
                        "categories": categories,
                    }
                    all_papers.append(paper)

                _logger.info(f"[arXiv MCP] 策略 '{search_query[:40]}...' 找到 {len(feed.entries)} 篇")
        except Exception as e:
            _logger.warning(f"[arXiv MCP] 搜索失败: {e}")
            continue

        # 如果标题搜索已有结果，不需要全文搜索
        if all_papers:
            break

    return all_papers[:limit]

async def fetch_paper_content(session: aiohttp.ClientSession, paper_id: str) -> str:
    """
    Attempt to fetch and extract the content of a paper.
    
    Args:
        session: The aiohttp session
        paper_id: The arXiv paper ID
        
    Returns:
        Extracted text from the paper or error message
    """
    try:
        pdf_url = f"http://arxiv.org/pdf/{paper_id}"
        
        async with session.get(pdf_url) as response:
            if response.status != 200:
                return f"Failed to fetch PDF: HTTP {response.status}"
            
            # Read PDF content
            pdf_content = await response.read()
            
            # Use PyPDF2 to extract text
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from first few pages (full extraction could be too large)
            max_pages = min(5, len(pdf_reader.pages))
            text = ""
            
            for i in range(max_pages):
                page = pdf_reader.pages[i]
                text += page.extract_text()
            
            # Truncate if too long
            if len(text) > 5000:
                text = text[:5000] + "... [truncated]"
            
            return text
    except Exception as e:
        return f"Error extracting paper content: {str(e)}"

def format_paper_to_markdown(paper: dict, content: str = None) -> str:
    """
    Format paper metadata and content to Markdown.
    
    Args:
        paper: Paper metadata
        content: Paper content (if available)
        
    Returns:
        Markdown formatted string
    """
    md = f"# {paper['title']}\n\n"
    
    # Authors
    md += "## Authors\n"
    md += ", ".join(paper['authors']) + "\n\n"
    
    # Publication info
    md += f"**Published:** {paper['published']}\n"
    md += f"**Last Updated:** {paper['updated']}\n"
    md += f"**arXiv ID:** {paper['id']}\n"
    md += f"**Categories:** {', '.join(paper['categories'])}\n\n"
    
    # Links
    md += f"**Paper Link:** [{paper['link']}]({paper['link']})\n"
    md += f"**PDF Link:** [{paper['pdf_link']}]({paper['pdf_link']})\n\n"
    
    # Summary
    md += "## Abstract\n"
    md += paper['summary'] + "\n\n"
    
    # Content (if available)
    if content and content.startswith("Error") is False:
        md += "## Content Preview\n"
        md += "```\n" + content + "\n```\n\n"
    
    return md

@mcp.tool()
async def search_arxiv(ctx: Context, query: str, limit: int = DEFAULT_RESULT_LIMIT) -> str:
    """Search for papers on arXiv based on keywords.
    
    This tool searches arXiv for papers matching the provided keywords and returns
    the top results with their metadata in a structured format.
    
    Args:
        ctx: The MCP server provided context
        query: Search keywords or phrases
        limit: Maximum number of results to return (default: 5)
    """
    try:
        session = ctx.request_context.lifespan_context.session
        papers = await fetch_arxiv_papers(session, query, limit)
        
        if not papers:
            return "No papers found matching your query."
        
        # Format results as JSON
        return json.dumps(papers, indent=2)
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"

@mcp.tool()
async def search_papers(
    ctx: Context,
    query: str,
    max_results: int = DEFAULT_RESULT_LIMIT,
    limit: int | None = None,
) -> str:
    """Search for papers on arXiv.

    Compatibility wrapper for clients that use the blazickjp/arxiv-mcp-server
    tool shape: search_papers(query, max_results=...).
    """
    return await search_arxiv(ctx, query=query, limit=limit or max_results)

@mcp.tool()
async def extract_paper_content(ctx: Context, paper_id: str) -> str:
    """Download a paper PDF and extract a bounded text preview."""
    session = ctx.request_context.lifespan_context.session
    return await fetch_paper_content(session, paper_id)

@mcp.tool()
async def get_paper_details(ctx: Context, paper_id: str, include_content: bool = False) -> str:
    """Get detailed information about a specific arXiv paper.
    
    This tool retrieves detailed metadata for a specific paper and optionally
    attempts to extract its content.
    
    Args:
        ctx: The MCP server provided context
        paper_id: The arXiv paper ID (e.g., "2104.08653")
        include_content: Whether to attempt to extract paper content (default: False)
    """
    try:
        session = ctx.request_context.lifespan_context.session
        
        # Fetch paper metadata
        query = f"id:{paper_id}"
        papers = await fetch_arxiv_papers(session, query, 1)
        
        if not papers:
            return f"Paper with ID {paper_id} not found."
        
        paper = papers[0]
        
        # Fetch content if requested
        content = None
        if include_content:
            content = await fetch_paper_content(session, paper_id)
        
        # Format as Markdown
        markdown_output = format_paper_to_markdown(paper, content)
        return markdown_output
    except Exception as e:
        return f"Error retrieving paper details: {str(e)}"

@mcp.tool()
async def search_and_summarize(ctx: Context, query: str, limit: int = DEFAULT_RESULT_LIMIT) -> str:
    """Search arXiv and provide a comprehensive summary of the top papers.
    
    This tool searches arXiv for papers matching the provided keywords, fetches
    their metadata and content, and returns a comprehensive summary in Markdown format.
    
    Args:
        ctx: The MCP server provided context
        query: Search keywords or phrases
        limit: Maximum number of results to return (default: 5)
    """
    try:
        session = ctx.request_context.lifespan_context.session
        papers = await fetch_arxiv_papers(session, query, limit)
        
        if not papers:
            return "No papers found matching your query."
        
        # Compile results
        results = f"# arXiv Search Results for: {query}\n\n"
        results += f"Found {len(papers)} papers matching your query.\n\n"
        
        # Process each paper
        for i, paper in enumerate(papers):
            results += f"## {i+1}. {paper['title']}\n\n"
            
            # Authors
            results += "### Authors\n"
            results += ", ".join(paper['authors']) + "\n\n"
            
            # Publication info
            results += f"**Published:** {paper['published']}\n"
            results += f"**arXiv ID:** {paper['id']}\n"
            results += f"**Categories:** {', '.join(paper['categories'])}\n\n"
            
            # Links
            results += f"**Paper Link:** [{paper['link']}]({paper['link']})\n"
            results += f"**PDF Link:** [{paper['pdf_link']}]({paper['pdf_link']})\n\n"
            
            # Abstract
            results += "### Abstract\n"
            results += paper['summary'] + "\n\n"
            
            # Add separator between papers
            if i < len(papers) - 1:
                results += "---\n\n"
        
        return results
    except Exception as e:
        return f"Error searching and summarizing arXiv papers: {str(e)}"

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())

from langgraph.graph import StateGraph, END
from state.graph_state import GraphState
from tools.arxiv_api import search_papers, fetch_paper_content
import re
from config import llm
def fetch_paper(state):
    """从arXiv获取论文内容"""
    paper_url = state["paper_url"]
    text = fetch_paper_content(paper_url)
    return {"paper_text": text, "current_step": "fetch"}

def parse_paper(state):
    text = state["paper_text"]

    # 更通用的章节标题模式（允许数字、冒号等）
    pattern = r'\n((?:[A-Z][A-Za-z ]{2,}|Abstract|Introduction|Methods|Results|Discussion|Conclusion)[\s\S]*?\n)'
    parts = re.split(pattern, text)

    sections = {}
    current_title = "Preamble"  # 第一个标题之前的内容
    for part in parts:
        if re.match(pattern, f"\n{part}"):
            current_title = part.strip()
            if current_title not in sections:
                sections[current_title] = ""
        else:
            if current_title not in sections:
                sections[current_title] = ""
            sections[current_title] += part  # 累加内容

    return {"sections": sections, "current_step": "parse"}
def summarize(state):
    """LLM总结论文"""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是论文分析助手，擅长提炼论文核心贡献。"),
        ("user", "请分析以下论文，给出：\n1. 研究问题\n2. 核心方法\n3. 主要贡献\n\n论文内容：\n{paper_text}")
    ])

    try:
        chain = prompt | llm
        result = chain.invoke({"paper_text": state["paper_text"]})
        return {"summary": result.content}
    except Exception as e:
        return {"summary": f"总结失败: {str(e)}"}

def analyze_methodology(state):
    return state

def generate_presentation(state):
    return state
def build_workflow():
    workflow = StateGraph(GraphState)

    # 添加节点（先把函数名挂上，后面再实现）
    workflow.add_node("fetch", fetch_paper)
    workflow.add_node("parse", parse_paper)
    workflow.add_node("summarize", summarize)
    workflow.add_node("analyze", analyze_methodology)
    workflow.add_node("present", generate_presentation)

    # 设置入口
    workflow.set_entry_point("fetch")

    # 连接边：fetch -> parse -> summarize -> analyze -> present -> END
    workflow.add_edge("fetch", "parse")
    workflow.add_edge("parse", "summarize")
    workflow.add_edge("summarize", "analyze")
    workflow.add_edge("analyze", "present")
    workflow.add_edge("present", END)

    return workflow.compile()


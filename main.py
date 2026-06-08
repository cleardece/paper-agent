"""
Paper Agent - 主入口
初始化所有组件，启动交互式问答
"""
import asyncio
from agents.analyzer import AnalyzerAgent
from agents.critic import CriticAgent
from agents.fetcher import FetcherAgent
from agents.presenter import PresenterAgent
from agents.retriever import RetrieverAgent
from agents.supervisor import SupervisorAgent
from config import get_llm, LLM_MODEL
from graph.workflow import build_workflow
from state.graph_state import AgentState
from storage.milvus import MilvusClient
from storage.mongodb import MongoDBClient
# 工具和存储层
from tools.arxiv_api import ArxivAPI
from tools.embeddings import EmbeddingService
from tools.pdf_parser import PDFParser
from tools.code_generator import CodeGenerator
def init_components():
    llm = get_llm()
    arxiv_api = ArxivAPI()
    pdf_parser = PDFParser()
    embedding_service = EmbeddingService()
    mongodb_client = MongoDBClient()
    milvus_client = MilvusClient()
    code_generator = CodeGenerator(llm)  # 加这行

    agents = {
        "supervisor": SupervisorAgent(llm),
        "fetcher": FetcherAgent(arxiv_api, pdf_parser, mongodb_client, embedding_service, milvus_client),
        "retriever": RetrieverAgent(embedding_service, milvus_client, mongodb_client),
        "analyzer": AnalyzerAgent(llm, mongodb_client),
        "critic": CriticAgent(llm),
        "presenter": PresenterAgent(llm, code_generator),  # 这里传进去
    }

    workflow = build_workflow(**agents)
    return workflow

def create_initial_state(query: str) -> AgentState:
    """创建初始状态"""
    return {
        "user_query": query,
        "search_query": None,
        "messages": [],
        "target_papers": [],
        "retrieved_chunks": [],
        "search_results": None,
        "analysis": None,
        "answer": None,
        "critic_score": None,
        "next_agent": None,
        "iteration": 0,
        "max_iterations": 2,
        "error": None,
    }


async def run(workflow, query: str):
    """执行一次问答"""
    state = create_initial_state(query)
    result = await workflow.ainvoke(state)
    return result

def main():
    print(f"Paper Agent 已启动 (LLM: {LLM_MODEL})")
    print("输入论文名搜索入库，或直接提问。输入 q 退出。\n")

    workflow = init_components()

    while True:
        query = input("你: ").strip()
        if not query or query.lower() == "q":
            print("再见！")
            break

        state = create_initial_state(query)
        result = asyncio.run(workflow.ainvoke(state))

        if result.get("answer"):
            print(f"\n助手: {result['answer']}\n")
        elif result.get("error"):
            print(f"\n提示: {result['error']}\n")
        else:
            print("\n提示: 未生成回答\n")

    if __name__ == "__main__":
        main()
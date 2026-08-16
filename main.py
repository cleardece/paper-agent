"""
Paper Agent - 主入口
初始化所有组件，启动交互式问答
"""
import asyncio
from agents.analyzer import AnalyzerAgent
from agents.critic import CriticAgent
from agents.direct_analyzer import DirectAnalyzerAgent
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
        "supervisor": SupervisorAgent(llm, mongodb_client),
        "fetcher": FetcherAgent(arxiv_api, pdf_parser, mongodb_client, embedding_service, milvus_client),
        "retriever": RetrieverAgent(embedding_service, milvus_client, mongodb_client),
        "analyzer": AnalyzerAgent(llm, mongodb_client),
        "critic": CriticAgent(llm),
        "presenter": PresenterAgent(llm, code_generator),
        "direct_analyzer": DirectAnalyzerAgent(llm, mongodb_client, embedding_service, milvus_client, pdf_parser, arxiv_api),
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
        "conversation_context": None,
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
    chat_history = []  # 保存最近对话用于上下文

    while True:
        query = input("你: ").strip()
        if not query or query.lower() == "q":
            print("再见！")
            break

        # 构建对话上下文
        context = ""
        if chat_history:
            recent = chat_history[-20:]  # 最近 10 轮（每轮 = 用户 + 助手 = 2条）
            context = "\n".join(recent)

        state = create_initial_state(query)
        state["conversation_context"] = context
        result = asyncio.run(workflow.ainvoke(state))

        # 记录对话历史
        chat_history.append(f"用户: {query[:100]}")
        answer = result.get("answer") or result.get("error") or "未生成回答"
        chat_history.append(f"助手: {answer[:100]}")

        if result.get("answer"):
            print(f"\n助手: {result['answer']}\n")
        elif result.get("error"):
            print(f"\n提示: {result['error']}\n")
        else:
            print("\n提示: 未生成回答\n")


if __name__ == "__main__":
    main()
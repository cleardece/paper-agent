#定义图状态
from typing import TypedDict,Annotated,List
from langgraph.graph.message import add_messages
class GraphState(TypedDict):
    # 输入
    paper_url: str                    # 论文链接或arXiv ID
    paper_text: str                   # 解析后的论文全文
    sections: dict                    # 按章节拆分 {title: content}

    # 中间结果
    summary: str                      # 摘要分析
    methodology: str                  # 方法论分析
    key_findings: List[str]           # 关键发现列表

    # 最终输出
    structured_output: dict           # 结构化分析结果
    presentation: str                 # 汇报文案

    # 流程控制
    current_step: str                 # 当前步骤
    error: str                        # 错误信息
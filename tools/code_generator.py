"""
Paper Agent - 代码生成工具
根据论文方法论内容生成可复现的PyTorch代码
"""

from langchain_core.messages import SystemMessage, HumanMessage


CODEGEN_SYSTEM_PROMPT = """你是AI研究代码生成专家。根据论文方法论内容，生成完整可运行的PyTorch实现。

规则：
1. 输出完整Python代码，不要省略任何部分
2. 必须包含：模型定义、数据集类、训练函数、评估函数、main入口
3. 代码注释标注对应论文的哪个部分
4. 使用标准PyTorch API，避免冷门依赖
5. 超参数使用论文中提到的值，未提到的使用合理默认值
6. 只输出代码，不要输出解释文字"""


class CodeGenerator:
    """论文代码生成器"""

    def __init__(self, llm):
        self.llm = llm

    def generate(self, paper_content: str, paper_title: str = "") -> dict:
        """
        输入论文方法论内容，生成PyTorch实现。

        Returns:
            {"code": str, "paper_title": str, "error": str|None}
        """
        if not paper_content.strip():
            return {"code": "", "paper_title": paper_title, "error": "论文内容为空"}

        user_message = f"论文标题：{paper_title}\n\n方法论内容：\n{paper_content}"

        messages = [
            SystemMessage(content=CODEGEN_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        try:
            response = self.llm.invoke(messages)
            code = self._extract_code(response.content)
            return {"code": code, "paper_title": paper_title, "error": None}
        except Exception as e:
            return {"code": "", "paper_title": paper_title, "error": str(e)}

    def _extract_code(self, raw: str) -> str:
        """从LLM输出中提取代码块"""
        if "```python" in raw:
            code = raw.split("```python", 1)[1]
            code = code.split("```", 1)[0]
        elif "```" in raw:
            code = raw.split("```", 1)[1]
            code = code.split("```", 1)[0]
        else:
            code = raw
        return code.strip()
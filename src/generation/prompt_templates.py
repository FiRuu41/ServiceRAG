"""
提示词模块
"""
from langchain_core.prompts import PromptTemplate


def get_qa_prompt() -> PromptTemplate:
    """构建一个提示模板，指导模型根据检索到的文本分块回答问题。"""
    template = """
    你是企业知识库助手。
    只能根据下面的资料回答，不要编造。
    {context}
    问题：{question}
    请用中文回答
    """
    prompt_template = PromptTemplate.from_template(template)
    return prompt_template
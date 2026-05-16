"""
生成模块（支持多轮对话）
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from src.utils.config import DEEPSEEK_MODEL, DEEPSEEK_TEMPERATURE, DEEPSEEK_BASE_URL, get_deepseek_api_key
from src.generation.prompt_templates import get_qa_prompt


def _get_model():
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=DEEPSEEK_TEMPERATURE,
        base_url=DEEPSEEK_BASE_URL,
        api_key=get_deepseek_api_key(),
    )


def generate_answer(question, retrieved_docs, history=None):
    """生成答案。
    history: list[dict]，每项 {"role": "user"|"assistant", "content": str}
    """
    context = "\n\n".join("\n" + doc.page_content for doc in retrieved_docs)

    if not history:
        # 单轮：保持原有简单链路
        chain = get_qa_prompt() | _get_model() | StrOutputParser()
        return chain.invoke({"question": question, "context": context})

    # 多轮：构建消息列表
    messages = [SystemMessage(content=(
        "你是企业知识库助手。\n"
        "请根据本次提供的资料回答用户的当前问题，可以参考之前的对话上下文，但不要编造资料外的信息。\n"
        "如果之前的对话和当前问题无关，请只依据当前资料回答。\n"
        "请用中文回答。"
    ))]
    # 注入历史
    for msg in history[-6:]:  # 只保留最近 3 轮（6 条消息）
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    # 当前问题（带本轮检索的资料）
    messages.append(HumanMessage(content=(
        f"以下是本次检索到的相关资料：\n{context}\n\n当前问题：{question}"
    )))

    response = _get_model().invoke(messages)
    return response.content



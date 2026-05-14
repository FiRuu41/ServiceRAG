"""
生成模块
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from src.utils.config import DEEPSEEK_MODEL, DEEPSEEK_TEMPERATURE, DEEPSEEK_BASE_URL, get_deepseek_api_key
from src.generation.prompt_templates import get_qa_prompt

def generate_answer(question, retrieved_docs):
    """拼接检索到的文本分块 初始化模型 构建提示模板 执行链式调用生成答案"""
    chunks = []
    for doc in retrieved_docs:
        chunks.append("\n" + doc.page_content)
    context = "\n\n".join(chunks)
    model = ChatOpenAI(
        model = DEEPSEEK_MODEL,
        temperature = DEEPSEEK_TEMPERATURE,
        base_url = DEEPSEEK_BASE_URL,
        api_key = get_deepseek_api_key()
    )
    prompt_template = get_qa_prompt()
    chain = prompt_template | model | StrOutputParser()
    answer = chain.invoke({
        "question": question,
        "context": context,
    })
    return answer


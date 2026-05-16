"""
MultiQuery 查询重写：用 LLM 生成同义问句，多路检索后合并去重，扩大召回
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from src.utils.config import (
    MULTI_QUERY_COUNT,
    MULTI_QUERY_TOP_K,
    DEEPSEEK_MODEL,
    DEEPSEEK_TEMPERATURE,
    DEEPSEEK_BASE_URL,
    get_deepseek_api_key,
)


MULTI_QUERY_PROMPT = PromptTemplate.from_template("""\
你是一个知识库助手。用户会提出一个问题，请生成 {count} 个不同角度或不同措辞的同义问题，\
帮助从文档中检索到更全面的信息。
每行一个问题，不要编号，不要多余文字。

原问题：{question}

{count} 个同义问题：""")


def _get_llm():
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=DEEPSEEK_TEMPERATURE,
        base_url=DEEPSEEK_BASE_URL,
        api_key=get_deepseek_api_key(),
    )


def generate_variant_questions(question: str) -> list:
    """用 LLM 对一个问题生成多个变体问句"""
    chain = MULTI_QUERY_PROMPT | _get_llm() | StrOutputParser()
    result = chain.invoke({"question": question, "count": MULTI_QUERY_COUNT})
    # 每行一个，过滤空行
    variants = [line.strip() for line in result.strip().split("\n") if line.strip()]
    # 去掉可能的编号前缀（如 "1. "）
    variants = [q.lstrip("0123456789.、) -") for q in variants]
    return variants


def multi_query_retrieve(question: str, vectorstore) -> list:
    """多查询检索：原问题 + LLM 变体，合并去重"""
    queries = [question] + generate_variant_questions(question)
    seen = set()
    all_docs = []
    for q in queries:
        docs = vectorstore.similarity_search(q, k=MULTI_QUERY_TOP_K)
        for doc in docs:
            # 用页面内容前 200 字符做去重指纹
            fingerprint = doc.page_content[:200]
            if fingerprint not in seen:
                seen.add(fingerprint)
                all_docs.append(doc)
    return all_docs

"""
检索模块
"""

from src.utils.config import RETRIEVAL_TOP_K

def retrieve(question:str, vectorstore) -> list:
    """根据用户的问题从向量库中检索相关文档"""
    retrieved_docs = vectorstore.similarity_search(question, k=RETRIEVAL_TOP_K)
    return retrieved_docs
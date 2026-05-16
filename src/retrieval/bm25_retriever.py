"""
BM25 关键词检索：与向量检索互补，擅长精确关键词匹配
"""
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class BM25Store:
    """BM25 索引，包装 BM25Retriever"""

    def __init__(self):
        self._retriever = None

    def build(self, documents: list[Document]):
        self._retriever = BM25Retriever.from_documents(documents)

    def search(self, query: str, k: int = 10) -> list[Document]:
        if self._retriever is None:
            return []
        return self._retriever.invoke(query)[:k]


def build_bm25_index(documents: list[Document]) -> BM25Store:
    store = BM25Store()
    store.build(documents)
    return store

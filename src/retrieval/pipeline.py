"""
统一检索管道：MultiQuery 查询重写 → 向量检索 + BM25 混合 → 重排序
"""
from src.utils.config import (
    RETRIEVAL_TOP_K,
    RERANK_ENABLED,
    RERANK_INITIAL_TOP_K,
    RERANK_FINAL_TOP_K,
    MULTI_QUERY_ENABLED,
    MULTI_QUERY_TOP_K,
)
from src.retrieval.multi_query import multi_query_retrieve
from src.retrieval.reranker import get_reranker


def retrieve_with_pipeline(question: str, vectorstore, bm25_store=None) -> list:
    """
    统一检索入口：
    1. MultiQuery 查询重写（可选）
    2. 向量检索
    3. BM25 关键词检索（可选）
    4. 合并去重
    5. CrossEncoder 重排序（可选）
    """

    # Step 1-2: 多查询向量检索，或直接向量检索
    if MULTI_QUERY_ENABLED:
        docs = multi_query_retrieve(question, vectorstore)
    else:
        k = RERANK_INITIAL_TOP_K if RERANK_ENABLED else RETRIEVAL_TOP_K
        docs = vectorstore.similarity_search(question, k=k)

    # Step 3: BM25 混合检索
    if bm25_store is not None:
        bm_k = RERANK_INITIAL_TOP_K if RERANK_ENABLED else RETRIEVAL_TOP_K
        bm25_docs = bm25_store.search(question, k=bm_k)
        seen = {doc.page_content[:200] for doc in docs}
        for d in bm25_docs:
            if d.page_content[:200] not in seen:
                seen.add(d.page_content[:200])
                docs.append(d)

    # Step 4: 重排序
    if RERANK_ENABLED and len(docs) > RERANK_FINAL_TOP_K:
        reranker = get_reranker()
        docs = reranker.rerank(question, docs)
    docs = docs[:RERANK_FINAL_TOP_K if RERANK_ENABLED else RETRIEVAL_TOP_K]

    return docs

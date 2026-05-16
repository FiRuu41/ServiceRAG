"""
重排序模块：用 CrossEncoder 对向量检索结果重新打分排序
"""
import os
from sentence_transformers import CrossEncoder
from src.utils.config import RERANK_MODEL_NAME

# 优先用 HF 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class Reranker:
    """Cross-Encoder 重排序器，加载一次可复用"""

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = CrossEncoder(RERANK_MODEL_NAME)
        return self._model

    def rerank(self, question: str, docs: list) -> list:
        """对文档列表重新排序，返回按相关性从高到低排列的文档列表"""
        if not docs:
            return docs
        pairs = [[question, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)
        # scores 是 NumPy 数组，argsort 降序排列
        ranked_indices = scores.argsort()[::-1]
        return [docs[i] for i in ranked_indices]


_reranker_instance = None


def get_reranker() -> Reranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance

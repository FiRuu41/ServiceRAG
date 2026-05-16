"""
对比评估：基线检索 vs 重排序检索
"""
import os
import json
from datetime import datetime

from src.utils.config import PROJECT_ROOT, RETRIEVAL_TOP_K, RERANK_INITIAL_TOP_K, RERANK_FINAL_TOP_K
from src.document_loader.load_local import load_pdfs
from src.processing.splitter import split_documents
from src.processing.embed_store import load_vectorstore, create_vectorstore
from src.retrieval.base_retriever import retrieve
from src.retrieval.reranker import get_reranker
from src.generation.answer_gen import generate_answer
from src.utils.config import CHROMA_PERSIST_DIR


def check_retrieval_hit(ground_truth, retrieved_docs):
    separators = ["。", "；", "，", "、", "：", ".", ";", ","]
    clauses = [ground_truth]
    for sep in separators:
        new_clauses = []
        for c in clauses:
            new_clauses.extend(c.split(sep))
        clauses = new_clauses
    key_clauses = [c.strip() for c in clauses if len(c.strip()) > 4]
    retrieved_text = " ".join(doc.page_content for doc in retrieved_docs)
    for clause in key_clauses:
        if clause in retrieved_text:
            return True
    return False


def run(mode: str):
    """mode = "baseline" 或 "rerank" """
    test_qa = json.load(open(PROJECT_ROOT / "test_qa.json", encoding="utf-8"))

    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        vs = load_vectorstore()
    else:
        print("向量库不存在，正在重建...")
        all_docs = load_pdfs()
        split_docs = split_documents(all_docs)
        vs = create_vectorstore(split_docs)

    reranker = get_reranker() if mode == "rerank" else None
    results = []
    hit_count = 0

    for i, qa in enumerate(test_qa):
        q = qa["question"]
        gt = qa["ground_truth"]

        if mode == "baseline":
            docs = retrieve(q, vs)
        else:
            docs = vs.similarity_search(q, k=RERANK_INITIAL_TOP_K)
            docs = reranker.rerank(q, docs)
            docs = docs[:RERANK_FINAL_TOP_K]

        is_hit = check_retrieval_hit(gt, docs)
        if is_hit:
            hit_count += 1

        answer = generate_answer(q, docs)
        results.append({
            "id": i + 1,
            "question": q,
            "ground_truth": gt,
            "system_answer": answer,
            "retrieval_hit": is_hit,
        })

    summary = {
        "mode": mode,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(test_qa),
        "retrieval_hit_count": hit_count,
        "retrieval_hit_rate": f"{hit_count / len(test_qa) * 100:.1f}%",
        "details": results,
    }

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"eval_{mode}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[{mode}] 检索命中率: {summary['retrieval_hit_rate']}  ({hit_count}/{len(test_qa)})")
    return summary


if __name__ == "__main__":
    print("=" * 50)
    r1 = run("baseline")
    r2 = run("rerank")
    print("=" * 50)
    print(f"基线命中率:     {r1['retrieval_hit_rate']}")
    print(f"重排序后命中率: {r2['retrieval_hit_rate']}")

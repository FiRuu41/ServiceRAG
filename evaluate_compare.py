"""
对比评估：基线 vs 优化后（重排序 + MultiQuery + BM25）
"""
import os
import json
from datetime import datetime

from src.utils.config import PROJECT_ROOT, CHROMA_PERSIST_DIR
from src.document_loader.load_local import load_pdfs
from src.processing.splitter import split_documents
from src.processing.embed_store import load_vectorstore, create_vectorstore
from src.retrieval.base_retriever import retrieve
from src.retrieval.pipeline import retrieve_with_pipeline
from src.retrieval.bm25_retriever import build_bm25_index
from src.generation.answer_gen import generate_answer


def check_hit(ground_truth, retrieved_docs):
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


def run(mode: str, vectorstore, bm25_store=None):
    test_qa = json.load(open(PROJECT_ROOT / "test_qa.json", encoding="utf-8"))
    results = []
    hit_count = 0

    for i, qa in enumerate(test_qa):
        q, gt = qa["question"], qa["ground_truth"]

        if mode == "baseline":
            docs = retrieve(q, vectorstore)
        else:
            docs = retrieve_with_pipeline(q, vectorstore, bm25_store)

        is_hit = check_hit(gt, docs)
        if is_hit:
            hit_count += 1

        answer = generate_answer(q, docs)
        results.append({
            "id": i + 1, "question": q, "ground_truth": gt,
            "system_answer": answer, "retrieval_hit": is_hit,
        })

    summary = {
        "mode": mode, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

    print(f"[{mode}] 命中率: {summary['retrieval_hit_rate']} ({hit_count}/{len(test_qa)})")
    return summary


if __name__ == "__main__":
    print("加载向量库...")
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        vs = load_vectorstore()
    else:
        print("重建向量库...")
        vs = create_vectorstore(split_documents(load_pdfs()))

    print("构建 BM25 索引...")
    all_docs = load_pdfs()
    bm25 = build_bm25_index(split_documents(all_docs))

    print("=" * 50)
    r1 = run("baseline", vs)
    r2 = run("optimized", vs, bm25)
    print("=" * 50)
    print(f"基线:     {r1['retrieval_hit_rate']}")
    print(f"优化后:   {r2['retrieval_hit_rate']}")

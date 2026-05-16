"""
评测基线模型的性能
"""
import os
from src.utils.config import PROJECT_ROOT, RETRIEVAL_TOP_K, CHROMA_PERSIST_DIR
from src.document_loader.load_local import load_pdfs
from src.processing.splitter import split_documents
from src.processing.embed_store import load_vectorstore, create_vectorstore
from src.retrieval.base_retriever import retrieve
from src.generation.answer_gen import generate_answer
import json
from datetime import datetime

def load_test_questions() -> list:
    """从 test_questions.json 文件中加载测试问题列表"""
    path = PROJECT_ROOT / "test_qa.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_retrieval_hit(ground_truth,retrieved_docs) -> bool:
    """检查检索到的文档片段中是否包含标准答案的关键信息"""
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

def evaluate():
    """主评估流程，对每个测试问题跑一遍 RAG 系统，记录结果并输出报告"""
    test_qa = load_test_questions()
    print(f"共加载 {len(test_qa)} 个测试问题\n")
    print("正在加载向量库...")
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        vectorstore = load_vectorstore()
    else:
        print("向量库不存在，正在重建...")
        all_docs = load_pdfs()
        split_docs = split_documents(all_docs)
        vectorstore = create_vectorstore(split_docs)
    print("向量库加载完成\n")
    results = []
    hit_count = 0
    for i, qa in enumerate(test_qa):
        question = qa["question"]
        ground_truth = qa["ground_truth"]

        print(f"{'=' * 60}")
        print(f"[{i + 1}/{len(test_qa)}] 问题: {question}")

        retrieved_docs = retrieve(question, vectorstore)
        is_hit = check_retrieval_hit(ground_truth, retrieved_docs)
        if is_hit:
            hit_count += 1
        print(f"检索命中: {'✓' if is_hit else '✗'}")

        answer = generate_answer(question, retrieved_docs)
        print(f"系统答案: {answer}")
        print(f"标准答案: {ground_truth}")

        while True:
            score_input = input("答案相关性评分 (1=相关, 0=不相关, s=跳过): ").strip().lower()
            if score_input in ("1", "0", "s"):
                break
            print("请输入 1、0 或 s")

        relevance_score = None if score_input == "s" else int(score_input)

        results.append({
            "id": i + 1,
            "question": question,
            "ground_truth": ground_truth,
            "system_answer": answer,
            "retrieval_hit": is_hit,
            "relevance_score": relevance_score,
            "retrieved_sources": [doc.page_content[:100] for doc in retrieved_docs],
        })
        print()

    answered = [r for r in results if r["relevance_score"] is not None]
    relevant_count = sum(1 for r in answered if r["relevance_score"] == 1)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(test_qa),
        "retrieval_hit_count": hit_count,
        "retrieval_hit_rate": f"{hit_count / len(test_qa) * 100:.1f}%",
        "answered_count": len(answered),
        "relevant_count": relevant_count,
        "answer_relevance_rate": f"{relevant_count / len(answered) * 100:.1f}%" if answered else "N/A",
        "top_k": RETRIEVAL_TOP_K,
        "details": results,
    }

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    output_path = os.path.join(log_dir, "baseline_eval.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("基线评估报告")
    print("=" * 60)
    print(f"总问题数:       {summary['total_questions']}")
    print(f"检索命中率:     {summary['retrieval_hit_rate']} ({hit_count}/{len(test_qa)})")
    print(f"已评分问题数:   {summary['answered_count']}")
    print(f"答案相关率:     {summary['answer_relevance_rate']}")
    print(f"Top-K:          {RETRIEVAL_TOP_K}")
    print(f"\n详细结果已保存至: {output_path}")

if __name__ == "__main__":
    evaluate()

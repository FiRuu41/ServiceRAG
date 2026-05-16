"""
RAGAS 自动评估：用 LLM 自动打分代替人工评分
评估维度：
- faithfulness（忠实度）：答案是否忠于检索资料，没编造
- answer_relevancy（相关性）：答案是否切中问题
- context_precision（上下文精确度）：检索到的内容相关性
- context_recall（上下文召回）：是否检索到了答案需要的信息
"""
import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import (
    PROJECT_ROOT, CHROMA_PERSIST_DIR,
    DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, get_deepseek_api_key,
    EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_KWARGS, ENCODE_KWARGS,
)
from src.document_loader.load_local import load_pdfs
from src.processing.splitter import split_documents
from src.processing.embed_store import load_vectorstore, create_vectorstore
from src.retrieval.pipeline import retrieve_with_pipeline
from src.retrieval.bm25_retriever import build_bm25_index
from src.generation.answer_gen import generate_answer


def build_evaluation_dataset(test_qa, vs, bm25):
    """跑一遍 RAG 系统，构建 RAGAS 需要的数据集格式"""
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for i, qa in enumerate(test_qa):
        q, gt = qa["question"], qa["ground_truth"]
        print(f"[{i+1}/{len(test_qa)}] {q[:30]}...")
        retrieved = retrieve_with_pipeline(q, vs, bm25)
        answer = generate_answer(q, retrieved)
        rows["question"].append(q)
        rows["answer"].append(answer)
        rows["contexts"].append([doc.page_content for doc in retrieved])
        rows["ground_truth"].append(gt)
    return Dataset.from_dict(rows)


def main():
    print("加载向量库...")
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        vs = load_vectorstore()
    else:
        vs = create_vectorstore(split_documents(load_pdfs()))
    bm25 = build_bm25_index(split_documents(load_pdfs()))

    test_qa = json.load(open(PROJECT_ROOT / "test_qa.json", encoding="utf-8"))

    print(f"构建评估数据集（{len(test_qa)} 题）...")
    dataset = build_evaluation_dataset(test_qa, vs, bm25)

    # 用 DeepSeek + 本地 embedding 做评估
    llm = LangchainLLMWrapper(ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=0,
        base_url=DEEPSEEK_BASE_URL,
        api_key=get_deepseek_api_key(),
    ))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_PATH),
        model_kwargs=EMBEDDING_MODEL_KWARGS,
        encode_kwargs=ENCODE_KWARGS,
    ))

    print("RAGAS 评估中（每题需调用 LLM 多次，可能较慢）...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n" + "=" * 50)
    print("RAGAS 评估结果")
    print("=" * 50)
    print(result)

    # 保存结果
    out_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(out_dir, exist_ok=True)
    df = result.to_pandas()
    df.to_csv(os.path.join(out_dir, "ragas_eval.csv"), index=False, encoding="utf-8-sig")

    summary = {k: float(v) for k, v in result._repr_dict.items()} if hasattr(result, "_repr_dict") else {}
    with open(os.path.join(out_dir, "ragas_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"scores": summary, "n_questions": len(test_qa)}, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果: {out_dir}/ragas_eval.csv")


if __name__ == "__main__":
    main()

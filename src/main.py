"""
主函数
"""
import os
from src.document_loader.load_local import load_pdfs
from src.generation.answer_gen import generate_answer
from src.processing.embed_store import load_vectorstore, create_vectorstore
from src.processing.splitter import split_documents
from src.retrieval.base_retriever import retrieve
from src.utils.config import CHROMA_PERSIST_DIR

if __name__ == "__main__":
    all_docs = load_pdfs()
    split_docs = split_documents(all_docs)
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        vectorstore = load_vectorstore()
    else:
        vectorstore = create_vectorstore(split_docs)
    question = input("请输入你的问题：")
    retrieved_docs = retrieve(question, vectorstore)
    print(generate_answer(question, retrieved_docs))












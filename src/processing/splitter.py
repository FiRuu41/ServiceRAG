"""
文本切分模块
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(pdf_documents: list) -> list:
    """把 Document 列表切成更小的 chunk，返回切分后的列表。"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(pdf_documents)
    print(f"切分完成: {len(pdf_documents)} 个文档 -> {len(split_docs)} 个 chunk")
    return split_docs


if __name__ == "__main__":
    from src.document_loader.load_local import load_pdfs
    docs = load_pdfs()
    chunks = split_documents(docs)
    print(f"\n第一个 chunk 预览:\n{chunks[0].page_content[:200]}")

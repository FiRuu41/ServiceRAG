"""
文档加载模块。
"""
from langchain_community.document_loaders import PyPDFLoader
from src.utils.config import DATA_DIR

def load_pdfs() -> list:
    """加载 data/raw 目录下的所有 PDF，返回 Document 列表。"""
    all_docs = []

    # 遍历目录下所有 .pdf 文件
    for pdf_file in DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        all_docs.extend(docs)
        print(f"已加载: {pdf_file.name} ({len(docs)} 页)")

    print(f"\n共加载 {len(all_docs)} 个文档块")
    return all_docs

if __name__ == "__main__":
    docs = load_pdfs()


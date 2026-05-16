"""
文档加载模块。支持 PDF / DOCX / TXT / Markdown。
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from src.utils.config import DATA_DIR


def _load_single_file(file_path: Path) -> list[Document]:
    """根据扩展名选择合适的 Loader 加载单个文件"""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    if ext == ".docx":
        return Docx2txtLoader(str(file_path)).load()
    if ext in (".txt", ".md"):
        return TextLoader(str(file_path), encoding="utf-8").load()
    return []


def load_pdfs() -> list:
    """加载 data/raw 目录下的所有支持格式（PDF / DOCX / TXT / MD），返回 Document 列表。

    保留旧函数名以兼容现有调用方。
    """
    all_docs = []
    supported_exts = {".pdf", ".docx", ".txt", ".md"}

    for file_path in DATA_DIR.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() not in supported_exts:
            continue
        try:
            docs = _load_single_file(file_path)
            all_docs.extend(docs)
            print(f"已加载: {file_path.name} ({len(docs)} 段)")
        except Exception as e:
            print(f"跳过 {file_path.name}：{e}")

    print(f"\n共加载 {len(all_docs)} 个文档块")
    return all_docs


# 保留对外别名
load_documents = load_pdfs


if __name__ == "__main__":
    docs = load_pdfs()

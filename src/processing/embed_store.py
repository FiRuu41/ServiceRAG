"""
向量化模块
"""
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.config import EMBEDDING_MODEL_KWARGS,ENCODE_KWARGS
from src.utils.config import CHROMA_PERSIST_DIR,EMBEDDING_MODEL_PATH
from langchain_chroma import Chroma

def create_vectorstore(split_docs:list) -> Chroma:
    """把切分后的文档向量化并存入 Chroma"""
    embedding_model =  HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_PATH),
        model_kwargs=EMBEDDING_MODEL_KWARGS,
        encode_kwargs=ENCODE_KWARGS,
    )
    vectorstore = Chroma.from_documents(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding=embedding_model,
        documents=split_docs
    )
    return vectorstore

def load_vectorstore() -> Chroma:
    """加载已经存在的 Chroma 向量库"""
    embedding_model = HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_PATH),
        model_kwargs=EMBEDDING_MODEL_KWARGS,
        encode_kwargs=ENCODE_KWARGS,
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_model,
    )
    return vectorstore
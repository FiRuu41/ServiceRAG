"""
统一配置文件。
所有可调整的参数集中在这里，其他模块从这里导入
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 自动读取项目根目录下的 .env 文件
load_dotenv(PROJECT_ROOT / ".env")

# 路径配置
DATA_DIR = PROJECT_ROOT / "data" / "raw"
EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models" / "BAAI" / "bge-base-zh-v1.5"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"

# 文本切分参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Embedding 模型参数
EMBEDDING_MODEL_KWARGS = {"device": "cpu"}
ENCODE_KWARGS = {"normalize_embeddings": True}

# 检索参数
RETRIEVAL_TOP_K = 4

# DeepSeek API 配置
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_TEMPERATURE = 0.5

"""
从环境变量读取 DeepSeek API Key，没有则报错。
"""
def get_deepseek_api_key() -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请先在 .env 文件中设置 DEEPSEEK_API_KEY")
    return api_key


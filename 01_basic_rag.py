# 下面导入的是用于：
# - 从 PDF 加载文档的 Loader
# - 将长文本切分为更小的 chunk（便于向量化与检索）
# - 用于生成文本向量的 Embedding 模型
# - Chroma 向量数据库，用于持久化和检索向量
# - ChatOpenAI（这里是 Deepseek 的兼容接口）用于生成回答
# - 提示模板与输出解析器，用于构建请求和解析模型输出
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
# 加载 PDF 文档
pdf_loader = PyPDFLoader("data/raw/员工手册.pdf")
docs1 = pdf_loader.load()
loader2 = PyPDFLoader("data/raw/网络与信息化常见问题指导手册.pdf")
docs2 = loader2.load()
pdf_documents = docs1 + docs2

# 文本分块
# 目的：把长文本拆成更小的 chunk 以便嵌入（embedding）与检索。
# chunk_size 控制每个块的大致字符数；chunk_overlap 控制相邻块之间的重叠（有助于保持上下文连续性）。
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
split_docs = text_splitter.split_documents(pdf_documents)

# 嵌入模型设置
# 说明：选择一个支持中文的 embedding 模型（这里举例 bge-base-zh-v1.5）。
# model_kwargs 中设置 device（'cpu' 或 'cuda'），encode_kwargs 用于额外选项（如标准化 embeddings）。
embedding_model_name = "models/BAAI/bge-base-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 向量库持久化目录
# 目的：将向量化后的数据持久化到磁盘，避免每次都重新计算 embedding。
PERSIST_DIR = "./chroma_db"

# 判断并加载 / 创建向量库
# 逻辑：
# - 如果持久化目录存在并且非空，直接加载已有的 Chroma 库（更快）。
# - 否则，使用 split_docs 创建库并持久化到 PERSIST_DIR。
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    )
else:
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR
    )

# 检索与问题设置
# 说明：这里示例问题是 "工作时间在什么时间段"，k=4 表示返回最相似的 4 段内容供模型参考。
question = input("请输入你的问题：")
retrieved_docs = vectorstore.similarity_search(question, k=4)

# 安全地读取 API Key
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请先在环境变量中设置 DEEPSEEK_API_KEY")
# 模型客户端设置
# 说明：这里使用 ChatOpenAI 的兼容接口，model、temperature、base_url 等根据服务提供方说明配置。
model = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.2,
    base_url="https://api.deepseek.com",
    api_key=api_key,
)

# 格式化检索到的上下文
# 说明：把检索到的文档按来源和页码拼成一个大的 context，供 LLM 参考回答。
# doc.metadata.get('page_label') 常见用于表示 PDF 的页码信息（不同 loader 名称可能不同）。
chunks = []
for i, doc in enumerate(retrieved_docs):
    chunks.append("\n" + doc.page_content)
context = "\n\n".join(chunks)

# 构建提示模板
# 说明：PromptTemplate 用来把 context 和 question 插入到模板中。
# 模板中明确要求模型只能根据给定资料回答并在最后写出引用页码，这有助于减少模型“编造”。
template = """
你是企业知识库助手。
只能根据下面的资料回答，不要编造。
{context}
问题：{question}
请用中文回答
"""
prompt_template = PromptTemplate.from_template(template)

# 链式调用与输出解析
# 这里用到了 prompt_template | model | StrOutputParser() 的组合（LangChain 的新式管道语法）。
# StrOutputParser 将模型输出作为字符串解析，便于直接打印或后续处理。
chain = prompt_template | model | StrOutputParser()

# 调用链并打印答案
# invoke 接受一个字典，键要与模板中使用的占位符一致（这里是 context 和 question）。
answer = chain.invoke({
    "context": context,
    "question": question
})
print(answer)

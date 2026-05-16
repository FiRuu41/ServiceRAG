# ServiceRAG — 企业服务台知识库智能问答系统

基于 RAG（检索增强生成）的企业内部知识库问答系统。将 HR 制度、法律法规、IT 手册等内部文档接入系统后，员工通过 ChatGPT 式聊天界面提问，系统自动检索相关文档并由大语言模型生成带**引用来源**的答案，避免幻觉。支持**多轮对话**与 **RAGAS 自动评估**。

## 技术架构

```
文档(PDF/DOCX/TXT/MD) → 切分 → Embedding (BGE-Chinese) → Chroma 向量库
                                                              ↓
用户提问 → 向量检索 + BM25 混合 → CrossEncoder 重排序 → LLM 生成答案（带历史上下文）
```

| 层级 | 技术选型 |
|------|---------|
| 语言 | Python 3.9+ |
| 框架 | LangChain |
| 文档加载 | PyPDFLoader · Docx2txtLoader · TextLoader（支持 PDF / DOCX / TXT / MD） |
| 文本切分 | RecursiveCharacterTextSplitter |
| 嵌入模型 | BAAI/bge-base-zh-v1.5（本地） |
| 向量库 | Chroma（本地持久化） |
| 检索优化 | BM25 关键词检索 + CrossEncoder 重排序（bge-reranker-v2-m3） |
| 查询重写 | MultiQueryRetriever（可选，默认关闭） |
| 生成模型 | DeepSeek-V4（API） |
| 界面 | Streamlit（ChatGPT 式聊天） |
| 评估 | 基线人工评分 + RAGAS 自动评估 |

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=你的密钥

# 3. 下载本地模型（任选一个 git 镜像）
cd models/BAAI
git lfs install
git clone https://www.modelscope.cn/BAAI/bge-reranker-v2-m3.git
git clone https://www.modelscope.cn/BAAI/bge-base-zh-v1.5.git
cd ../..

# 4. 放入文档
# 将 PDF / DOCX / TXT / MD 文件放入 data/raw/ 目录

# 5. 启动界面
streamlit run src/ui/app.py
```

## 项目结构

```
ServiceRAG/
├── src/
│   ├── document_loader/    # 文档加载（PDF / DOCX / TXT / MD）
│   ├── processing/         # 文本切分、向量化
│   ├── retrieval/          # 检索管道（向量 / BM25 / 重排序 / MultiQuery）
│   ├── generation/         # LLM 答案生成（含多轮历史）
│   ├── evaluation/         # 基线评估 + RAGAS 自动评估
│   ├── ui/                 # Streamlit 聊天界面
│   └── utils/              # 配置、日志
├── data/raw/               # 原始文档（gitignore）
├── models/                 # 本地模型（gitignore）
├── logs/                   # 评估结果与问答日志（gitignore）
├── test_qa.json            # 20 题测试问答对
├── evaluate_compare.py     # 基线 vs 优化对比评估
└── README.md
```

## 评估结果

| 方案 | 检索命中率 |
|------|-----------|
| 基线（向量检索 Top-4） | 70.0% |
| 优化后（CrossEncoder 重排序 + BM25 混合） | **85.0%** |

```bash
# 检索方案对比评估
python evaluate_compare.py

# RAGAS 自动评估（4 维分数）
python -m src.evaluation.evaluate_ragas
```

RAGAS 评估 4 个维度：
- **faithfulness** 忠实度：答案是否忠于检索资料，避免幻觉
- **answer_relevancy** 相关性：答案是否切中问题
- **context_precision** 上下文精确度：检索质量
- **context_recall** 上下文召回：是否找到答案所需信息

结果保存在 `logs/ragas_eval.csv`。

## 主要功能

- ✅ 多格式文档加载（PDF / DOCX / TXT / MD）
- ✅ 中文 Embedding + 持久化向量库
- ✅ BM25 + 向量混合检索
- ✅ CrossEncoder 重排序（bge-reranker-v2-m3）
- ✅ MultiQuery 查询重写（可配置开关）
- ✅ ChatGPT 式聊天界面，引用来源溯源
- ✅ 多轮对话上下文记忆（最近 3 轮）
- ✅ 问答日志 CSV 自动归档
- ✅ 20 题测试集 + 基线/优化对比评估
- ✅ RAGAS LLM 自动评估

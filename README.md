# ServiceRAG — 企业服务台知识库智能问答系统

基于 RAG（检索增强生成）的企业内部知识库问答系统。上传企业 PDF 文档后，输入问题即可获得带引用来源的智能答案。

## 技术架构

```
文档 → 切分 → Embedding (BGE-Chinese) → Chroma 向量库
                                            ↓
用户提问 → 向量检索 + BM25 混合 → CrossEncoder 重排序 → LLM 生成答案
```

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.9+ |
| 文档处理 | PyPDFLoader + RecursiveCharacterTextSplitter |
| 嵌入模型 | BAAI/bge-base-zh-v1.5（本地） |
| 向量库 | Chroma（本地） |
| 检索优化 | BM25 + CrossEncoder(bge-reranker-v2-m3) |
| 生成模型 | DeepSeek-V4（API） |
| 界面 | Streamlit |

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=你的密钥

# 3. 下载模型
cd models/BAAI
git lfs install
git clone https://www.modelscope.cn/BAAI/bge-reranker-v2-m3.git
# bge-base-zh-v1.5 同理

# 4. 放入文档
# 将 PDF 文件放入 data/raw/ 目录

# 5. 启动界面
streamlit run src/ui/app.py
```

## 项目结构

```
ServiceRAG/
├── src/
│   ├── document_loader/    # 文档加载
│   ├── processing/         # 文本切分、向量化
│   ├── retrieval/          # 检索管道（向量/BM25/重排序）
│   ├── generation/         # LLM 答案生成
│   ├── evaluation/         # 基线评估
│   ├── ui/                 # Streamlit 界面
│   └── utils/              # 配置、日志
├── data/raw/               # 原始文档（不提交 git）
├── models/                 # 本地模型（不提交 git）
├── test_qa.json            # 测试问答对
└── evaluate_compare.py     # 检索优化对比评估
```

## 评估结果

| 方案 | 检索命中率 |
|------|-----------|
| 基线（向量 Top-4） | 70.0% |
| 优化后（重排序 + BM25，不含 MultiQuery） | **85.0%** |

```bash
# 运行对比评估
python evaluate_compare.py
```

## 路线图

- [x] 基础 RAG 链路（文档→检索→生成）
- [x] 测试集 + 基线评估
- [x] CrossEncoder 重排序
- [x] BM25 混合检索
- [x] MultiQuery 查询重写
- [x] Streamlit 界面（ChatGPT 式聊天）
- [x] 多轮对话支持
- [x] RAGAS 自动评估

## RAGAS 自动评估

支持用 LLM 自动打分代替人工评分，评估四个维度：

```bash
python -m src.evaluation.evaluate_ragas
```

- **faithfulness** 忠实度：答案是否忠于检索资料
- **answer_relevancy** 相关性：答案是否切中问题
- **context_precision** 上下文精确度：检索质量
- **context_recall** 上下文召回：是否找到答案所需信息

结果保存在 `logs/ragas_eval.csv`

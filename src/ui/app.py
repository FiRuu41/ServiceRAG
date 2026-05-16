"""
ServiceRAG — 企业服务台知识库问答系统
运行方式：streamlit run src/ui/app.py
"""
import os
import sys
import time
import csv
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from src.utils.config import PROJECT_ROOT, CHROMA_PERSIST_DIR
from src.document_loader.load_local import load_pdfs
from src.processing.splitter import split_documents
from src.processing.embed_store import load_vectorstore, create_vectorstore
from src.retrieval.pipeline import retrieve_with_pipeline
from src.retrieval.bm25_retriever import build_bm25_index
from src.generation.answer_gen import generate_answer

# ── 页面配置 ──
st.set_page_config(
    page_title="ServiceRAG · 企业知识库问答",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 自定义样式 ──
st.markdown("""
<style>
    /* 隐藏 Streamlit 默认装饰 */
    #MainMenu, footer, header {visibility: hidden;}

    /* 主容器 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 960px;
    }

    /* 标题 */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #1a1a1a;
        margin-bottom: 0.25rem;
    }
    .main-subtitle {
        font-size: 0.95rem;
        color: #888;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* 答案卡片 */
    .answer-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-top: 1rem;
        border: 1px solid #e8e8ea;
        line-height: 1.8;
        font-size: 1rem;
    }
    .answer-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #999;
        margin-bottom: 0.5rem;
    }

    /* 指标栏 */
    .metrics-row {
        display: flex;
        gap: 2rem;
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
    .metric-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.85rem;
        color: #888;
    }
    .metric-value {
        font-weight: 600;
        color: #1a1a1a;
    }

    /* 输入区 */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1.5px solid #e0e0e0;
        padding: 0.7rem 1rem;
        font-size: 1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #333;
        box-shadow: none;
    }

    /* 按钮 */
    .stButton > button {
        border-radius: 10px;
        background: #1a1a1a;
        color: white;
        font-weight: 600;
        padding: 0.5rem 2rem;
        border: none;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #333;
    }
</style>
""", unsafe_allow_html=True)

# ── 系统初始化 ──
@st.cache_resource(show_spinner=False)
def init_system():
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        vs = load_vectorstore()
    else:
        vs = create_vectorstore(split_documents(load_pdfs()))
    docs = split_documents(load_pdfs())
    bm25 = build_bm25_index(docs)
    return vs, bm25


def log_qa(question, answer, sources, elapsed):
    p = os.path.join(PROJECT_ROOT, "logs", "qa_history.csv")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    exists = os.path.exists(p)
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "question", "answer", "sources", "time_taken"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     question, answer, " | ".join(sources), f"{elapsed:.1f}s"])

# ── 页面渲染 ──
st.markdown('<div class="main-title">知识库问答</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">基于内部文档的智能检索与生成 · 所有答案均溯源至原始资料</div>',
            unsafe_allow_html=True)

vs, bm25 = init_system()

col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        "问题",
        placeholder="输入你的问题，例如：员工试用期是多长时间？",
        label_visibility="collapsed",
    )
with col_btn:
    submitted = st.button("提问", type="primary", use_container_width=True)

if submitted and question.strip():
    t0 = time.time()
    try:
        retrieved = retrieve_with_pipeline(question, vs, bm25)
        answer = generate_answer(question, retrieved)
        elapsed = time.time() - t0

        # 答案卡片
        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-label">回答</div>
            {answer}
        </div>
        """, unsafe_allow_html=True)

        # 指标栏
        cols = st.columns(3)
        cols[0].metric("检索来源", f"{len(retrieved)} 条")
        cols[1].metric("处理耗时", f"{elapsed:.1f}s")
        cols[2].metric("文档范围", "2 份")

        # 引用来源
        with st.expander("查看引用来源"):
            for i, doc in enumerate(retrieved):
                st.caption(f"来源 {i+1}")
                st.text(doc.page_content[:300])

        log_qa(question, answer, [d.page_content[:80] for d in retrieved], elapsed)

    except Exception as e:
        st.error(f"处理出错：{e}")

# 页脚
st.markdown("---")
st.caption("ServiceRAG · 复星旅文集团")

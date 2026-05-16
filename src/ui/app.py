"""
ServiceRAG — 企业知识库智能问答系统
运行方式：streamlit run src/ui/app.py
"""
import os
import sys
import time
import csv
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st

st.set_page_config(
    page_title="ServiceRAG · 企业知识库问答",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── 样式 ──
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding-top: 1.5rem; padding-bottom: 6rem; max-width: 820px; }

    /* 标题区 */
    .app-header {
        text-align: center;
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid #f0f0f0;
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 1.5rem; font-weight: 700; color: #1a1a1a; letter-spacing: -0.02em;
    }
    .app-subtitle { font-size: 0.85rem; color: #999; margin-top: 0.25rem; }

    /* 引用来源样式 */
    .source-chip {
        display: inline-block;
        background: #f1f3f5;
        border: 1px solid #e8eaec;
        border-radius: 6px;
        padding: 0.3rem 0.7rem;
        font-size: 0.78rem;
        color: #555;
        margin: 0.25rem 0.3rem 0 0;
    }
    .source-detail {
        background: #fafbfc;
        border-left: 3px solid #1a1a1a;
        padding: 0.6rem 0.9rem;
        font-size: 0.82rem;
        color: #555;
        line-height: 1.6;
        margin-top: 0.5rem;
        border-radius: 0 6px 6px 0;
    }

    .meta-line {
        font-size: 0.75rem; color: #999;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── 系统初始化 ──
def init_system():
    if "system_ready" not in st.session_state:
        from src.utils.config import CHROMA_PERSIST_DIR
        from src.document_loader.load_local import load_pdfs
        from src.processing.splitter import split_documents
        from src.processing.embed_store import load_vectorstore, create_vectorstore
        from src.retrieval.bm25_retriever import build_bm25_index

        with st.status("正在初始化系统...", expanded=True) as status:
            st.write("加载向量库...")
            if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
                vs = load_vectorstore()
            else:
                vs = create_vectorstore(split_documents(load_pdfs()))
            st.write("构建 BM25 索引...")
            docs = split_documents(load_pdfs())
            bm25 = build_bm25_index(docs)
            st.write("✓ 就绪")
            status.update(label="系统就绪", state="complete", expanded=False)

        st.session_state.vs = vs
        st.session_state.bm25 = bm25
        st.session_state.system_ready = True
        st.session_state.messages = []
    return st.session_state.vs, st.session_state.bm25


def log_qa(question, answer, sources, elapsed):
    from src.utils.config import PROJECT_ROOT
    p = os.path.join(PROJECT_ROOT, "logs", "qa_history.csv")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    exists = os.path.exists(p)
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "question", "answer", "sources", "time_taken"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     question, answer, " | ".join(sources), f"{elapsed:.1f}s"])


# ── 侧边栏：历史 + 操作 ──
with st.sidebar:
    st.markdown("### 📚 ServiceRAG")
    st.caption("企业内部知识库智能问答")
    st.markdown("---")

    if st.button("➕ 新对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("**对话历史**")
    if "messages" in st.session_state and st.session_state.messages:
        user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
        for i, m in enumerate(user_msgs):
            st.caption(f"• {m['content'][:30]}{'...' if len(m['content']) > 30 else ''}")
    else:
        st.caption("_暂无对话_")

    st.markdown("---")
    st.caption("**数据源**")
    st.caption("• 员工手册")
    st.caption("• 个人信息保护法")


# ── 主页面 ──
st.markdown("""
<div class="app-header">
    <div class="app-title">知识库问答</div>
    <div class="app-subtitle">所有答案均溯源至内部文档</div>
</div>
""", unsafe_allow_html=True)

vs, bm25 = init_system()

# 初始化对话
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "◈"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"📎 引用来源（{len(msg['sources'])}条）"):
                for i, src in enumerate(msg["sources"]):
                    st.markdown(f'<div class="source-detail"><b>来源 {i+1}</b><br>{src}</div>',
                                unsafe_allow_html=True)
            st.markdown(f'<div class="meta-line">⏱ {msg.get("elapsed", "?")}s</div>',
                        unsafe_allow_html=True)

# 聊天输入框（固定底部）
prompt = st.chat_input("输入你的问题，例如：员工试用期是多长时间？")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("_正在检索文档..._")

        t0 = time.time()
        try:
            from src.retrieval.pipeline import retrieve_with_pipeline
            from src.generation.answer_gen import generate_answer

            retrieved = retrieve_with_pipeline(prompt, vs, bm25)
            placeholder.markdown("_正在生成答案..._")
            # 传入历史（不含当前用户消息和系统占位）
            history = st.session_state.messages[:-1]
            answer = generate_answer(prompt, retrieved, history=history)
            elapsed = time.time() - t0

            placeholder.markdown(answer)

            sources_full = [doc.page_content[:400] for doc in retrieved]
            with st.expander(f"📎 引用来源（{len(retrieved)}条）"):
                for i, src in enumerate(sources_full):
                    st.markdown(f'<div class="source-detail"><b>来源 {i+1}</b><br>{src}</div>',
                                unsafe_allow_html=True)

            st.markdown(f'<div class="meta-line">⏱ {elapsed:.1f}s</div>',
                        unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_full,
                "elapsed": f"{elapsed:.1f}",
            })

            log_qa(prompt, answer, [s[:80] for s in sources_full], elapsed)

        except Exception as e:
            placeholder.error(f"处理出错：{e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ 处理出错：{e}",
            })

"""
Streamlit 前端 —— 论文上传、解析状态追踪、问答交互。

启动命令：streamlit run ui/streamlit_app.py --server.port 8501

页面布局：
  - 左侧边栏：PDF 上传 + 已入库论文列表（可勾选作为问答上下文）
  - 主区域：聊天式问答界面，回答附带来源引用的折叠块
"""

import time

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Multimodal Research Agent", page_icon="📄", layout="wide")

# ── 初始化 session_state ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = set()  # 已成功上传的文件名集合，防止重复上传


# ── 工具函数 ──────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs) -> dict:
    """统一的 API 调用封装，超时设为 120 秒（问答可能较慢）。"""
    url = f"{API_BASE}{path}"
    resp = httpx.request(method, url, timeout=120, **kwargs)
    return resp.json()


# ── 侧边栏：上传 PDF + 论文列表 ──────────────────────────────────────────

with st.sidebar:
    st.header("Papers")

    uploaded = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    # 只上传尚未处理过的新文件（用文件名+大小做唯一标识）
    if uploaded:
        new_files = [
            f for f in uploaded
            if f"{f.name}_{f.size}" not in st.session_state["uploaded_files"]
        ]
        if new_files:
            for f in new_files:
                file_key = f"{f.name}_{f.size}"
                with st.status(f"Uploading {f.name}...") as upload_status:
                    resp = api(
                        "POST", "/upload",
                        files={"file": (f.name, f.getvalue(), "application/pdf")},
                    )
                    if resp.get("success"):
                        d = resp["data"]
                        st.session_state["uploaded_files"].add(file_key)
                        upload_status.update(label=f"Uploaded {f.name}", state="complete")

                        # 轮询解析进度，直到 ready 或 failed
                        progress = st.empty()
                        while True:
                            status_data = api("GET", f"/status/{d['task_id']}")
                            if not status_data.get("success"):
                                progress.error(f"Status check failed: {status_data.get('error')}")
                                break
                            info = status_data["data"]
                            status = info["status"]
                            progress.info(f"**{status}** — {info['progress_msg']}")
                            if status == "ready":
                                progress.success("Paper indexed and ready for Q&A!")
                                break
                            elif status == "failed":
                                progress.error(f"Processing failed: {info['progress_msg']}")
                                break
                            time.sleep(2)
                    else:
                        upload_status.update(label=f"Failed: {resp.get('error')}", state="error")

            st.rerun()

    st.divider()

    # 获取已入库论文列表，显示为可勾选项
    papers_resp = api("GET", "/papers")
    papers = papers_resp.get("data", []) if papers_resp.get("success") else []

    if papers:
        selected_ids = []
        for p in papers:
            status_icon = {"ready": "✅", "failed": "❌", "pending": "⏳"}.get(p["status"], "🔄")
            if st.checkbox(
                f"{status_icon} {p['title']} ({p['page_count']}p)",
                key=p["paper_id"],
                value=p["status"] == "ready",
            ):
                selected_ids.append(p["paper_id"])
        st.session_state["selected_ids"] = selected_ids
    else:
        st.caption("No papers uploaded yet.")
        st.session_state["selected_ids"] = []

# ── 主区域：聊天式问答 ──────────────────────────────────────────────────

st.title("Multimodal Research Agent")

# 渲染历史消息
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- Paper `{s['paper_id']}`, Page {s['page_num']}" +
                             (f", Section: {s['section']}" if s.get("section") else ""))

query = st.chat_input("Ask a question about your papers...")

if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    selected = st.session_state.get("selected_ids", [])
    ready_papers = [p for p in papers if p["paper_id"] in selected and p["status"] == "ready"]

    if not ready_papers:
        assistant_msg = "Please upload and select at least one paper (in ready state) to ask questions."
        st.session_state["messages"].append({"role": "assistant", "content": assistant_msg})
        with st.chat_message("assistant"):
            st.markdown(assistant_msg)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = api("POST", "/ask", json={
                    "paper_ids": [p["paper_id"] for p in ready_papers],
                    "query": query,
                })

            if resp.get("success"):
                data = resp["data"]
                st.markdown(data["answer"])
                sources = data.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.write(f"- Paper `{s['paper_id']}`, Page {s['page_num']}" +
                                     (f", Section: {s['section']}" if s.get("section") else ""))
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": sources,
                })
            else:
                err = resp.get("error", "Unknown error")
                st.error(f"Error: {err}")
                st.session_state["messages"].append({"role": "assistant", "content": f"Error: {err}"})

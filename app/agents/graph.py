"""
Agent 编排层 —— 用 LangGraph 构建的任务调度图。

整体流程：
  classify（意图分类）→ retrieve（向量检索）→ 按类型分发到三个 Agent 之一

三个 Agent：
  - DocAgent：普通问答和跨论文对比，基于检索到的 chunk 生成答案
  - FigureAgent：图表解释，把图片编码为 base64 发给 GPT-4o Vision
  - WriterAgent：Related Work 生成，两轮检索 + 主题聚合 + 学术风格写作

State 是一个 TypedDict，在各节点之间传递，每个节点读取需要的字段、写回更新的字段。
"""

import base64
import json
import logging
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from app.config import CHAT_MODEL, VISION_MODEL
from app.retrieval.retriever import retrieve, RetrievedChunk

logger = logging.getLogger(__name__)

client = OpenAI()


class AgentState(TypedDict, total=False):
    """
    在 LangGraph 各节点之间流转的状态对象。
    total=False 表示所有字段都是可选的，初始调用时只需提供 query 和 paper_ids。
    """
    query: str
    paper_ids: list[str]
    task_type: str                          # 路由结果："doc" / "figure" / "writer"
    retrieved_chunks: list[RetrievedChunk]  # 向量检索返回的相关片段
    figure_paths: list[str]                 # 检索结果中附带的图片路径
    final_answer: str                       # 最终回答文本
    sources: list[dict]                     # 引用来源列表


# ═══════════════════════════════════════════════════════════════════════════
# 节点 1：意图分类（Router）
# ═══════════════════════════════════════════════════════════════════════════

CLASSIFY_SYSTEM = """You are a task classifier for a research paper Q&A system.
Given the user's query, classify it into exactly one category:
- "doc": general question answering or cross-paper comparison
- "figure": the user asks to explain, describe, or analyze a figure, chart, plot, or table image
- "writer": the user asks to generate a related work section, literature review, or survey paragraph

Respond with ONLY a JSON object: {"task_type": "<category>"}"""


def classify_task(state: AgentState) -> AgentState:
    """
    用一次 LLM 调用判断用户意图。
    temperature=0 确保分类结果稳定，解析失败时默认走 doc 路径。
    """
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": state["query"]},
        ],
        temperature=0,
        max_tokens=50,
    )
    raw = resp.choices[0].message.content or '{"task_type":"doc"}'
    try:
        parsed = json.loads(raw)
        task_type = parsed.get("task_type", "doc")
    except json.JSONDecodeError:
        task_type = "doc"
    return {**state, "task_type": task_type}


# ═══════════════════════════════════════════════════════════════════════════
# 节点 2：向量检索（三个 Agent 共享）
# ═══════════════════════════════════════════════════════════════════════════

def retrieve_context(state: AgentState) -> AgentState:
    """
    调用 retriever 获取与 query 最相关的 chunk。
    同时提取结果中附带的图片路径，供 FigureAgent 使用。
    """
    chunks = retrieve(state["query"], state.get("paper_ids") or None)
    figure_paths = [c.image_path for c in chunks if c.image_path]
    return {**state, "retrieved_chunks": chunks, "figure_paths": figure_paths}


# ═══════════════════════════════════════════════════════════════════════════
# 节点 3a：DocAgent —— 普通问答
# ═══════════════════════════════════════════════════════════════════════════

DOC_SYSTEM = """You are a research assistant that answers questions based ONLY on the provided context.
Rules:
- If the context is insufficient, say so explicitly. Do NOT fabricate information.
- Cite sources as [Paper: <paper_id>, Page <page_num>] after relevant statements.
- Be concise and precise."""


def doc_agent(state: AgentState) -> AgentState:
    """
    把检索到的 chunk 拼成上下文，要求 LLM 只基于上下文回答。
    每段 chunk 前标注来源，LLM 在回答中引用时可以直接复用。
    """
    chunks = state.get("retrieved_chunks", [])
    context = "\n\n".join(
        f"[Paper: {c.paper_id}, Page {c.page_num}]\n{c.text}" for c in chunks
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": DOC_SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['query']}"},
        ],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content or ""
    sources = [
        {"paper_id": c.paper_id, "page_num": c.page_num, "section": c.section}
        for c in chunks
    ]
    return {**state, "final_answer": answer, "sources": sources}


# ═══════════════════════════════════════════════════════════════════════════
# 节点 3b：FigureAgent —— 图表解释
# ═══════════════════════════════════════════════════════════════════════════

FIGURE_SYSTEM = """You are analyzing a figure from a research paper.
Describe: 1) the type of figure (bar chart, line plot, table, diagram, etc.),
2) the main conclusion or finding, 3) notable data trends or patterns.
If text context is provided, use it to give a richer explanation."""


def figure_agent(state: AgentState) -> AgentState:
    """
    将图片转为 base64 编码，连同文本上下文一起发送给 Vision 模型。
    最多发送 3 张图片，避免 token 过多。
    """
    chunks = state.get("retrieved_chunks", [])
    text_context = "\n".join(c.text for c in chunks[:4])

    messages: list[dict] = [{"role": "system", "content": FIGURE_SYSTEM}]

    # 构建多模态消息：图片(base64) + 文字
    figure_paths = state.get("figure_paths", [])
    content_parts: list[dict] = []
    if figure_paths:
        for fp in figure_paths[:3]:
            p = Path(fp)
            if p.exists():
                b64 = base64.b64encode(p.read_bytes()).decode()
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                )

    content_parts.append(
        {"type": "text", "text": f"Context:\n{text_context}\n\nQuestion: {state['query']}"}
    )
    messages.append({"role": "user", "content": content_parts})

    resp = client.chat.completions.create(
        model=VISION_MODEL, messages=messages, temperature=0.3, max_tokens=1024
    )
    answer = resp.choices[0].message.content or ""
    sources = [
        {"paper_id": c.paper_id, "page_num": c.page_num, "section": c.section}
        for c in chunks
    ]
    return {**state, "final_answer": answer, "sources": sources}


# ═══════════════════════════════════════════════════════════════════════════
# 节点 3c：WriterAgent —— Related Work 生成
# ═══════════════════════════════════════════════════════════════════════════

WRITER_SYSTEM = """You are a scientific writer generating a Related Work section.
Given research paper excerpts, organize them by theme, synthesize the key findings,
and write coherent paragraphs in academic style.
Cite as [Paper: <paper_id>, Page <page_num>]."""


def writer_agent(state: AgentState) -> AgentState:
    """
    两轮检索策略：
      第一轮：用原始 query 检索 top-10
      第二轮：从第一轮结果中提取长词作为关键词，拼成新 query 再检索 top-6
    合并去重后，交给 LLM 按主题组织成 Related Work 段落。
    """
    first_chunks = retrieve(state["query"], state.get("paper_ids") or None, top_k=10)

    # 从第一轮结果中提取长度 > 6 的词作为补充关键词
    keywords = set()
    for c in first_chunks[:5]:
        for word in c.text.split():
            if len(word) > 6:
                keywords.add(word.lower())
    extra_query = " ".join(list(keywords)[:8])
    second_chunks = retrieve(extra_query, state.get("paper_ids") or None, top_k=6) if extra_query else []

    # 合并两轮结果并去重
    seen_texts = set()
    merged: list[RetrievedChunk] = []
    for c in first_chunks + second_chunks:
        if c.text not in seen_texts:
            seen_texts.add(c.text)
            merged.append(c)

    context = "\n\n".join(
        f"[Paper: {c.paper_id}, Page {c.page_num}]\n{c.text}" for c in merged
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": WRITER_SYSTEM},
            {"role": "user", "content": f"Topic: {state['query']}\n\nExcerpts:\n{context}"},
        ],
        temperature=0.4,
        max_tokens=2048,
    )
    answer = resp.choices[0].message.content or ""
    sources = [
        {"paper_id": c.paper_id, "page_num": c.page_num, "section": c.section}
        for c in merged
    ]
    return {**state, "final_answer": answer, "sources": sources}


# ═══════════════════════════════════════════════════════════════════════════
# 图组装：classify → retrieve → 条件分发 → 结束
# ═══════════════════════════════════════════════════════════════════════════

def _route(state: AgentState) -> str:
    """根据 classify_task 写入的 task_type 字段决定走哪个分支。"""
    return state.get("task_type", "doc")


def build_graph():
    """
    构建 LangGraph 状态图：

        classify → retrieve ─┬─ doc    → END
                             ├─ figure → END
                             └─ writer → END
    """
    g = StateGraph(AgentState)

    g.add_node("classify", classify_task)
    g.add_node("retrieve", retrieve_context)
    g.add_node("doc", doc_agent)
    g.add_node("figure", figure_agent)
    g.add_node("writer", writer_agent)

    g.set_entry_point("classify")
    g.add_edge("classify", "retrieve")
    # 条件边：retrieve 之后根据 task_type 分发到三个 Agent
    g.add_conditional_edges(
        "retrieve",
        _route,
        {"doc": "doc", "figure": "figure", "writer": "writer"},
    )
    g.add_edge("doc", END)
    g.add_edge("figure", END)
    g.add_edge("writer", END)

    return g.compile()


# 模块加载时就编译好 graph，后续调用 graph.invoke() 即可
graph = build_graph()

"""
Agent 编排层 —— 用 LangGraph 构建的任务调度图。

整体流程：
  classify（意图分类）→ retrieve（向量检索）→ 按类型分发到三个 Agent 之一

模型分工：
  - 文本 LLM（classify / doc / writer）：Google Gemini 2.5 Flash Lite（云端 API）
  - 视觉 LLM（figure）：Qwen2.5-VL-3B-Instruct-4bit（本地 MLX，Apple Silicon）

State 是一个 TypedDict，在各节点之间传递，每个节点读取需要的字段、写回更新的字段。
"""

import json
import logging
from pathlib import Path
from typing import TypedDict

from google import genai
from google.genai import types as genai_types
from langgraph.graph import END, StateGraph
from PIL import Image

from app.config import GEMINI_API_KEY, GEMINI_MODEL, VLM_MODEL_ID
from app.retrieval.retriever import retrieve, RetrievedChunk

logger = logging.getLogger(__name__)

# ── Gemini 客户端初始化（新版 google-genai SDK）─────────────────────────────
_gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def _call_gemini(
    system: str,
    user_msg: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> str:
    """
    封装 Gemini API 调用（使用新版 google-genai >= 1.0 SDK）。
    system_instruction 通过 GenerateContentConfig 传入。
    """
    resp = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_msg,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return resp.text or ""


# ── MLX VLM 单例（懒加载，首次调用时才下载/加载模型）─────────────────────
_vlm_model = None
_vlm_processor = None
_vlm_config = None


def _get_vlm():
    """
    延迟加载 Qwen2.5-VL MLX 模型。
    首次调用会从 HuggingFace 下载量化权重（约 2GB），之后复用缓存。
    """
    global _vlm_model, _vlm_processor, _vlm_config
    if _vlm_model is None:
        logger.info("首次加载 MLX VLM 模型，正在下载/加载：%s", VLM_MODEL_ID)
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        _vlm_model, _vlm_processor = load(VLM_MODEL_ID)
        _vlm_config = load_config(VLM_MODEL_ID)
        logger.info("MLX VLM 模型加载完成")
    return _vlm_model, _vlm_processor, _vlm_config


def _call_vlm(prompt: str, image_paths: list[str]) -> str:
    """
    调用本地 MLX VLM 模型进行图文推理。
    - 有图片：多模态输入（最多 3 张）
    - 无图片：退化为纯文本模式
    """
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = _get_vlm()

    # 加载存在的图片文件，过滤掉找不到的路径
    images: list[Image.Image] = []
    for fp in image_paths[:3]:
        p = Path(fp)
        if p.exists():
            images.append(Image.open(str(p)))

    num_images = len(images)
    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images=num_images
    )

    output = generate(
        model,
        processor,
        formatted_prompt,
        images if images else [],
        verbose=False,
        max_tokens=1024,
    )
    return output or ""


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph State
# ═══════════════════════════════════════════════════════════════════════════

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
# 节点 1：意图分类（Router）—— Gemini
# ═══════════════════════════════════════════════════════════════════════════

CLASSIFY_SYSTEM = """You are a task classifier for a research paper Q&A system.
Given the user's query, classify it into exactly one category:
- "doc": general question answering or cross-paper comparison
- "figure": the user asks to explain, describe, or analyze a figure, chart, plot, or table image
- "writer": the user asks to generate a related work section, literature review, or survey paragraph

Respond with ONLY a JSON object: {"task_type": "<category>"}"""


def classify_task(state: AgentState) -> AgentState:
    """
    用 Gemini 一次调用判断用户意图。
    temperature=0 确保分类结果稳定，JSON 解析失败时默认走 doc 路径。
    """
    raw = _call_gemini(CLASSIFY_SYSTEM, state["query"], temperature=0, max_tokens=50)
    # Gemini 有时会在 JSON 外包裹 markdown 代码块，需要清洗
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
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
# 节点 3a：DocAgent —— 普通问答（Gemini）
# ═══════════════════════════════════════════════════════════════════════════

DOC_SYSTEM = """You are a research assistant that answers questions based ONLY on the provided context.
Rules:
- If the context is insufficient, say so explicitly. Do NOT fabricate information.
- Cite sources as [Paper: <paper_id>, Page <page_num>] after relevant statements.
- Be concise and precise."""


def doc_agent(state: AgentState) -> AgentState:
    """
    把检索到的 chunk 拼成上下文，要求 Gemini 只基于上下文回答。
    每段 chunk 前标注来源，LLM 在回答中引用时可以直接复用。
    """
    chunks = state.get("retrieved_chunks", [])
    context = "\n\n".join(
        f"[Paper: {c.paper_id}, Page {c.page_num}]\n{c.text}" for c in chunks
    )
    answer = _call_gemini(
        DOC_SYSTEM,
        f"Context:\n{context}\n\nQuestion: {state['query']}",
        temperature=0.2,
    )
    sources = [
        {"paper_id": c.paper_id, "page_num": c.page_num, "section": c.section}
        for c in chunks
    ]
    return {**state, "final_answer": answer, "sources": sources}


# ═══════════════════════════════════════════════════════════════════════════
# 节点 3b：FigureAgent —— 图表解释（本地 MLX Qwen2.5-VL）
# ═══════════════════════════════════════════════════════════════════════════

FIGURE_PROMPT_TEMPLATE = """You are analyzing a figure from a research paper.
Describe: 1) the type of figure (bar chart, line plot, table, diagram, etc.),
2) the main conclusion or finding, 3) notable data trends or patterns.
If text context is provided, use it to give a richer explanation.

Context:
{context}

Question: {query}"""


def figure_agent(state: AgentState) -> AgentState:
    """
    使用本地 Qwen2.5-VL-3B-Instruct（MLX 量化版）分析图表。
    - 首次调用会触发模型下载（约 2GB），之后复用缓存
    - 最多传入 3 张图片
    - 若无图片，退化为纯文本描述
    """
    chunks = state.get("retrieved_chunks", [])
    text_context = "\n".join(c.text for c in chunks[:4])

    figure_paths = state.get("figure_paths", [])
    prompt = FIGURE_PROMPT_TEMPLATE.format(
        context=text_context,
        query=state["query"],
    )

    answer = _call_vlm(prompt, figure_paths)

    sources = [
        {"paper_id": c.paper_id, "page_num": c.page_num, "section": c.section}
        for c in chunks
    ]
    return {**state, "final_answer": answer, "sources": sources}


# ═══════════════════════════════════════════════════════════════════════════
# 节点 3c：WriterAgent —— Related Work 生成（Gemini）
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
    合并去重后，交给 Gemini 按主题组织成 Related Work 段落。
    """
    first_chunks = retrieve(state["query"], state.get("paper_ids") or None, top_k=10)

    # 从第一轮结果中提取长度 > 6 的词作为补充关键词
    keywords = set()
    for c in first_chunks[:5]:
        for word in c.text.split():
            if len(word) > 6:
                keywords.add(word.lower())
    extra_query = " ".join(list(keywords)[:8])
    second_chunks = (
        retrieve(extra_query, state.get("paper_ids") or None, top_k=6)
        if extra_query
        else []
    )

    # 合并两轮结果并去重
    seen_texts: set[str] = set()
    merged: list[RetrievedChunk] = []
    for c in first_chunks + second_chunks:
        if c.text not in seen_texts:
            seen_texts.add(c.text)
            merged.append(c)

    context = "\n\n".join(
        f"[Paper: {c.paper_id}, Page {c.page_num}]\n{c.text}" for c in merged
    )
    answer = _call_gemini(
        WRITER_SYSTEM,
        f"Topic: {state['query']}\n\nExcerpts:\n{context}",
        temperature=0.4,
        max_tokens=4096,
    )
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

        classify(Gemini) → retrieve → ─┬─ doc(Gemini)      → END
                                        ├─ figure(MLX VLM)  → END
                                        └─ writer(Gemini)   → END
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

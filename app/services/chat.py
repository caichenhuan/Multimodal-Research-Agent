"""
问答服务 —— 实时问答的调度中心。

作为 API 层和 Agent 层之间的薄封装：
接收 query + paper_ids → 调用 LangGraph 编译好的 graph → 返回结构化结果。

注意：graph 返回的 state 可能包含 MLX 等非原生 Python 类型，
必须在此层全部转为 str/int/float 再传给 FastAPI，否则 JSON 序列化会崩溃。
"""

import logging

from app.agents.graph import graph, AgentState

logger = logging.getLogger(__name__)


def answer_query(query: str, paper_ids: list[str] | None = None) -> dict:
    """
    问答入口。graph.invoke() 会依次执行 classify → retrieve → agent，
    返回最终 state，从中提取 answer 和 sources。
    """
    initial_state: AgentState = {
        "query": query,
        "paper_ids": paper_ids or [],
    }
    result = graph.invoke(initial_state)

    # 强制转为纯 Python 类型，防止 MLX/numpy 等对象导致 FastAPI 序列化失败
    sources = []
    for s in result.get("sources", []):
        sources.append({
            "paper_id": str(s.get("paper_id", "")),
            "page_num": int(s.get("page_num", 0)),
            "section": str(s.get("section", "")),
        })

    return {
        "answer": str(result.get("final_answer", "")),
        "sources": sources,
        "task_type": str(result.get("task_type", "doc")),
    }

"""
问答服务 —— 实时问答的调度中心。

作为 API 层和 Agent 层之间的薄封装：
接收 query + paper_ids → 调用 LangGraph 编译好的 graph → 返回结构化结果。
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
    return {
        "answer": result.get("final_answer", ""),
        "sources": result.get("sources", []),
        "task_type": result.get("task_type", "doc"),
    }

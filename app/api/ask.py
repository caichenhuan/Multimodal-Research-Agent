"""
POST /ask —— 接收用户的自然语言问题，返回有来源引用的回答。
请求体：{ paper_ids: [...], query: "..." }
响应体：{ answer: "...", sources: [...], task_type: "doc/figure/writer" }
"""

from pydantic import BaseModel
from fastapi import APIRouter

from app.services.chat import answer_query

router = APIRouter()


class AskRequest(BaseModel):
    paper_ids: list[str] = []   # 限定检索范围的论文列表，为空则搜索全部
    query: str


@router.post("/ask")
async def ask(req: AskRequest):
    try:
        result = answer_query(req.query, req.paper_ids or None)
        return {
            "success": True,
            "data": result,
            "error": None,
        }
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}

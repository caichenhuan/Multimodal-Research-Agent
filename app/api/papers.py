"""
GET /papers —— 返回所有已入库论文的列表（供前端侧边栏展示）。
"""

from fastapi import APIRouter

from app.db.sqlite import list_papers

router = APIRouter()


@router.get("/papers")
async def get_papers():
    try:
        papers = list_papers()
        return {
            "success": True,
            "data": [
                {
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "filename": p.filename,
                    "page_count": p.page_count,
                    "status": p.status,
                    "created_at": p.created_at,
                }
                for p in papers
            ],
            "error": None,
        }
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}

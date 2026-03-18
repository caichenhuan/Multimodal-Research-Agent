"""
GET /figures/{paper_id} —— 返回某篇论文所有提取出的图片路径列表。
图片在 ingest 阶段由 pdf_parser 保存到 storage/figures/{paper_id}/ 下。
"""

from pathlib import Path

from fastapi import APIRouter

from app.config import FIGURE_DIR

router = APIRouter()


@router.get("/figures/{paper_id}")
async def get_figures(paper_id: str):
    try:
        fig_dir = FIGURE_DIR / paper_id
        if not fig_dir.exists():
            return {"success": True, "data": [], "error": None}

        images = sorted(fig_dir.glob("*.png"))
        return {
            "success": True,
            "data": [{"filename": img.name, "path": str(img)} for img in images],
            "error": None,
        }
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}

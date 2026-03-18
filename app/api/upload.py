"""
POST /upload —— 接收 PDF 文件，保存到本地，触发后台解析任务。

流程：校验文件类型 → 写入 DB → 保存文件 → 启动 BackgroundTask → 返回 task_id。
前端拿到 task_id 后可以轮询 /status 获取解析进度。
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, UploadFile

from app.config import PDF_DIR
from app.db.sqlite import create_paper
from app.services.ingest import ingest_paper

router = APIRouter()


@router.post("/upload")
async def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks):
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            return {"success": False, "data": None, "error": "Only PDF files are accepted"}

        PDF_DIR.mkdir(parents=True, exist_ok=True)
        paper, task = create_paper(file.filename)
        # 用 paper_id 重命名文件，避免中文 / 特殊字符导致的路径问题
        save_path = PDF_DIR / f"{paper.paper_id}.pdf"

        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 把解析任务放入后台执行，接口立即返回
        background_tasks.add_task(ingest_paper, paper.paper_id, str(save_path))

        return {
            "success": True,
            "data": {
                "paper_id": paper.paper_id,
                "task_id": task.task_id,
                "filename": file.filename,
            },
            "error": None,
        }
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}

"""
GET /status/{task_id} —— 查询解析任务的当前状态。
前端上传 PDF 后会每 2 秒轮询此接口，直到状态变为 ready 或 failed。
"""

from fastapi import APIRouter

from app.db.sqlite import get_task

router = APIRouter()


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    try:
        task = get_task(task_id)
        if not task:
            return {"success": False, "data": None, "error": "Task not found"}
        return {
            "success": True,
            "data": {
                "task_id": task.task_id,
                "paper_id": task.paper_id,
                "status": task.status,
                "progress_msg": task.progress_msg,
                "updated_at": task.updated_at,
            },
            "error": None,
        }
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}

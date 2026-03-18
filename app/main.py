"""
FastAPI 应用入口 —— 注册路由、中间件，启动时初始化数据库。

启动命令：uvicorn app.main:app --reload --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.sqlite import init_db
from app.api.upload import router as upload_router
from app.api.status import router as status_router
from app.api.ask import router as ask_router
from app.api.papers import router as papers_router
from app.api.figures import router as figures_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="Multimodal Research Agent", version="0.1.0")

# Streamlit 默认跑在 8501 端口，和 FastAPI (8000) 不同源，需要开放 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(status_router)
app.include_router(ask_router)
app.include_router(papers_router)
app.include_router(figures_router)


@app.on_event("startup")
def on_startup():
    """服务启动时确保 SQLite 表结构存在。"""
    init_db()

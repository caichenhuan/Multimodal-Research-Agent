"""
解析服务 —— 整个系统的离线处理主链路。

被 BackgroundTasks 在后台调用，完成以下流水线：
  PDF → 解析(parse) → 分块(chunk) → 向量化(embed) → 入库(ready)

每个阶段结束后都会更新 SQLite 中的状态，前端通过轮询 /status 接口
可以实时看到当前进度。任何步骤失败会把状态置为 failed。
"""

import logging
from dataclasses import asdict

from app.db.sqlite import update_paper_status, update_task_status, get_task_by_paper
from app.parser.pdf_parser import parse_pdf
from app.parser.chunker import chunk_pages, Chunk
from app.retrieval.retriever import add_chunks, add_figures

logger = logging.getLogger(__name__)


def _set_progress(paper_id: str, task_id: str | None, status: str, msg: str) -> None:
    """同时更新 papers 表和 tasks 表的状态，保持两张表一致。"""
    update_paper_status(paper_id, status)
    if task_id:
        update_task_status(task_id, status, msg)
    logger.info("[%s] %s – %s", paper_id, status, msg)


def ingest_paper(paper_id: str, pdf_path: str) -> None:
    """
    解析入口，由 upload 接口的 BackgroundTasks 触发。
    整个函数串行执行（已经在后台线程中，不需要内部并发）。
    """
    task = get_task_by_paper(paper_id)
    task_id = task.task_id if task else None

    try:
        # ── 阶段 1：PDF 解析 → 提取每页的文字和图片 ────────────────
        _set_progress(paper_id, task_id, "parsing", "Extracting text and figures from PDF")
        pages = parse_pdf(pdf_path, paper_id)

        # 解析完成后记录页数
        update_paper_status(paper_id, "parsing", page_count=len(pages))

        # ── 阶段 2：文本分块 ───────────────────────────────────────
        _set_progress(paper_id, task_id, "chunking", "Splitting text into chunks")
        page_dicts = []
        for p in pages:
            page_dicts.append({
                "page_num": p.page_num,
                "text": p.text,
                "figures": p.figures,
            })
        chunks = chunk_pages(page_dicts, paper_id)

        # ── 阶段 3：向量化 + 写入 ChromaDB ────────────────────────
        _set_progress(paper_id, task_id, "embedding", "Generating embeddings and indexing")

        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        caption_chunks = [c for c in chunks if c.chunk_type == "figure_caption"]

        if text_chunks:
            add_chunks([asdict(c) for c in text_chunks])

        # 图注单独写入 figure_captions collection，附带图片路径
        figure_records = []
        for p in pages:
            for fig in p.figures:
                if fig.caption:
                    figure_records.append({
                        "paper_id": paper_id,
                        "page_num": p.page_num,
                        "caption": fig.caption,
                        "image_path": fig.image_path,
                    })
        if figure_records:
            add_figures(figure_records)

        # ── 完成 ──────────────────────────────────────────────────
        _set_progress(paper_id, task_id, "ready", "Paper indexed and ready for Q&A")

    except Exception as e:
        logger.error("Ingest failed for %s: %s", paper_id, e, exc_info=True)
        update_paper_status(paper_id, "failed", error_msg=str(e))
        if task_id:
            update_task_status(task_id, "failed", f"Error: {e}")

"""
数据库层 —— 用 SQLite 存储论文元数据和任务状态。

两张表：
  - papers：论文基本信息（标题、文件名、页数、处理状态）
  - tasks：每次解析任务的进度追踪（状态、进度描述）

不使用 ORM，直接用 sqlite3 + dataclass，保持轻量。
"""

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import DB_PATH


@dataclass
class Paper:
    paper_id: str
    title: str
    filename: str
    page_count: int
    status: str                # pending → parsing → chunking → embedding → ready / failed
    error_msg: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class Task:
    task_id: str
    paper_id: str
    status: str
    progress_msg: str          # 前端轮询时展示给用户的进度说明
    created_at: str
    updated_at: str


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    # WAL 模式允许读写并发，避免后台任务写入时阻塞 API 读取
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """应用启动时调用，确保表结构存在。"""
    conn = _connect()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            paper_id   TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            filename   TEXT NOT NULL,
            page_count INTEGER DEFAULT 0,
            status     TEXT DEFAULT 'pending',
            error_msg  TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks (
            task_id      TEXT PRIMARY KEY,
            paper_id     TEXT NOT NULL,
            status       TEXT DEFAULT 'pending',
            progress_msg TEXT DEFAULT '',
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
        );
        """
    )
    conn.close()


def _now() -> str:
    return datetime.utcnow().isoformat()


def create_paper(filename: str, title: str = "") -> tuple[Paper, Task]:
    """
    新建一条论文记录和对应的解析任务。
    paper_id / task_id 使用 UUID 前 12 位，足够避免冲突且便于展示。
    """
    paper_id = uuid.uuid4().hex[:12]
    task_id = uuid.uuid4().hex[:12]
    now = _now()
    if not title:
        title = Path(filename).stem  # 没有手动标题时，用文件名做默认标题

    conn = _connect()
    conn.execute(
        "INSERT INTO papers (paper_id, title, filename, page_count, status, created_at, updated_at) "
        "VALUES (?, ?, ?, 0, 'pending', ?, ?)",
        (paper_id, title, filename, now, now),
    )
    conn.execute(
        "INSERT INTO tasks (task_id, paper_id, status, progress_msg, created_at, updated_at) "
        "VALUES (?, ?, 'pending', 'Queued', ?, ?)",
        (task_id, paper_id, now, now),
    )
    conn.commit()
    conn.close()

    paper = Paper(paper_id, title, filename, 0, "pending", None, now, now)
    task = Task(task_id, paper_id, "pending", "Queued", now, now)
    return paper, task


def update_paper_status(
    paper_id: str,
    status: str,
    error_msg: Optional[str] = None,
    page_count: Optional[int] = None,
) -> None:
    """更新论文处理状态，ingest 流水线每个阶段结束后调用。"""
    now = _now()
    conn = _connect()
    if page_count is not None:
        conn.execute(
            "UPDATE papers SET status=?, error_msg=?, page_count=?, updated_at=? WHERE paper_id=?",
            (status, error_msg, page_count, now, paper_id),
        )
    else:
        conn.execute(
            "UPDATE papers SET status=?, error_msg=?, updated_at=? WHERE paper_id=?",
            (status, error_msg, now, paper_id),
        )
    conn.commit()
    conn.close()


def update_task_status(task_id: str, status: str, progress_msg: str = "") -> None:
    now = _now()
    conn = _connect()
    conn.execute(
        "UPDATE tasks SET status=?, progress_msg=?, updated_at=? WHERE task_id=?",
        (status, progress_msg, now, task_id),
    )
    conn.commit()
    conn.close()


def get_paper(paper_id: str) -> Optional[Paper]:
    conn = _connect()
    row = conn.execute("SELECT * FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
    conn.close()
    return Paper(*row) if row else None


def get_task(task_id: str) -> Optional[Task]:
    conn = _connect()
    row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    conn.close()
    return Task(*row) if row else None


def get_task_by_paper(paper_id: str) -> Optional[Task]:
    """取某篇论文最新的一条解析任务（支持重新解析场景）。"""
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM tasks WHERE paper_id=? ORDER BY created_at DESC LIMIT 1",
        (paper_id,),
    ).fetchone()
    conn.close()
    return Task(*row) if row else None


def list_papers() -> list[Paper]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM papers ORDER BY created_at DESC").fetchall()
    conn.close()
    return [Paper(*r) for r in rows]


def delete_paper(paper_id: str) -> None:
    """删除论文及其所有关联任务，用于重新解析前的清理。"""
    conn = _connect()
    conn.execute("DELETE FROM tasks WHERE paper_id=?", (paper_id,))
    conn.execute("DELETE FROM papers WHERE paper_id=?", (paper_id,))
    conn.commit()
    conn.close()

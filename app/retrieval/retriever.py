"""
检索层 —— 基于 ChromaDB 的向量存储与检索。

维护两个 collection：
  - text_chunks：正文 chunk 的向量索引
  - figure_captions：图注的向量索引（单独存储，便于图表类问题专项检索）

embedding 由 ChromaDB 内置的 OpenAIEmbeddingFunction 自动管理，
写入时自动 embed，查询时也自动 embed query，不需要手动调用 embedding API。

ChromaDB 使用本地持久化模式（PersistentClient），数据存在 storage/chroma/ 目录。
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from app.config import CHROMA_DIR, EMBEDDING_MODEL, RETRIEVE_TOP_K

logger = logging.getLogger(__name__)

# 模块级单例，避免每次调用都重新连接
_client: chromadb.ClientAPI | None = None
_embed_fn: Any = None

# 用于判断用户 query 是否涉及图表，触发额外检索 figure_captions collection
_FIGURE_KW = re.compile(r"(图|表|figure|table|chart|plot|diagram)", re.IGNORECASE)


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embed_fn


def _text_collection():
    return _get_client().get_or_create_collection(
        name="text_chunks",
        embedding_function=_get_embed_fn(),
    )


def _figure_collection():
    return _get_client().get_or_create_collection(
        name="figure_captions",
        embedding_function=_get_embed_fn(),
    )


def add_chunks(chunks: list[dict]) -> None:
    """批量写入正文 chunk 到向量库，每 100 条一批避免超 API 限制。"""
    col = _text_collection()
    ids = [f"{c['paper_id']}_c{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "paper_id": c["paper_id"],
            "page_num": c["page_num"],
            "chunk_index": c["chunk_index"],
            "section": c.get("section", ""),
        }
        for c in chunks
    ]
    for i in range(0, len(ids), 100):
        col.add(
            ids=ids[i : i + 100],
            documents=documents[i : i + 100],
            metadatas=metadatas[i : i + 100],
        )
    logger.info("Added %d text chunks to ChromaDB", len(ids))


def add_figures(figures: list[dict]) -> None:
    """将图注文字写入 figure_captions collection，metadata 里保存图片路径。"""
    if not figures:
        return
    col = _figure_collection()
    ids = [f"{f['paper_id']}_fig{i}" for i, f in enumerate(figures)]
    documents = [f["caption"] for f in figures]
    metadatas = [
        {
            "paper_id": f["paper_id"],
            "page_num": f["page_num"],
            "image_path": f.get("image_path", ""),
        }
        for f in figures
    ]
    col.add(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Added %d figure captions to ChromaDB", len(ids))


@dataclass
class RetrievedChunk:
    text: str
    paper_id: str
    page_num: int
    section: str
    score: float        # 相似度得分（1 - cosine_distance），越高越相关
    source: str         # "text" 或 "figure"，标识来自哪个 collection
    image_path: str = ""


def retrieve(
    query: str,
    paper_ids: list[str] | None = None,
    top_k: int = RETRIEVE_TOP_K,
) -> list[RetrievedChunk]:
    """
    核心检索函数。
    1. 始终检索 text_chunks
    2. 如果 query 包含图表关键词（中英文），额外检索 figure_captions
    3. 合并结果按相似度降序排列，返回 top_k 条
    """
    # 构造 paper_id 过滤条件（ChromaDB 的 where 语法）
    where_filter = None
    if paper_ids:
        if len(paper_ids) == 1:
            where_filter = {"paper_id": paper_ids[0]}
        else:
            where_filter = {"paper_id": {"$in": paper_ids}}

    results: list[RetrievedChunk] = []

    # ── 正文检索 ────────────────────────────────────────────────────────
    text_col = _text_collection()
    kwargs: dict[str, Any] = {"query_texts": [query], "n_results": top_k}
    if where_filter:
        kwargs["where"] = where_filter
    try:
        res = text_col.query(**kwargs)
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            results.append(
                RetrievedChunk(
                    text=doc,
                    paper_id=meta["paper_id"],
                    page_num=meta["page_num"],
                    section=meta.get("section", ""),
                    score=1.0 - dist,  # ChromaDB 返回的是距离，转为相似度
                    source="text",
                )
            )
    except Exception:
        logger.warning("Text retrieval failed", exc_info=True)

    # ── 图表检索（仅在 query 包含图表关键词时触发）─────────────────────
    if _FIGURE_KW.search(query):
        fig_col = _figure_collection()
        fig_kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(top_k, 4),  # 图注条目通常较少，限制返回数
        }
        if where_filter:
            fig_kwargs["where"] = where_filter
        try:
            fig_res = fig_col.query(**fig_kwargs)
            for doc, meta, dist in zip(
                fig_res["documents"][0],
                fig_res["metadatas"][0],
                fig_res["distances"][0],
            ):
                results.append(
                    RetrievedChunk(
                        text=doc,
                        paper_id=meta["paper_id"],
                        page_num=meta["page_num"],
                        section="",
                        score=1.0 - dist,
                        source="figure",
                        image_path=meta.get("image_path", ""),
                    )
                )
        except Exception:
            logger.warning("Figure retrieval failed", exc_info=True)

    # 合并后按相似度降序排列，截取 top_k
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def delete_paper(paper_id: str) -> None:
    """删除某篇论文在两个 collection 中的所有向量，用于重新解析前的清理。"""
    where = {"paper_id": paper_id}
    try:
        _text_collection().delete(where=where)
    except Exception:
        pass
    try:
        _figure_collection().delete(where=where)
    except Exception:
        pass
    logger.info("Deleted all vectors for paper %s", paper_id)

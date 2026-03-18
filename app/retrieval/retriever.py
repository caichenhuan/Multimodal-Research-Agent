"""
检索层 —— 基于 ChromaDB 的向量存储与检索。

维护两个 collection：
  - text_chunks：正文 chunk 的向量索引
  - figure_captions：图注的向量索引（单独存储，便于图表类问题专项检索）

Embedding 使用 Google 的 text-embedding-004 模型（通过 GEMINI_API_KEY 调用），
自定义 ChromaDB EmbeddingFunction 封装批量调用逻辑。

ChromaDB 使用本地持久化模式（PersistentClient），数据存在 storage/chroma/ 目录。
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb import EmbeddingFunction
from google import genai
from google.genai.errors import ClientError

from app.config import CHROMA_DIR, EMBEDDING_MODEL, GEMINI_API_KEY, RETRIEVE_TOP_K

logger = logging.getLogger(__name__)

# 模块级单例，避免每次调用都重新连接或重新创建客户端
_client: chromadb.ClientAPI | None = None
_embed_fn: EmbeddingFunction | None = None

# 用于判断用户 query 是否涉及图表，触发额外检索 figure_captions collection
_FIGURE_KW = re.compile(r"(图|表|figure|table|chart|plot|diagram)", re.IGNORECASE)


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    自定义 ChromaDB EmbeddingFunction，调用 Gemini embedding API。

    免费版限制：每分钟 100 次 embedding 请求。
    应对策略：小批量（每批 50 条） + 批间间隔 + 429 自动重试等待。
    """

    BATCH_SIZE = 50        # 每批条数，留出余量避免刚好踩线
    BATCH_DELAY = 1.0      # 批间等待秒数，平滑请求速率
    MAX_RETRIES = 5        # 429 重试次数上限

    def __init__(self, api_key: str, model_name: str):
        self._genai_client = genai.Client(api_key=api_key)
        self._model = model_name

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """发送单批 embedding 请求，遇到 429 自动等待重试。"""
        for attempt in range(self.MAX_RETRIES):
            try:
                result = self._genai_client.models.embed_content(
                    model=self._model,
                    contents=texts,
                )
                return [e.values for e in result.embeddings]
            except ClientError as e:
                if e.status_code == 429:
                    # 从错误信息中提取建议等待时间，兜底 60 秒
                    wait = 62
                    msg = str(e)
                    if "retryDelay" in msg:
                        import re as _re
                        m = _re.search(r"retry(?:Delay)?.*?(\d+)", msg, _re.IGNORECASE)
                        if m:
                            wait = int(m.group(1)) + 2
                    logger.warning(
                        "Embedding rate limited (attempt %d/%d), waiting %ds...",
                        attempt + 1, self.MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Embedding failed after {self.MAX_RETRIES} retries due to rate limiting")

    def __call__(self, input: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        total_batches = (len(input) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        for batch_idx in range(total_batches):
            start = batch_idx * self.BATCH_SIZE
            batch = input[start : start + self.BATCH_SIZE]
            logger.info(
                "Embedding batch %d/%d (%d texts)", batch_idx + 1, total_batches, len(batch)
            )
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)
            # 批间等待，避免连续请求打满配额
            if batch_idx < total_batches - 1:
                time.sleep(self.BATCH_DELAY)
        return all_embeddings


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _get_embed_fn() -> GeminiEmbeddingFunction:
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = GeminiEmbeddingFunction(
            api_key=GEMINI_API_KEY,
            model_name=EMBEDDING_MODEL,
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

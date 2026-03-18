"""
全局配置模块
所有路径、模型名称、超参数集中在此管理，其他模块统一从这里导入。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── 路径配置 ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # 项目根目录
STORAGE_DIR = BASE_DIR / "storage"
PDF_DIR = STORAGE_DIR / "pdfs"         # 上传的 PDF 原文件
FIGURE_DIR = STORAGE_DIR / "figures"   # 从 PDF 中提取的图片
CHROMA_DIR = STORAGE_DIR / "chroma"    # ChromaDB 持久化存储
DB_PATH = STORAGE_DIR / "papers.db"    # SQLite 元数据库

# ── API Keys ────────────────────────────────────────────────────────────────
# 全系统只需要一个 Key，Gemini 同时承担文本 LLM 和 Embedding 两个职责
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── 模型配置 ────────────────────────────────────────────────────────────────
# 文本 LLM：Google Gemini，处理意图分类、普通问答、Related Work 生成
GEMINI_MODEL = "gemini-2.5-flash-lite"

# 视觉 LLM：本地 MLX 量化模型，处理图表解释任务（仅支持 Apple Silicon）
VLM_MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

# Embedding 模型：Google gemini-embedding-001，用于 chunk 向量化和 ChromaDB 检索
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ── 分块与检索参数 ──────────────────────────────────────────────────────────
CHUNK_SIZE = 800            # 每个 chunk 的最大字符数
CHUNK_OVERLAP = 100         # 相邻 chunk 之间的重叠字符数，防止语义被截断
EMBEDDING_BATCH_SIZE = 100  # 每批次调用 embedding API 的条数上限
RETRIEVE_TOP_K = 8          # 默认检索返回的最相关 chunk 数量

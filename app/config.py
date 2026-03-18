"""
全局配置模块
所有路径、模型名称、超参数集中在此管理，其他模块统一从这里导入。
"""

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

# ── 模型配置 ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"  # 向量化模型，用于 chunk 入库和检索
CHAT_MODEL = "gpt-4o"                       # 文本对话模型，用于问答和分类
VISION_MODEL = "gpt-4o"                     # 多模态模型，用于图表解释（接受图片输入）

# ── 分块与检索参数 ──────────────────────────────────────────────────────────
CHUNK_SIZE = 800            # 每个 chunk 的最大字符数
CHUNK_OVERLAP = 100         # 相邻 chunk 之间的重叠字符数，防止语义被截断
EMBEDDING_BATCH_SIZE = 100  # 每批次调用 embedding API 的条数上限
RETRIEVE_TOP_K = 8          # 默认检索返回的最相关 chunk 数量

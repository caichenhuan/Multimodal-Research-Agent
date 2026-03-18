"""
文本分块器 —— 将每页文字切成适合向量检索的 chunk。

分块策略：使用 LangChain 的 RecursiveCharacterTextSplitter，
按 段落(\n\n) → 换行(\n) → 句号(. ) → 空格 → 字符 的优先级切分，
尽量保持语义完整性。

每个 chunk 带有元数据（paper_id、page_num、section 等），
便于检索后追溯来源。
"""

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE

# 匹配常见论文章节标题格式：
#   - 关键词型：Abstract、Introduction、Conclusion 等
#   - 编号型：1 Introduction、2.1 Background 等
#   - Markdown 型：## Related Work
_SECTION_RE = re.compile(
    r"^(?:"
    r"(?:Abstract|Introduction|Conclusion|Acknowledgements?|References|Appendix)"
    r"|(?:\d+(?:\.\d+)*\s+[A-Z].*)"
    r"|(?:#{1,3}\s+.+)"
    r")$",
    re.MULTILINE,
)


@dataclass
class Chunk:
    text: str
    paper_id: str
    page_num: int
    chunk_index: int
    section: str        # 该 chunk 所属的章节名（如 "Introduction"），用于检索时过滤
    chunk_type: str     # "text" 或 "figure_caption"


def _detect_section(text: str) -> str:
    """从页面文字中尝试识别章节标题，返回第一个匹配项。"""
    matches = _SECTION_RE.findall(text)
    if matches:
        return matches[0].strip().lstrip("#").strip()
    return ""


def chunk_pages(
    pages: list[dict],
    paper_id: str,
) -> list[Chunk]:
    """
    输入：每页的 dict（含 page_num, text, figures）
    输出：扁平的 Chunk 列表，包含正文 chunk 和图注 chunk
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    idx = 0
    current_section = ""  # 章节标签会一直向后继承，直到检测到新的章节

    for page in pages:
        page_num: int = page["page_num"]
        text: str = page.get("text", "")

        detected = _detect_section(text)
        if detected:
            current_section = detected

        # 正文分块
        if text.strip():
            splits = splitter.split_text(text)
            for s in splits:
                chunks.append(
                    Chunk(
                        text=s,
                        paper_id=paper_id,
                        page_num=page_num,
                        chunk_index=idx,
                        section=current_section,
                        chunk_type="text",
                    )
                )
                idx += 1

        # 图注单独作为 chunk 入库，方便按图表检索
        for fig in page.get("figures", []):
            caption = fig.get("caption", "") if isinstance(fig, dict) else fig.caption
            if caption and caption.strip():
                chunks.append(
                    Chunk(
                        text=caption,
                        paper_id=paper_id,
                        page_num=page_num,
                        chunk_index=idx,
                        section=current_section,
                        chunk_type="figure_caption",
                    )
                )
                idx += 1

    return chunks

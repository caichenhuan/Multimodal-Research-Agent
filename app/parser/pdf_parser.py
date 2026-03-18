"""
PDF 解析器 —— 从 PDF 中提取每一页的文字、图片和表格区域。

采用双引擎策略：
  - PyMuPDF (fitz)：擅长提取嵌入式图片（位图），速度快
  - pdfplumber：擅长提取带结构的文字（能识别多栏布局），也用于表格检测

局限：纯矢量图（如 LaTeX 绘制的折线图）无法通过 get_images() 检测到，
MVP 阶段暂不处理，仅记录到日志。
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from app.config import FIGURE_DIR

logger = logging.getLogger(__name__)


@dataclass
class FigureInfo:
    image_path: str                                # 提取后保存的 PNG 路径
    bbox: tuple[float, float, float, float]        # 图片在页面中的位置 (x0, y0, x1, y1)
    caption: str                                   # 自动识别的图注文字


@dataclass
class PageData:
    page_num: int
    text: str
    figures: list[FigureInfo] = field(default_factory=list)
    table_bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)


# 匹配常见图注格式，如 "Figure 1: ..." 或 "Table 2. ..."
_CAPTION_RE = re.compile(
    r"(Fig(?:ure)?|Table)\s*\.?\s*\d+[.:]\s*.+", re.IGNORECASE
)


def _find_caption(plumber_page, img_bottom_y: float) -> str:
    """
    在图片 bbox 下方 40pt 范围内搜索文字，拼接后用正则判断是否为图注。
    论文图注通常紧跟在图片下方，格式为 "Figure X: description"。
    """
    words = plumber_page.extract_words()
    candidates: list[tuple[float, str]] = []
    for w in words:
        if w["top"] >= img_bottom_y and w["top"] <= img_bottom_y + 40:
            candidates.append((w["x0"], w["text"]))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0])
    line = " ".join(t for _, t in candidates)
    if _CAPTION_RE.search(line):
        return line.strip()
    return ""


def parse_pdf(pdf_path: str, paper_id: str) -> list[PageData]:
    """
    解析入口：输入 PDF 路径，输出每页的结构化数据列表。
    图片会保存到 storage/figures/{paper_id}/ 目录下。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    figure_dir = FIGURE_DIR / paper_id
    figure_dir.mkdir(parents=True, exist_ok=True)

    # 同时打开两个引擎，逐页并行使用
    doc = fitz.open(str(pdf_path))
    plumber_pdf = pdfplumber.open(str(pdf_path))

    pages: list[PageData] = []

    for page_idx in range(len(doc)):
        fitz_page = doc[page_idx]
        plumber_page = plumber_pdf.pages[page_idx]
        page_num = page_idx + 1

        # ── 文字提取（用 pdfplumber，结构性更好）──────────────────────
        text = plumber_page.extract_text() or ""

        # ── 图片提取（用 PyMuPDF）─────────────────────────────────────
        figures: list[FigureInfo] = []
        try:
            image_list = fitz_page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]  # PDF 内部对象引用 ID
                try:
                    pix = fitz.Pixmap(doc, xref)
                    # CMYK 等非 RGB 色彩空间需要转换，否则保存 PNG 会报错
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    # 过滤掉太小的图片（通常是 icon、装饰性元素）
                    if pix.width < 50 or pix.height < 50:
                        continue

                    img_filename = f"p{page_num}_img{img_idx}.png"
                    img_path = figure_dir / img_filename
                    pix.save(str(img_path))

                    # 获取图片在页面上的位置矩形
                    rects = fitz_page.get_image_rects(xref)
                    bbox = (0.0, 0.0, 0.0, 0.0)
                    if rects:
                        r = rects[0]
                        bbox = (r.x0, r.y0, r.x1, r.y1)

                    # 尝试在图片下方查找图注
                    caption = _find_caption(plumber_page, bbox[3])

                    figures.append(
                        FigureInfo(
                            image_path=str(img_path),
                            bbox=bbox,
                            caption=caption,
                        )
                    )
                except Exception:
                    logger.warning(
                        "Failed to extract image xref=%s on page %s",
                        xref,
                        page_num,
                        exc_info=True,
                    )
        except Exception:
            logger.warning(
                "Failed to enumerate images on page %s", page_num, exc_info=True
            )

        # ── 表格区域检测（MVP 阶段只记录位置，不解析内容）────────────
        table_bboxes: list[tuple[float, float, float, float]] = []
        try:
            tables = plumber_page.find_tables()
            for t in tables:
                table_bboxes.append(tuple(t.bbox))  # type: ignore[arg-type]
        except Exception:
            pass

        pages.append(
            PageData(
                page_num=page_num,
                text=text,
                figures=figures,
                table_bboxes=table_bboxes,
            )
        )

    doc.close()
    plumber_pdf.close()
    return pages

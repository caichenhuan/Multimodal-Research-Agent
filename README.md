# Multimodal Research Agent

面向科研场景的论文解析与问答系统。上传 PDF 论文后，系统自动完成文本抽取、图表提取、向量化入库，之后可以通过自然语言提问，得到有文献来源的回答。支持单篇问答、跨论文对比、图表解释、Related Work 生成等任务。

---

## 一、环境准备

### 1.1 前置要求

- **Python 3.11+**
- **Apple Silicon Mac**（M1/M2/M3/M4，本地 VLM 依赖 MLX 框架）
- **Gemini API Key**（用于文本 LLM 和 Embedding）
- **conda 环境**（推荐）

### 1.2 安装依赖

```bash
conda activate mllmagent
cd "/Users/raphael/Documents/Projects/Multimodal Agent"
pip install -r requirements.txt
```

涉及的主要依赖：

| 包 | 用途 |
|---|------|
| fastapi + uvicorn | 后端 API 服务 |
| streamlit | 前端界面 |
| google-genai | Gemini 文本 LLM（分类、问答、写作）+ Embedding 向量化 |
| mlx-vlm | 本地视觉 LLM（图表解释，仅 Apple Silicon）|
| torch + torchvision | VLM 图片预处理依赖 |
| langgraph | Agent 编排状态图 |
| PyMuPDF (fitz) | PDF 图片提取 |
| pdfplumber | PDF 文字提取 |
| chromadb | 本地向量数据库 |
| langchain-text-splitters | 文本分块 |

### 1.3 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，只需一个 Key：

```
GEMINI_API_KEY=AIzaSy-xxxxxxxxxxxxxxxxxxxxxxx
```

> 本地 VLM 模型（Qwen2.5-VL-3B）在首次提问图表类问题时自动从 HuggingFace 下载（约 2GB），不需要额外配置。

---

## 二、启动服务

需要打开 **两个终端窗口**，分别启动后端和前端。

### 终端 1：启动 FastAPI 后端

```bash
conda activate mllmagent
cd "/Users/raphael/Documents/Projects/Multimodal Agent"

uvicorn app.main:app --reload --port 8000
```

看到以下输出说明启动成功：

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Started reloader process
```

### 终端 2：启动 Streamlit 前端

```bash
conda activate mllmagent
cd "/Users/raphael/Documents/Projects/Multimodal Agent"

streamlit run ui/streamlit_app.py --server.port 8501
```

浏览器会自动打开 `http://localhost:8501`。

---

## 三、使用流程

### 3.1 上传论文

1. 在页面 **左侧边栏** 点击 「Upload PDF(s)」
2. 选择一个或多个 PDF 文件
3. 上传后会实时显示处理进度：
   - `parsing` — 正在提取文字和图片
   - `chunking` — 正在分块
   - `embedding` — 正在生成向量并入库
   - `ready` ✅ — 完成，可以开始提问
   - `failed` ❌ — 处理失败，查看错误信息

### 3.2 选择论文

- 侧边栏的论文列表中，**勾选**你想作为问答上下文的论文
- 可以同时勾选多篇，实现跨论文对比
- 只有状态为 ✅ ready 的论文可以用于提问

### 3.3 提问

在页面底部的输入框中输入问题，按回车发送。系统会自动判断问题类型：

| 问题类型 | 触发条件 | 示例 |
|---------|---------|------|
| **普通问答** (DocAgent) | 默认类型 | "这篇论文的主要贡献是什么？" |
| **图表解释** (FigureAgent) | 提到图、表、figure、chart | "请解释 Figure 3 中的实验结果" |
| **Related Work 生成** (WriterAgent) | 要求写综述/文献回顾 | "帮我写一段关于 attention mechanism 的 related work" |

### 3.4 查看回答

- 回答正文会直接显示在聊天区域
- 点击回答下方的 **「Sources」** 折叠块，可以看到引用来源（论文 ID + 页码 + 章节）
- 对话历史会保留在当前会话中

---

## 四、技术方案

### 4.1 整体架构

系统分为**离线处理**（上传时）和**在线问答**（提问时）两条链路，共用同一套向量存储和元数据库。

```
                         ┌──────────────────────────────────────────────────────┐
                         │                    在线问答链路                       │
                         │                                                      │
用户提问 ──→ Streamlit ──→ FastAPI /ask ──→ LangGraph Agent Graph               │
                         │                      │                               │
                         │              ┌───────┴────────┐                      │
                         │              ▼                ▼                      │
                         │     Gemini 意图分类    ChromaDB 向量检索              │
                         │              │                │                      │
                         │       ┌──────┼──────┐         │                      │
                         │       ▼      ▼      ▼         │                      │
                         │     Doc   Figure  Writer      │                      │
                         │    Agent   Agent   Agent       │                      │
                         │   (Gemini) (MLX)  (Gemini)     │                      │
                         │       └──────┼──────┘         │                      │
                         │              ▼                │                      │
                         │         回答 + 引用来源 ──→ 前端展示                  │
                         └──────────────────────────────────────────────────────┘

                         ┌──────────────────────────────────────────────────────┐
                         │                    离线处理链路                       │
                         │                                                      │
上传 PDF ──→ FastAPI /upload ──→ BackgroundTask                                 │
                         │            │                                         │
                         │    ┌───────┴────────────┐                            │
                         │    ▼                    ▼                            │
                         │  PyMuPDF            pdfplumber                       │
                         │  (提取图片)          (提取文字)                       │
                         │    │                    │                            │
                         │    ▼                    ▼                            │
                         │  保存 PNG          文本分块 (800 chars)               │
                         │    │                    │                            │
                         │    ▼                    ▼                            │
                         │  图注识别          Gemini Embedding                   │
                         │    │                    │                            │
                         │    ▼                    ▼                            │
                         │  figure_captions    text_chunks                      │
                         │  (ChromaDB)         (ChromaDB)                       │
                         └──────────────────────────────────────────────────────┘
```

### 4.2 模型分工

本系统使用三种模型，各司其职：

| 模型 | 运行位置 | 职责 | 何时调用 |
|------|---------|------|---------|
| **Gemini 2.5 Flash Lite** | 云端 API | 意图分类、普通问答、Related Work 生成 | 每次提问时 |
| **Qwen2.5-VL-3B-Instruct-4bit** | 本地 MLX (Apple Silicon) | 图表/图片视觉理解 | 仅图表类问题时 |
| **Gemini Embedding (gemini-embedding-001)** | 云端 API | 文本向量化，用于入库和检索 | 上传 PDF 时 + 每次提问时 |

选择此方案的考量：
- **Gemini Flash Lite**：响应速度快、成本低，适合高频的分类和问答任务
- **本地 VLM**：图表解释涉及图片传输，本地推理避免了上传大图到云端的延迟和隐私顾虑；Qwen2.5-VL-3B 4bit 量化后仅占约 2GB 显存，在 Apple Silicon 上可流畅运行
- **Gemini Embedding**：统一使用 Gemini 生态，只需一个 API Key

### 4.3 离线处理流水线

用户上传 PDF 后，系统在后台自动执行以下步骤：

1. **PDF 解析**（双引擎策略）
   - **PyMuPDF (fitz)**：遍历每页的嵌入式图片，提取为 PNG 文件，过滤 50×50 以下的小图（logo/icon）
   - **pdfplumber**：提取带结构的文字（能识别多栏排版），检测表格区域
   - **图注识别**：在图片 bbox 下方 40pt 范围内搜索 "Figure X: ..." 格式的文字

2. **文本分块**
   - 使用 `RecursiveCharacterTextSplitter`，每块 800 字符、重叠 100 字符
   - 切分优先级：段落 `\n\n` → 换行 `\n` → 句号 `. ` → 空格 → 字符
   - 自动识别章节标题（"1 Introduction"、"Abstract" 等），作为元数据标签

3. **向量化入库**
   - 正文 chunk → `text_chunks` collection（ChromaDB）
   - 图注文字 → `figure_captions` collection（ChromaDB，附带图片路径）
   - 使用 Gemini Embedding API，每批 100 条，429 自动重试

4. **状态追踪**
   - 全程更新 SQLite 中的状态：`pending → parsing → chunking → embedding → ready / failed`
   - 前端每 2 秒轮询 `/status` 接口，实时显示进度

### 4.4 在线问答流程（LangGraph）

用户提问后，系统通过 LangGraph 状态图依次执行：

```
classify(Gemini) → retrieve(ChromaDB) ─┬─ DocAgent(Gemini)         → 返回回答
                                        ├─ FigureAgent(本地 MLX VLM) → 返回回答
                                        └─ WriterAgent(Gemini)      → 返回回答
```

**Step 1 — 意图分类**：调用 Gemini，将用户 query 分为 `doc` / `figure` / `writer` 三类

**Step 2 — 向量检索**：
- 始终检索 `text_chunks` collection，返回 top-8 相关片段
- 如果 query 含"图/表/figure/table"等关键词，额外检索 `figure_captions` collection
- 检索结果附带 paper_id、页码、章节等元数据，用于回答时标注来源

**Step 3 — Agent 执行**（根据分类结果走不同分支）：

| Agent | 模型 | 策略 |
|-------|------|------|
| **DocAgent** | Gemini | 将检索到的 chunk 拼成上下文，要求 LLM 只基于上下文回答，不足则明确说明，每句标注 `[Paper: id, Page N]` |
| **FigureAgent** | 本地 Qwen2.5-VL | 将检索到的图片（最多 3 张）编码后传入 VLM，结合文本上下文分析图表类型、主要结论、数据趋势 |
| **WriterAgent** | Gemini | 两轮检索策略：第一轮用原始 query 检索 top-10，第二轮从结果中提取关键词再检索 top-6，合并去重后按主题组织成学术风格的 Related Work 段落 |

### 4.5 API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/upload` | POST | 上传 PDF，触发后台解析，返回 `task_id` |
| `/status/{task_id}` | GET | 查询解析进度（前端轮询用） |
| `/ask` | POST | 提交问题 `{paper_ids, query}`，返回 `{answer, sources}` |
| `/papers` | GET | 获取所有已入库论文列表 |
| `/figures/{paper_id}` | GET | 获取某篇论文的图片列表 |

所有接口统一返回格式：`{ success: bool, data: any, error: string | null }`

```bash
# 上传
curl -X POST http://localhost:8000/upload -F "file=@paper.pdf"

# 查询状态
curl http://localhost:8000/status/{task_id}

# 提问
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"paper_ids": ["a1b2c3"], "query": "What is the main contribution?"}'

# 论文列表
curl http://localhost:8000/papers

# 图片列表
curl http://localhost:8000/figures/{paper_id}
```

---

## 五、数据存储位置

所有运行时数据都在 `storage/` 目录下：

```
storage/
├── pdfs/           # 上传的 PDF 原文件
├── figures/        # 提取出的图片（按 paper_id 分目录）
├── chroma/         # ChromaDB 向量索引（持久化）
└── papers.db       # SQLite 元数据（论文信息 + 任务状态）
```

如需完全重置，删除 `storage/` 目录即可：

```bash
rm -rf storage/
```

---

## 六、项目结构

```
app/
├── main.py              # FastAPI 入口，注册路由和中间件
├── config.py            # 路径、模型名称、超参数集中配置
├── api/                 # 5 个 API 路由
│   ├── upload.py        # POST /upload — 上传 PDF，触发后台解析
│   ├── status.py        # GET /status/{task_id} — 查询解析进度
│   ├── ask.py           # POST /ask — 提交问题，返回回答
│   ├── papers.py        # GET /papers — 论文列表
│   └── figures.py       # GET /figures/{paper_id} — 图片列表
├── services/
│   ├── ingest.py        # 离线流水线：PDF → parse → chunk → embed → ready
│   └── chat.py          # 在线问答：query → LangGraph → answer
├── agents/
│   └── graph.py         # LangGraph 状态图 + 3 个 Agent（Doc/Figure/Writer）
├── parser/
│   ├── pdf_parser.py    # PyMuPDF + pdfplumber 双引擎 PDF 解析
│   └── chunker.py       # 文本分块 + 章节识别
├── retrieval/
│   └── retriever.py     # ChromaDB 向量检索 + Gemini Embedding
└── db/
    └── sqlite.py        # SQLite 元数据管理（papers + tasks 两张表）
ui/
└── streamlit_app.py     # Streamlit 前端界面
storage/                 # 运行时数据（自动创建，已加入 .gitignore）
```

---

## 七、常见问题

**Q: 上传后一直卡在 parsing 状态？**
检查终端 1 的 FastAPI 日志，可能是 PDF 文件损坏或格式不兼容。

**Q: 提问时报 Gemini API 错误？**
确认 `.env` 中的 `GEMINI_API_KEY` 正确，且账号有 Gemini API 访问权限。

**Q: Embedding 报 429 RESOURCE_EXHAUSTED？**
Gemini 免费版每分钟 100 次 embedding 请求限制。系统会自动等待重试，但如果频繁触发建议升级付费版。

**Q: 如何更换模型？**
编辑 `app/config.py`，修改 `GEMINI_MODEL`（文本）、`VLM_MODEL_ID`（视觉）、`EMBEDDING_MODEL`（向量化）。

**Q: 图表解释第一次很慢？**
首次触发 FigureAgent 时，系统从 HuggingFace 下载 Qwen2.5-VL-3B 模型（约 2GB），后续直接从缓存加载。图表解释本身需要约 30 秒（Prefill 阶段处理图片的 vision tokens 较多）。

**Q: MLX VLM 在 Intel Mac / Linux 上能用吗？**
`mlx-vlm` 仅支持 Apple Silicon（M1/M2/M3/M4）。如需跨平台，可将 `figure_agent` 改为调用 Gemini 的 vision 接口。

**Q: 想重新解析某篇论文？**
目前需要手动删除 `storage/` 下对应数据后重新上传。后续可扩展删除接口。

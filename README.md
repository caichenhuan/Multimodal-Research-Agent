# Multimodal Research Agent

面向科研场景的论文解析与问答系统。上传 PDF 论文后，系统自动完成文本抽取、图表提取、向量化入库，之后可以通过自然语言提问，得到有文献来源的回答。

---

## 一、环境准备

### 1.1 前置要求

- **Python 3.11+**（当前环境为 3.13）
- **OpenAI API Key**（需要能访问 `gpt-4o` 和 `text-embedding-3-small`）

### 1.2 创建虚拟环境（推荐）

```bash
cd "/Users/raphael/Documents/Projects/Multimodal Agent"

python3 -m venv .venv
source .venv/bin/activate
```

### 1.3 安装依赖

```bash
pip install -r requirements.txt
```

涉及的主要依赖：

| 包 | 用途 |
|---|------|
| fastapi + uvicorn | 后端 API 服务 |
| streamlit | 前端界面 |
| openai | GPT-4o 对话 + embedding |
| langgraph | Agent 编排状态图 |
| PyMuPDF (fitz) | PDF 图片提取 |
| pdfplumber | PDF 文字提取 |
| chromadb | 本地向量数据库 |
| langchain-text-splitters | 文本分块 |

### 1.4 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 OpenAI Key：

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 二、启动服务

需要打开 **两个终端窗口**，分别启动后端和前端。

### 终端 1：启动 FastAPI 后端

```bash
cd "/Users/raphael/Documents/Projects/Multimodal Agent"
source .venv/bin/activate

uvicorn app.main:app --reload --port 8000
```

看到以下输出说明启动成功：

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Started reloader process
```

### 终端 2：启动 Streamlit 前端

```bash
cd "/Users/raphael/Documents/Projects/Multimodal Agent"
source .venv/bin/activate

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

## 四、API 接口说明

如果你想直接调用后端 API（不通过 Streamlit），以下是五个可用接口：

### POST /upload

上传 PDF 文件，触发后台解析。

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@my_paper.pdf"
```

返回：

```json
{ "success": true, "data": { "paper_id": "a1b2c3d4e5f6", "task_id": "f6e5d4c3b2a1" } }
```

### GET /status/{task_id}

查询解析进度。

```bash
curl http://localhost:8000/status/f6e5d4c3b2a1
```

### POST /ask

提问。

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"paper_ids": ["a1b2c3d4e5f6"], "query": "What is the main contribution?"}'
```

### GET /papers

获取所有论文列表。

```bash
curl http://localhost:8000/papers
```

### GET /figures/{paper_id}

获取某篇论文的图片列表。

```bash
curl http://localhost:8000/figures/a1b2c3d4e5f6
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

## 六、项目架构

```
用户 ──→ Streamlit (8501) ──→ FastAPI (8000) ──→ LangGraph Agent
                                   │                    │
                                   ▼                    ▼
                              BackgroundTask       OpenAI API
                                   │              (GPT-4o / Embedding)
                                   ▼
                          PDF Parser (fitz + pdfplumber)
                                   │
                                   ▼
                          Chunker (RecursiveCharacterTextSplitter)
                                   │
                                   ▼
                          ChromaDB (向量入库)
```

```
app/
├── main.py              # FastAPI 入口，注册路由和中间件
├── config.py            # 路径、模型名称、超参数集中配置
├── api/                 # 5 个 API 路由
│   ├── upload.py        # POST /upload
│   ├── status.py        # GET /status/{task_id}
│   ├── ask.py           # POST /ask
│   ├── papers.py        # GET /papers
│   └── figures.py       # GET /figures/{paper_id}
├── services/
│   ├── ingest.py        # 离线流水线：PDF → chunks → embeddings
│   └── chat.py          # 在线问答：query → Agent graph → answer
├── agents/
│   └── graph.py         # LangGraph 状态图 + 3 个 Agent
├── parser/
│   ├── pdf_parser.py    # PyMuPDF + pdfplumber 双引擎解析
│   └── chunker.py       # 文本分块 + 章节识别
├── retrieval/
│   └── retriever.py     # ChromaDB 向量检索
└── db/
    └── sqlite.py        # SQLite 元数据管理
ui/
└── streamlit_app.py     # 前端界面
storage/                 # 运行时数据（自动创建）
```

---

## 七、常见问题

**Q: 上传后一直卡在 parsing 状态？**
检查终端 1 的 FastAPI 日志，可能是 PDF 文件损坏或格式不兼容。

**Q: 提问时报 OpenAI API 错误？**
确认 `.env` 中的 `OPENAI_API_KEY` 正确，且账号有 `gpt-4o` 访问权限。

**Q: 如何更换模型？**
编辑 `app/config.py`，修改 `CHAT_MODEL`、`VISION_MODEL`、`EMBEDDING_MODEL`。

**Q: 想重新解析某篇论文？**
目前需要手动删除 `storage/` 下对应数据后重新上传。后续可扩展删除接口。

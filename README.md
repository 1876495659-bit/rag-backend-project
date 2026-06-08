# RAG 企业知识库问答系统

基于 LangChain + FAISS + Qwen3(Ollama) 的企业级知识检索增强生成(Q&A)系统。

## 技术架构

```
┌──────────────────────────────────────────────────────────────┐
│                    RAG 问答流程                               │
│                                                              │
│  用户问题 ──→ 文本向量化 ──→ FAISS 检索 ──→ 重排序           │
│                        ──→ 拼接上下文 ──→ Qwen3 生成回答      │
│                                                              │
│  关键设计：                                                   │
│  - 防幻觉 Prompt：严格限制模型仅基于上下文回答                │
│  - 重排序优化：CrossEncoder 精排                             │
│  - 流式输出：SSE 逐 token 推送，实现打字机效果               │
│  - 持久化存储：索引保存到磁盘，避免重复向量化                 │
└──────────────────────────────────────────────────────────────┘
```

## 功能模块

| 模块 | 说明 |
|------|------|
| 文档上传 | 支持 PDF / Word 文档上传 |
| 文档解析 | 自动提取文本并切分为文本块 |
| 向量化 | 使用中文嵌入模型生成向量 |
| 向量检索 | FAISS 向量数据库，支持 Top-K 检索 |
| 重排序 | CrossEncoder 精排，提升检索精度 |
| 问答生成 | Qwen3 模型生成回答，防幻觉 Prompt |
| 流式输出 | SSE 协议逐 token 推送，打字机效果 |
| 聊天界面 | Vue3 + Element Plus 企业级 UI |

## 快速开始

### 1. 后端服务

```bash
# 安装依赖
pip install -r backend/requirements.txt

# 启动后端服务（http://localhost:8000）
python backend/project.py
```

### 2. 前端界面

```bash
cd frontend
npm install
npm run dev
# 访问 http://localhost:3000
```

### 3. Ollama 本地模型

```bash
# 安装 Ollama: https://ollama.com
ollama pull qwen3
```

## API 文档

启动后端后访问: http://localhost:8000/docs

| 接口 | 方法 | 说明 |
|------|------|------|
| `/upload` | POST | 上传并解析文档 |
| `/upload_batch` | POST | 批量上传并入库 |
| `/query` | POST | 普通问答接口 |
| `/query/stream` | POST | 流式问答接口 |
| `/search` | POST | 检索接口 |
| `/reset` | POST | 重置知识库 |
| `/list` | GET | 列出文档 |

## 项目结构

```
rag-backend-project/
├── backend/
│   ├── project.py          # FastAPI 主服务
│   ├── rag_core.py         # RAG 核心链路
│   ├── upload_handler.py   # 文档解析与切分
│   └── requirements.txt    # Python 依赖
├── frontend/
│   ├── package.json        # 前端依赖
│   ├── vite.config.js      # Vite 配置
│   └── src/
│       ├── main.js         # Vue 入口
│       └── App.vue         # 聊天界面组件
└── README.md
```

## 许可证

MIT

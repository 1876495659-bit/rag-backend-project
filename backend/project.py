"""
RAG 企业知识库问答系统 - FastAPI 后端主服务

本模块提供以下核心 API 接口：
1. POST /upload        - 上传单个文档（PDF/Word）
2. POST /upload_batch   - 上传并入库（解析+向量化+FAISS存储）
3. POST /query          - 问答接口（普通输出）
4. POST /query/stream   - 问答接口（流式输出，打字机效果）
5. POST /search         - 检索接口（查找相似文档）
6. POST /reset          - 重置知识库
7. GET  /list           - 列出知识库中的文档
8. GET  /               - 服务健康检查
"""

import os
import uuid
import asyncio
import json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# 导入 RAG 核心管道和文档解析器
# 兼容两种启动方式：从 backend/ 目录运行 & 从项目根目录运行
import sys, os
_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from backend.rag_core import RAGPipeline
from backend.upload_handler import DocumentParser

# ==========================================
# 创建 FastAPI 应用实例
# ==========================================
app = FastAPI(
    title="企业知识库 RAG 问答系统",
    description="基于 LangChain + FAISS + Qwen3 的检索增强生成知识库系统",
    version="1.0.0"
)

# ==========================================
# 配置跨域中间件，允许前端跨域访问
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # 生产环境建议限制为前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 配置数据目录
# ==========================================
BASE_DIR = Path(__file__).parent.parent / "data"
UPLOAD_DIR = BASE_DIR / "documents"       # 文档上传目录
INDEX_DIR = BASE_DIR / "faiss_index"      # FAISS 向量索引目录

# 创建必要的目录（如果不存在）
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 初始化 RAG 管道和文档解析器全局实例
# ==========================================
# RAGPipeline 负责向量化、检索、生成等核心逻辑
rag_pipeline = RAGPipeline(vectorstore_dir=str(INDEX_DIR))
# DocumentParser 负责解析 PDF/Word 文档并切分为文本块
doc_parser = DocumentParser(
    upload_dir=str(UPLOAD_DIR),
    chunk_size=500,
    chunk_overlap=100
)


# ==========================================
# 1. 服务健康检查接口
# ==========================================
@app.get("/")
async def root():
    """服务健康检查接口"""
    return {
        "status": "running",
        "title": "企业知识库 RAG 问答系统",
        "version": "1.0.0",
        "description": "基于 LangChain + FAISS + Qwen3(Ollama) 的检索增强生成知识库系统"
    }


# ==========================================
# 2. 文档上传接口（仅解析和切分，不入库）
# ==========================================
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    top_k: int = Form(default=3, description="检索返回的相似文档数量")
):
    """
    上传文档接口：接收文件并解析切分为多个文本块（chunk）

    - 支持格式：PDF、Word (.docx)
    - 每个 chunk 带有元数据：文件名、页码、原文
    - 返回切分后的 chunk 列表

    Args:
        file: 上传的文件对象
        top_k: 检索返回的相似文档数量

    Returns:
        包含切分后的 chunk 列表及总数
    """
    try:
        # 检查文件扩展名是否合法
        file_name = file.filename or ""
        if not (file_name.endswith(".pdf") or file_name.endswith(".docx")):
            return JSONResponse(
                status_code=400,
                content={"error": "仅支持 PDF 和 Word 文档格式"}
            )

        # 解析文档并切分为文本块
        result = await doc_parser.parse_and_chunk(
            file=file,
            top_k=top_k
        )

        return JSONResponse(content={
            "status": "ok",
            "file_name": file_name,
            "chunk_count": result.get("count", 0),
            "chunks": result.get("chunks", [])
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"文档解析失败: {str(e)}"}
        )


# ==========================================
# 3. 文档上传并入库接口（完整流程）
# ==========================================
@app.post("/upload_batch")
async def upload_batch(
    files: List[UploadFile] = File(...),       # 支持批量上传
    top_k: int = Form(default=3, description="检索返回的相似文档数量")
):
    """
    上传并入库接口：接收文件 → 解析切分 → 向量化 → 存入 FAISS 索引

    这是完整的 RAG 文档入库流程，包括：
    1. 读取文件内容
    2. 解析并切分为文本块
    3. 将文本块向量化
    4. 存入 FAISS 向量索引库
    5. 持久化保存索引

    Args:
        files: 批量上传的文件列表
        top_k: 检索返回的相似文档数量

    Returns:
        入库结果，包含成功/失败的文件数和总 chunk 数
    """
    try:
        total_chunks = 0
        success_files = []
        failed_files = []

        for file in files:
            file_name = file.filename or "unknown"

            # 检查文件扩展名
            if not (file_name.endswith(".pdf") or file_name.endswith(".docx")):
                failed_files.append({"file": file_name, "reason": "不支持的文件格式"})
                continue

            try:
                # 第一步：解析文档并切分为文本块
                parse_result = await doc_parser.parse_and_chunk(
                    file=file,
                    top_k=top_k
                )

                chunks = parse_result.get("chunks", [])
                if not chunks:
                    failed_files.append({"file": file_name, "reason": "文档解析后无内容"})
                    continue

                # 第二步：初始化或加载 FAISS 索引
                await rag_pipeline.load_or_init_index()

                # 第三步：将文本块向量化并存入 FAISS
                await rag_pipeline.add_chunks(chunks)

                # 第四步：持久化保存索引到磁盘
                rag_pipeline.save_index()

                total_chunks += len(chunks)
                success_files.append(file_name)

            except Exception as e:
                failed_files.append({"file": file_name, "reason": str(e)})

        return JSONResponse(content={
            "status": "ok",
            "success_files": success_files,
            "failed_files": failed_files,
            "total_chunks": total_chunks,
            "index_size": rag_pipeline.get_index_size()
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"文档入库失败: {str(e)}"}
        )


# ==========================================
# 4. 普通问答接口（一次性返回完整答案）
# ==========================================
@app.post("/query")
async def query(
    question: str = Form(..., description="用户提出的问题"),
    top_k: int = Form(default=3, description="检索返回的相似文档数量"),
    use_rerank: bool = Form(default=True, description="是否使用重排序优化")
):
    """
    问答接口（非流式）：接收问题 → 检索知识库 → 生成回答

    流程：
    1. 将用户问题向量化
    2. 在 FAISS 中检索 Top-K 相似文档
    3. 可选：对检索结果进行重排序
    4. 拼接检索结果作为上下文
    5. 调用 Qwen3 模型生成回答
    6. 返回完整回答和引用来源

    Args:
        question: 用户问题
        top_k: 检索返回的相似文档数量
        use_rerank: 是否启用重排序优化

    Returns:
        回答内容、引用来源和生成耗时
    """
    try:
        # 检查问题是否为空
        if not question.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "问题不能为空"}
            )

        # 执行完整的 RAG 流程
        result = await rag_pipeline.query(
            question=question,
            top_k=top_k,
            use_rerank=use_rerank
        )

        return JSONResponse(content={
            "status": "ok",
            "question": question,
            "answer": result.get("answer", "抱歉，未能生成回答。"),
            "references": result.get("references", []),    # 引用来源
            "context_used": result.get("context_used", ""), # 使用的上下文
            "response_time": result.get("response_time", 0) # 响应耗时（秒）
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"问答失败: {str(e)}"}
        )


# ==========================================
# 5. 流式问答接口（打字机效果）
# ==========================================
async def generate_answer_stream(question: str, top_k: int, use_rerank: bool):
    """
    流式生成回答的异步生成器

    将回答的每个 token 作为独立的 SSE（Server-Sent Event）数据流发送，
    前端可以实时逐字显示，实现 ChatGPT 式的打字机效果。

    数据格式：data: {"type": "answer", "content": "..."}/n/n
             data: {"type": "done", "content": ""}/n/n
    """
    try:
        # 执行 RAG 流程，使用生成器模式逐 token 产出
        for token in rag_pipeline.query_streaming(
            question=question,
            top_k=top_k,
            use_rerank=use_rerank
        ):
            # 将每个 token 封装为 SSE 格式发送
            sse_data = json.dumps({
                "type": "answer",
                "content": token
            }, ensure_ascii=False)
            yield f"data: {sse_data}\n\n"
            # 短暂延迟，模拟打字机效果（避免过快）
            await asyncio.sleep(0.02)

        # 发送完成信号
        yield f"data: {json.dumps({'type': 'done', 'content': ''}, ensure_ascii=False)}\n\n"

    except Exception as e:
        # 发送错误信号
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"


@app.post("/query/stream")
async def query_stream(
    question: str = Form(..., description="用户提出的问题"),
    top_k: int = Form(default=3, description="检索返回的相似文档数量"),
    use_rerank: bool = Form(default=True, description="是否使用重排序优化")
):
    """
    流式问答接口：实现 ChatGPT 式的打字机效果

    使用 Server-Sent Events (SSE) 协议，将回答逐 token 实时推送给前端。
    前端使用 EventSource 或 fetch API 接收流式数据并实时显示。

    Args:
        question: 用户问题
        top_k: 检索返回的相似文档数量
        use_rerank: 是否启用重排序优化

    Returns:
        SSE 数据流
    """
    try:
        # 检查问题是否为空
        if not question.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "问题不能为空"}
            )

        return StreamingResponse(
            generate_answer_stream(
                question=question,
                top_k=top_k,
                use_rerank=use_rerank
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"    # 禁用 Nginx 缓冲
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"流式生成失败: {str(e)}"}
        )


# ==========================================
# 6. 检索接口（仅搜索，不调用大模型）
# ==========================================
@app.post("/search")
async def search(
    query: str = Form(..., description="搜索关键词"),
    top_k: int = Form(default=5, description="返回的相似文档数量")
):
    """
    检索接口：仅执行向量搜索，返回相似文档片段

    适用于：
    - 搜索知识库中的相关内容
    - 展示检索结果供用户预览
    - 不调用大模型，节省资源

    Args:
        query: 搜索关键词
        top_k: 返回的相似文档数量

    Returns:
        包含相似文档片段列表
    """
    try:
        # 加载索引（如果尚未加载）
        await rag_pipeline.load_or_init_index()

        # 执行向量搜索
        results = rag_pipeline.search(query=query, top_k=top_k)

        return JSONResponse(content={
            "status": "ok",
            "query": query,
            "results": results,
            "total_results": len(results)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"检索失败: {str(e)}"}
        )


# ==========================================
# 7. 重置知识库接口
# ==========================================
@app.post("/reset")
async def reset_knowledge_base():
    """
    重置知识库：清空所有索引和文档数据

    谨慎操作！此操作将永久删除所有已上传的文档和向量索引。
    重置后需要重新上传文档才能使用问答功能。

    Returns:
        重置结果
    """
    try:
        # 重置 RAG 管道（清空向量索引和内存）
        rag_pipeline.reset()
        # 清空文档解析器的缓存
        doc_parser.reset()

        return JSONResponse(content={
            "status": "ok",
            "message": "知识库已重置，所有文档和索引已清空"
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"重置失败: {str(e)}"}
        )


# ==========================================
# 8. 列出知识库文档接口
# ==========================================
@app.get("/list")
async def list_documents():
    """
    列出知识库中所有已上传的文档及其统计信息

    返回：
    - 文档文件名列表
    - 每个文档的 chunk 数量
    - 总 chunk 数量
    - 向量索引大小
    """
    try:
        # 获取所有已解析的文档
        all_chunks = doc_parser.get_all_chunks()

        # 统计每个文档的 chunk 数量
        doc_stats = []
        total_chunks = 0
        for file_name, chunks in all_chunks.items():
            doc_stats.append({
                "file_name": file_name,
                "chunk_count": len(chunks)
            })
            total_chunks += len(chunks)

        return JSONResponse(content={
            "status": "ok",
            "documents": doc_stats,
            "total_documents": len(doc_stats),
            "total_chunks": total_chunks,
            "index_size": rag_pipeline.get_index_size()
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"获取文档列表失败: {str(e)}"}
        )


# ==========================================
# 启动入口
# ==========================================
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  🚀 企业知识库 RAG 问答系统 启动中...")
    print(f"  📡 服务地址: http://localhost:8000")
    print(f"  📚 API 文档: http://localhost:8000/docs")
    print(f"  📂 上传目录: {UPLOAD_DIR.absolute()}")
    print(f"  🗄️ 索引目录: {INDEX_DIR.absolute()}")
    print("=" * 60)

    # 使用 Uvicorn 启动 ASGI 服务器
    uvicorn.run(
        "backend.project:app",     # 应用模块路径
        host="0.0.0.0",           # 监听所有网络接口
        port=8000,                # 服务端口
        reload=True,              # 开发模式：代码自动热重载
        log_level="info"          # 日志级别
    )

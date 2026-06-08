"""
Backend project.py - FastAPI Core
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.project_rag_core import RAGPipeline
from backend.upload_handler import DocumentParser

app = FastAPI(title="Knowledge Base RAG", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./data/documents")
INDEX_DIR = Path("./data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

rag_pipeline = RAGPipeline(vectorstore_dir=str(INDEX_DIR))
doc_parser = DocumentParser(UPLOAD_DIR=str(UPLOAD_DIR))


@app.get("/")
async def status():
    return {"status": "running", "title": "Knowledge Base RAG Service"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    result = await doc_parser.parse_and_chunk(file=file)
    return {"status": "ok", "message": f"Extracted {len(result.get('chunks', []))} chunks"}


@app.post("/upload_batch")
async def upload_batch(file: UploadFile = File(...), user_id: str = Form(None)):
    result = await doc_parser.parse_and_chunk(file=file)
    await rag_pipeline.load_or_init_index()
    chunks_data = []
    for chunk in result.get("chunks", []):
        chunks_data.append(chunk)
    await rag_pipeline.add_chunks(chunks_data)
    return {"status": "ok", "chunk_count": len(chunks_data)}


@app.post("/search")
async def search(query: str = Form("")).dict():
    await rag_pipeline.load_or_init_index()
    results = await rag_pipeline.search(query=query)
    return results


@app.post("/reset")
async def reset_kb():
    rag_pipeline.reset()
    return {"message": "Knowledge base reset"}


@app.get("/list/{user_id}")
async def list_docs(user_id: str):
    return await doc_parser.list_chunks(user_id=user_id)


async def generate_answer_streaming(query: str):
    """Streaming generator for real-time output"""
    for chunk in rag_pipeline.generate_streaming(query):
        yield f"data: {chunk}\n\n"
        await asyncio.sleep(0.01)


@app.get("/query/{query}/stream/{user_id}")
async def stream_response(query: str, user_id: str):
    return StreamingResponse(generate_answer_streaming(query), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

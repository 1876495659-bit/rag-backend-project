"""
Upload Handler - Document Parsing and Chunking
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict
from fastapi import UploadFile


class DocumentParser:
    def __init__(self, upload_dir="./data/documents"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.storage = {}

    async def parse_and_chunk(self, file: UploadFile) -> Dict:
        content = await file.read()
        file_name = file.filename
        chunks = []
        if file_name.endswith('.pdf'):
            chunks = self._parse_pdf(content)
        elif file_name.endswith('.docx'):
            chunks = self._parse_docx(content)
        else:
            raise ValueError("Unsupported file type")
        result = self._split_chunks(chunks, file_name)
        return result

    def _parse_pdf(self, content: bytes) -> List[str]:
        from pypdf import PdfReader
        from io import BytesIO
        pdf_reader = PdfReader(BytesIO(content))
        texts = []
        for i, page in enumerate(pdf_reader.pages):
            texts.append(f"[Page {i+1}] {page.extract_text()}")
        return texts

    def _parse_docx(self, content: bytes) -> List[str]:
        from docx import Document
        from io import BytesIO
        doc = Document(BytesIO(content))
        texts = []
        for i, para in enumerate(doc.paragraphs):
            texts.append(f"[Paragraph {i+1}] {para.text}")
        return texts

    def _split_chunks(self, texts: List[str], file_name: str) -> Dict:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n", ". ", "。"]
        )
        chunks = []
        for text in texts:
            chunk_list = splitter.split_text(text)
            for i, chunk in enumerate(chunk_list):
                chunk_meta = {
                    'file_name': file_name,
                    'page_num': i + 1,
                    'text': chunk,
                    'metadata': {'source': file_name, 'page': i + 1}
                }
                chunks.append(chunk_meta)
        self.storage[file_name] = chunks
        return {'chunks': chunks, 'count': len(chunks)}

    async def list_chunks(self, user_id: str) -> List[Dict]:
        return [v for k, v in self.storage.items() if k.startswith(user_id)]

    async def delete_chunk(self, user_id: str, id: str) -> bool:
        for chunks in self.storage.values():
            for chunk in chunks:
                if chunk.get('id') == id:
                    chunks.remove(chunk)
                    return True
        return False

    def reset(self):
        self.storage.clear()

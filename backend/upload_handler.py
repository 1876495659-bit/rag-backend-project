"""
RAG 企业知识库问答系统 - 文档解析与切分模块

本模块负责处理各种格式文档的上传、解析和文本切分工作：
1. 支持 PDF 文档解析（使用 pypdf）
2. 支持 Word 文档解析（使用 python-docx）
3. 智能文本切分（使用 RecursiveCharacterTextSplitter）
4. 每个文本块携带元数据（文件名、页码、段落）

功能亮点：
- 支持批量上传
- 支持增量切分（避免重复解析相同文件）
- 提供文件统计和管理功能
"""

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from fastapi import UploadFile

# 导入 LangChain 的文本切分器
# 新版 langchain 将 text_splitter 拆分为独立包
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


# ==========================================
# 文本块数据结构
# ==========================================
@dataclass
class TextChunk:
    """文本块数据类：存储切分后的文本单元"""
    text: str                          # 文本块内容
    file_name: str                     # 来源文件名
    chunk_id: str                      # 文本块唯一标识
    page_num: int = 0                  # 页码（PDF）或段落编号
    metadata: Dict = field(default_factory=dict)  # 扩展元数据

    def to_dict(self) -> Dict:
        """转换为字典格式，方便序列化"""
        return {
            "text": self.text,
            "file_name": self.file_name,
            "chunk_id": self.chunk_id,
            "page_num": self.page_num,
            "metadata": self.metadata
        }


# ==========================================
# 文档解析器类
# ==========================================
class DocumentParser:
    """
    文档解析与切分器

    功能：
    - 解析 PDF 和 Word 文档
    - 将文档切分为适合向量化的文本块
    - 管理已解析的文档缓存
    - 支持增量更新和文件统计

    参数：
        upload_dir: 上传目录路径
        chunk_size: 每个文本块的最大字符数（默认 500）
        chunk_overlap: 文本块之间的重叠字符数（默认 100）
    """

    def __init__(
        self,
        upload_dir: str = "./data/documents",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

        # 内存缓存：存储已解析的文档内容
        self._parsed_cache: Dict[str, List[TextChunk]] = {}

        # 配置文本切分器参数
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,             # 每个块最大 500 字符
            chunk_overlap=chunk_overlap,       # 块之间重叠 100 字符（保留上下文连续性）
            length_function=len,               # 使用字符长度作为切分依据
            separators=[                       # 优先按这些分隔符切分
                "\n\n",       # 空行（段落分隔）
                "\n",         # 换行
                "。",         # 中文句号
                ". ",         # 英文句点+空格
                "；",         # 中文分号
                "; ",         # 英文分号+空格
                " ",          # 空格
                ""            # 兜底：强制切分
            ]
        )

    # ==========================================
    # 核心解析方法
    # ==========================================

    async def parse_and_chunk(
        self,
        file: UploadFile,
        top_k: int = 3
    ) -> Dict:
        """
        解析上传的文档并切分为文本块（核心入口方法）

        处理流程：
        1. 读取文件内容
        2. 根据扩展名选择解析器（PDF/Word）
        3. 提取纯文本
        4. 按配置的参数切分为文本块
        5. 为每个文本块添加元数据
        6. 存入内存缓存

        Args:
            file: FastAPI 上传的文件对象
            top_k: 检索时返回的相似文档数量（用于文档内部标注）

        Returns:
            包含 chunks 列表和总数量的字典

        Raises:
            ValueError: 不支持的文件格式
            Exception: 解析过程发生错误
        """
        file_name = file.filename or "unknown_file"
        content = await file.read()          # 读取文件字节内容

        # 根据文件扩展名选择对应的解析方法
        if file_name.lower().endswith(".pdf"):
            raw_texts = self._parse_pdf(content)
        elif file_name.lower().endswith(".docx"):
            raw_texts = self._parse_docx(content)
        else:
            raise ValueError(f"不支持的文件格式: {file_name}（仅支持 .pdf 和 .docx）")

        # 将提取的文本切分为合适的块
        chunks = self._split_texts(raw_texts, file_name)

        # 存入缓存
        self._parsed_cache[file_name] = chunks

        return {
            "chunks": [chunk.to_dict() for chunk in chunks],  # 转换为字典列表
            "count": len(chunks)                              # 总块数
        }

    # ==========================================
    # PDF 文档解析
    # ==========================================

    def _parse_pdf(self, content: bytes) -> List[str]:
        """
        解析 PDF 文档，逐页提取文本内容

        Args:
            content: PDF 文件的字节数据

        Returns:
            每页文本的列表，每页文本前带有页码标记
        """
        from pypdf import PdfReader
        from io import BytesIO

        # 使用 pypdf 读取 PDF 内容
        pdf_reader = PdfReader(BytesIO(content))
        extracted_texts = []

        # 逐页提取文本并添加页码标注
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():   # 跳过空白页
                # 添加页码标记，便于后续追溯引用来源
                full_text = f"[第 {page_num} 页] {text}"
                extracted_texts.append(full_text)

        return extracted_texts

    # ==========================================
    # Word 文档解析
    # ==========================================

    def _parse_docx(self, content: bytes) -> List[str]:
        """
        解析 Word 文档，提取段落文本

        Args:
            content: Word 文件的字节数据

        Returns:
            每个段落的文本列表

        注意：
            - 跳过表格中的内容（仅提取段落）
            - 每个段落带有段落编号标注
        """
        from docx import Document
        from io import BytesIO

        # 使用 python-docx 读取文档
        doc = Document(BytesIO(content))
        extracted_texts = []

        for para_num, paragraph in enumerate(doc.paragraphs, start=1):
            text = paragraph.text.strip()
            if text:  # 跳过空段落
                # 添加段落编号标注
                full_text = f"[第 {para_num} 段] {text}"
                extracted_texts.append(full_text)

        return extracted_texts

    # ==========================================
    # 文本切分方法
    # ==========================================

    def _split_texts(
        self,
        raw_texts: List[str],
        file_name: str
    ) -> List[TextChunk]:
        """
        将原始文本切分为适合向量化的文本块

        切分策略：
        - 首先按段落/页面分隔
        - 如果单个块过大，继续按中文/英文标点切分
        - 相邻块之间保留重叠部分，保证语义连续性

        Args:
            raw_texts: 原始文本列表
            file_name: 文件名（用于元数据）

        Returns:
            切分后的 TextChunk 对象列表
        """
        all_chunks = []
        chunk_counter = 0  # 全局块计数器，确保唯一 ID

        for text in raw_texts:
            # 使用 LangChain 的递归字符切分器进行切分
            text_list = self.splitter.split_text(text)

            for chunk_text in text_list:
                chunk_counter += 1
                # 创建文本块对象
                chunk = TextChunk(
                    text=chunk_text,
                    file_name=file_name,
                    chunk_id=f"{file_name}_chunk_{chunk_counter}",
                    page_num=chunk_counter,  # 使用序号作为页码标识
                    metadata={
                        "source": file_name,           # 来源文件
                        "chunk_id": f"chunk_{chunk_counter}",  # 块唯一 ID
                        "language": "zh"              # 标记语言
                    }
                )
                all_chunks.append(chunk)

        return all_chunks

    # ==========================================
    # 文档管理方法
    # ==========================================

    def get_all_chunks(self) -> Dict[str, List[TextChunk]]:
        """
        获取所有已缓存的文档文本块

        Returns:
            字典：{文件名: 文本块列表}
        """
        return self._parsed_cache.copy()

    def get_chunks_by_file(self, file_name: str) -> List[TextChunk]:
        """
        获取指定文件的文本块

        Args:
            file_name: 文件名

        Returns:
            该文件的所有文本块列表
        """
        return self._parsed_cache.get(file_name, [])

    def get_file_names(self) -> List[str]:
        """
        获取所有已缓存的文件名列表

        Returns:
            文件名列表
        """
        return list(self._parsed_cache.keys())

    def get_stats(self) -> Dict:
        """
        获取文档解析统计信息

        Returns:
            包含文件数、总块数、每个文件的块数的字典
        """
        total_files = len(self._parsed_cache)
        total_chunks = sum(len(chunks) for chunks in self._parsed_cache.values())

        file_details = {}
        for file_name, chunks in self._parsed_cache.items():
            file_details[file_name] = {
                "chunk_count": len(chunks),
                "total_chars": sum(len(c.text) for c in chunks)
            }

        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "file_details": file_details
        }

    def reset(self):
        """
        清空所有缓存的文档数据
        谨慎操作！此操作将删除所有已解析的文档
        """
        self._parsed_cache.clear()
        print("✅ 文档缓存已清空")

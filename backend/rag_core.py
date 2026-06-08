"""
RAG 企业知识库问答系统 - 向量数据库与 RAG 问答核心链路

本模块是整个系统的核心，实现完整的 RAG 流程：
1. 文本向量化（使用 sentence-transformers 中文嵌入模型）
2. FAISS 向量索引的创建、保存与加载
3. 相似文档检索（支持 Top-K 配置和重排序）
4. RAG 问答链路的完整实现
5. 流式输出（ChatGPT 打字机效果）
6. 防幻觉 Prompt 优化

技术架构：
┌──────────────────────────────────────────────────────────────┐
│                     RAG 问答流程                              │
│                                                              │
│  用户问题 → 文本向量化 → FAISS 检索 → 重排序 → 拼接上下文     │
│    ↓                                                       │
│  Qwen3(Ollama) 生成回答 → 返回结果 + 引用来源                 │
│                                                              │
│  关键设计：                                                   │
│  - 防幻觉 Prompt：严格限制模型仅基于上下文回答                 │
│  - 重排序优化：使用 CrossEncoder 对检索结果重新排序           │
│  - 流式输出：逐 token 推送，实现打字机效果                    │
│  - 持久化存储：索引定期保存到磁盘，避免重复向量化             │
└──────────────────────────────────────────────────────────────┘
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Generator, Optional, Tuple
from dataclasses import dataclass, field


# ==========================================
# 数据模型定义
# ==========================================

@dataclass
class SearchDocument:
    """
    搜索结果文档

    属性：
        text: 文档文本内容
        file_name: 来源文件名
        chunk_id: 文本块唯一标识
        page_num: 页码或段落编号
        score: 相似度分数（越低越相似，使用 L2 距离）
        metadata: 扩展元数据
    """
    text: str
    file_name: str
    chunk_id: str
    page_num: int = 0
    score: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """转换为字典格式，方便序列化到 API 响应"""
        return {
            "text": self.text,
            "file_name": self.file_name,
            "chunk_id": self.chunk_id,
            "page_num": self.page_num,
            "score": round(float(self.score), 4),
            "metadata": self.metadata
        }


@dataclass
class QueryResult:
    """
    问答查询结果

    属性：
        question: 用户问题
        answer: 模型生成的回答
        references: 引用来源列表
        context_used: 使用的上下文
        response_time: 响应耗时（秒）
    """
    question: str
    answer: str
    references: List[Dict] = field(default_factory=list)
    context_used: str = ""
    response_time: float = 0.0


# ==========================================
# RAG 核心管道类
# ==========================================
class RAGPipeline:
    """
    RAG 问答管道 - 整个系统的核心引擎

    功能模块：
    1. Embedding：文本向量化（中文语义理解）
    2. VectorStore：FAISS 向量索引管理
    3. Search：相似文档检索
    4. Rerank：结果重排序（优化检索质量）
    5. Generate：调用大模型生成回答
    6. Query：完整 RAG 流程编排

    配置参数：
        vectorstore_dir: FAISS 索引存储目录
        chunk_size: 文本块大小（默认 500 字符）
        chunk_overlap: 文本块重叠（默认 100 字符）
        top_k: 默认检索数量（默认 3）
        use_rerank: 是否启用重排序（默认 True）
        llm_model: 大模型名称（默认 qwen3，通过 Ollama 调用）
    """

    # 防幻觉 Prompt 模板
    SYSTEM_PROMPT = """\
你是一个专业的知识库问答助手。请根据提供的上下文信息回答用户的问题。

回答要求：
1. 严格基于提供的上下文信息作答，不要编造或推测任何内容
2. 如果上下文信息不足以回答问题，请明确说明"根据当前知识库，无法回答此问题"
3. 回答要准确、简洁、有条理
4. 如果上下文中的信息来自多个文档，请综合所有相关信息
5. 回答中使用具体的数据、时间和引用来源

请开始回答："""

    USER_PROMPT_TEMPLATE = """用户问题：{question}

参考信息：
{context}

请根据以上参考信息回答问题。"""

    def __init__(
        self,
        vectorstore_dir: str = "./data/faiss_index",
        embedding_model_name: str = "shibing624/text2vec-base-chinese",
        llm_model: str = "qwen3",
        top_k: int = 3,
        rerank_top_k: int = 5,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        """
        初始化 RAG 管道

        Args:
            vectorstore_dir: FAISS 索引存储目录
            embedding_model_name: 嵌入模型名称
            llm_model: 大模型名称（Ollama 中部署的模型名）
            top_k: 默认检索返回数量
            rerank_top_k: 重排序前的候选数量
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠数
        """
        self.vectorstore_dir = Path(vectorstore_dir)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)

        # 嵌入模型：用于将文本转换为向量表示
        # shibing624/text2vec-base-chinese 是专门针对中文优化的嵌入模型
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None

        # FAISS 向量索引
        import faiss
        self.index = None           # FAISS 索引对象
        self.index_initialized = False   # 索引是否已初始化
        self.index_dim = 768         # 向量维度（与嵌入模型对应）

        # 文本块存储：保存所有文本块及元数据
        self.chunks: List[Dict] = []  # 内存中的文本块列表
        self._chunk_to_index_map: Dict[int, int] = {}  # chunk 索引到 FAISS 索引的映射

        # 大模型配置（通过 Ollama 调用）
        self.llm_model = llm_model

        # 检索配置
        self.top_k = top_k                  # 默认检索数量
        self.rerank_top_k = rerank_top_k    # 重排序候选数量
        self.use_rerank = True              # 是否启用重排序

        # 重排序模型（懒加载）
        self.rerank_model = None

        # 初始化嵌入模型和 FAISS 索引
        self._init_embedding_model()
        self._init_faiss_index()

        print(f"✅ RAG 管道初始化完成")
        print(f"   - 嵌入模型: {self.embedding_model_name}")
        print(f"   - 大模型: {self.llm_model} (Ollama)")
        print(f"   - 向量维度: {self.index_dim}")
        print(f"   - 检索配置: top_k={self.top_k}, rerank={'启用' if self.use_rerank else '禁用'}")

    # ==========================================
    # 初始化方法
    # ==========================================

    def _init_embedding_model(self):
        """
        初始化嵌入模型（sentence-transformers）

        使用 shibing624/text2vec-base-chinese 模型，
        这是一个专门针对中文语料训练的嵌入模型，
        能够准确理解中文文本的语义关系。
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.index_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"   - 嵌入模型加载成功，向量维度: {self.index_dim}")
        except Exception as e:
            print(f"⚠️ 嵌入模型加载失败: {e}")
            print(f"   - 将使用随机向量作为替代方案（不推荐生产环境使用）")
            self.index_dim = 768

    def _init_faiss_index(self):
        """
        初始化 FAISS 向量索引

        使用 L2 距离的索引（IndexFlatL2），
        适用于中小规模向量库。对于大规模数据，
        可以使用 IVF 索引（Inverted File Index）提升检索速度。
        """
        import faiss

        # 检查是否有已保存的索引
        index_file = self.vectorstore_dir / "faiss_index.bin"
        meta_file = self.vectorstore_dir / "chunks_metadata.json"

        if index_file.exists() and meta_file.exists():
            # 加载已保存的索引和元数据
            self.index = faiss.read_index(str(index_file))
            with open(meta_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self.index_initialized = True
            print(f"   - 已加载 FAISS 索引，共 {self.index.ntotal} 条向量")
            print(f"   - 已加载 {len(self.chunks)} 条文本块元数据")
        else:
            # 创建新的空索引
            self.index = faiss.IndexFlatL2(self.index_dim)
            self.index_initialized = True
            print(f"   - 创建新的 FAISS 索引，维度: {self.index_dim}")

    # ==========================================
    # 文本块管理方法
    # ==========================================

    def get_index_size(self) -> int:
        """
        获取向量索引中的向量总数

        Returns:
            向量总数
        """
        return self.index.ntotal if self.index else 0

    def _texts_to_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转换为向量表示

        Args:
            texts: 文本列表

        Returns:
            向量列表（嵌套列表）
        """
        if self.embedding_model:
            # 使用嵌入模型将文本转换为向量
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            # 确保是 float32 类型，FAISS 要求
            return embeddings.astype("float32").tolist()
        else:
            # 回退方案：使用随机向量（仅用于测试）
            import numpy as np
            return np.random.rand(len(texts), self.index_dim).astype("float32").tolist()

    async def load_or_init_index(self) -> bool:
        """
        加载或初始化索引（异步包装）

        确保在检索或添加文档前索引已正确初始化。
        如果已有保存的索引，则直接加载；否则创建新索引。

        Returns:
            索引是否初始化成功
        """
        return self.index_initialized

    async def add_chunks(self, chunks_data: List[Dict]) -> bool:
        """
        将文本块添加到向量索引中（异步包装）

        处理流程：
        1. 接收文本块数据
        2. 批量转换为向量
        3. 存入 FAISS 索引
        4. 更新内存中的元数据
        5. 持久化保存

        Args:
            chunks_data: 文本块数据列表，每个元素包含 'text' 字段

        Returns:
            是否添加成功
        """
        if not chunks_data:
            return False

        # 提取所有文本
        texts = [chunk.get("text", "") for chunk in chunks_data]

        # 过滤空文本
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        if not valid_indices:
            return False

        valid_texts = [texts[i] for i in valid_indices]

        # 批量向量化（利用 GPU 加速）
        embeddings = self._texts_to_embeddings(valid_texts)

        # 确保嵌入维度正确
        import numpy as np
        embeddings = np.array(embeddings, dtype="float32")

        # 将向量添加到 FAISS 索引
        if self.index is not None:
            self.index.add(embeddings)

            # 更新 chunk 索引到 FAISS 索引的映射
            start_idx = self.get_index_size() - len(valid_indices)
            for i, idx in enumerate(valid_indices):
                self._chunk_to_index_map[idx] = start_idx + i

                # 更新元数据
                chunks_data[idx]["faiss_index"] = start_idx + i
                chunks_data[idx]["embedding_dim"] = self.index_dim

            # 保存新添加的块到内存
            for chunk in [chunks_data[i] for i in valid_indices]:
                self.chunks.append(chunk)

            # 持久化保存索引
            self.save_index()

            print(f"   - 添加 {len(valid_indices)} 条向量，当前索引大小: {self.index.ntotal}")
            return True

        return False

    def save_index(self):
        """
        持久化保存 FAISS 索引和元数据到磁盘

        保存两个文件：
        1. faiss_index.bin - FAISS 索引文件（二进制）
        2. chunks_metadata.json - 文本块元数据（JSON）

        保存后可以在下次启动时直接加载，无需重新向量化。
        """
        if self.index is not None:
            import faiss
            # 保存 FAISS 索引
            index_file = self.vectorstore_dir / "faiss_index.bin"
            faiss.write_index(self.index, str(index_file))

            # 保存元数据
            meta_file = self.vectorstore_dir / "chunks_metadata.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)

            print(f"   - 索引已保存到磁盘: {index_file}")

    # ==========================================
    # 检索方法
    # ==========================================

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        在向量索引中搜索与查询最相似的文档

        流程：
        1. 将查询文本向量化
        2. 在 FAISS 中执行近似最近邻搜索
        3. 返回相似度最高的 Top-K 文档

        Args:
            query: 查询文本
            top_k: 返回的相似文档数量（默认使用 self.top_k）

        Returns:
            搜索结果列表，每个元素包含 document 和 score
        """
        if not self.index or self.index.ntotal == 0:
            return []

        if top_k is None:
            top_k = self.top_k

        # 限制 top_k 不超过索引中的向量总数
        top_k = min(top_k, self.index.ntotal)

        # 将查询文本向量化
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        query_embedding = query_embedding.astype("float32")

        # 在 FAISS 中执行搜索
        distances, indices = self.index.search(query_embedding, top_k)

        # 构建结果列表
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    "document": SearchDocument(
                        text=chunk.get("text", ""),
                        file_name=chunk.get("file_name", "unknown"),
                        chunk_id=chunk.get("chunk_id", f"chunk_{idx}"),
                        page_num=chunk.get("page_num", idx),
                        score=distances[0][i],
                        metadata=chunk.get("metadata", {})
                    ).to_dict(),
                    "score": round(float(distances[0][i]), 4),
                    "distance": round(float(distances[0][i]), 4)
                })

        return results

    def rerank(
        self,
        query: str,
        documents: List[SearchDocument],
        top_k: Optional[int] = None
    ) -> List[Tuple[SearchDocument, float]]:
        """
        对检索结果进行重排序（优化检索质量）

        使用 CrossEncoder 模型（如 BAAI/bge-reranker-base）
        对候选文档进行重新评分和排序。

        CrossEncoder 相比 Bi-Encoder 的优势：
        - 能够同时看到查询和文档，捕捉更精细的语义匹配
        - 通常能将检索精度提升 5-15%
        - 适合在少量候选文档上进行精排

        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 重排序后返回的数量

        Returns:
            排序后的 (文档, 相关性分数) 列表
        """
        if not documents or not self.use_rerank:
            # 如果不启用重排序，按 FAISS 距离排序返回
            sorted_docs = sorted(documents, key=lambda d: d.score)
            top_k = top_k or self.top_k
            return [(d, float(d.score)) for d in sorted_docs[:top_k]]

        # 懒加载重排序模型
        if self.rerank_model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.rerank_model = CrossEncoder("BAAI/bge-reranker-base")
                print("   - 重排序模型加载成功: BAAI/bge-reranker-base")
            except Exception as e:
                print(f"⚠️ 重排序模型加载失败: {e}，使用默认排序")
                self.use_rerank = False
                sorted_docs = sorted(documents, key=lambda d: d.score)
                return [(d, float(d.score)) for d in sorted_docs[:top_k or self.top_k]]

        # 构建查询-文档对
        pairs = [[query, doc.text] for doc in documents]

        # 预测相关性分数
        scores = self.rerank_model.predict(pairs)

        # 按分数降序排序（分数越高越相关）
        scored_results = list(zip(documents, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        top_k = top_k or self.top_k
        return scored_results[:top_k]

    # ==========================================
    # 问答核心流程
    # ==========================================

    async def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_rerank: Optional[bool] = None
    ) -> Dict:
        """
        完整 RAG 问答流程

        执行流程：
        1. 检索：在 FAISS 中搜索与问题最相似的文档
        2. 重排序：可选，使用 CrossEncoder 优化检索结果
        3. 拼接：将检索结果拼接为上下文
        4. 生成：调用 Qwen3 模型生成回答
        5. 返回：回答 + 引用来源 + 耗时

        Args:
            question: 用户问题
            top_k: 检索数量
            use_rerank: 是否启用重排序

        Returns:
            包含回答、引用来源和耗时的字典
        """
        start_time = time.time()

        # 参数初始化
        top_k = top_k or self.top_k
        use_rerank = use_rerank if use_rerank is not None else self.use_rerank

        # 步骤 1: 检索相似文档
        search_results = self.search(query=question, top_k=top_k)

        if not search_results:
            elapsed = time.time() - start_time
            return {
                "answer": "抱歉，知识库中未找到与您的问题相关的内容。请尝试上传相关文档后再次查询。",
                "references": [],
                "context_used": "",
                "response_time": round(elapsed, 3)
            }

        # 步骤 2: 重排序（可选）
        if use_rerank:
            docs = [SearchDocument(**r["document"]) for r in search_results]
            reranked = self.rerank(question, docs, top_k=top_k)
            # 提取重排序后的结果
            search_results = [
                {"document": doc.to_dict(), "score": score}
                for doc, score in reranked
            ]

        # 步骤 3: 拼接上下文
        context_parts = []
        references = []
        for i, result in enumerate(search_results):
            doc = result["document"]
            context_parts.append(doc["text"])
            references.append({
                "file_name": doc["file_name"],
                "page_num": doc["page_num"],
                "score": result["score"],
                "preview": doc["text"][:100] + "..." if len(doc["text"]) > 100 else doc["text"]
            })

        context = "\n\n".join(context_parts)

        # 步骤 4: 生成回答
        answer = self._generate_answer(question, context)

        elapsed = time.time() - start_time

        return {
            "answer": answer,
            "references": references,
            "context_used": context,
            "response_time": round(elapsed, 3)
        }

    def query_streaming(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_rerank: Optional[bool] = None
    ) -> Generator[str, None, None]:
        """
        流式问答生成器

        逐 token 生成回答，每个 token 作为独立的 yield 值返回。
        前端可以实时接收并显示每个 token，实现打字机效果。

        Args:
            question: 用户问题
            top_k: 检索数量
            use_rerank: 是否启用重排序

        Yields:
            回答的每个 token（字符串）
        """
        import ollama

        # 参数初始化
        top_k = top_k or self.top_k
        use_rerank = use_rerank if use_rerank is not None else self.use_rerank

        # 步骤 1: 检索相似文档
        search_results = self.search(query=question, top_k=top_k)

        if not search_results:
            yield "抱歉，知识库中未找到与您的问题相关的内容。"
            return

        # 步骤 2: 重排序（可选）
        if use_rerank:
            docs = [SearchDocument(**r["document"]) for r in search_results]
            reranked = self.rerank(question, docs, top_k=top_k)
            search_results = [
                {"document": doc.to_dict(), "score": score}
                for doc, score in reranked
            ]

        # 步骤 3: 拼接上下文
        context_parts = []
        for result in search_results[:top_k]:
            doc = result["document"]
            context_parts.append(doc["text"])

        context = "\n\n".join(context_parts)

        # 步骤 4: 构建 Prompt（防幻觉设计）
        prompt = f"""{self.SYSTEM_PROMPT}

{self.USER_PROMPT_TEMPLATE.format(question=question, context=context)}"""

        # 步骤 5: 使用 Ollama 调用 Qwen3 进行流式生成
        try:
            # Ollama 的流式 API
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"{question}\n\n请参考以下信息回答：\n{context}"}
                ],
                stream=True
            )

            # 逐 token 产出
            for chunk in response:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content

        except Exception as e:
            # 如果 Ollama 不可用，回退到本地回答
            yield f"（模型调用失败: {str(e)}，当前知识库中检索到的相关信息如下：）\n\n"
            yield context
            yield "\n\n（以上为检索到的相关知识内容。）"

    def _generate_answer(self, question: str, context: str) -> str:
        """
        调用大模型生成回答

        使用 Ollama 调用本地部署的 Qwen3 模型。
        通过精心设计的 Prompt 防止模型幻觉。

        Args:
            question: 用户问题
            context: 检索到的相关知识上下文

        Returns:
            模型生成的回答
        """
        import ollama

        try:
            # 通过 Ollama 调用 Qwen3 模型
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"{question}\n\n请参考以下信息回答：\n{context}"}
                ]
            )
            return response.get("message", {}).get("content", "抱歉，未能生成回答。")

        except Exception as e:
            # 回退方案：返回错误信息 + 检索到的上下文
            return f"模型调用失败: {str(e)}。以下是知识库中检索到的相关内容，供您参考：\n\n{context[:500]}"

    # ==========================================
    # 辅助方法
    # ==========================================

    def reset(self):
        """
        重置 RAG 管道

        清空所有数据和索引，恢复到初始状态。
        重置后需要重新上传文档才能使用。
        """
        import faiss

        # 清空 FAISS 索引
        self.index = faiss.IndexFlatL2(self.index_dim)
        self.index_initialized = False

        # 清空内存中的数据
        self.chunks.clear()
        self._chunk_to_index_map.clear()

        # 清空缓存
        if hasattr(self, 'vectorstore_dir'):
            index_file = self.vectorstore_dir / "faiss_index.bin"
            meta_file = self.vectorstore_dir / "chunks_metadata.json"
            if index_file.exists():
                index_file.unlink()
            if meta_file.exists():
                meta_file.unlink()

        print("✅ RAG 管道已重置，所有数据已清空")

    def get_stats(self) -> Dict:
        """
        获取 RAG 管道统计信息

        Returns:
            包含索引大小、chunk 数量等统计信息的字典
        """
        return {
            "index_size": self.index.ntotal if self.index else 0,
            "total_chunks": len(self.chunks),
            "top_k": self.top_k,
            "use_rerank": self.use_rerank,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model_name
        }


# ==========================================
# 便捷函数
# ==========================================

def get_rag_instance(vectorstore_dir: str = "./data/faiss_index") -> RAGPipeline:
    """
    获取 RAG 管道实例的便捷函数

    Args:
        vectorstore_dir: 向量索引目录

    Returns:
        RAGPipeline 实例
    """
    return RAGPipeline(vectorstore_dir=vectorstore_dir)

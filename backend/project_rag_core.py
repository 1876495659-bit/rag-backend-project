import os
import asyncio
import numpy as np
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from typing import List, Generator


class RAGPipeline:
    def __init__(self, vectorstore_dir="./data/index"):
        self.vectorstore_dir = vectorstore_dir
        os.makedirs(vectorstore_dir, exist_ok=True)
        self.index_path = os.path.join(vectorstore_dir, "faiss_index")
        self.chunks_store = []
        self.vectorstore = None
        self.embedding_model = None
        self.llm = "qwen3"  # default model name
        self.rerank_model = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        self._load_or_init_model()

    def _load_or_init_model(self):
        if not self.vectorstore:
            self.vectorstore = self._load_or_create_store()
        if not self.embedding_model:
            self.embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')

    async def load_or_init_index(self):
        if self.vectorstore:
            return True
        if self.chunks_store:
            await self.add_chunks(self.chunks_store)
        return True

    def _load_or_create_store(self):
        import faiss
        return faiss.IndexFlatL2(768)

    async def add_chunks(self, chunks_data: list):
        import numpy as np
        import faiss
        if not chunks_data:
            return
        for chunk in chunks_data:
            self.chunks_store.append(chunk)
        texts = [c.get('text', '') for c in self.chunks_store]
        if self.embedding_model:
            embeddings = self.embedding_model.encode(texts)
        else:
            embeddings = np.array([np.random.random(768) for _ in range(len(texts))])
        embeddings = embeddings.astype(np.float32)
        if self.vectorstore:
            if self.vectorstore.ntotal > 0:
                self.vectorstore.add(embeddings)
            else:
                self.vectorstore.add(embeddings)
        return True

    def search(self, query: str, top_k: int = 3) -> list:
        import numpy as np
        if self.embedding_model:
            query_vec = self.embedding_model.encode([query])
        else:
            query_vec = np.random.randn(1, 768).astype(np.float32)
        if self.vectorstore:
            distances, indices = self.vectorstore.search(query_vec, top_k)
        else:
            return []
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks_store):
                results.append({
                    'chunk': self.chunks_store[idx],
                    'distance': distances[0][i]
                })
        return results

    def generate(self, query: str, context: str) -> str:
        from langchain_community.llms import Ollama
        llm = Ollama(model=self.llm)
        prompt = f"""
        You are a knowledgeable assistant helping to answer questions based on the provided context.
        User Question: {query}
        Context: {context}
        Instructions:
        1. Answer ONLY using the provided context.
        2. If the context does not contain enough information, reply: "I cannot answer this question based on the provided information."
        3. Do not invent or hallucinate information.
        4. Answer in a concise, professional manner.
        """
        return llm.invoke(prompt)

    def generate_streaming(self, query: str, top_k: int = 3) -> Generator:
        results = self.search(query, top_k=top_k)
        context = "\n".join([r.get('chunk', {}).get('text', '') for r in results])
        if not context:
            yield "No information found in the knowledge base."
            return
        from langchain_community.llms import Ollama
        llm = Ollama(model=self.llm, stream=True)
        prompt = f"""
        You are a knowledgeable assistant. Based on the context, answer the user's question.
        Question: {query}
        Context: {context}
        Instructions:
        1. Answer ONLY using the provided context.
        2. If the context is insufficient, state that.
        3. Do not invent or hallucinate information.
        """
        for chunk in llm.stream(prompt):
            yield chunk

    def rerank(self, query: str, candidates: list, top_k: int = 3) -> list:
        if not candidates:
            return []
        from sentence_transformers import CrossEncoder
        self.rerank_model = CrossEncoder('BAAI/bge-reranker-base')
        pairs = [[query, c.get('text', '')] for c in candidates]
        scores = self.rerank_model.predict(pairs)
        scored_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]


def get_rag_instance(vectorstore_dir="./data/index"):
    return RAGPipeline(vectorstore_dir=vectorstore_dir)

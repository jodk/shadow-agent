"""Chroma 中文向量检索。

这个模块故意与 SQLiteMemory 分开：

1. SQLite 保存完整原文并负责 FTS5/BM25；
2. Chroma 保存向量索引；
3. SentenceTransformer 负责把中文文本转换成向量。

模型和 Chroma 数据都放在 memory.db 同级目录，便于学习和迁移整个数据目录。
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


logger = logging.getLogger(__name__)


class ChineseEmbeddingModel:
    """延迟加载中文 embedding 模型。

    SentenceTransformer 在 ``model_path`` 已存在时只读取本地文件；目录不存在时，
    会按 ``model_name`` 从 Hugging Face 下载，并保存到该目录。模型加载和编码都
    是 CPU/GPU 密集型同步操作，因此通过 ``asyncio.to_thread`` 放到工作线程。
    """

    def __init__(
        self,
        model_name: str,
        model_path: Path,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_path = Path(model_path)
        # 不把 None 交给 SentenceTransformer 自动选择设备。Apple Silicon 上自动
        # 选择通常会启用 MPS，而 MPS 在 asyncio 后台线程中运行时可能导致底层
        # PyTorch/Metal 直接 segmentation fault。学习项目默认优先保证稳定性。
        self.configured_device = device
        self.device = device or "cpu"
        self._model: Any = None
        self._operation_lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def _get_model(self) -> Any:
        """返回已经加载的模型。

        调用方必须先取得 ``_operation_lock``。加载和 encode 共用同一把锁，可以
        防止后台索引与在线向量检索同时操作同一个 PyTorch 模型实例。
        """
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_model)
        return self._model

    def _get_operation_lock(self) -> asyncio.Lock:
        # asyncio.Lock 绑定事件循环。API 进程通常只有一个 loop，但测试或嵌入式
        # 使用可能先后创建多个 loop；发现 loop 变化时重新创建锁，避免跨 loop 报错。
        current_loop = asyncio.get_running_loop()
        if self._loop is not current_loop:
            self._loop = current_loop
            self._operation_lock = asyncio.Lock()
        return self._operation_lock

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            logger.exception(
                "embedding_dependency_missing model=%s model_path=%s",
                self.model_name,
                self.model_path,
            )
            raise RuntimeError(
                "缺少 sentence-transformers，请执行 pip install -e '.[vector]'"
            ) from exc

        logger.info(
            "embedding_device_selected model=%s configured=%s actual=%s",
            self.model_name,
            self.configured_device or "default",
            self.device,
        )
        local_model_exists = (self.model_path / "modules.json").exists()
        if local_model_exists:
            logger.info(
                "embedding_model_local_found model=%s path=%s action=offline_load",
                self.model_name,
                self.model_path,
            )
            try:
                model = SentenceTransformer(
                    str(self.model_path),
                    device=self.device,
                )
            except Exception:
                logger.exception(
                    "embedding_model_local_load_failed model=%s path=%s",
                    self.model_name,
                    self.model_path,
                )
                raise
            logger.info(
                "embedding_model_loaded model=%s path=%s source=local device=%s",
                self.model_name,
                self.model_path,
                self.device,
            )
            return model

        logger.info(
            "embedding_model_local_missing model=%s path=%s action=download",
            self.model_name,
            self.model_path,
        )
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info(
                "embedding_model_download_started model=%s target=%s device=%s",
                self.model_name,
                self.model_path,
                self.device,
            )
            model = SentenceTransformer(self.model_name, device=self.device)
            model.save(str(self.model_path))
        except Exception:
            logger.exception(
                "embedding_model_download_failed model=%s target=%s",
                self.model_name,
                self.model_path,
            )
            raise
        logger.info(
            "embedding_model_download_finished model=%s path=%s",
            self.model_name,
            self.model_path,
        )
        logger.info(
            "embedding_model_loaded model=%s path=%s source=downloaded device=%s",
            self.model_name,
            self.model_path,
            self.device,
        )
        return model

    async def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """批量生成文档向量，并归一化以适合 cosine 距离。"""
        if not texts:
            return []
        # 模型加载和编码严格串行。即使后台索引 worker 与聊天检索同时运行，也不会
        # 从两个线程并发调用同一个 SentenceTransformer/PyTorch 模型。
        async with self._get_operation_lock():
            model = await self._get_model()
            logger.info(
                "embedding_documents_started model=%s document_count=%d device=%s",
                self.model_name,
                len(texts),
                self.device,
            )
            try:
                vectors = await asyncio.to_thread(
                    model.encode,
                    list(texts),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception:
                logger.exception(
                    "embedding_documents_failed model=%s document_count=%d device=%s",
                    self.model_name,
                    len(texts),
                    self.device,
                )
                raise
            logger.info(
                "embedding_documents_finished model=%s document_count=%d "
                "vector_dimension=%d device=%s",
                self.model_name,
                len(texts),
                len(vectors[0]) if len(vectors) else 0,
                self.device,
            )
        # SentenceTransformer 通常返回 NumPy 数组；这里也兼容普通 Python
        # 列表，方便替换模型和编写不依赖 NumPy 的单元测试。
        return [
            vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for vector in vectors
        ]

    async def embed_query(self, text: str) -> List[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]


class ChromaVectorStore:
    """Chroma 持久化向量索引。

    Chroma 的同步客户端调用统一放到线程中，避免阻塞 FastAPI 的事件循环。
    文档 id 直接使用 SQLite message id，因此两套存储能稳定关联。
    """

    def __init__(
        self,
        persist_directory: Path,
        embedding_model: ChineseEmbeddingModel,
        collection_name: str = "messages",
    ):
        self.persist_directory = Path(persist_directory)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self._collection: Any = None
        self._client_lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def _get_collection(self) -> Any:
        # 与 embedding 模型相同，避免把绑定旧事件循环的锁带到新 loop。
        current_loop = asyncio.get_running_loop()
        if self._loop is not current_loop:
            self._loop = current_loop
            self._client_lock = asyncio.Lock()
        if self._collection is not None:
            return self._collection
        async with self._client_lock:
            if self._collection is None:
                self._collection = await asyncio.to_thread(self._create_collection)
        return self._collection

    def _create_collection(self) -> Any:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "缺少 chromadb，请执行 pip install -e '.[vector]'"
            ) from exc

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_directory))
        collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "chroma_collection_ready path=%s collection=%s",
            self.persist_directory,
            self.collection_name,
        )
        return collection

    async def upsert(
        self,
        message_id: int,
        session_id: str,
        role: str,
        text: str,
    ) -> None:
        vector = (await self.embedding_model.embed_documents([text]))[0]
        collection = await self._get_collection()
        await asyncio.to_thread(
            collection.upsert,
            ids=[str(message_id)],
            documents=[text],
            embeddings=[vector],
            metadatas=[{"session_id": session_id, "role": role}],
        )
        logger.info(
            "chroma_message_indexed message_id=%d session_id=%s",
            message_id,
            session_id,
        )

    async def search(
        self,
        query: str,
        limit: int,
        session_id: Optional[str] = None,
        exclude_session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        collection = await self._get_collection()
        if await asyncio.to_thread(collection.count) == 0:
            return []
        query_vector = await self.embedding_model.embed_query(query)

        where: Optional[Dict[str, Any]] = None
        if session_id is not None:
            where = {"session_id": session_id}
        elif exclude_session_id is not None:
            where = {"session_id": {"$ne": exclude_session_id}}

        result = await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_vector],
            n_results=limit,
            where=where,
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            {
                "message_id": int(message_id),
                "score": 1.0 - float(distance),
            }
            for message_id, distance in zip(ids, distances)
        ]

    async def delete_session(self, session_id: str) -> None:
        collection = await self._get_collection()
        await asyncio.to_thread(
            collection.delete,
            where={"session_id": session_id},
        )
        logger.info("chroma_session_deleted session_id=%s", session_id)

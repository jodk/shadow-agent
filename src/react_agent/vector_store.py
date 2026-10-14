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
        # ``Path(model_path)`` 允许调用方传入字符串或 Path，内部统一成 Path。
        self.model_name = model_name
        self.model_path = Path(model_path)
        # 不把 None 交给 SentenceTransformer 自动选择设备。Apple Silicon 上自动
        # 选择通常会启用 MPS，而 MPS 在 asyncio 后台线程中运行时可能导致底层
        # PyTorch/Metal 直接 segmentation fault。学习项目默认优先保证稳定性。
        self.configured_device = device
        self.device = device or "cpu"
        # ``Any`` 表示这里暂时不限制具体模型类型，因为 SentenceTransformer
        # 是可选依赖，导入失败时项目仍然可以使用 SQLite/BM25。
        self._model: Any = None
        # 模型是同步对象，而 Agent 的主流程是异步的；这把锁保证同一个模型
        # 不会被多个协程同时 encode。
        self._operation_lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def _get_model(self) -> Any:
        """返回已经加载的模型。

        调用方必须先取得 ``_operation_lock``。加载和 encode 共用同一把锁，可以
        防止后台索引与在线向量检索同时操作同一个 PyTorch 模型实例。
        """
        # ``is None`` 是判断“没有对象”的惯用写法，比 ``if not self._model``
        # 更准确，因为某些对象可能定义了特殊的真假值行为。
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_model)
        return self._model

    def _get_operation_lock(self) -> asyncio.Lock:
        # asyncio.Lock 绑定事件循环。API 进程通常只有一个 loop，但测试或嵌入式
        # 使用可能先后创建多个 loop；发现 loop 变化时重新创建锁，避免跨 loop 报错。
        # ``get_running_loop`` 只能在异步事件循环中调用；这里的调用位置都在
        # async 方法内，因此可以得到当前正在运行的 loop。
        current_loop = asyncio.get_running_loop()
        if self._loop is not current_loop:
            self._loop = current_loop
            self._operation_lock = asyncio.Lock()
        return self._operation_lock

    def _load_model(self) -> Any:
        # 延迟导入：只有真正启用向量功能时才需要安装 sentence-transformers。
        # 这样纯 BM25 模式不会因为缺少可选依赖而无法启动。
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
        # 不只判断目录是否存在，还检查 SentenceTransformer 的关键文件，
        # 避免把一个空目录误认为“已有完整模型”。
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
        # ``async with`` 会在进入时获取锁，在离开代码块时自动释放锁，
        # 即使中途抛出异常也能释放，避免后续请求永久等待。
        async with self._get_operation_lock():
            model = await self._get_model()
            logger.info(
                "embedding_documents_started model=%s document_count=%d device=%s",
                self.model_name,
                len(texts),
                self.device,
            )
            try:
                # SentenceTransformer.encode 是同步函数。to_thread 将其放到
                # 工作线程，避免阻塞 asyncio 事件循环处理聊天请求。
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
        # 列表推导式把每个 NumPy 向量转换成普通 list，便于 JSON/Chroma 使用。
        return [
            vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for vector in vectors
        ]

    async def embed_query(self, text: str) -> List[float]:
        # 查询向量和文档向量必须使用同一个 embedding 模型生成，才能比较距离。
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
        # Chroma 的 collection 是同步 SDK 对象，保存起来后后续请求可以复用。
        self._collection: Any = None
        self._client_lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def _get_collection(self) -> Any:
        # 与 embedding 模型相同，避免把绑定旧事件循环的锁带到新 loop。
        current_loop = asyncio.get_running_loop()
        if self._loop is not current_loop:
            self._loop = current_loop
            self._client_lock = asyncio.Lock()
        # 这是“缓存 + 双重检查”的简单形式：已经创建过就直接返回，
        # 未创建时才获取锁，避免每次查询都重复初始化 Chroma。
        if self._collection is not None:
            return self._collection
        async with self._client_lock:
            if self._collection is None:
                self._collection = await asyncio.to_thread(self._create_collection)
        return self._collection

    def _create_collection(self) -> Any:
        # Chroma 是可选依赖，所以同样采用函数内部导入，支持纯 BM25 启动。
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "缺少 chromadb，请执行 pip install -e '.[vector]'"
            ) from exc

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        # PersistentClient 会把向量数据落盘，而不是只保存在内存中。
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
        # embed_documents 返回“多个向量组成的列表”，这里只提交一条文本，
        # 所以用 [0] 取出第一条向量。
        vector = (await self.embedding_model.embed_documents([text]))[0]
        collection = await self._get_collection()
        # upsert = update or insert：id 已存在时更新，不存在时插入。
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
        # 负数或 0 没有合理的检索结果，直接返回空列表。
        if limit <= 0:
            return []
        collection = await self._get_collection()
        if await asyncio.to_thread(collection.count) == 0:
            return []
        query_vector = await self.embedding_model.embed_query(query)

        # Optional[Dict[...]] 表示 where 要么是字典，要么是 None。
        # None 表示不增加元数据过滤条件。
        where: Optional[Dict[str, Any]] = None
        if session_id is not None:
            where = {"session_id": session_id}
        elif exclude_session_id is not None:
            where = {"session_id": {"$ne": exclude_session_id}}

        # Chroma query 返回的是嵌套列表，例如 ids[0] 对应本次查询。
        result = await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_vector],
            n_results=limit,
            where=where,
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        # zip 会把 id 和 distance 按位置配对；列表推导式再转换成项目内部
        # 统一的结果格式。距离越小越相似，因此用 1-distance 变成相似度。
        return [
            {
                "message_id": int(message_id),
                "score": 1.0 - float(distance),
            }
            for message_id, distance in zip(ids, distances)
        ]

    async def delete_session(self, session_id: str) -> None:
        collection = await self._get_collection()
        # where 是按 metadata 过滤，删除指定会话的全部向量。
        await asyncio.to_thread(
            collection.delete,
            where={"session_id": session_id},
        )
        logger.info("chroma_session_deleted session_id=%s", session_id)

    async def sync_capabilities(self, documents: Sequence[Any]) -> None:
        """同步能力文档到独立的 Chroma collection。

        能力数量通常远小于聊天消息，因此这里采用一次批量 upsert。向量生成失败
        会由调用方记录并降级到 BM25，不影响主流程。
        """
        collection = await self._get_collection()
        # f-string 将变量嵌入字符串；列表推导式为每个能力生成稳定 id。
        active_ids = [f"capability:{item.capability_id}" for item in documents]
        if documents:
            # 先提取搜索文本，再一次性生成向量，减少模型调用次数。
            vectors = await self.embedding_model.embed_documents(
                [item.search_text for item in documents]
            )
            await asyncio.to_thread(
                collection.upsert,
                ids=active_ids,
                documents=[item.search_text for item in documents],
                embeddings=vectors,
                # 这里的列表推导式与 documents 一一对应，保证每个向量的元数据
                # 与它的 capability_id 保持同一顺序。
                metadatas=[
                    {
                        "capability_id": item.capability_id,
                        "kind": item.kind,
                        "name": item.name,
                        "skill_name": item.skill_name or "",
                    }
                    for item in documents
                ],
            )
        # 不传 include，兼容不同版本 Chroma 对 include=[] 的参数校验。
        # get 不传 id 时读取当前 collection 中已有的能力记录，用于清理旧数据。
        existing = await asyncio.to_thread(collection.get)
        existing_ids = set(existing.get("ids", []))
        # 集合差集 ``A - B`` 得到“旧的但不再活跃”的 id；转成 list 后交给 Chroma。
        stale_ids = list(existing_ids - set(active_ids))
        if stale_ids:
            await asyncio.to_thread(collection.delete, ids=stale_ids)
        logger.info(
            "chroma_capabilities_synchronized count=%d removed_count=%d",
            len(documents),
            len(stale_ids),
        )

    async def search_capabilities(
        self, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        collection = await self._get_collection()
        if await asyncio.to_thread(collection.count) == 0:
            return []
        query_vector = await self.embedding_model.embed_query(query)
        result = await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_vector],
            n_results=limit,
            include=["distances", "metadatas"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        # ``zip`` 同时遍历三个同长度序列；metadata 用 get 提供默认值，
        # 兼容旧索引中可能缺少 capability_id 的记录。
        return [
            {
                "capability_id": metadata.get("capability_id", item_id.removeprefix("capability:")),
                "score": 1.0 - float(distance),
            }
            for item_id, distance, metadata in zip(ids, distances, metadatas)
        ]

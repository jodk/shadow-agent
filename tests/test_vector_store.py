import asyncio
import logging
from pathlib import Path

import pytest

from react_agent.vector_store import ChineseEmbeddingModel


class FakeSentenceTransformer:
    """测试用的本地模型，避免测试联网下载真实模型。"""

    loaded_sources = []
    saved_paths = []

    def __init__(self, source, device=None):
        self.source = source
        self.device = device
        self.loaded_sources.append(source)

    def save(self, path):
        self.saved_paths.append(path)
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "modules.json").write_text("{}", encoding="utf-8")

    def encode(self, texts, normalize_embeddings, show_progress_bar):
        assert normalize_embeddings is True
        assert show_progress_bar is False
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_embedding_model_loads_existing_directory_offline(
    tmp_path: Path, monkeypatch, caplog
):
    model_path = tmp_path / "models" / "local-model"
    model_path.mkdir(parents=True)
    (model_path / "modules.json").write_text("{}", encoding="utf-8")
    FakeSentenceTransformer.loaded_sources.clear()

    import sentence_transformers

    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )
    embedding_model = ChineseEmbeddingModel("remote/model", model_path)

    with caplog.at_level(logging.DEBUG, logger="react_agent.vector_store"):
        vectors = await embedding_model.embed_documents(["测试文本"])

    assert vectors == [[1.0, 0.0]]
    assert FakeSentenceTransformer.loaded_sources == [str(model_path)]
    assert "embedding_model_local_found" in caplog.text
    assert "action=offline_load" in caplog.text
    assert "embedding_model_loaded" in caplog.text
    assert "source=local" in caplog.text
    assert "configured=default actual=cpu" in caplog.text
    assert FakeSentenceTransformer.loaded_sources == [str(model_path)]


@pytest.mark.asyncio
async def test_embedding_model_downloads_and_saves_when_directory_missing(
    tmp_path: Path, monkeypatch, caplog
):
    model_path = tmp_path / "models" / "downloaded-model"
    FakeSentenceTransformer.loaded_sources.clear()
    FakeSentenceTransformer.saved_paths.clear()

    import sentence_transformers

    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )
    embedding_model = ChineseEmbeddingModel("remote/model", model_path)

    with caplog.at_level(logging.INFO, logger="react_agent.vector_store"):
        await embedding_model.embed_documents(["测试文本"])

    assert FakeSentenceTransformer.loaded_sources == ["remote/model"]
    assert FakeSentenceTransformer.saved_paths == [str(model_path)]
    assert (model_path / "modules.json").exists()
    assert "embedding_model_local_missing" in caplog.text
    assert "embedding_model_download_started" in caplog.text
    assert "embedding_model_download_finished" in caplog.text
    assert "source=downloaded" in caplog.text


@pytest.mark.asyncio
async def test_embedding_model_serializes_concurrent_encode_calls(
    tmp_path: Path, monkeypatch
):
    """后台索引和在线检索并发时，同一个模型不能被两个线程同时调用。"""

    model_path = tmp_path / "models" / "local-model"
    model_path.mkdir(parents=True)
    (model_path / "modules.json").write_text("{}", encoding="utf-8")

    class ConcurrencyCheckingTransformer(FakeSentenceTransformer):
        active_calls = 0
        max_active_calls = 0

        def encode(self, texts, normalize_embeddings, show_progress_bar):
            type(self).active_calls += 1
            type(self).max_active_calls = max(
                type(self).max_active_calls,
                type(self).active_calls,
            )
            try:
                # encode 运行在线程池中，用短暂阻塞扩大两个任务重叠的概率。
                import time

                time.sleep(0.05)
                return super().encode(
                    texts,
                    normalize_embeddings,
                    show_progress_bar,
                )
            finally:
                type(self).active_calls -= 1

    import sentence_transformers

    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        ConcurrencyCheckingTransformer,
    )
    embedding_model = ChineseEmbeddingModel("remote/model", model_path)

    first, second = await asyncio.gather(
        embedding_model.embed_documents(["第一条"]),
        embedding_model.embed_documents(["第二条"]),
    )

    assert first == [[1.0, 0.0]]
    assert second == [[1.0, 0.0]]
    assert ConcurrencyCheckingTransformer.max_active_calls == 1

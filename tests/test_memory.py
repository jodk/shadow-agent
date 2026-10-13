import json
import sqlite3
from pathlib import Path

import pytest

from react_agent.memory import MemoryHit, SQLiteMemory, reciprocal_rank_fusion
from react_agent.model import TokenUsage


class FakeVectorStore:
    def __init__(self):
        self.documents = {}
        self.upsert_started = False

    async def upsert(self, message_id, session_id, role, text):
        self.upsert_started = True
        self.documents[message_id] = {
            "session_id": session_id,
            "role": role,
            "text": text,
        }

    async def search(self, query, limit, session_id=None, exclude_session_id=None):
        results = []
        for message_id, item in self.documents.items():
            if session_id is not None and item["session_id"] != session_id:
                continue
            if exclude_session_id is not None and item["session_id"] == exclude_session_id:
                continue
            if query.lower() in item["text"].lower():
                results.append({"message_id": message_id, "score": 0.9})
        return results[:limit]

    async def delete_session(self, session_id):
        self.documents = {
            key: item
            for key, item in self.documents.items()
            if item["session_id"] != session_id
        }


def test_sqlite_memory_preserves_original_messages(tmp_path: Path):
    memory = SQLiteMemory(tmp_path / "memory.db")

    memory.append_message("session-1", {"role": "user", "content": "我喜欢使用 Python"})
    memory.append_message(
        "session-1",
        {
            "role": "assistant",
            "content": "好的",
            "tool_calls": [{"id": "call-1", "name": "demo"}],
        },
    )

    messages = memory.get_messages("session-1")

    assert messages == [
        {"role": "user", "content": "我喜欢使用 Python"},
        {
            "role": "assistant",
            "content": "好的",
            "tool_calls": [{"id": "call-1", "name": "demo"}],
        },
    ]


def test_sqlite_memory_keeps_messages_and_fts_in_one_database_file(tmp_path: Path):
    database_path = tmp_path / "memory.db"
    memory = SQLiteMemory(database_path)
    memory.append_message("session-1", {"role": "user", "content": "SQLite BM25"})

    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert "messages" in tables
    assert "messages_fts" in tables
    # Chroma 是独立持久化目录，不再要求 SQLite 支持动态扩展。
    assert "message_vectors" not in tables


def test_bm25_search_returns_matching_message(tmp_path: Path):
    memory = SQLiteMemory(tmp_path / "memory.db")
    memory.append_message("session-1", {"role": "user", "content": "我喜欢 Python 和 SQLite"})
    memory.append_message("session-1", {"role": "user", "content": "今天讨论天气"})

    results = memory.search_keyword("session-1", "Python SQLite")

    assert len(results) == 1
    assert results[0].message["content"] == "我喜欢 Python 和 SQLite"
    assert results[0].score > 0


def test_cross_session_keyword_search_excludes_current_session(tmp_path: Path):
    memory = SQLiteMemory(tmp_path / "memory.db")
    memory.append_message("old", {"role": "user", "content": "我叫张德坤"})
    memory.append_message("new", {"role": "user", "content": "我叫什么名字"})

    hits = memory.search_keyword(None, "张德坤", exclude_session_id="new")

    assert [hit.message["content"] for hit in hits] == ["我叫张德坤"]


@pytest.mark.asyncio
async def test_context_builder_keeps_recent_and_retrieves_old_messages(tmp_path: Path):
    memory = SQLiteMemory(tmp_path / "memory.db")
    memory.append_message(
        "session-1",
        {"role": "user", "content": "项目必须使用 Python 3.9"},
    )
    memory.append_message(
        "session-1",
        {"role": "assistant", "content": "我会兼容 Python 3.9"},
    )
    memory.append_message("session-1", {"role": "user", "content": "今天讨论网页颜色"})

    context = await memory.build_context(
        "session-1",
        current_message="请继续实现 Python 3.9 的功能",
        recent_limit=1,
        keyword_limit=3,
        cross_session=False,
    )

    assert context[0]["role"] == "system"
    assert "Python 3.9" in context[0]["content"]
    assert context[-1]["content"] == "今天讨论网页颜色"


@pytest.mark.asyncio
async def test_context_builder_retrieves_memory_from_another_session(tmp_path: Path):
    memory = SQLiteMemory(tmp_path / "memory.db")
    memory.append_message(
        "old-session",
        {"role": "user", "content": "请记住，我叫张德坤"},
    )
    memory.append_message(
        "old-session",
        {"role": "assistant", "content": "好的，我记住你叫张德坤了"},
    )
    memory.append_message(
        "new-session",
        {"role": "user", "content": "你还记得我吗"},
    )

    context = await memory.build_context(
        "new-session",
        current_message="我叫什么名字",
        recent_limit=8,
        keyword_limit=5,
        vector_limit=0,
        fused_limit=5,
        cross_session=True,
    )

    assert any(
        "张德坤" in str(message.get("content"))
        for message in context
    )


@pytest.mark.asyncio
async def test_cross_session_context_contains_name_for_new_question(tmp_path: Path):
    memory = SQLiteMemory(tmp_path / "memory.db")
    memory.append_message(
        "old-session",
        {"role": "user", "content": "请记住，我叫张德坤"},
    )
    memory.append_message(
        "old-session",
        {"role": "assistant", "content": "好的，我记住你叫张德坤了"},
    )

    context = await memory.build_context(
        "new-session",
        current_message="我叫什么名字",
        recent_limit=8,
        keyword_limit=5,
        vector_limit=0,
        fused_limit=5,
        cross_session=True,
    )
    context_text = json.dumps(context, ensure_ascii=False)

    assert "张德坤" in context_text
    assert "我叫什么名字" not in context_text


def test_session_usage_is_accumulated_and_deleted_with_session(tmp_path: Path):
    memory = SQLiteMemory(tmp_path / "memory.db")

    memory.add_usage("session-1", TokenUsage(prompt_tokens=10, completion_tokens=4))
    total = memory.add_usage(
        "session-1",
        TokenUsage(prompt_tokens=3, completion_tokens=2, estimated=True),
    )

    assert total.as_dict() == {
        "prompt_tokens": 13,
        "completion_tokens": 6,
        "total_tokens": 19,
        "estimated": True,
    }
    memory.delete_session("session-1")
    assert memory.get_usage("session-1").total_tokens == 0


def test_list_sessions_returns_latest_first_with_usage(tmp_path: Path):
    memory = SQLiteMemory(tmp_path / "memory.db")
    memory.append_message("old", {"role": "user", "content": "旧会话"})
    memory.append_message("new", {"role": "user", "content": "新会话"})
    memory.append_message("new", {"role": "assistant", "content": "最新回答"})
    memory.add_usage("new", TokenUsage(prompt_tokens=7, completion_tokens=3))

    sessions = memory.list_sessions()

    assert [item["session_id"] for item in sessions] == ["new", "old"]
    assert sessions[0]["title"] == "新会话"
    assert sessions[0]["preview"] == "最新回答"
    assert sessions[0]["usage"]["total_tokens"] == 10


def test_rrf_fuses_two_ranked_lists_by_rank_not_raw_score():
    keyword_hits = [
        MemoryHit(1, {"role": "user", "content": "exact"}, 0.01, "bm25"),
        MemoryHit(2, {"role": "user", "content": "keyword-second"}, 0.02, "bm25"),
    ]
    vector_hits = [
        MemoryHit(2, {"role": "user", "content": "keyword-second"}, 0.99, "vector"),
        MemoryHit(3, {"role": "user", "content": "semantic"}, 0.98, "vector"),
    ]

    fused = reciprocal_rank_fusion(
        [keyword_hits, vector_hits],
        k=60,
        limit=3,
    )

    assert [hit.message_id for hit in fused] == [2, 1, 3]
    assert fused[0].source == "rrf:bm25+vector"
    assert fused[0].score > fused[1].score


@pytest.mark.asyncio
async def test_vector_index_is_generated_by_background_worker(tmp_path: Path):
    vector_store = FakeVectorStore()
    memory = SQLiteMemory(tmp_path / "memory.db", vector_store=vector_store)

    memory.append_message("session-1", {"role": "user", "content": "Python agent"})

    # append_message 返回时只保证 SQLite 原文已保存，向量任务在后台执行。
    await memory.wait_for_indexing()
    hits = await memory.search_vector("session-1", "Python", limit=1)

    assert vector_store.upsert_started is True
    assert len(hits) == 1
    assert hits[0].source == "vector_chroma"
    assert hits[0].message["content"] == "Python agent"
    await memory.close()

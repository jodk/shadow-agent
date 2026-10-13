import json

from fastapi.testclient import TestClient

from react_agent.agent import ReActAgent
from react_agent.api import create_app
from react_agent.model import AssistantMessage, ModelResponse, TokenUsage
from react_agent.tools import ToolRegistry, tool


class EchoModel:
    calls = 0

    async def complete(self, messages, tools):
        self.calls += 1
        user_content = next(
            item["content"] for item in reversed(messages) if item["role"] == "user"
        )
        return ModelResponse(
            message=AssistantMessage(content=f"Echo: {user_content}"),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )


@tool
def demo_tool(value: str) -> str:
    """A demo tool."""
    return value


def make_client():
    agent = ReActAgent(EchoModel(), ToolRegistry([demo_tool]))
    return TestClient(create_app(agent=agent))


def parse_sse(text):
    """把测试客户端收集到的完整 SSE 响应还原为事件列表。"""
    events = []
    for frame in text.strip().split("\n\n"):
        event_type = "message"
        data_lines = []
        for line in frame.splitlines():
            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").lstrip())
        if data_lines:
            events.append(
                {
                    "type": event_type,
                    "data": json.loads("\n".join(data_lines)),
                }
            )
    return events


def test_stream_chat_keeps_session_history():
    client = make_client()

    first = client.post(
        "/api/chat/stream",
        json={"session_id": "demo", "message": "hello"},
    )
    second = client.post(
        "/api/chat/stream",
        json={"session_id": "demo", "message": "again"},
    )

    first_events = parse_sse(first.text)
    assert first.headers["content-type"].startswith("text/event-stream")
    assert first.headers["cache-control"] == "no-cache, no-transform"
    assert first.headers["x-accel-buffering"] == "no"
    assert first_events[0]["type"] == "connected"
    assert first_events[-1]["type"] == "done"
    assert first_events[-1]["data"]["content"] == "Echo: hello"
    assert first_events[-1]["data"]["usage"]["total_tokens"] == 15
    assert first_events[-1]["data"]["session_usage"]["total_tokens"] == 15
    assert second.status_code == 200
    history_response = client.get("/api/sessions/demo")
    history = history_response.json()["messages"]
    assert [message["role"] for message in history] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert history_response.json()["usage"]["total_tokens"] == 30
    assert client.get("/api/sessions/demo/usage").json()["total_tokens"] == 30


def test_lists_tools_and_clears_session():
    client = make_client()
    client.post("/api/chat/stream", json={"session_id": "demo", "message": "hello"})

    tools = client.get("/api/tools").json()
    cleared = client.delete("/api/sessions/demo")

    assert tools[0]["name"] == "demo_tool"
    assert cleared.status_code == 204
    cleared_session = client.get("/api/sessions/demo").json()
    assert cleared_session["messages"] == []
    assert cleared_session["usage"]["total_tokens"] == 0


def test_lists_sessions():
    client = make_client()
    client.post(
        "/api/chat/stream",
        json={"session_id": "session-a", "message": "first"},
    )
    client.post(
        "/api/chat/stream",
        json={"session_id": "session-b", "message": "second"},
    )

    response = client.get("/api/sessions")

    assert response.status_code == 200
    session_ids = {item["session_id"] for item in response.json()}
    assert {"session-a", "session-b"}.issubset(session_ids)


def test_delete_session_removes_messages_usage_and_list_item():
    client = make_client()
    client.post(
        "/api/chat/stream",
        json={"session_id": "to-delete", "message": "temporary"},
    )

    response = client.delete("/api/sessions/to-delete")

    assert response.status_code == 204
    assert client.get("/api/sessions/to-delete").json() == {
        "messages": [],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated": False,
        },
    }
    assert all(
        item["session_id"] != "to-delete"
        for item in client.get("/api/sessions").json()
    )

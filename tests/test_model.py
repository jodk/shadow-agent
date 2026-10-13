import logging

import httpx
import pytest

from react_agent.config import ModelConfig
from react_agent.model import OpenAICompatibleClient


@pytest.mark.asyncio
async def test_openai_client_parses_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert payload["model"] == "demo"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "add",
                                        "arguments": '{"left": 1, "right": 2}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(
        ModelConfig(
            base_url="https://model.example.test/v1",
            api_key="secret",
            model="demo",
        ),
        client=http_client,
    )

    response = await client.complete([{"role": "user", "content": "add"}], [])

    assert response.message.tool_calls[0].name == "add"
    assert response.message.tool_calls[0].arguments == {"left": 1, "right": 2}
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )
    assert response.usage.estimated is True
    await http_client.aclose()


@pytest.mark.asyncio
async def test_openai_client_uses_api_usage_when_available():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(
        ModelConfig(
            base_url="https://model.example.test/v1",
            api_key="secret",
            model="demo",
        ),
        client=http_client,
    )

    response = await client.complete([{"role": "user", "content": "hello"}], [])

    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 8
    assert response.usage.total_tokens == 20
    assert response.usage.estimated is False
    await http_client.aclose()


@pytest.mark.asyncio
async def test_openai_client_streams_text_and_parses_final_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert payload["stream"] is True
        body = (
            'data: {"choices":[{"delta":{"role":"assistant","content":"你好"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"，世界"}}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode("utf-8"),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(
        ModelConfig(
            base_url="https://model.example.test/v1",
            api_key="secret",
            model="demo",
        ),
        client=http_client,
    )

    events = [event async for event in client.stream([{"role": "user", "content": "hi"}], [])]

    assert [event.delta for event in events if event.delta] == ["你好", "，世界"]
    final = events[-1].response
    assert final is not None
    assert final.message.content == "你好，世界"
    assert final.usage.as_dict() == {
        "prompt_tokens": 4,
        "completion_tokens": 3,
        "total_tokens": 7,
        "estimated": False,
    }
    await http_client.aclose()


@pytest.mark.asyncio
async def test_model_logs_do_not_include_api_key_or_message(caplog):
    """模型日志只能记录元数据，不能泄露密钥或完整用户消息。"""
    secret_key = "secret-key-must-not-appear"
    secret_message = "private-message-must-not-appear"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = OpenAICompatibleClient(
        ModelConfig(
            base_url="https://example.test/v1",
            api_key=secret_key,
            model="demo-model",
        ),
        client=client,
    )

    with caplog.at_level(logging.INFO, logger="react_agent.model"):
        await model.complete(
            [{"role": "user", "content": secret_message}],
            [],
        )
    await client.aclose()

    assert "model_request_started" in caplog.text
    assert "total_tokens=4" in caplog.text
    assert secret_key not in caplog.text
    assert secret_message not in caplog.text

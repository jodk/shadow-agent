import pytest

from react_agent.agent import ReActAgent
from react_agent.model import (
    AssistantMessage,
    ModelResponse,
    ModelStreamEvent,
    TokenUsage,
    ToolCall,
)
from react_agent.tools import ToolRegistry, tool


@tool
def multiply(left: int, right: int) -> int:
    """Multiply two integers."""
    return left * right


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete(self, messages, tools):
        self.calls.append((list(messages), list(tools)))
        return self.responses.pop(0)


class StreamingModel:
    async def stream(self, messages, tools):
        yield ModelStreamEvent(delta="流式")
        yield ModelStreamEvent(delta="回答")
        yield ModelStreamEvent(
            response=ModelResponse(
                message=AssistantMessage(content="流式回答"),
                usage=TokenUsage(prompt_tokens=2, completion_tokens=2),
            )
        )


@pytest.mark.asyncio
async def test_agent_forwards_streaming_deltas_before_final_answer():
    agent = ReActAgent(model=StreamingModel())

    events = [event async for event in agent.stream([{"role": "user", "content": "hi"}])]

    assert [event.data["content"] for event in events if event.type == "assistant_delta"] == [
        "流式",
        "回答",
    ]
    assert events[-1].type == "assistant_completed"
    assert events[-1].data["content"] == "流式回答"


@pytest.mark.asyncio
async def test_agent_executes_tool_and_returns_follow_up_answer():
    model = ScriptedModel(
        [
            ModelResponse(
                message=AssistantMessage(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="multiply",
                            arguments={"left": 6, "right": 7},
                        )
                    ],
                ),
                usage=TokenUsage(prompt_tokens=10, completion_tokens=3),
            ),
            ModelResponse(
                message=AssistantMessage(content="The answer is 42."),
                usage=TokenUsage(prompt_tokens=20, completion_tokens=5),
            ),
        ]
    )
    agent = ReActAgent(model=model, tools=ToolRegistry([multiply]), max_steps=4)

    events = [event async for event in agent.stream([{"role": "user", "content": "6 * 7?"}])]

    assert events[-1].type == "assistant_completed"
    assert events[-1].data["content"] == "The answer is 42."
    assert events[-1].data["usage"] == {
        "prompt_tokens": 30,
        "completion_tokens": 8,
        "total_tokens": 38,
        "estimated": False,
    }
    usage_events = [event for event in events if event.type == "model_usage"]
    assert [event.data["total_tokens"] for event in usage_events] == [13, 25]
    second_messages = model.calls[1][0]
    assert second_messages[-1]["role"] == "tool"
    assert '"value": 42' in second_messages[-1]["content"]


@pytest.mark.asyncio
async def test_agent_reports_tool_failure_to_model():
    model = ScriptedModel(
        [
            ModelResponse(
                message=AssistantMessage(
                    tool_calls=[ToolCall(id="call-1", name="missing", arguments={})]
                )
            ),
            ModelResponse(message=AssistantMessage(content="That tool is unavailable.")),
        ]
    )
    agent = ReActAgent(model=model, tools=ToolRegistry(), max_steps=3)

    answer = await agent.run([{"role": "user", "content": "Use missing."}])

    assert answer == "That tool is unavailable."
    assert '"ok": false' in model.calls[1][0][-1]["content"]


@pytest.mark.asyncio
async def test_agent_stops_after_max_steps():
    response = ModelResponse(
        message=AssistantMessage(
            tool_calls=[ToolCall(id="call-1", name="multiply", arguments={"left": 1, "right": 1})]
        )
    )
    model = ScriptedModel([response, response])
    agent = ReActAgent(model=model, tools=ToolRegistry([multiply]), max_steps=2)

    events = [event async for event in agent.stream([{"role": "user", "content": "loop"}])]

    assert events[-1].type == "error"
    assert "maximum" in events[-1].data["message"].lower()


@pytest.mark.asyncio
async def test_agent_only_exposes_allowed_tools_and_adds_skill_instructions():
    model = ScriptedModel(
        [ModelResponse(message=AssistantMessage(content="done"))]
    )
    agent = ReActAgent(
        model=model,
        tools=ToolRegistry([multiply]),
        system_prompt="base",
    )

    await agent.run(
        [{"role": "user", "content": "hi"}],
        additional_instructions="本轮请使用精确工具。",
        allowed_tool_names=set(),
    )

    sent_messages, sent_tools = model.calls[0]
    assert "本轮请使用精确工具。" in sent_messages[0]["content"]
    assert sent_tools == []

import pytest

from react_agent.tools import ToolRegistry, tool


@tool
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


@tool(name="echo_text", description="Echo text asynchronously.")
async def echo(value: str) -> str:
    return value


def test_decorator_builds_openai_tool_schema():
    definition = add.openai_definition()

    assert definition["function"]["name"] == "add"
    assert definition["function"]["parameters"]["required"] == ["left", "right"]
    assert definition["function"]["parameters"]["properties"]["left"]["type"] == "integer"


@pytest.mark.asyncio
async def test_registry_executes_sync_and_async_tools():
    registry = ToolRegistry([add, echo])

    add_result = await registry.execute("add", {"left": 2, "right": 3})
    echo_result = await registry.execute("echo_text", {"value": "hello"})

    assert add_result.ok is True
    assert add_result.value == 5
    assert echo_result.value == "hello"


@pytest.mark.asyncio
async def test_registry_returns_structured_error_for_invalid_arguments():
    registry = ToolRegistry([add])

    result = await registry.execute("add", {"left": "not-an-int", "right": 3})

    assert result.ok is False
    assert result.error

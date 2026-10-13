from pathlib import Path

import httpx
import pytest

from react_agent.tools.rest import load_rest_tools


@pytest.mark.asyncio
async def test_rest_tool_maps_arguments_to_request(tmp_path: Path):
    definition = tmp_path / "rest.yaml"
    definition.write_text(
        """
tools:
  - name: find_user
    description: Find a user.
    method: GET
    url: https://api.example.test/users/{user_id}
    parameters:
      type: object
      properties:
        user_id: {type: integer}
        verbose: {type: boolean}
      required: [user_id]
    path_params: [user_id]
    query_params: [verbose]
    response_path: data
""",
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/users/7"
        assert request.url.params["verbose"] == "true"
        return httpx.Response(200, json={"data": {"name": "Ada"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tools = load_rest_tools(definition, client=client)

    result = await tools[0].execute({"user_id": 7, "verbose": True})

    assert result.ok is True
    assert result.value == {"name": "Ada"}
    await client.aclose()


@pytest.mark.asyncio
async def test_rest_tool_surfaces_http_error(tmp_path: Path):
    definition = tmp_path / "rest.yaml"
    definition.write_text(
        """
tools:
  - name: broken
    description: Broken endpoint.
    method: GET
    url: https://api.example.test/broken
    parameters: {type: object, properties: {}}
""",
        encoding="utf-8",
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, json={"error": "offline"})
        )
    )
    tool = load_rest_tools(definition, client=client)[0]

    result = await tool.execute({})

    assert result.ok is False
    assert "503" in result.error
    await client.aclose()

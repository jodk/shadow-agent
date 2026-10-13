from fastapi.testclient import TestClient

from react_agent.agent import ReActAgent
from react_agent.api import create_app
from react_agent.model import AssistantMessage, ModelResponse


class Model:
    async def complete(self, messages, tools):
        return ModelResponse(message=AssistantMessage(content="ok"))


def test_home_page_contains_chat_controls():
    client = TestClient(create_app(agent=ReActAgent(Model())))

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="message-input"' in response.text
    assert 'id="send-button"' in response.text
    assert 'id="clear-button"' in response.text
    assert 'id="session-list"' in response.text
    assert 'id="new-session-button"' in response.text
    assert 'id="turn-input-tokens"' in response.text
    assert 'id="turn-output-tokens"' in response.text
    assert 'id="turn-total-tokens"' in response.text
    assert 'id="session-total-tokens"' in response.text
    assert "/static/app.js" in response.text

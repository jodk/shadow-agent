from pathlib import Path

import pytest

from react_agent.skills import SkillLoader


@pytest.fixture
def text_analysis_tools():
    skill_root = (
        Path(__file__).parents[1]
        / "examples"
        / "skills"
        / "text-analysis"
    )
    loaded = SkillLoader([skill_root]).load()
    return {item.name: item for item in loaded.tools}, loaded.instructions


def test_text_analysis_skill_is_discovered(text_analysis_tools):
    tools, instructions = text_analysis_tools

    assert "文本分析" in instructions
    assert set(tools) == {
        "count_text",
        "count_keyword",
        "summarize_text_info",
    }


@pytest.mark.asyncio
async def test_count_text_returns_chinese_and_english_statistics(text_analysis_tools):
    tools, _ = text_analysis_tools

    result = await tools["count_text"].execute(
        {"text": "你好，ReAct Agent! Hello world."}
    )

    assert result.ok is True
    assert result.value["chinese_characters"] == 2
    assert result.value["english_words"] == 4
    assert result.value["total_characters"] == 28


@pytest.mark.asyncio
async def test_count_keyword_can_ignore_case(text_analysis_tools):
    tools, _ = text_analysis_tools

    result = await tools["count_keyword"].execute(
        {
            "text": "Agent calls tools. The AGENT gets results.",
            "keyword": "agent",
            "case_sensitive": False,
        }
    )

    assert result.value == {
        "keyword": "agent",
        "count": 2,
        "case_sensitive": False,
    }

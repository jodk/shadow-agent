from pathlib import Path

import pytest

from react_agent.skills import SkillLoader


@pytest.mark.asyncio
async def test_skill_loader_combines_instructions_and_python_tools(tmp_path: Path):
    skill_dir = tmp_path / "weather"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "# Weather\nUse this skill for weather questions.", encoding="utf-8"
    )
    (skill_dir / "skill.yaml").write_text(
        "enabled: true\npython_module: tools.py\n", encoding="utf-8"
    )
    (skill_dir / "tools.py").write_text(
        """
from react_agent.tools import tool

@tool
def current_weather(city: str) -> str:
    \"\"\"Return demo weather.\"\"\"
    return f"Sunny in {city}"

TOOLS = [current_weather]
""",
        encoding="utf-8",
    )

    loaded = SkillLoader([tmp_path]).load()

    assert "Use this skill for weather questions." in loaded.instructions
    result = await loaded.tools[0].execute({"city": "Shanghai"})
    assert result.value == "Sunny in Shanghai"


def test_skill_loader_ignores_disabled_skill(tmp_path: Path):
    skill_dir = tmp_path / "disabled"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("Do not load.", encoding="utf-8")
    (skill_dir / "skill.yaml").write_text("enabled: false\n", encoding="utf-8")

    loaded = SkillLoader([tmp_path]).load()

    assert loaded.instructions == ""
    assert loaded.tools == []

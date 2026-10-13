import logging
from pathlib import Path
from typing import List, Tuple

from .agent import ReActAgent
from .config import AppConfig
from .model import OpenAICompatibleClient
from .skills import SkillLoader
from .tools import ToolRegistry
from .tools.mcp import MCPToolProvider
from .tools.rest import load_rest_tools


logger = logging.getLogger(__name__)


async def build_agent(config: AppConfig, base_dir: Path) -> Tuple[ReActAgent, List[MCPToolProvider]]:
    """创建完整 Agent，并返回需要在应用关闭时释放的 MCP Provider。

    ``Tuple[A, B]`` 表示函数返回两个值。调用方可写成
    ``agent, providers = await build_agent(...)`` 进行解包。
    """
    registry = ToolRegistry()
    # 相对路径以 config.yaml 所在目录为基准，而不是进程当前目录。
    skill_dirs = [
        path if path.is_absolute() else base_dir / path
        for path in map(Path, config.skills.directories)
    ]
    skills = SkillLoader(skill_dirs).load()
    logger.info(
        "skills_loaded directory_count=%d tool_count=%d instruction_length=%d",
        len(skill_dirs),
        len(skills.tools),
        len(skills.instructions),
    )
    for item in skills.tools:
        registry.register(item)

    for configured_path in config.tools.rest.files:
        path = Path(configured_path)
        if not path.is_absolute():
            path = base_dir / path
        rest_tools = load_rest_tools(path)
        logger.info("rest_tools_loaded path=%s tool_count=%d", path, len(rest_tools))
        for item in rest_tools:
            registry.register(item)

    providers = []
    for server in config.tools.mcp_servers:
        provider = MCPToolProvider(server)
        logger.info(
            "mcp_connection_started server=%s transport=%s",
            server.name,
            server.transport,
        )
        mcp_tools = await provider.connect()
        logger.info(
            "mcp_connection_completed server=%s tool_count=%d",
            server.name,
            len(mcp_tools),
        )
        for item in mcp_tools:
            registry.register(item)
        providers.append(provider)

    prompt = config.agent.system_prompt
    if skills.instructions:
        # Skill 说明最终成为 system prompt 的一部分，帮助模型判断何时调用工具。
        prompt = f"{prompt}\n\nAvailable skills:\n\n{skills.instructions}"
    logger.info(
        "agent_built tool_count=%d system_prompt_length=%d",
        len(registry.list()),
        len(prompt),
    )
    return (
        ReActAgent(
            model=OpenAICompatibleClient(config.model),
            tools=registry,
            system_prompt=prompt,
            max_steps=config.agent.max_steps,
        ),
        providers,
    )
"""应用装配层：根据配置创建模型、工具、Skill 和 Agent。"""

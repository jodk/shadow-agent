import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .agent import ReActAgent
from .capabilities import CapabilityCatalog, CapabilityRouter
from .config import AppConfig
from .model import OpenAICompatibleClient
from .skills import LoadedSkills, SkillLoader
from .tools import ToolRegistry
from .tools.mcp import MCPToolProvider
from .tools.rest import load_rest_tools


logger = logging.getLogger(__name__)


@dataclass
class BuiltAgent:
    """完整应用装配结果。

    把这些对象放在一个结果对象里，API 和 CLI 可以共享完全相同的工具、
    Skill 和能力路由配置，避免两个入口各自实现一套容易漂移的逻辑。
    """

    # dataclass 让装配结果像一个有名字的容器，比长元组更容易阅读。
    agent: ReActAgent
    providers: List[MCPToolProvider]
    skills: LoadedSkills
    capability_catalog: CapabilityCatalog
    capability_router: CapabilityRouter


async def build_agent(
    config: AppConfig,
    base_dir: Path,
    *,
    # * 后面的参数只能按名称传入，避免路径和向量库参数传反。
    memory_path: Path | None = None,
    capability_vector_store=None,
) -> BuiltAgent:
    """创建完整 Agent、Skill 能力目录和能力路由。

    ``memory_path`` 和向量库由 API 在创建共享 embedding 模型后传入。
    没有传入向量库时仍然会建立 SQLite/BM25 能力索引，适合 CLI 或无向量环境。
    """
    registry = ToolRegistry()
    # 相对路径以 config.yaml 所在目录为基准，而不是进程当前目录。
    # map(Path, ...) 逐个转换路径；列表推导式再根据是否绝对路径拼接 base_dir。
    skill_dirs = [
        path if path.is_absolute() else base_dir / path
        for path in map(Path, config.skills.directories)
    ]
    skills = SkillLoader(skill_dirs).load()
    logger.info(
        "skills_loaded directory_count=%d tool_count=%d skill_count=%d",
        len(skill_dirs),
        len(skills.tools),
        len(skills.skills),
    )
    # Skill、REST、MCP 虽然来源不同，最终都注册进同一个 Registry。
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

    # providers 保存应用关闭时需要释放的 MCP 资源。
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

    logger.info(
        "agent_built tool_count=%d system_prompt_length=%d",
        len(registry.list()),
        len(config.agent.system_prompt),
    )
    agent = ReActAgent(
        model=OpenAICompatibleClient(config.model),
        tools=registry,
        system_prompt=config.agent.system_prompt,
        max_steps=config.agent.max_steps,
    )
    if memory_path is None:
        memory_path = base_dir / config.agent.memory_db
    capability_catalog = CapabilityCatalog(
        memory_path,
        skills.skills,
        registry,
        vector_store=capability_vector_store,
    )
    capability_router = CapabilityRouter(capability_catalog, registry, config.agent)
    # 用关键字参数构造 dataclass，可读性高于依赖字段顺序。
    return BuiltAgent(
        agent=agent,
        providers=providers,
        skills=skills.skills,
        capability_catalog=capability_catalog,
        capability_router=capability_router,
    )
"""应用装配层：根据配置创建模型、工具、Skill 和 Agent。"""

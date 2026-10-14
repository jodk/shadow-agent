from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """模型 API 配置。

    ``BaseModel`` 是 Pydantic 提供的基类：类属性上的类型标注会同时用于
    文档说明和运行时校验。例如 ``temperature: float`` 表示这里期望一个
    浮点数；YAML 中写错类型时，Pydantic 会在启动阶段尽早报错。
    """

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    timeout_seconds: float = 60


class AgentConfig(BaseModel):
    """Agent 行为配置。

    配置类把原始 YAML 字典转换成结构化对象。后续代码可以使用
    ``config.agent.max_steps`` 这样的点号语法访问字段，而不必反复写
    ``data["agent"]["max_steps"]``。
    """

    system_prompt: str = "You are a helpful assistant."
    max_steps: int = Field(default=10, ge=1)
    memory_db: str = "./data/memory.db"
    recent_messages: int = Field(default=8, ge=1)
    keyword_memory_limit: int = Field(default=5, ge=0)
    vector_memory_limit: int = Field(default=5, ge=0)
    fused_memory_limit: int = Field(default=8, ge=0)
    rrf_k: int = Field(default=60, ge=1)
    cross_session_memory: bool = True
    vector_enabled: bool = True
    embedding_model: str = "shibing624/text2vec-base-chinese"
    # 默认固定使用 CPU。macOS Apple Silicon 虽然支持 MPS，但
    # SentenceTransformer/PyTorch 在后台线程中使用 MPS 时，某些版本组合会触发
    # 原生 segmentation fault；这种崩溃无法被 Python 的 try/except 捕获。
    embedding_device: str = "cpu"
    vector_workers: int = Field(default=1, ge=1, le=4)
    # 无论能力检索结果如何都向模型暴露的高频核心工具。
    always_tools: List[str] = Field(default_factory=list)
    capability_keyword_limit: int = Field(default=10, ge=0)
    capability_vector_limit: int = Field(default=10, ge=0)
    capability_skill_limit: int = Field(default=3, ge=0)
    capability_tool_limit: int = Field(default=8, ge=0)
    capability_rrf_k: int = Field(default=60, ge=1)
    capability_skill_token_budget: int = Field(default=2000, ge=0)
    capability_tool_token_budget: int = Field(default=3000, ge=0)


class RestToolsConfig(BaseModel):
    """REST 工具文件列表。"""

    # ``default_factory=list`` 每次创建对象时都会新建一个列表。
    # 如果直接写 ``files: List[str] = []``，多个配置对象可能意外共享同一个可变列表。
    files: List[str] = Field(default_factory=list)


class MCPServerConfig(BaseModel):
    """一个 MCP Server 的启动信息。

    ``Optional[str]`` 表示值可以是字符串，也可以是 ``None``。这适合
    ``stdio`` 和 HTTP 两类传输方式：前者通常需要 command，后者通常需要 url。
    """

    name: str
    transport: str = "stdio"
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    url: Optional[str] = None


class ToolsConfig(BaseModel):
    """所有工具来源的配置。"""

    # ``Field(default_factory=RestToolsConfig)`` 会在缺少 rest 配置时，
    # 自动创建一个默认的 RestToolsConfig 对象。
    rest: RestToolsConfig = Field(default_factory=RestToolsConfig)
    mcp_servers: List[MCPServerConfig] = Field(default_factory=list)


class SkillsConfig(BaseModel):
    """Skill 目录配置。"""

    directories: List[str] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    """应用日志配置。

    日志同时支持控制台和滚动文件。文件达到 ``max_bytes`` 后会自动切分，
    最多保留 ``backup_count`` 个历史文件，避免学习项目长期运行后占满磁盘。
    """

    level: str = "INFO"
    console: bool = True
    file: Optional[str] = "./data/agent.log"
    max_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    backup_count: int = Field(default=5, ge=0)

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        """把日志级别统一为大写，并拒绝拼写错误的值。

        ``@field_validator("level")`` 是装饰器：它把下面的普通函数注册成
        ``level`` 字段的校验函数。``@classmethod`` 表示调用时通过 ``cls``
        接收当前模型类，而不是通过实例 ``self`` 接收对象。
        """
        normalized = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"日志级别必须是以下值之一: {', '.join(sorted(allowed))}")
        return normalized


class AppConfig(BaseModel):
    """应用总配置。

    嵌套的 Pydantic 模型让 YAML 的层级结构映射为 Python 对象层级。
    例如 YAML 的 ``agent.max_steps`` 最终对应 ``config.agent.max_steps``。
    """

    model: ModelConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: Path) -> AppConfig:
    """从 YAML 文件加载配置并交给 Pydantic 校验。

    ``Path`` 是 pathlib 的路径对象；``read_text`` 负责打开、读取并关闭文件。
    ``yaml.safe_load`` 把 YAML 文本解析为 Python 字典、列表、字符串等基础对象，
    ``model_validate`` 再把这个对象转换成 AppConfig 并执行字段校验。
    """
    # ``or {}`` 保证空 YAML 文件也会得到字典，而不是 None。
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)
"""读取和校验 ``config.yaml``。

Pydantic 的 ``BaseModel`` 会把普通字典转换成有类型提示的配置对象，并在字段缺失
或类型错误时给出明确异常。
"""

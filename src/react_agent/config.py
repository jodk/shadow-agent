from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """模型 API 配置。"""

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    timeout_seconds: float = 60


class AgentConfig(BaseModel):
    """Agent 行为配置。``ge=1`` 表示 max_steps 必须大于等于 1。"""

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


class RestToolsConfig(BaseModel):
    files: List[str] = Field(default_factory=list)


class MCPServerConfig(BaseModel):
    """一个 MCP Server 的启动信息。Optional 表示字段允许为 None。"""

    name: str
    transport: str = "stdio"
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    url: Optional[str] = None


class ToolsConfig(BaseModel):
    rest: RestToolsConfig = Field(default_factory=RestToolsConfig)
    mcp_servers: List[MCPServerConfig] = Field(default_factory=list)


class SkillsConfig(BaseModel):
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
        """把日志级别统一为大写，并拒绝拼写错误的值。"""
        normalized = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"日志级别必须是以下值之一: {', '.join(sorted(allowed))}")
        return normalized


class AppConfig(BaseModel):
    model: ModelConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: Path) -> AppConfig:
    """从 YAML 文件加载配置并交给 Pydantic 校验。"""
    # ``or {}`` 保证空 YAML 文件也会得到字典，而不是 None。
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)
"""读取和校验 ``config.yaml``。

Pydantic 的 ``BaseModel`` 会把普通字典转换成有类型提示的配置对象，并在字段缺失
或类型错误时给出明确异常。
"""

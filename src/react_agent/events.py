from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class AgentEvent:
    """``dataclass`` 会自动生成 __init__ 等样板代码。

    ``default_factory=dict`` 为每个事件创建独立字典，避免多个实例共享可变默认值。
    """

    type: str
    data: Dict[str, Any] = field(default_factory=dict)
"""Agent 运行过程中的事件数据结构。"""

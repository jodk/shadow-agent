import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set

from pydantic import ValidationError


logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """统一的工具执行结果，失败也作为数据返回，而不是向上抛出异常。"""

    # ``@dataclass`` 会根据这些字段自动生成 __init__ 等样板方法。
    # 因此 ToolResult(ok=True, value=123) 就能直接创建对象，不必手写构造函数。
    ok: bool
    value: Any = None
    error: Optional[str] = None

    def to_json(self) -> str:
        # ``json.dumps`` 把 Python 字典转换为 JSON 字符串。
        # ensure_ascii=False 能保留中文，而 default=str 处理不能直接 JSON
        # 序列化的对象，避免日志或工具消息再次报错。
        return json.dumps(
            {"ok": self.ok, "value": self.value, "error": self.error},
            ensure_ascii=False,
            default=str,
        )


class Tool:
    """Agent 可执行的一个动作。

    ``handler`` 统一接收参数字典。Python、REST 和 MCP 工具都先适配成这个形式，
    因此 ReActAgent 无需知道具体工具来源。
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Awaitable[Any]],
        *,
        # 这里的 ``*`` 表示 keywords 和 always_expose 必须用名称传入，
        # 例如 Tool(..., keywords=["天气"])，不能依赖位置顺序。
        keywords: Optional[Iterable[str]] = None,
        always_expose: bool = False,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        # ``Iterable`` 表示只要对象可以被 for 遍历即可，例如 list、tuple 或生成器。
        # 转成 list 后，Tool 内部可以重复使用这些关键词。
        self.keywords = list(keywords or [])
        self.always_expose = always_expose
        self.skill_name: Optional[str] = None

    def openai_definition(self) -> Dict[str, Any]:
        """生成 OpenAI tool calling 协议需要的函数定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """调用工具并把异常转换成 ToolResult。

        将失败结果返回模型后，模型可以换参数重试，也可以向用户解释失败原因。
        """
        # ``try/except`` 把可能失败的代码包起来。
        # 工具失败会变成 ToolResult，而不是让整个 Agent 进程崩溃。
        try:
            value = await self.handler(arguments)
            return ToolResult(ok=True, value=value)
        except (TypeError, ValueError, ValidationError) as exc:
            logger.warning(
                "tool_validation_failed name=%s error_type=%s",
                self.name,
                type(exc).__name__,
            )
            return ToolResult(ok=False, error=str(exc))
        except Exception as exc:
            logger.exception(
                "tool_execution_failed name=%s error_type=%s",
                self.name,
                type(exc).__name__,
            )
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")


class ToolRegistry:
    """按名称保存和查找工具。"""

    def __init__(self, tools: Optional[Iterable[Tool]] = None):
        self._tools: Dict[str, Tool] = {}
        # ``tools or []`` 让 None 表示“没有初始工具”。
        # 这里的 for 循环会依次调用 register，统一执行重复名称检查。
        for item in tools or []:
            self.register(item)

    def register(self, item: Tool) -> None:
        # 禁止静默覆盖同名工具，否则模型看到的 Schema 可能和实际执行函数不一致。
        if item.name in self._tools:
            raise ValueError(f"Tool already registered: {item.name}")
        self._tools[item.name] = item
        logger.debug("tool_registered name=%s", item.name)

    def definitions(
        self, allowed_names: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """只返回当前请求允许暴露给模型的工具 Schema。"""
        # 这是列表推导式。``if`` 是筛选条件：
        # allowed_names 为 None 时保留全部工具，否则只保留集合中的名称。
        return [
            item.openai_definition()
            for item in self._tools.values()
            if allowed_names is None or item.name in allowed_names
        ]

    def list(self) -> List[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    async def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        allowed_names: Optional[Set[str]] = None,
    ) -> ToolResult:
        """根据模型给出的工具名执行对应工具。"""
        if allowed_names is not None and name not in allowed_names:
            logger.warning("tool_not_exposed name=%s", name)
            return ToolResult(
                ok=False,
                error=f"Tool is not exposed for the current request: {name}",
            )
        item = self._tools.get(name)
        if item is None:
            logger.warning("unknown_tool_requested name=%s", name)
            return ToolResult(ok=False, error=f"Unknown tool: {name}")
        return await item.execute(arguments)
"""所有工具类型共享的基础抽象。"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from pydantic import ValidationError


logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """统一的工具执行结果，失败也作为数据返回，而不是向上抛出异常。"""

    ok: bool
    value: Any = None
    error: Optional[str] = None

    def to_json(self) -> str:
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
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

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
        for item in tools or []:
            self.register(item)

    def register(self, item: Tool) -> None:
        # 禁止静默覆盖同名工具，否则模型看到的 Schema 可能和实际执行函数不一致。
        if item.name in self._tools:
            raise ValueError(f"Tool already registered: {item.name}")
        self._tools[item.name] = item
        logger.debug("tool_registered name=%s", item.name)

    def definitions(self) -> List[Dict[str, Any]]:
        return [item.openai_definition() for item in self._tools.values()]

    def list(self) -> List[Tool]:
        return list(self._tools.values())

    async def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """根据模型给出的工具名执行对应工具。"""
        item = self._tools.get(name)
        if item is None:
            logger.warning("unknown_tool_requested name=%s", name)
            return ToolResult(ok=False, error=f"Unknown tool: {name}")
        return await item.execute(arguments)
"""所有工具类型共享的基础抽象。"""

from typing import Any, Dict, List

from ..config import MCPServerConfig
from .base import Tool


class MCPToolProvider:
    """Load MCP tools lazily so the core project works without the MCP extra."""

    def __init__(self, config: MCPServerConfig):
        # ``Any`` 表示这里暂时不限制 SDK 对象的具体类型，因为 MCP 是可选依赖。
        self.config = config
        self._session: Any = None
        self._stack: Any = None

    async def connect(self) -> List[Tool]:
        """启动 stdio MCP Server、初始化会话并发现工具。"""
        # 延迟导入：只有用户真的配置 MCP 时才加载可选依赖。
        try:
            # MCP 是可选依赖，只有真正启用 MCP 时才导入。
            from contextlib import AsyncExitStack

            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "MCP support is optional. Install it with: pip install -e '.[mcp]'"
            ) from exc

        if self.config.transport != "stdio":
            raise ValueError("This learning implementation currently supports stdio MCP.")
        if not self.config.command:
            raise ValueError("MCP stdio server requires a command.")

        # AsyncExitStack 是异步版的资源管理器集合。
        # 多个网络连接或子进程可以统一注册，最后由 aclose 逆序释放。
        self._stack = AsyncExitStack()
        # AsyncExitStack 集中管理多个异步上下文，close 时按相反顺序释放资源。
        read, write = await self._stack.enter_async_context(
            stdio_client(
                StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args,
                )
            )
        )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        response = await self._session.list_tools()
        # 服务端工具通过 _adapt 转为本项目统一工具对象。
        # 列表推导式把 SDK 的每个工具对象转换为内部 Tool。
        return [self._adapt(item) for item in response.tools]

    def _adapt(self, mcp_tool: Any) -> Tool:
        """把 MCP SDK 的工具对象包装成内部 Tool。"""
        # handler 是闭包，会记住当前 mcp_tool 和 self._session，
        # 因此 Agent 只需调用内部统一的 handler。
        async def handler(arguments: Dict[str, Any]) -> Any:
            response = await self._session.call_tool(mcp_tool.name, arguments)
            content = []
            for item in response.content:
                if hasattr(item, "text"):
                    content.append(item.text)
                else:
                    content.append(str(item))
            return {"is_error": response.isError, "content": content}

        return Tool(
            name=mcp_tool.name,
            description=mcp_tool.description or mcp_tool.name,
            parameters=mcp_tool.inputSchema,
            handler=handler,
        )

    async def close(self) -> None:
        """关闭 MCP 会话及其子进程输入输出流。"""
        if self._stack is not None:
            await self._stack.aclose()
"""MCP 工具适配器。

MCP Server 负责暴露工具清单和执行工具。本模块将远端 MCP Tool 转成项目内部的
``Tool``，这样 ReActAgent 不需要包含任何 MCP 专用逻辑。
"""

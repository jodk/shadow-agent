import argparse
import asyncio
import json
import logging
from pathlib import Path

import uvicorn

from .config import load_config
from .factory import build_agent
from .logging_config import configure_logging


logger = logging.getLogger(__name__)


async def _chat(config_path: Path, message: str = "") -> None:
    """运行一次性或交互式终端聊天。"""
    # async def 定义协程函数；它必须由 await 或 asyncio.run 驱动。
    config = load_config(config_path)
    configure_logging(config.logging, config_path.resolve().parent)
    logger.info("cli_chat_started config_path=%s", config_path.resolve())
    built = await build_agent(config, config_path.resolve().parent)
    agent = built.agent
    providers = built.providers
    history = []
    try:
        while True:
            # 有命令行 message 时使用该消息，否则等待用户输入。
            question = message or input("You> ").strip()
            if not question:
                break
            history.append({"role": "user", "content": question})
            # await 等待能力路由完成；selection 决定本轮暴露哪些工具。
            selection = await built.capability_router.route(question)
            # Agent.stream 是异步生成器，async for 可以边收到边打印。
            async for event in agent.stream(
                history,
                additional_instructions=selection.skill_instructions,
                allowed_tool_names=set(selection.tool_names),
            ):
                if event.type == "tool_started":
                    print(
                        f"[tool] {event.data['name']} "
                        f"{json.dumps(event.data['arguments'], ensure_ascii=False)}"
                    )
                elif event.type == "tool_finished":
                    print(
                        f"[tool result] {json.dumps(event.data, ensure_ascii=False, default=str)}"
                    )
                elif event.type == "assistant_completed":
                    answer = event.data["content"]
                    print(f"Agent> {answer}")
                    history.append({"role": "assistant", "content": answer})
                elif event.type == "error":
                    print(f"[error] {event.data['message']}")
            if message:
                break
    finally:
        for provider in providers:
            await provider.close()
        logger.info("cli_chat_stopped")


async def _tools(config_path: Path) -> None:
    """列出配置加载后实际注册的工具。"""
    config = load_config(config_path)
    configure_logging(config.logging, config_path.resolve().parent)
    logger.info("cli_tools_started config_path=%s", config_path.resolve())
    built = await build_agent(config, config_path.resolve().parent)
    agent = built.agent
    providers = built.providers
    # try/finally 确保即使打印过程中异常，也会关闭 MCP Provider。
    try:
        for item in agent.tools.list():
            print(f"{item.name}: {item.description}")
    finally:
        for provider in providers:
            await provider.close()
        logger.info("cli_tools_stopped")


def main() -> None:
    """解析命令行参数并分派到 chat、tools 或 serve。"""
    # argparse 将命令行参数解析成 Namespace 对象，属性名来自 add_argument。
    parser = argparse.ArgumentParser(description="Learning ReAct Agent")
    parser.add_argument("--config", default="config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    chat = subparsers.add_parser("chat", help="Chat in the terminal")
    chat.add_argument("message", nargs="?", default="")
    subparsers.add_parser("tools", help="List configured tools")
    serve = subparsers.add_parser("serve", help="Start the web server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config_path = Path(args.config)

    if args.command == "serve":
        from .api import create_app

        # uvicorn 是运行 ASGI/FastAPI 应用的 Web 服务器。
        config = load_config(config_path)
        configure_logging(config.logging, config_path.resolve().parent)
        logger.info(
            "cli_server_starting config_path=%s host=%s port=%d",
            config_path.resolve(),
            args.host,
            args.port,
        )
        # uvicorn.run 是同步入口；它内部负责创建和运行 ASGI 事件循环。
        uvicorn.run(create_app(config_path=config_path), host=args.host, port=args.port)
    elif args.command == "tools":
        asyncio.run(_tools(config_path))
    else:
        # asyncio.run 创建事件循环，用于执行最外层 async 函数。
        asyncio.run(_chat(config_path, args.message))


if __name__ == "__main__":
    main()
"""命令行入口：``python -m react_agent ...``。"""

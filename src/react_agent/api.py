import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import ReActAgent
from .config import load_config
from .factory import build_agent
from .logging_config import configure_logging
from .memory import SQLiteMemory
from .model import TokenUsage
from .vector_store import ChineseEmbeddingModel, ChromaVectorStore


STATIC_DIR = Path(__file__).with_name("static")
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """POST /api/chat/stream 的请求体。"""

    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


def format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """把一个 Agent 事件编码为标准 Server-Sent Events 帧。

    SSE 使用空行分隔事件；事件名称放在 ``event:``，JSON 数据放在 ``data:``。
    """
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {payload}\n\n"


def create_app(
    agent: Optional[ReActAgent] = None,
    config_path: Path = Path("config.yaml"),
) -> FastAPI:
    """创建 FastAPI 应用。

    测试时可直接传入假的 Agent；真实启动时根据 config_path 构建 Agent。
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """管理整个 Web 应用的启动和关闭资源。

        ``yield`` 之前是启动阶段，之后是关闭阶段。这种写法能保证 MCP 连接被释放。
        """
        if app.state.agent is None:
            config = load_config(config_path)
            configure_logging(config.logging, config_path.resolve().parent)
            logger.info("application_starting config_path=%s", config_path.resolve())
            app.state.agent, app.state.mcp_providers = await build_agent(
                config, config_path.resolve().parent
            )
            memory_path = Path(config.agent.memory_db)
            if not memory_path.is_absolute():
                memory_path = config_path.resolve().parent / memory_path
            vector_store = None
            if config.agent.vector_enabled:
                data_dir = memory_path.parent
                model_directory_name = (
                    config.agent.embedding_model.replace("/", "--")
                )
                model_path = data_dir / "models" / model_directory_name
                logger.info(
                    "embedding_model_configured model=%s model_path=%s "
                    "local_exists=%s chroma_path=%s",
                    config.agent.embedding_model,
                    model_path,
                    (model_path / "modules.json").exists(),
                    data_dir / "chroma",
                )
                embedding_model = ChineseEmbeddingModel(
                    model_name=config.agent.embedding_model,
                    model_path=model_path,
                    device=config.agent.embedding_device,
                )
                vector_store = ChromaVectorStore(
                    persist_directory=data_dir / "chroma",
                    embedding_model=embedding_model,
                )
            else:
                logger.info("embedding_model_disabled reason=vector_enabled_false")
            app.state.memory = SQLiteMemory(
                memory_path,
                vector_store=vector_store,
                vector_workers=config.agent.vector_workers,
            )
            app.state.memory_recent_limit = config.agent.recent_messages
            app.state.memory_keyword_limit = config.agent.keyword_memory_limit
            app.state.memory_vector_limit = config.agent.vector_memory_limit
            app.state.memory_fused_limit = config.agent.fused_memory_limit
            app.state.memory_rrf_k = config.agent.rrf_k
            app.state.memory_cross_session = config.agent.cross_session_memory
            logger.info(
                "application_started memory_db=%s tool_count=%d mcp_provider_count=%d",
                memory_path,
                len(app.state.agent.tools.list()),
                len(app.state.mcp_providers),
            )
        else:
            # 测试或嵌入式调用传入 Agent 时没有完整 YAML 配置。此时使用独立的
            # 测试数据库，但只在真正进入 lifespan 后创建，不影响生产 serve。
            app.state.memory = SQLiteMemory(Path("data/test-memory.db"))
            logger.info("test_memory_initialized path=data/test-memory.db")
        yield
        if app.state.memory is not None:
            await app.state.memory.close()
        for provider in app.state.mcp_providers:
            await provider.close()
        logger.info("application_stopped")

    app = FastAPI(title="Learning ReAct Agent", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.agent = agent
    # 只有测试或嵌入式调用传入 Agent 时才创建测试数据库。生产模式的 agent
    # 为 None，会在 lifespan 中根据 config.yaml 创建正式 memory.db，因此不会
    # 再产生 data/test-memory.db。
    app.state.memory = (
        SQLiteMemory(Path("data/test-memory.db")) if agent is not None else None
    )
    app.state.memory_recent_limit = 8
    app.state.memory_keyword_limit = 5
    app.state.memory_vector_limit = 5
    app.state.memory_fused_limit = 8
    app.state.memory_rrf_k = 60
    app.state.memory_cross_session = True
    app.state.mcp_providers = []

    @app.get("/")
    async def home():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/tools")
    async def list_tools():
        """给前端或开发者查看当前可用工具，不包含工具处理函数本身。"""
        logger.debug("tools_list_requested")
        return [
            {
                "name": item.name,
                "description": item.description,
                "parameters": item.parameters,
            }
            for item in app.state.agent.tools.list()
        ]

    @app.get("/api/sessions")
    async def list_sessions():
        """返回左侧会话列表。"""
        logger.debug("sessions_list_requested")
        return app.state.memory.list_sessions()

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        logger.debug("session_requested session_id=%s", session_id)
        return {
            "messages": app.state.memory.get_messages(session_id),
            "usage": app.state.memory.get_usage(session_id).as_dict(),
        }

    @app.get("/api/sessions/{session_id}/usage")
    async def get_session_usage(session_id: str):
        """单独查询会话累计 token，便于前端或其他客户端刷新统计。"""
        logger.debug("session_usage_requested session_id=%s", session_id)
        return app.state.memory.get_usage(session_id).as_dict()

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def clear_session(session_id: str):
        app.state.memory.delete_session(session_id)
        if app.state.memory.vector_store is not None:
            try:
                await app.state.memory.vector_store.delete_session(session_id)
            except Exception:
                logger.exception(
                    "session_vector_delete_failed session_id=%s", session_id
                )
        logger.info("session_deleted session_id=%s", session_id)
        return Response(status_code=204)

    @app.post("/api/chat/stream")
    async def chat(request: ChatRequest):
        logger.info(
            "chat_started session_id=%s message_length=%d",
            request.session_id,
            len(request.message),
        )
        app.state.memory.append_message(
            request.session_id,
            {"role": "user", "content": request.message},
        )

        async def generate():
            """异步生成标准 SSE，每个事件都以空行作为边界。"""
            final_content = ""
            run_usage = TokenUsage()
            # 先立即发送一个小事件，让浏览器尽快收到响应头并确认连接已建立。
            yield format_sse_event(
                "connected",
                {"session_id": request.session_id},
            )
            context = await app.state.memory.build_context(
                request.session_id,
                current_message=request.message,
                recent_limit=app.state.memory_recent_limit,
                keyword_limit=app.state.memory_keyword_limit,
                vector_limit=app.state.memory_vector_limit,
                fused_limit=app.state.memory_fused_limit,
                rrf_k=app.state.memory_rrf_k,
                cross_session=app.state.memory_cross_session,
            )
            logger.info(
                "chat_context_built session_id=%s context_message_count=%d",
                request.session_id,
                len(context),
            )
            try:
                async for event in app.state.agent.stream(context):
                    if event.type == "assistant_completed":
                        final_content = event.data["content"]
                        run_usage = TokenUsage(**event.data.get("usage", {}))
                    yield format_sse_event(event.type, event.data)
            except Exception:
                logger.exception("chat_failed session_id=%s", request.session_id)
                yield format_sse_event(
                    "error",
                    {"message": "聊天处理失败，请查看服务端日志"},
                )
                return
            if final_content:
                # 完整回答落库，下一轮可以通过最近消息或检索找回。
                app.state.memory.append_message(
                    request.session_id,
                    {"role": "assistant", "content": final_content},
                )
                session_usage = app.state.memory.add_usage(
                    request.session_id, run_usage
                )
                logger.info(
                    "chat_completed session_id=%s answer_length=%d "
                    "prompt_tokens=%d completion_tokens=%d total_tokens=%d "
                    "session_total_tokens=%d",
                    request.session_id,
                    len(final_content),
                    run_usage.prompt_tokens,
                    run_usage.completion_tokens,
                    run_usage.total_tokens,
                    session_usage.total_tokens,
                )
                yield format_sse_event(
                    "done",
                    {
                        "content": final_content,
                        "usage": run_usage.as_dict(),
                        "session_usage": session_usage.as_dict(),
                    },
                )

        # 禁用缓存和常见反向代理缓冲，避免代理攒齐完整回答后才交给浏览器。
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app
"""FastAPI 服务和内存会话管理。"""

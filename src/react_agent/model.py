"""模型层：把 OpenAI 兼容 HTTP 响应转换成 Agent 能理解的 Python 对象。"""

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol

import httpx

from .config import ModelConfig


logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """模型发出的单次工具调用请求。"""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class AssistantMessage:
    """模型返回的 assistant 消息。

    普通回答只有 ``content``；需要工具时，``tool_calls`` 中会有一个或多个请求。
    """

    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)


@dataclass
class TokenUsage:
    """一次或多次模型调用消耗的 token 数量。

    OpenAI 兼容接口通常会返回真实 usage；如果某个兼容服务没有返回 usage，
    模型层会使用一个保守的字符数估算，并把 ``estimated`` 标记为 True。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = False

    def __post_init__(self) -> None:
        """保证 total_tokens 与输入、输出之和一致。"""
        # dataclass 创建对象后会自动调用 __post_init__。
        # 这里适合做依赖多个字段的补充计算。
        if self.total_tokens <= 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def add(self, other: "TokenUsage") -> None:
        """把另一次模型调用的 usage 累加到当前对象。"""
        # "TokenUsage" 是字符串形式的前向引用，允许在类定义内部引用本类。
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.estimated = self.estimated or other.estimated

    def as_dict(self) -> Dict[str, Any]:
        """转换为 API 和前端都容易使用的字典。"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated": self.estimated,
        }


@dataclass
class ModelResponse:
    message: AssistantMessage
    usage: TokenUsage = field(default_factory=TokenUsage)


def estimate_tokens(value: Any) -> int:
    """在模型服务不返回 usage 时做一个简单估算。

    这不是 tokenizer，只是为了让学习项目在不同兼容服务上仍然能显示成本。
    中文字符通常接近一个 token，英文和 JSON 符号按约四个字符一个 token 估计。
    真正计费或精确限额时，应优先使用模型 API 返回的 usage。
    """
    text = json.dumps(value, ensure_ascii=False, default=str)
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - chinese
    return max(1, chinese + math.ceil(other / 4))


class ChatModel(Protocol):
    """模型接口协议。

    ``Protocol`` 类似 Java/TypeScript 的接口。任何对象只要实现同样的 ``complete``
    方法，就可以传给 ReActAgent。测试中的 ScriptedModel 就利用了这一点。
    """

    # Protocol 用结构描述接口，而不是要求继承某个基类。
    # 只要对象实现下面的方法，类型检查器就可以把它当作 ChatModel。
    async def complete(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> ModelResponse:
        ...

    async def stream(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> AsyncIterator["ModelStreamEvent"]:
        ...


@dataclass
class ModelStreamEvent:
    """模型流式响应中的一个事件。

    ``delta`` 是本次新产生的文本片段。模型通常会把完整回答拆成很多个
    小片段返回；``response`` 在流结束时提供最终的完整 ModelResponse。
    """

    delta: str = ""
    response: Optional[ModelResponse] = None


class OpenAICompatibleClient:
    """通过 OpenAI 兼容的 ``/chat/completions`` 接口调用模型。"""

    def __init__(self, config: ModelConfig, client: Optional[httpx.AsyncClient] = None):
        self.config = config
        self.client = client

    async def complete(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> ModelResponse:
        # Dict[str, Any] 表示键是字符串、值可以是任意类型的字典。
        # 字典字面量创建请求 JSON；类型标注只说明预期结构。
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if tools:
            # 没有工具时不发送 tools 字段，兼容部分实现不完整的模型服务。
            payload["tools"] = tools

        # 测试可以注入 MockTransport 客户端；真实运行时由这里临时创建客户端。
        # is None 是判断“确实没有对象”的推荐写法。
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.config.timeout_seconds)
        started_at = time.perf_counter()
        logger.info(
            "model_request_started model=%s base_url=%s message_count=%d tool_count=%d",
            self.config.model,
            self.config.base_url,
            len(messages),
            len(tools),
        )
        # 外层 try/finally 负责释放客户端，内层 try/except 负责记录请求失败。
        try:
            try:
                response = await client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
            except Exception:
                logger.exception(
                    "model_request_failed model=%s elapsed_ms=%.2f",
                    self.config.model,
                    (time.perf_counter() - started_at) * 1000,
                )
                raise
            response_data = response.json()
            message = response_data["choices"][0]["message"]
            tool_calls = []
            # get 找不到 key 时返回 None；or [] 将 None 转成可遍历空列表。
            for item in message.get("tool_calls") or []:
                function = item["function"]
                raw_arguments = function.get("arguments") or "{}"
                # OpenAI 协议中的 arguments 通常是 JSON 字符串，而不是字典。
                arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else raw_arguments
                )
                tool_calls.append(
                    ToolCall(
                        id=item["id"],
                        name=function["name"],
                        arguments=arguments,
                    )
                )
            raw_usage = response_data.get("usage") or {}
            has_usage = any(
                key in raw_usage
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            if has_usage:
                prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
                completion_tokens = int(raw_usage.get("completion_tokens") or 0)
                total_tokens = int(
                    raw_usage.get("total_tokens")
                    or prompt_tokens + completion_tokens
                )
                usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            else:
                # 兼容不提供 usage 的服务：分别估算请求和响应部分。
                usage = TokenUsage(
                    prompt_tokens=estimate_tokens(
                        {"messages": messages, "tools": tools}
                    ),
                    completion_tokens=estimate_tokens(message),
                    estimated=True,
                )

            logger.info(
                "model_request_completed model=%s status_code=%d elapsed_ms=%.2f "
                "prompt_tokens=%d completion_tokens=%d total_tokens=%d estimated=%s",
                self.config.model,
                response.status_code,
                (time.perf_counter() - started_at) * 1000,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
                usage.estimated,
            )
            return ModelResponse(
                message=AssistantMessage(
                    content=message.get("content"),
                    tool_calls=tool_calls,
                ),
                usage=usage,
            )
        finally:
            # 只有本方法创建的客户端才由本方法关闭；外部注入的客户端由外部管理。
            if owns_client:
                await client.aclose()

    async def stream(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> AsyncIterator[ModelStreamEvent]:
        """以 SSE 流式方式调用 OpenAI 兼容接口。

        工具调用的参数也可能被拆成多个 delta，因此这里先把 tool call
        片段累积完整，最后组装成和 ``complete`` 相同的 ModelResponse。
        """
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.config.timeout_seconds)
        started_at = time.perf_counter()
        logger.info(
            "model_stream_started model=%s base_url=%s message_count=%d tool_count=%d",
            self.config.model,
            self.config.base_url,
            len(messages),
            len(tools),
        )

        # 流式文本先收集到 list，最后用 join 一次性合并，避免反复拼接字符串。
        content_parts: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        usage: Optional[TokenUsage] = None
        status_code = 0
        try:
            try:
                request = client.build_request(
                    "POST",
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response = await client.send(request, stream=True)
                response.raise_for_status()
                status_code = response.status_code
            except Exception:
                logger.exception(
                    "model_stream_failed model=%s elapsed_ms=%.2f",
                    self.config.model,
                    (time.perf_counter() - started_at) * 1000,
                )
                raise

            # HTTPX 的 aiter_lines 是异步生成器；收到一行就恢复一次循环。
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw_data = line[5:].strip()
                if raw_data == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw_data)
                except json.JSONDecodeError:
                    logger.warning("model_stream_invalid_chunk model=%s", self.config.model)
                    continue

                raw_usage = chunk.get("usage") or {}
                if raw_usage:
                    usage = TokenUsage(
                        prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
                        completion_tokens=int(raw_usage.get("completion_tokens") or 0),
                        total_tokens=int(raw_usage.get("total_tokens") or 0),
                    )
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {})
                text = delta.get("content") or ""
                if text:
                    content_parts.append(text)
                    yield ModelStreamEvent(delta=text)

                # 同一工具调用可能跨多个 SSE chunk，因此按 index 累积参数片段。
                for item in delta.get("tool_calls") or []:
                    index = int(item.get("index", 0))
                    current = tool_calls.setdefault(
                        index,
                        {"id": item.get("id", ""), "name": "", "arguments": ""},
                    )
                    if item.get("id"):
                        current["id"] = item["id"]
                    function = item.get("function") or {}
                    current["name"] += function.get("name") or ""
                    current["arguments"] += function.get("arguments") or ""

            if usage is None:
                usage = TokenUsage(
                    prompt_tokens=estimate_tokens({"messages": messages, "tools": tools}),
                    completion_tokens=estimate_tokens("".join(content_parts)),
                    estimated=True,
                )
            parsed_tool_calls: List[ToolCall] = []
            for item in tool_calls.values():
                raw_arguments = item["arguments"] or "{}"
                parsed_tool_calls.append(
                    ToolCall(
                        id=item["id"],
                        name=item["name"],
                        arguments=(
                            json.loads(raw_arguments)
                            if isinstance(raw_arguments, str)
                            else raw_arguments
                        ),
                    )
                )
            final_response = ModelResponse(
                message=AssistantMessage(
                    content="".join(content_parts) or None,
                    tool_calls=parsed_tool_calls,
                ),
                usage=usage,
            )
            logger.info(
                "model_stream_completed model=%s status_code=%d elapsed_ms=%.2f "
                "prompt_tokens=%d completion_tokens=%d total_tokens=%d estimated=%s",
                self.config.model,
                status_code,
                (time.perf_counter() - started_at) * 1000,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
                usage.estimated,
            )
            yield ModelStreamEvent(response=final_response)
        finally:
            # finally 即使流式请求中途抛异常也会执行，避免 HTTP 连接泄漏。
            await response.aclose() if "response" in locals() else None
            if owns_client:
                await client.aclose()

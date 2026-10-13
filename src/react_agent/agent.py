"""ReAct Agent 的核心循环。

建议学习时首先阅读这个文件。这里没有使用 LangGraph，而是直接用 ``for`` 循环
实现下面的流程：

1. 把对话和工具说明发送给模型；
2. 如果模型给出普通文本，就结束本轮对话；
3. 如果模型要求调用工具，就执行工具；
4. 把工具结果追加到对话，再回到第 1 步。

``async`` 表示这是异步代码：等待模型或工具执行时，Web 服务可以继续处理其他请求。
"""

import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from .events import AgentEvent
from .model import ChatModel, ModelStreamEvent, TokenUsage
from .tools import ToolRegistry


logger = logging.getLogger(__name__)


class ReActAgent:
    """协调模型和工具的 Agent。

    这个类本身不知道模型来自哪个厂商，也不知道工具是 Python、REST 还是 MCP。
    它只依赖 ``ChatModel`` 和 ``ToolRegistry`` 两个统一接口。
    """

    def __init__(
        self,
        model: ChatModel,
        tools: Optional[ToolRegistry] = None,
        system_prompt: str = "You are a helpful assistant.",
        max_steps: int = 10,
    ):
        # ``or`` 在左侧为空时使用右侧值。调用者不传工具时，创建一个空注册表。
        self.model = model
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    async def stream(
        self, messages: List[Dict[str, Any]]
    ) -> AsyncIterator[AgentEvent]:
        """运行 Agent，并逐个产出过程事件。

        ``AsyncIterator`` 可以理解为异步版本的迭代器。这里使用 ``yield``，所以调用者
        能在最终答案生成前先收到 ``model_started``、``tool_started`` 等过程事件。
        Web 页面正是利用这些事件实时显示工具执行过程。
        """
        # 星号 ``*messages`` 会把列表中的元素展开到新列表中。
        conversation = [{"role": "system", "content": self.system_prompt}, *messages]
        # 模型不能直接读取 Python 函数，因此要先转成 OpenAI tools JSON Schema。
        definitions = self.tools.definitions()
        # 一次 Agent run 可能因为工具调用而调用模型多次，因此必须累计每一步 usage。
        run_usage = TokenUsage()
        logger.info(
            "agent_run_started message_count=%d tool_count=%d max_steps=%d",
            len(messages),
            len(definitions),
            self.max_steps,
        )

        # 一次“步骤”代表调用一次模型。限制步骤数可以避免模型无限调用工具。
        for step in range(1, self.max_steps + 1):
            logger.info("model_step_started step=%d", step)
            yield AgentEvent("model_started", {"step": step})
            try:
                # 优先使用模型的流式接口；旧的测试模型或自定义模型如果只实现
                # complete，则继续走兼容分支。
                if hasattr(self.model, "stream"):
                    response = None
                    async for model_event in self.model.stream(conversation, definitions):
                        if model_event.delta:
                            yield AgentEvent(
                                "assistant_delta",
                                {"content": model_event.delta, "step": step},
                            )
                        if model_event.response is not None:
                            response = model_event.response
                    if response is None:
                        raise RuntimeError("模型流式响应没有返回最终结果")
                else:
                    response = await self.model.complete(conversation, definitions)
            except Exception as exc:
                logger.exception("model_step_failed step=%d", step)
                # type(exc).__name__ 得到异常类型名，例如 HTTPStatusError。
                yield AgentEvent("error", {"message": f"{type(exc).__name__}: {exc}"})
                return

            assistant = response.message
            run_usage.add(response.usage)
            logger.info(
                "model_step_completed step=%d tool_call_count=%d "
                "prompt_tokens=%d completion_tokens=%d total_tokens=%d estimated=%s",
                step,
                len(assistant.tool_calls),
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
                response.usage.estimated,
            )
            yield AgentEvent("model_usage", {
                **response.usage.as_dict(),
                "step": step,
            })
            # 无论有没有工具调用，都要先把 assistant 消息写回对话历史。
            # 有些模型要求后续 tool 消息之前必须存在对应的 assistant tool_calls。
            assistant_payload: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant.content,
            }
            if assistant.tool_calls:
                # 列表推导式：遍历所有 ToolCall，生成 OpenAI 接口要求的字典。
                assistant_payload["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in assistant.tool_calls
                ]
            conversation.append(assistant_payload)

            # 没有 tool_calls 表示模型已经给出了最终回答，结束 ReAct 循环。
            if not assistant.tool_calls:
                content = assistant.content or ""
                logger.info(
                    "agent_run_completed answer_length=%d prompt_tokens=%d "
                    "completion_tokens=%d total_tokens=%d estimated=%s",
                    len(content),
                    run_usage.prompt_tokens,
                    run_usage.completion_tokens,
                    run_usage.total_tokens,
                    run_usage.estimated,
                )
                yield AgentEvent(
                    "assistant_completed",
                    {"content": content, "usage": run_usage.as_dict()},
                )
                return

            # 一个 assistant 消息可能同时请求多个工具，因此这里逐个执行。
            for call in assistant.tool_calls:
                started_at = time.perf_counter()
                logger.info(
                    "tool_call_started name=%s call_id=%s argument_keys=%s",
                    call.name,
                    call.id,
                    sorted(call.arguments.keys()),
                )
                yield AgentEvent(
                    "tool_started",
                    {"id": call.id, "name": call.name, "arguments": call.arguments},
                )
                result = await self.tools.execute(call.name, call.arguments)
                logger.info(
                    "tool_call_completed name=%s call_id=%s ok=%s elapsed_ms=%.2f",
                    call.name,
                    call.id,
                    result.ok,
                    (time.perf_counter() - started_at) * 1000,
                )
                yield AgentEvent(
                    "tool_finished",
                    {
                        "id": call.id,
                        "name": call.name,
                        "ok": result.ok,
                        "value": result.value,
                        "error": result.error,
                    },
                )
                conversation.append(
                    {
                        "role": "tool",
                        # tool_call_id 用来告诉模型：这个结果对应哪个工具请求。
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result.to_json(),
                    }
                )

        # for 循环正常耗尽，说明每一步模型都在继续调用工具，没有生成最终答案。
        logger.error("agent_run_failed reason=max_steps max_steps=%d", self.max_steps)
        yield AgentEvent(
            "error",
            {"message": f"Maximum agent steps reached ({self.max_steps})."},
        )

    async def run(self, messages: List[Dict[str, Any]]) -> str:
        """只返回最终文本，适合不关心中间事件的调用方。"""
        answer = ""
        # ``async for`` 用来遍历异步生成器 ``stream`` 产生的事件。
        async for event in self.stream(messages):
            if event.type == "assistant_completed":
                answer = event.data["content"]
            elif event.type == "error":
                raise RuntimeError(event.data["message"])
        return answer

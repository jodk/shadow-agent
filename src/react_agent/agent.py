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
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from .events import AgentEvent
from .model import ChatModel, ModelStreamEvent, TokenUsage
from .tools import ToolRegistry


logger = logging.getLogger(__name__)
# ``getLogger(__name__)`` 使用当前模块名创建日志记录器。
# 这样调用 ``logger.info(...)`` 时，日志系统可以知道消息来自哪个文件。


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
        # 参数后的 ``: 类型`` 是类型标注，``-> 返回类型`` 也是类型标注。
        # Python 运行时通常不会仅凭类型标注阻止错误，它主要帮助阅读者、
        # IDE 和静态检查工具理解数据类型。
        #
        # ``or`` 在左侧对象为假值时使用右侧对象。这里表示调用者不传工具
        # （传入 None）时，自动创建一个空的 ToolRegistry。
        self.model = model
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        additional_instructions: str = "",
        allowed_tool_names: Optional[Set[str]] = None,
    ) -> AsyncIterator[AgentEvent]:
        """运行 Agent，并逐个产出过程事件。

        ``AsyncIterator`` 可以理解为异步版本的迭代器。这里使用 ``yield``，所以调用者
        能在最终答案生成前先收到 ``model_started``、``tool_started`` 等过程事件。
        Web 页面正是利用这些事件实时显示工具执行过程。
        """
        # 形参列表中的单独 ``*`` 表示：后面的参数只能使用关键字传入。
        # 例如必须写 ``agent.stream(messages, allowed_tool_names=names)``，
        # 不能把 names 作为第三个位置参数传入。这样可避免参数顺序误用。
        #
        # 列表中的 ``*messages`` 是“可迭代对象解包”语法：
        # [{"role": "system"}, *messages] 会把 messages 中的每条消息展开到
        # 新列表，而不是把整个 messages 列表嵌套成一个元素。
        system_prompt = self.system_prompt
        if additional_instructions.strip():
            # 能力路由只把本轮命中的 Skill 说明放进来，避免启动时把所有
            # SKILL.md 都塞进每一次模型请求。
            system_prompt = (
                f"{system_prompt}\n\n"
                "本轮相关 Skill 说明如下。只有在与当前问题相关时才参考：\n\n"
                f"{additional_instructions.strip()}"
            )
        conversation = [{"role": "system", "content": system_prompt}, *messages]
        # 模型不能直接读取 Python 函数，因此要先转成 OpenAI tools JSON Schema。
        # ``definitions`` 是一个普通列表；如果 allowed_tool_names 是集合，
        # ToolRegistry 会只筛选本轮允许暴露的工具。
        definitions = self.tools.definitions(allowed_tool_names)
        # 一次 Agent run 可能因为工具调用而调用模型多次，因此必须累计每一步 usage。
        run_usage = TokenUsage()
        logger.info(
            "agent_run_started message_count=%d tool_count=%d max_steps=%d",
            len(messages),
            len(definitions),
            self.max_steps,
        )

        # 一次“步骤”代表调用一次模型。``range(start, stop)`` 包含 start，
        # 但不包含 stop，所以这里最多执行 max_steps 次。
        # 限制步骤数可以避免模型无限调用工具。
        for step in range(1, self.max_steps + 1):
            logger.info("model_step_started step=%d", step)
            yield AgentEvent("model_started", {"step": step})
            try:
                # 优先使用模型的流式接口；旧的测试模型或自定义模型如果只实现
                # complete，则继续走兼容分支。
                # ``hasattr(object, name)`` 用来判断对象是否具有某个属性。
                # 这里允许模型只实现 complete()，方便编写简单测试模型。
                if hasattr(self.model, "stream"):
                    response = None
                    # ``async for`` 逐个读取异步生成器产生的事件。
                    # 每次循环都可能立刻把增量文本转发给浏览器。
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
            # ``Dict[str, Any]`` 表示字典键是字符串、值可以是任意类型。
            assistant_payload: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant.content,
            }
            if assistant.tool_calls:
                # 列表推导式是 for 循环创建列表的简写：
                # [表达式 for item in iterable]
                # 这里遍历每个 ToolCall，生成 OpenAI 接口要求的字典。
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
            # 即使只有一个工具调用，也保持列表循环结构，便于支持并行请求扩展。
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
                result = await self.tools.execute(
                    call.name,
                    call.arguments,
                    allowed_names=allowed_tool_names,
                )
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
                # ``append`` 会把一个完整对象追加到列表末尾。
                # tool 消息必须放回 conversation，下一次模型调用才能看到结果。
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

    async def run(
        self,
        messages: List[Dict[str, Any]],
        *,
        additional_instructions: str = "",
        allowed_tool_names: Optional[Set[str]] = None,
    ) -> str:
        """只返回最终文本，适合不关心中间事件的调用方。"""
        answer = ""
        # ``async for`` 用来遍历异步生成器 ``stream`` 产生的事件。
        async for event in self.stream(
            messages,
            additional_instructions=additional_instructions,
            allowed_tool_names=allowed_tool_names,
        ):
            if event.type == "assistant_completed":
                answer = event.data["content"]
            elif event.type == "error":
                raise RuntimeError(event.data["message"])
        return answer

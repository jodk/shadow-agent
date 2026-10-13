import inspect
from typing import Any, Callable, Dict, Optional, Union, get_args, get_origin

from pydantic import TypeAdapter

from .base import Tool


def _json_type(annotation: Any) -> Dict[str, Any]:
    """把常见 Python 类型标注转换成 JSON Schema 类型。"""
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _json_type(args[0])
    mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return {"type": mapping.get(origin or annotation, "string")}


def tool(
    function: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
):
    """工具装饰器。

    两种用法都支持：

    ``@tool`` 直接使用函数名和文档字符串；
    ``@tool(name="search", description="搜索")`` 显式指定元数据。

    装饰器执行后，原函数名绑定的不再是函数本身，而是一个 ``Tool`` 对象。
    """

    def decorate(fn: Callable[..., Any]) -> Tool:
        # inspect.signature 可以在运行时读取参数名、类型标注和默认值。
        signature = inspect.signature(fn)
        properties: Dict[str, Any] = {}
        required = []
        adapters: Dict[str, TypeAdapter] = {}
        for parameter_name, parameter in signature.parameters.items():
            annotation = (
                parameter.annotation
                if parameter.annotation is not inspect.Parameter.empty
                else str
            )
            properties[parameter_name] = _json_type(annotation)
            adapters[parameter_name] = TypeAdapter(annotation)
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter_name)

        async def handler(arguments: Dict[str, Any]) -> Any:
            """闭包：这里仍能访问外层函数创建的 signature 和 adapters。"""
            unknown = set(arguments) - set(signature.parameters)
            if unknown:
                raise ValueError(f"Unexpected arguments: {sorted(unknown)}")
            parsed = {}
            for parameter_name, parameter in signature.parameters.items():
                if parameter_name not in arguments:
                    if parameter.default is inspect.Parameter.empty:
                        raise ValueError(f"Missing argument: {parameter_name}")
                    continue
                parsed[parameter_name] = adapters[parameter_name].validate_python(
                    arguments[parameter_name], strict=True
                )
            result = fn(**parsed)
            # 同一个装饰器同时支持 def 和 async def；异步结果需要 await。
            if inspect.isawaitable(result):
                return await result
            return result

        return Tool(
            name=name or fn.__name__,
            description=description or inspect.getdoc(fn) or fn.__name__,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            handler=handler,
        )

    if function is not None:
        # 对应不带括号的 ``@tool``。
        return decorate(function)
    # 对应带参数的 ``@tool(...)``，先返回真正的装饰函数。
    return decorate
"""把普通 Python 函数包装成 Agent 工具。"""

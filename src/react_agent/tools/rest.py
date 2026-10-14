from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml

from .base import Tool


def _extract_path(value: Any, path: Optional[str]) -> Any:
    """按 ``data.items.0`` 这样的路径，从 JSON 响应中提取局部数据。"""
    if not path:
        return value
    current = value
    # ``for part in ...`` 依次处理路径中的每一段。
    # 三元表达式 ``A if condition else B`` 根据条件选择一个值。
    for part in path.split("."):
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def load_rest_tools(
    path: Path, client: Optional[httpx.AsyncClient] = None
) -> List[Tool]:
    """读取一个 REST 工具 YAML 文件。

    URL 来自配置而不是模型，因此模型只能填写参数，不能决定访问任意主机。
    ``client`` 参数主要用于测试时注入 MockTransport。
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    tools: List[Tool] = []
    # ``dict.get("tools", [])`` 在键不存在时返回空列表，因此配置缺少 tools
    # 时不会抛出 KeyError。
    for definition in data.get("tools", []):
        item = dict(definition)
        shared_client = client

        async def handler(arguments: Dict[str, Any], spec=item, http_client=shared_client):
            """真正发送 HTTP 请求的闭包。

            默认参数 ``spec=item`` 很重要：它会固定本次循环的配置。若直接引用 item，
            所有闭包最终都可能使用循环中的最后一项，这是 Python 常见的闭包陷阱。
            """
            url = spec["url"]
            # 路径参数：/users/{user_id} -> /users/7
            # 默认参数 spec=item 会在创建闭包时保存当前 item。
            # 这是解决 Python 循环闭包晚绑定问题的常见写法。
            for name in spec.get("path_params", []):
                if name not in arguments:
                    raise ValueError(f"Missing path argument: {name}")
                url = url.replace("{" + name + "}", str(arguments[name]))

            # 查询参数：{"verbose": True} -> ?verbose=true
            # 字典推导式会边遍历边构造字典。
            query = {
                name: str(arguments[name]).lower()
                if isinstance(arguments.get(name), bool)
                else arguments[name]
                for name in spec.get("query_params", [])
                if name in arguments
            }
            body_names = spec.get("body_params", [])
            # body_params 中列出的参数会组成 JSON 请求体。
            body = {name: arguments[name] for name in body_names if name in arguments}
            headers = dict(spec.get("headers", {}))
            owns_client = http_client is None
            active_client = http_client or httpx.AsyncClient(
                timeout=spec.get("timeout_seconds", 30)
            )
            try:
                response = await active_client.request(
                    spec.get("method", "GET").upper(),
                    url,
                    params=query or None,
                    json=body or None,
                    headers=headers or None,
                )
                response.raise_for_status()
                if not response.content:
                    value = None
                else:
                    value = response.json()
                return _extract_path(value, spec.get("response_path"))
            finally:
                # finally 无论请求成功还是抛出异常都会执行，适合释放 HTTP 客户端。
                if owns_client:
                    await active_client.aclose()

        # ``append`` 将新建的 Tool 放入结果列表。
        tools.append(
            Tool(
                name=item["name"],
                description=item["description"],
                parameters=item.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
                handler=handler,
                keywords=item.get("keywords", []),
                always_expose=bool(item.get("always_expose", False)),
            )
        )
    return tools
"""把 YAML 中声明的 REST API 包装成统一 Tool。"""

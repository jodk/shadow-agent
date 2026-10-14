"""一个最小 Skill 工具示例。

函数类型标注会被 ``@tool`` 转换成 JSON Schema，文档字符串会成为工具描述。
"""

from react_agent.tools import tool


@tool
def add_numbers(left: float, right: float) -> float:
    """Add two numbers."""
    # ``-> float`` 是返回值类型标注，告诉装饰器和阅读者结果应为浮点数。
    return left + right


@tool
def multiply_numbers(left: float, right: float) -> float:
    """Multiply two numbers."""
    return left * right


# SkillLoader 会读取这个约定名称，把其中的 Tool 注册给 Agent。
TOOLS = [add_numbers, multiply_numbers]

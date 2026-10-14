"""文本分析 Skill 的工具实现。

这个示例特意只使用 Python 标准库，复制项目后无需安装额外依赖即可运行。
模型看不到函数内部实现，只能看到 @tool 生成的名称、说明和参数 Schema。
"""

import re
from typing import Dict, Union

from react_agent.tools import tool


# 这是一个“类型别名”：给复杂类型起一个容易阅读的名字。
# Dict[str, Union[int, str]] 表示键是字符串，值可以是 int 或 str。
# ``Union`` 是“类型并集”，意思是允许多个候选类型，而不是只能选一个。
TextStatistics = Dict[str, Union[int, str]]


# ``@tool`` 是装饰器语法，等价于先定义函数，再执行
# ``count_text = tool(count_text)``。项目用它把普通函数注册为 Agent 工具。
@tool
def count_text(text: str) -> TextStatistics:
    """精确统计文本的字符数、非空白字符数、中文字符数、英文单词数和行数。"""
    # len 会统计所有字符，包括空格、标点和换行。
    total_characters = len(text)

    # str.isspace() 可识别空格、制表符和换行符等空白字符。
    # ``1 for character in text if ...`` 是生成器表达式：它逐个产生 1，
    # 不会先创建完整的中间列表；sum 再把这些 1 相加。
    non_whitespace_characters = sum(
        1 for character in text if not character.isspace()
    )

    # 常用汉字主要位于 Unicode 的 \u4e00 到 \u9fff 区间。
    chinese_characters = len(re.findall(r"[\u4e00-\u9fff]", text))

    # 这里把连续的英文字母视为一个英文单词。
    english_words = len(re.findall(r"[A-Za-z]+", text))

    # splitlines() 能处理 \n 和 \r\n。空字符串按 0 行处理。
    lines = len(text.splitlines()) if text else 0

    return {
        "total_characters": total_characters,
        "non_whitespace_characters": non_whitespace_characters,
        "chinese_characters": chinese_characters,
        "english_words": english_words,
        "lines": lines,
    }


# 函数参数的 ``= False`` 是默认值：调用方不传 case_sensitive 时自动使用 False。
@tool
def count_keyword(
    text: str,
    keyword: str,
    case_sensitive: bool = False,
) -> Dict[str, Union[str, int, bool]]:
    """统计关键词或短语在文本中出现的次数，可选择是否区分英文大小写。"""
    if not keyword:
        # 空字符串在 Python 中会匹配到很多位置，通常不符合用户意图，因此明确拒绝。
        raise ValueError("keyword 不能为空")

    if case_sensitive:
        count = text.count(keyword)
    else:
        # casefold 比 lower 更适合做不区分大小写的 Unicode 文本比较。
        count = text.casefold().count(keyword.casefold())

    return {
        "keyword": keyword,
        "count": count,
        "case_sensitive": case_sensitive,
    }


# 返回类型仍然复用 TextStatistics，因为这里的字典值同样包含 int 和 str。
@tool
def summarize_text_info(text: str, preview_length: int = 80) -> TextStatistics:
    """返回文本总长度、行数、首行和指定长度的内容预览。"""
    if preview_length < 1:
        raise ValueError("preview_length 必须大于等于 1")

    lines = text.splitlines()
    first_line = lines[0] if lines else ""

    # 文本超过预览长度时添加省略号；否则返回完整文本。
    # 这是条件表达式（三元表达式）：条件成立取左边，否则取右边。
    preview = (
        text[:preview_length] + "..."
        if len(text) > preview_length
        else text
    )

    return {
        "total_characters": len(text),
        "lines": len(lines),
        "first_line": first_line,
        "preview": preview,
    }


# 列表保存三个函数对象，而不是函数调用结果；因此加载 Skill 时可以读取
# 它们的名称、文档字符串和参数类型，再交给 Agent 决定何时调用。
TOOLS = [count_text, count_keyword, summarize_text_info]

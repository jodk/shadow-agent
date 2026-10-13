"""文本分析 Skill 的工具实现。

这个示例特意只使用 Python 标准库，复制项目后无需安装额外依赖即可运行。
模型看不到函数内部实现，只能看到 @tool 生成的名称、说明和参数 Schema。
"""

import re
from typing import Dict, Union

from react_agent.tools import tool


# 返回值中的数值和字符串放在同一个字典中，因此值类型写成 Union[int, str]。
TextStatistics = Dict[str, Union[int, str]]


@tool
def count_text(text: str) -> TextStatistics:
    """精确统计文本的字符数、非空白字符数、中文字符数、英文单词数和行数。"""
    # len 会统计所有字符，包括空格、标点和换行。
    total_characters = len(text)

    # str.isspace() 可识别空格、制表符和换行符等空白字符。
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


@tool
def summarize_text_info(text: str, preview_length: int = 80) -> TextStatistics:
    """返回文本总长度、行数、首行和指定长度的内容预览。"""
    if preview_length < 1:
        raise ValueError("preview_length 必须大于等于 1")

    lines = text.splitlines()
    first_line = lines[0] if lines else ""

    # 文本超过预览长度时添加省略号；否则返回完整文本。
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


# SkillLoader 约定读取 TOOLS。只有放进这个列表的工具才会注册给 Agent。
TOOLS = [count_text, count_keyword, summarize_text_info]

"""加载 Skill 元数据和 Skill 附带的 Python 工具。

完整 ``SKILL.md`` 不再在启动时统一拼入系统提示词。启动阶段只登记名称、描述、
关键词、说明文件路径和所属工具；每轮请求经过能力检索后，才读取命中的说明。
"""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import yaml

from .tools import Tool


@dataclass
class LoadedSkill:
    """一个可被能力路由检索和按需加载的 Skill。"""

    name: str
    description: str
    keywords: List[str]
    instruction_path: Path
    tool_names: List[str]

    def load_instructions(self) -> str:
        """只在本轮检索命中 Skill 后读取完整 SKILL.md。"""
        # Path.read_text 是 pathlib 提供的便捷文件读取方法；
        # with open(...) 类似，但 Path 对象把路径操作封装成了方法。
        return self.instruction_path.read_text(encoding="utf-8").strip()


@dataclass
class LoadedSkills:
    """汇总所有启用 Skill 的元数据和工具。"""

    skills: List[LoadedSkill] = field(default_factory=list)
    tools: List[Tool] = field(default_factory=list)

    @property
    def instructions(self) -> str:
        """旧测试和学习代码可读取的摘要属性。

        生产请求不会使用这个属性拼接全量提示词。它只返回 Skill 文件路径，
        需要真正内容时必须调用单个 ``LoadedSkill.load_instructions()``。
        """
        # 这是只为旧学习示例保留的显式兼容读取。Agent 主链路不会访问它，
        # 因此不会把所有 Skill 文档发送给模型。
        return "\n\n".join(
            skill.load_instructions() for skill in self.skills
        )


class SkillLoader:
    def __init__(self, directories: Sequence[Path]):
        # Sequence 表示“有顺序、可遍历的序列”，例如 list 或 tuple。
        # Path(item) 会把字符串路径统一转换为 pathlib.Path 对象。
        self.directories = [Path(item) for item in directories]

    def load(self) -> LoadedSkills:
        """扫描目录并动态导入启用的 Skill。"""
        loaded_skills: List[LoadedSkill] = []
        tools: List[Tool] = []
        # 这里连续使用多层 for，是“遍历目录 -> 遍历 Skill 子目录”的结构。
        for root in self.directories:
            if not root.exists():
                continue
            # 条件表达式：如果 root 自己就是 Skill，就只处理 root；
            # 否则处理 root 下的每个子目录。
            candidates = (
                [root] if (root / "skill.yaml").exists() else sorted(root.iterdir())
            )
            for directory in candidates:
                if not directory.is_dir():
                    continue
                manifest_path = directory / "skill.yaml"
                instruction_path = directory / "SKILL.md"
                if not manifest_path.exists() or not instruction_path.exists():
                    continue
                # yaml.safe_load 把 YAML 文本解析成 Python 字典/列表。
                manifest = yaml.safe_load(
                    manifest_path.read_text(encoding="utf-8")
                ) or {}
                if not manifest.get("enabled", True):
                    continue

                skill_name = str(manifest.get("name") or directory.name)
                raw_instructions = instruction_path.read_text(encoding="utf-8").strip()
                first_line = raw_instructions.splitlines()[0] if raw_instructions else skill_name
                description = str(
                    manifest.get("description") or first_line
                ).lstrip("# ").strip()
                keywords = [str(item) for item in manifest.get("keywords", [])]

                skill_tools: List[Tool] = []
                module_name = manifest.get("python_module")
                if module_name:
                    module_path = directory / module_name
                    unique_name = (
                        f"react_agent_skill_{directory.name}_{abs(hash(module_path))}"
                    )
                    spec = importlib.util.spec_from_file_location(unique_name, module_path)
                    if spec is None or spec.loader is None:
                        raise RuntimeError(f"Cannot load skill module: {module_path}")
                    # 动态导入：模块文件路径由配置决定，不要求它提前安装成包。
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    skill_tools = list(getattr(module, "TOOLS", []))
                    for item in skill_tools:
                        item.skill_name = skill_name
                    tools.extend(skill_tools)

                loaded_skills.append(
                    LoadedSkill(
                        name=skill_name,
                        description=description,
                        keywords=keywords,
                        instruction_path=instruction_path,
                        tool_names=[item.name for item in skill_tools],
                    )
                )
        return LoadedSkills(skills=loaded_skills, tools=tools)

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import yaml

from .tools import Tool


@dataclass
class LoadedSkills:
    """汇总所有启用 Skill 后得到的提示词和工具。"""

    instructions: str = ""
    tools: List[Tool] = field(default_factory=list)


class SkillLoader:
    def __init__(self, directories: Sequence[Path]):
        self.directories = [Path(item) for item in directories]

    def load(self) -> LoadedSkills:
        """扫描目录并动态导入启用的 Skill。"""
        instructions = []
        tools: List[Tool] = []
        for root in self.directories:
            if not root.exists():
                continue
            candidates = [root] if (root / "skill.yaml").exists() else sorted(root.iterdir())
            for directory in candidates:
                if not directory.is_dir():
                    continue
                manifest_path = directory / "skill.yaml"
                instruction_path = directory / "SKILL.md"
                if not manifest_path.exists() or not instruction_path.exists():
                    continue
                manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
                if not manifest.get("enabled", True):
                    continue
                instructions.append(instruction_path.read_text(encoding="utf-8").strip())
                module_name = manifest.get("python_module")
                if module_name:
                    module_path = directory / module_name
                    # 每个动态模块使用唯一名称，避免两个 Skill 的 tools.py 相互覆盖。
                    unique_name = f"react_agent_skill_{directory.name}_{abs(hash(module_path))}"
                    spec = importlib.util.spec_from_file_location(unique_name, module_path)
                    if spec is None or spec.loader is None:
                        raise RuntimeError(f"Cannot load skill module: {module_path}")
                    module = importlib.util.module_from_spec(spec)
                    # exec_module 会真正执行 tools.py；其顶层 @tool 也在此时执行。
                    spec.loader.exec_module(module)
                    tools.extend(getattr(module, "TOOLS", []))
        return LoadedSkills(
            instructions="\n\n".join(instructions),
            tools=tools,
        )
"""加载 Skill 说明和 Skill 附带的 Python 工具。

这里的 Skill 不是可执行协议，而是一个组织方式：

- ``SKILL.md``：提供给模型的使用说明；
- ``skill.yaml``：控制是否启用以及 Python 模块入口；
- ``tools.py``：可选的工具实现。
"""

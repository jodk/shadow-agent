import logging
from pathlib import Path

from react_agent.config import LoggingConfig
from react_agent.logging_config import configure_logging


def test_configure_logging_writes_utf8_file(tmp_path: Path):
    """日志配置应自动创建目录，并能够写入中文内容。"""
    config = LoggingConfig(
        level="INFO",
        console=False,
        file="./logs/agent.log",
        max_bytes=1024,
        backup_count=1,
    )

    configure_logging(config, tmp_path)
    logging.getLogger("react_agent.test").info("测试日志")

    # handler 会缓冲输出，显式 flush 后再读取文件，避免测试偶发失败。
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = (tmp_path / "logs" / "agent.log").read_text(encoding="utf-8")
    assert "INFO" in content
    assert "react_agent.test" in content
    assert "测试日志" in content

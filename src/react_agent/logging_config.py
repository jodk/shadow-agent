"""统一配置项目日志。

项目只使用 Python 标准库 ``logging``。业务代码通过
``logging.getLogger(__name__)`` 获取 logger，不需要关心日志最终输出到哪里。
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import LoggingConfig


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(config: LoggingConfig, base_dir: Path) -> None:
    """根据 YAML 配置初始化根日志记录器。

    ``force=True`` 会替换之前的 handler，防止测试、CLI 或服务重复初始化时，
    同一条日志被输出多次。相对日志路径以配置文件所在目录为基准。
    """
    handlers: list[logging.Handler] = []
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if config.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if config.file:
        log_path = Path(config.file)
        if not log_path.is_absolute():
            log_path = base_dir / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, config.level),
        handlers=handlers,
        force=True,
    )

"""统一配置项目日志。

项目只使用 Python 标准库 ``logging``。业务代码通过
``logging.getLogger(__name__)`` 获取 logger，不需要关心日志最终输出到哪里。
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import LoggingConfig


# logging 会用 LogRecord 中的字段替换 ``%(...)s`` 占位符。
# 例如 ``%(levelname)s`` 会变成 INFO、ERROR 等日志级别文字。
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(config: LoggingConfig, base_dir: Path) -> None:
    """根据 YAML 配置初始化根日志记录器。

    ``force=True`` 会替换之前的 handler，防止测试、CLI 或服务重复初始化时，
    同一条日志被输出多次。相对日志路径以配置文件所在目录为基准。
    """
    # ``list[logging.Handler]`` 是 Python 3.9+ 的内置泛型写法，表示列表中
    # 的每个元素都应该是 logging.Handler 或其子类对象。
    handlers: list[logging.Handler] = []

    # Formatter 只负责“日志长什么样”，真正把日志写到终端或文件的是 Handler。
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if config.console:
        # StreamHandler 默认写入 stderr，适合在命令行实时查看日志。
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if config.file:
        # 配置文件中的相对路径不能依赖当前工作目录，否则从不同目录启动
        # 程序时日志可能被写到不同位置；这里统一相对于配置文件目录解析。
        log_path = Path(config.file)
        if not log_path.is_absolute():
            log_path = base_dir / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # RotatingFileHandler 会在文件达到 maxBytes 后轮换旧文件，并按
        # backupCount 保留有限数量的历史日志。
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # getattr(logging, config.level) 根据字符串动态读取 logging.INFO 等常量。
    # ``force=True`` 会移除旧 handler，避免重复初始化时一条日志打印多次。
    logging.basicConfig(
        level=getattr(logging, config.level),
        handlers=handlers,
        force=True,
    )

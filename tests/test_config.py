from pathlib import Path

from react_agent.config import load_config


def test_load_config_reads_model_and_agent_settings(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model:
  base_url: https://example.test/v1
  api_key: secret
  model: demo-model
agent:
  system_prompt: Be useful.
  max_steps: 5
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model.base_url == "https://example.test/v1"
    assert config.model.api_key == "secret"
    assert config.model.model == "demo-model"
    assert config.agent.max_steps == 5
    assert config.logging.level == "INFO"
    assert config.agent.cross_session_memory is True


def test_load_config_reads_logging_settings(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model:
  base_url: https://example.test/v1
  api_key: secret
  model: demo-model
logging:
  level: debug
  console: false
  file: ./logs/test.log
  max_bytes: 2048
  backup_count: 2
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.logging.level == "DEBUG"
    assert config.logging.console is False
    assert config.logging.file == "./logs/test.log"
    assert config.logging.max_bytes == 2048
    assert config.logging.backup_count == 2

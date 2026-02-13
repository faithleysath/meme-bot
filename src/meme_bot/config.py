import logging
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Self

import tomli_w

logger = logging.getLogger(__name__)


@dataclass
class Config:
    env: Literal["default", "prod", "dev"] = "default"
    ws_url: str = r"ws://{host}:{port}"
    token: str = "your_token"

    @classmethod
    def _get_config_path(cls, env_name: str) -> Path:
        """根据环境名称动态获取配置文件路径"""
        return Path.cwd() / f"config.{env_name}.toml"

    @classmethod
    def _load_or_create_toml(cls, file_path: Path, default_data: dict[str, Any]) -> dict[str, Any]:
        """通用辅助方法：加载 TOML，如果不存在则创建"""
        try:
            with open(file_path, "rb") as f:
                return tomllib.load(f)
        except FileNotFoundError:
            logger.warning(f"Cannot find {file_path}, auto-generating one!")
            with open(file_path, "wb") as f:
                tomli_w.dump(default_data, f)
            logger.info(f"Successfully generated {file_path}!")
            return default_data
        except (TypeError, tomllib.TOMLDecodeError) as e:
            logger.critical(f"{file_path} is invalid, please check syntax/fields.")
            raise e

    @classmethod
    def load_config(cls) -> Self:
        default_config = cls()
        default_path = cls._get_config_path("default")

        # --- 1. 处理 Default 配置 ---
        raw_config = cls._load_or_create_toml(default_path, asdict(default_config))
        config = cls(**raw_config)

        # --- 2. 检查环境分支 ---
        if config.env not in ["default", "prod", "dev"]:
            msg = f"{default_path} contains invalid field value env = '{config.env}', should be 'default', 'prod' or 'dev'."
            logger.critical(msg)
            raise ValueError(msg)

        if config.env == "default":
            return config

        # --- 3. 处理 Extend 配置 ---
        extend_path = cls._get_config_path(config.env)

        # 准备生成扩展配置时的默认字典（剔除 env，保持纯净）
        extend_default_dict = asdict(default_config)
        extend_default_dict.pop("env", None)

        raw_extend_config = cls._load_or_create_toml(extend_path, extend_default_dict)

        # 仅做验证，确保扩展配置的字段都是合法的
        cls(**(asdict(default_config) | raw_extend_config))

        # --- 4. 合并与最终校验 ---
        final_config = cls(**(raw_config | raw_extend_config))

        # 检查是否和 default 完全一致
        final_dict = asdict(final_config)
        base_dict = asdict(default_config)
        final_dict.pop("env", None)
        base_dict.pop("env", None)

        if final_dict == base_dict:
            # 这里改成 warning 体验更好，不会阻断程序启动
            logger.warning(
                f"{extend_path} has the same actual values as {default_path}. You might want to modify it."
            )

        return final_config

    def save(self) -> None:
        """
        保存配置功能。
        根据当前的 env 将配置保存到对应的文件中。
        """
        file_path = self._get_config_path(self.env)
        data_to_save = asdict(self)

        # 如果不是 default 环境，保存时剔除 env 字段，保持环境配置文件的纯净
        if self.env != "default":
            data_to_save.pop("env", None)

        try:
            with open(file_path, "wb") as f:
                tomli_w.dump(data_to_save, f)
            logger.info(f"Successfully saved current configuration to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            raise


config = Config.load_config()

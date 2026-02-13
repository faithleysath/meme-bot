import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DbConfig:
    host: str = "localhost"
    port: int = 5432
    password: str = ""


@dataclass(frozen=True)
class Config:
    api_key: str = ""
    debug: bool = False
    database: DbConfig = field(default_factory=DbConfig)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_settings() -> Config:
    # 定位根目录 (meme_bot/config.py -> meme_bot/ -> 根目录)
    root = Path(__file__).resolve().parent.parent

    # 1. 加载默认配置 (必须存在)
    default_path = root / "config.default.toml"
    with open(default_path, "rb") as f:
        data = tomllib.load(f)

    # 2. 合并本地开发配置 (如果存在)
    dev_path = root / "config.dev.toml"
    if dev_path.exists():
        with open(dev_path, "rb") as f:
            data = _deep_merge(data, tomllib.load(f))

    # 3. 环境变量覆盖 (例如 export MEME_API_KEY=xxx)
    # 这种写法可以让 Docker 部署极其方便
    data["api_key"] = os.getenv("MEME_API_KEY", data.get("api_key"))
    data["debug"] = os.getenv("MEME_DEBUG", "").lower() == "true" or data.get("debug", False)

    # 4. 组装对象 (利用 standard 模式的类型提示)
    db_data = data.get("database", {})
    return Config(
        api_key=data.get("api_key", ""),
        debug=data.get("debug", False),
        database=DbConfig(**db_data) if db_data else DbConfig(),
    )


# 全局单例：在 main.py 中使用 from meme_bot.config import settings
settings = load_settings()

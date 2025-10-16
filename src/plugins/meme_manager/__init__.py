from nonebot import get_plugin_config, on_message, logger, get_bots
from nonebot.rule import Rule
from nonebot.permission import Permission
from nonebot.adapters.onebot.v11 import MessageEvent
from nonebot.plugin import PluginMetadata
from datetime import datetime
from pydantic import BaseModel
from nonebot import require
import json

from pathlib import Path

require("nonebot_plugin_localstore")

import nonebot_plugin_localstore as store

from .config import Config

__plugin_meta__ = PluginMetadata(
    name="meme-manager",
    description="",
    usage="",
    config=Config,
)

config: Config = get_plugin_config(Config)

self_sent_time: float = 0
meme_reciev: tuple[int, bytes, str] | None = None # (message_id, image_bytes, image_hash)

with open(__file__, "r", encoding="utf-8") as f:
    __plugin_meta__.extra["source_code"] = f.read()

class Meme(BaseModel):
    hash: str
    description: str | None = None # 由VLM自动生成的描述
    short_term: str | None = None # 用户添加的简称，全局唯一
    tags: list[str] = [] # 用户添加的标签，用于辅助搜索
    usage_count: int = 0 # 使用次数
    last_used: float | None = None # 上次使用时间戳
    created_at: float = datetime.now().timestamp() # 创建时间戳
    updated_at: float = datetime.now().timestamp() # 更新时间戳

class MemeManagerStore():
    def __init__(self):
        self.db_file: Path = store.get_plugin_data_file("memes.json")
        # try to create the file if not exists
        if not self.db_file.exists():
            self.db_file.parent.mkdir(parents=True, exist_ok=True)
            self.db_file.write_text("{}", encoding="utf-8")
        self.memes: dict[str, Meme] = {}
        self.readonly = False
        self.load()
    def load(self):
        try:
            with self.db_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    self.memes[k] = Meme(hash=k, **v)
        except Exception as e:
            logger.error(f"Failed to load memes from {self.db_file}: {e}")
            self.readonly = True
    def save(self):
        if self.readonly:
            logger.warning("MemeManagerStore is readonly, cannot save.")
            return
        try:
            with self.db_file.open("w", encoding="utf-8") as f:
                json.dump({k: v.model_dump(exclude={"hash"}) for k, v in self.memes.items()}, f, ensure_ascii=False) # 不需要indent为了节省空间
        except Exception as e:
            logger.error(f"Failed to save memes to {self.db_file}: {e}")
            self.readonly = True
    def add_meme(self, meme: Meme):
        if meme.hash in self.memes:
            logger.warning(f"Meme with hash {meme.hash} already exists, not adding.")
            return
        self.memes[meme.hash] = meme
        self.save()
    def get_meme(self, hash: str) -> Meme | None:
        return self.memes.get(hash)
    def update_meme(self, hash: str, **kwargs):
        meme = self.get_meme(hash)
        if not meme:
            logger.warning(f"Meme with hash {hash} not found, cannot update.")
            return
        for k, v in kwargs.items():
            if hasattr(meme, k):
                setattr(meme, k, v)
        meme.updated_at = datetime.now().timestamp()
        self.save()
    def delete_meme(self, hash: str):
        if hash in self.memes:
            del self.memes[hash]
            self.save()
        else:
            logger.warning(f"Meme with hash {hash} not found, cannot delete.")
    def get_meme_by_short_term(self, short_term: str) -> Meme | None:
        for meme in self.memes.values():
            if meme.short_term == short_term:
                return meme
        logger.warning(f"Meme with short_term {short_term} not found.")
        return None
            
def rule_wrapper(func):
    return Rule(func)

def permission_wrapper(func):
    return Permission(func)

@permission_wrapper
async def is_target_user(event: MessageEvent) -> bool:
    """检查是否为目标用户并节流"""
    if config.meme_listen_user_id is None:
        bots = get_bots()
        if not bots:
            logger.warning("No bots are currently connected.")
            return False
        bot_id = int(list(bots.keys())[0])
        config.meme_listen_user_id = bot_id
    return (event.user_id == config.meme_listen_user_id) and (datetime.now().timestamp() - self_sent_time > config.meme_self_sent_timeout)

@rule_wrapper
async def is_reply_message(event: MessageEvent) -> bool:
    """检查是否为回复消息"""
    return event.reply is not None

# 监听来自目标用户的私聊消息
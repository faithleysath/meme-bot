from nonebot import get_plugin_config, on_message, logger, get_bots
from nonebot.rule import Rule
from nonebot.permission import Permission
from nonebot.adapters.onebot.v11 import MessageEvent, PrivateMessageEvent, MessageSegment
from nonebot.plugin import PluginMetadata
from datetime import datetime
from pydantic import BaseModel
from nonebot import require
import json
import httpx

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
meme_reciev: tuple[int, bytes, str, str, float] | None = None # (message_id, image_bytes, image_hash, ext, timestamp)

with open(__file__, "r", encoding="utf-8") as f:
    __plugin_meta__.extra["source_code"] = f.read()

class Meme(BaseModel):
    hash: str # 这里用md5哈希作为唯一标识
    pHash: str | None = None # 感知哈希，用于相似图片搜索
    ext: str # 文件扩展名，如 jpg, png, gif
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

@rule_wrapper
async def to_me(event: PrivateMessageEvent) -> bool:
    """检查是否为私聊消息"""
    bots = get_bots()
    if not bots:
        logger.warning("No bots are currently connected.")
        return False
    bot_id = int(list(bots.keys())[0])
    return event.target_id == bot_id

@rule_wrapper
async def not_to_me(event: MessageEvent) -> bool:
    """检查是否为非私聊消息"""
    bots = get_bots()
    if not bots:
        logger.warning("No bots are currently connected.")
        return False
    bot_id = int(list(bots.keys())[0])
    return event.target_id != bot_id

# 监听来自目标用户的私聊消息
target_private_matcher = on_message(permission=is_target_user, rule=to_me, priority=5)

@target_private_matcher.handle()
async def handle_target_private(event: PrivateMessageEvent):
    global self_sent_time, meme_reciev
    # 检查图片数量
    image_num = len(event.message['image'])
    if image_num > 1:
        await target_private_matcher.finish("一次只能发送一张图片哦~")
    # 分为无图片和有图片两种情况
    if image_num == 1:
        # 处理仅包含图片的消息
        if event.message.only("image"):
            segment: MessageSegment = event.message[0]
            hash, ext = segment.data.get("file", "").split(".")
            # 下载图片
            async with httpx.AsyncClient() as client:
                resp = await client.get(segment.data.get("url", ""))
                if resp.status_code != 200:
                    await target_private_matcher.finish("图片下载失败，请稍后再试~")
                image_bytes = resp.content
            meme_reciev = (event.message_id, image_bytes, hash, ext, datetime.now().timestamp())
            await target_private_matcher.finish("图片已接收，请发送描述或标签~")
        else:
            await target_private_matcher.finish("请只发送一张图片哦~")
    else:
        # 处理无图片的消息
        if meme_reciev is None:
            await target_private_matcher.finish("请先发送一张图片哦~")
        description = event.message.extract_plain_text().strip()
        if not description:
            await target_private_matcher.finish("描述不能为空，请重新发送~")
        message_id, image_bytes, image_hash, ext, timestamp = meme_reciev
        meme = Meme(hash=image_hash, ext=ext, description=description)
        store = MemeManagerStore()
        if store.get_meme(image_hash):
            await target_private_matcher.finish("这张图片已经存在于数据库中啦~")
        store.add_meme(meme)
        meme_reciev = None
        self_sent_time = datetime.now().timestamp()
        await target_private_matcher.finish(f"图片已保存，hash: {image_hash}，描述: {description}")
    self_sent_time = datetime.now().timestamp()
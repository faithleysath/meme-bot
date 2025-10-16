from nonebot import get_plugin_config, on_message, logger, get_bots
from nonebot.rule import Rule
from nonebot.adapters.onebot.v11 import MessageEvent
from nonebot.plugin import PluginMetadata
from datetime import datetime
from pydantic import BaseModel

from .config import Config

__plugin_meta__ = PluginMetadata(
    name="meme-manager",
    description="",
    usage="",
    config=Config,
)

config: Config = get_plugin_config(Config)

self_sent_time: float = 0

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

async def is_reply_message(event: MessageEvent) -> bool:
    """检查是否为回复消息"""
    return event.reply is not None

reply_matcher = on_message(permission=is_target_user, rule=is_reply_message, priority=5, block=False)

@reply_matcher.handle()
async def handle_reply(event: MessageEvent):
    """处理回复消息"""
    assert event.reply is not None  # 确保 event.reply 不为 None
    logger.debug(f"Received reply message: {event.message}")
    logger.debug(f"Reply to message: {event.reply.message}")
    for segment in event.reply.message:
        logger.debug(f"Segment type: {segment.type}, data: {segment.data}")
from nonebot import get_plugin_config, on_message, logger, get_bots
from nonebot.rule import Rule
from nonebot.permission import Permission
from nonebot.adapters.onebot.v11 import MessageEvent, PrivateMessageEvent, MessageSegment
from nonebot.plugin import PluginMetadata
from nonebot.matcher import Matcher
from datetime import datetime
from pydantic import BaseModel
from nonebot import require
import json
import httpx
import asyncio
from openai import AsyncOpenAI

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

llm_client = AsyncOpenAI(base_url=config.meme_llm_base_url, api_key=config.meme_llm_api_key)

self_sent_time: float = 0
meme_reciev: tuple[int, bytes, str, str, float] | None = None # (message_id, image_bytes, image_hash, ext, timestamp)
meme_manager_lock = asyncio.Lock()

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
        print(self.db_file)
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

meme_store = MemeManagerStore()
            
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
    return (event.user_id == config.meme_listen_user_id) and (datetime.now().timestamp() - self_sent_time > config.meme_self_sent_timeout) # 防止bot响应的消息被当成请求，默认为2秒

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

async def finish_and_throttle(matcher: Matcher, message: str):
    global self_sent_time
    self_sent_time = datetime.now().timestamp()
    await matcher.finish(message)

# 管理表情包
@target_private_matcher.handle()
async def handle_target_private(matcher: Matcher, event: PrivateMessageEvent):
    global meme_reciev
    MERGE_TIME_WINDOW = 10 # 合并时间窗口，单位为秒
    async with meme_manager_lock:
        image_segment: MessageSegment | None = None
        # 检查图片数量
        image_num = len(event.message['image'])
        if image_num > 1:
            await finish_and_throttle(matcher, "一次只能发送一张图片哦~")
            return
        
        # 缓存图片
        if image_num == 1:
            image_segment = event.message["image", 0]
        elif event.reply is not None and len(event.reply.message['image']) == 1:
            image_segment = event.reply.message["image", 0]

        if image_segment:
            hash, ext = image_segment.data.get("file", "").split(".")
            # 下载图片
            async with httpx.AsyncClient() as client:
                resp = await client.get(image_segment.data.get("url", ""))
                if resp.status_code != 200:
                    await finish_and_throttle(matcher, "图片下载失败，请稍后再试~")
                    return
                image_bytes = resp.content
            meme_reciev = (event.message_id, image_bytes, hash, ext, datetime.now().timestamp())

        # 获取并合并可能存在的引用消息的纯文本内容
        current_msg_text = event.message.extract_plain_text().strip()
        reply_msg_text = ""
        if event.reply is not None:
            reply_msg_text = event.reply.message.extract_plain_text().strip()
        
        # 检查缓存图片是否过期
        image_enable = meme_reciev is not None and (datetime.now().timestamp() - meme_reciev[4] <= MERGE_TIME_WINDOW)

        # 构建提示词
        prompt = f"你是一个qq机器人插件的运行组成部分，下面是该插件的源代码。你目前被rfegtds处的代码调用了，请你理解插件的运行原理，返回符合要求的json字符串，使得插件能正确运行。\n\n Source Code:\n\n{__plugin_meta__.extra['source_code']}\n\n一些运行时变量：\n\n"

        # 提示词的一部分：调用代码地址rfegtds
        prompts = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                ]
            }
        ]

        if image_enable:
            assert meme_reciev is not None
            prompts[0]['content'].append({"type": "input_image", "image_url": b2s64(meme_reciev[1])})

        resp = await llm_client.responses.create(
            model=config.meme_llm_model,
            input=prompts
        )
from nonebot import get_plugin_config, on_message, logger, get_bots
from nonebot.rule import Rule
from nonebot.permission import Permission
from nonebot.adapters.onebot.v11 import MessageEvent, PrivateMessageEvent, MessageSegment
from nonebot.plugin import PluginMetadata
from nonebot.matcher import Matcher
from datetime import datetime
from pydantic import BaseModel, Field
from typing import cast
from nonebot import require
import json
import httpx
import asyncio
import base64
from openai import AsyncOpenAI
from enum import Enum
from time import time
import re
import uuid

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

config: Config = cast(Config, get_plugin_config(Config))

llm_client = AsyncOpenAI(base_url=config.meme_llm_base_url, api_key=config.meme_llm_api_key)

self_sent_time: float = 0
meme_reciev: tuple[int, bytes, str, str, float] | None = None # (message_id, image_bytes, image_hash, ext, timestamp)
meme_manager_lock = asyncio.Lock()
pending_confirmation: dict | None = None # 待确认的操作

with open(__file__, "r", encoding="utf-8") as f:
    __plugin_meta__.extra["source_code"] = f.read()

def b2s64(image_bytes: bytes, ext: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{ext};base64,{encoded}"

def extract_json(text: str) -> dict:
    """从可能被三引号等包裹的文本中提取出JSON对象"""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("在文本中未找到JSON对象")
    return json.loads(match.group(0))

class Meme(BaseModel):
    hash: str # 这里用md5哈希作为唯一标识
    pHash: str | None = None # 感知哈希，用于相似图片搜索
    ext: str # 文件扩展名，如 jpg, png, gif
    description: str | None = None # 由VLM自动生成的描述
    short_term: str | None = None # 用户添加的简称，全局唯一
    tags: list[str] = Field(default_factory=list) # 用户添加的标签，用于辅助搜索
    prompt: str | None = None # 告诉llm何时、如何使用该表情包
    usage_count: int = 0 # 使用次数
    last_used: float | None = None # 上次使用时间戳
    created_at: float = Field(default_factory=time) # 创建时间戳
    updated_at: float = Field(default_factory=time) # 更新时间戳

# --- LLM交互协议模型 ---

class Action(str, Enum):
    ADD_MEME = "ADD_MEME"
    SEARCH_MEME = "SEARCH_MEME"
    UPDATE_MEME = "UPDATE_MEME"
    DELETE_MEME = "DELETE_MEME"
    CONFIRM_ACTION = "CONFIRM_ACTION"
    CANCEL_ACTION = "CANCEL_ACTION"
    NO_ACTION = "NO_ACTION"

class AddPayload(BaseModel):
    short_term: str | None = None
    tags: list[str] = Field(default_factory=list)
    prompt: str | None = None

class SearchPayload(BaseModel):
    hash: str

class UpdatePayload(BaseModel):
    hash: str
    update_data: dict
    uuid: str

class DeletePayload(BaseModel):
    hash: str
    uuid: str

class ConfirmPayload(BaseModel):
    uuid: str

class CancelPayload(BaseModel):
    uuid: str

class LLMResult(BaseModel):
    action: Action
    payload: dict = Field(default_factory=dict)
    response: str | None = None

class MemeManagerStore():
    def __init__(self):
        # 元数据文件
        self.db_file: Path = store.get_plugin_data_file("memes.json")
        # 图片存储目录
        self.image_storage_path: Path = store.get_plugin_data_dir() / "images"
        self.image_storage_path.mkdir(parents=True, exist_ok=True)

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

    def save_meme_image(self, hash: str, ext: str, image_bytes: bytes):
        """保存表情图片文件"""
        image_file = self.image_storage_path / f"{hash}.{ext}"
        image_file.write_bytes(image_bytes)

    def get_meme_image_path(self, hash: str, ext: str) -> Path | None:
        """获取表情图片文件的路径"""
        image_file = self.image_storage_path / f"{hash}.{ext}"
        return image_file if image_file.exists() else None

    def delete_meme_image(self, hash: str, ext: str):
        """删除表情图片文件"""
        image_path = self.get_meme_image_path(hash, ext)
        if image_path:
            image_path.unlink()

    def add_meme(self, meme: Meme, image_bytes: bytes):
        if meme.hash in self.memes:
            logger.warning(f"Meme with hash {meme.hash} already exists, not adding.")
            return
        self.save_meme_image(meme.hash, meme.ext, image_bytes)
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
        meme = self.get_meme(hash)
        if meme:
            self.delete_meme_image(meme.hash, meme.ext)
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

    def get_memes_as_json_string(self) -> str:
        """将表情包数据库序列化为JSON字符串"""
        return json.dumps({k: v.model_dump(exclude={"hash"}) for k, v in self.memes.items()}, ensure_ascii=False)

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

async def call_llm(prompts: list) -> str:
    """调用LLM并返回其原始输出"""
    try:
        resp = await llm_client.responses.create(
            model=config.meme_llm_model,
            input=prompts,  # type: ignore
        )
        # 假设响应对象有 output_text 属性，根据用户反馈进行调整
        if hasattr(resp, "output_text"):
            return resp.output_text
        raise ValueError("无法从LLM响应中提取文本内容，响应对象缺少'output_text'属性")
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise

async def process_llm_response(matcher: Matcher, llm_output: str, current_uuid: str):
    """解析LLM的响应并执行相应的操作"""
    global meme_reciev, pending_confirmation
    try:
        json_data = extract_json(llm_output)
        llm_result = LLMResult.model_validate(json_data)
        action = llm_result.action
        response_msg = llm_result.response

        if action == Action.ADD_MEME:
            if meme_reciev is None:
                await finish_and_throttle(matcher, "请先发送一张图片才能添加哦~")
                return
            
            payload = AddPayload.model_validate(llm_result.payload)
            _, image_bytes, hash, ext, _ = meme_reciev
            
            new_meme = Meme(
                hash=hash,
                ext=ext,
                short_term=payload.short_term,
                tags=payload.tags,
                prompt=payload.prompt
            )
            meme_store.add_meme(new_meme, image_bytes)
            meme_reciev = None # 清空缓存
            await finish_and_throttle(matcher, response_msg or "表情已添加。")

        elif action == Action.SEARCH_MEME:
            payload = SearchPayload.model_validate(llm_result.payload)
            meme = meme_store.get_meme(payload.hash)
            
            if meme:
                image_path = meme_store.get_meme_image_path(meme.hash, meme.ext)
                if image_path:
                    await matcher.send(MessageSegment.image(image_path.as_uri()))
                    meme_store.update_meme(meme.hash, usage_count=meme.usage_count + 1, last_used=time())
                    if response_msg:
                        await finish_and_throttle(matcher, response_msg)
                else:
                    await finish_and_throttle(matcher, "找到了表情记录，但图片文件丢失了！")
            else:
                await finish_and_throttle(matcher, response_msg or "数据库记录和你说的表情对不上号呢。")

        elif action == Action.UPDATE_MEME:
            payload = UpdatePayload.model_validate(llm_result.payload)
            if payload.uuid != current_uuid:
                await finish_and_throttle(matcher, "操作已过期，请重新发起。")
                return
            pending_confirmation = {
                "uuid": payload.uuid,
                "action": action.value,
                "payload": payload.model_dump()
            }
            await finish_and_throttle(matcher, response_msg or f"请确认是否要执行更新操作？\nUUID: {payload.uuid}")

        elif action == Action.DELETE_MEME:
            payload = DeletePayload.model_validate(llm_result.payload)
            if payload.uuid != current_uuid:
                await finish_and_throttle(matcher, "操作已过期，请重新发起。")
                return
            pending_confirmation = {
                "uuid": payload.uuid,
                "action": action.value,
                "payload": payload.model_dump()
            }
            await finish_and_throttle(matcher, response_msg or f"请确认是否要执行删除操作？\nUUID: {payload.uuid}")

        elif action == Action.CONFIRM_ACTION:
            payload = ConfirmPayload.model_validate(llm_result.payload)
            if not pending_confirmation or pending_confirmation["uuid"] != payload.uuid:
                await finish_and_throttle(matcher, "没有找到待确认的操作或UUID不匹配。")
                return
            
            # 执行暂存的操作
            pending_action_str = pending_confirmation["action"]
            pending_action = Action(pending_action_str)
            pending_payload = pending_confirmation["payload"]
            
            if pending_action == Action.UPDATE_MEME:
                update_payload = UpdatePayload.model_validate(pending_payload)
                meme_store.update_meme(update_payload.hash, **update_payload.update_data)
                await finish_and_throttle(matcher, response_msg or "表情已更新。")
            
            elif pending_action == Action.DELETE_MEME:
                delete_payload = DeletePayload.model_validate(pending_payload)
                meme_store.delete_meme(delete_payload.hash)
                await finish_and_throttle(matcher, response_msg or "表情已删除。")
            
            pending_confirmation = None # 清空状态

        elif action == Action.CANCEL_ACTION:
            payload = CancelPayload.model_validate(llm_result.payload)
            if pending_confirmation and pending_confirmation["uuid"] == payload.uuid:
                pending_confirmation = None
                await finish_and_throttle(matcher, response_msg or "操作已取消。")
            else:
                await finish_and_throttle(matcher, "没有找到需要取消的操作或UUID不匹配。")

        elif action == Action.NO_ACTION:
            if response_msg:
                await finish_and_throttle(matcher, response_msg)

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse LLM response: {e}\nRaw output: {llm_output}")
        await matcher.send("LLM响应解析失败，请检查后台日志。")
        await finish_and_throttle(matcher, f"原始输出:\n{llm_output}")
    except Exception as e:
        logger.error(f"An error occurred in handler: {e}")
        await finish_and_throttle(matcher, f"处理时发生未知错误: {e}")

# 管理表情包
@target_private_matcher.handle()
async def handle_target_private(matcher: Matcher, event: PrivateMessageEvent):
    """
    通过LLM解析用户消息来管理表情包

    LLM被期望分析用户的文本、附带的图片（如有），以及上下文，
    然后返回一个JSON对象来指令本插件如何操作表情包数据库。

    工作流程:
    1.  对于`UPDATE_MEME`和`DELETE_MEME`操作，LLM必须返回当前交互的`uuid`。插件会暂存此操作并向用户请求确认。
    2.  用户通过**回复**该确认消息来发送“确认”或“取消”指令。
    3.  LLM需要从用户回复的原文（即插件发送的确认消息）中提取出`uuid`，然后生成`CONFIRM_ACTION`或`CANCEL_ACTION`并附带上这个提取出的`uuid`。
    4.  插件校验UUID后，执行或取消操作。

    LLM返回的JSON格式规范:
    {
      "action": "操作名称",
      "payload": {
        // 操作所需的数据
      },
      "response": "一句对用户的友好回复，在闲聊时也应提供此字段"
    }

    可用的 "操作名称" 及其 "payload" 结构:
    - "ADD_MEME": 保存一个新表情包。
      - payload: {"short_term": "可选的简称", "tags": ["标签", "列表"], "prompt": "可选的触发词"}
    - "SEARCH_MEME": 查找并发送一个表情包。
      - payload: {"hash": "要发送的表情包哈希"}
    - "UPDATE_MEME": **提议**修改一个已有的表情包，等待用户确认。
      - payload: {"hash": "表情包哈希", "update_data": {...}, "uuid": "当前交互UUID"}
    - "DELETE_MEME": **提议**删除一个表情包，等待用户确认。
      - payload: {"hash": "表情包哈希", "uuid": "当前交互UUID"}
    - "CONFIRM_ACTION": 确认执行暂存的操作。
      - payload: {"uuid": "待确认操作的UUID"}
    - "CANCEL_ACTION": 取消暂存的操作。
      - payload: {"uuid": "待确认操作的UUID"}
    - "NO_ACTION": 用户只是在闲聊或指令无法识别。
      - payload: {}
    """
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
        
        current_uuid = str(uuid.uuid4())

        # 构建提示词
        prompt = f"""你是一个qq机器人插件的运行组成部分。请你理解插件的运行原理，分析用户的意图，并严格按照 handle_target_private 函数文档字符串中定义的格式返回一个JSON字符串，使得插件能正确运行。

# 插件源代码:
{__plugin_meta__.extra['source_code']}

# 当前表情包数据库 (memes.json):
{meme_store.get_memes_as_json_string()}

# 运行时信息:
- 当前交互UUID: {current_uuid}
- 当前消息ID: {event.message_id}
- 回复的消息ID(如有): {event.reply.message_id if event.reply else "无"}
- 用户的当前消息: "{current_msg_text}"
- 用户回复的消息(如有): "{reply_msg_text}"
- 当前缓存的图片是否存在: {"是" if image_enable else "否"}
- 缓存图片的消息ID(如有): {meme_reciev[0] if image_enable and meme_reciev else "无"}
- 缓存图片的哈希(如有): {meme_reciev[2] if image_enable and meme_reciev else "无"}
- 缓存图片的扩展名(如有): {meme_reciev[3] if image_enable and meme_reciev else "无"}
"""

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
            image_bytes = meme_reciev[1]
            ext = meme_reciev[3]
            prompts[0]['content'].append({"type": "input_image", "image_url": b2s64(image_bytes, ext)})

        try:
            llm_output = await call_llm(prompts)
            await process_llm_response(matcher, llm_output, current_uuid)
        except Exception as e:
            await finish_and_throttle(matcher, f"LLM调用失败: {e}")
            return

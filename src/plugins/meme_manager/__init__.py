from nonebot.plugin import PluginMetadata
from .config import Config

__plugin_meta__ = PluginMetadata(
    name="meme-manager",
    description="",
    usage="",
    config=Config,
)

with open(__file__, "r", encoding="utf-8") as f:
    __plugin_meta__.extra["source_code"] = f.read()

from datetime import datetime
from nonebot.adapters.onebot.v11 import MessageEvent, Message, MessageSegment
from nonebot import on_message, logger, get_plugin_config
from nonebot.exception import FinishedException
from curl_cffi import AsyncSession, CurlError
import asyncio

from openai import AsyncClient, NotFoundError, APIStatusError, RateLimitError, APITimeoutError, BadRequestError, APIConnectionError, AuthenticationError, InternalServerError, PermissionDeniedError

conf = get_plugin_config(Config)

llm_client = AsyncClient(base_url=conf.meme_llm_base_url, api_key=conf.meme_llm_api_key)

import nonebot.adapters.onebot.v11.message as ob11_message  # noqa: F401  # for docstring
with open(ob11_message.__file__, "r", encoding="utf-8") as f:
    message_doc = f.read() # 准备给llm看的代码

from collections import OrderedDict, deque
from typing import Generic, TypeVar

KT = TypeVar("KT")
VT = TypeVar("VT")

class LRUCache(Generic[KT, VT]):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache: OrderedDict[KT, VT] = OrderedDict()

    def get(self, key: KT, default: VT | None = None) -> VT | None:
        """获取键值，若不存在返回 None，并将该键标记为最近使用"""
        if key not in self._cache:
            return default
        # 将键移到末尾（最近使用）
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: KT, value: VT):
        """插入键值对，若容量超限则删除最久未使用的键"""
        if key in self._cache:
            # 若键已存在，先移到末尾（更新为最近使用）
            self._cache.move_to_end(key)
        # 插入/更新值
        self._cache[key] = value
        # 若容量超限，删除最前面的键（最久未使用）
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    # 重载一些字典方法
    def __contains__(self, key: KT) -> bool:
        return key in self._cache
    
    def __getitem__(self, key: KT) -> VT:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result
    
    def __setitem__(self, key: KT, value: VT):
        self.put(key, value)

image_cache: LRUCache[str, bytes] = LRUCache(capacity=conf.meme_images_cache_capacity)  # 图片缓存，容量为128

class SessionHistory:
    """存储每个会话的消息历史记录"""
    def __init__(self, max_history: int = 20):
        super().__init__()
        self.max_history = max_history
        self._history: dict[int, deque[MessageEvent]] = {}

    def add_event(self, session_id: int, event: MessageEvent):
        """将事件添加到指定会话的历史记录中"""
        if session_id not in self._history:
            self._history[session_id] = deque(maxlen=self.max_history)
        self._history[session_id].append(event)

    def get_history(self, session_id: int) -> list[MessageEvent]:
        """获取指定会话的历史记录"""
        return list(self._history.get(session_id, deque()))
    
    def clear_history(self, session_id: int):
        """清除指定会话的历史记录"""
        if session_id in self._history:
            del self._history[session_id]

    # 重载一些字典方法
    def __contains__(self, key: int) -> bool:
        return key in self._history and len(self._history[key]) > 0
    def __getitem__(self, key: int) -> list[MessageEvent]:
        return self.get_history(key)
    def __setitem__(self, key: int, value: list[MessageEvent]):
        self._history[key] = deque(value, maxlen=self.max_history)
    # 重载一些运算符
    def __len__(self) -> int:
        return len(self._history)

session_history = SessionHistory(max_history=conf.meme_max_history_messages)  # 会话历史记录，最多存储20条消息

async def fetch_image(filename: str, url: str | None = None) -> bytes | None:
    """
    Fetch image from cache or URL.
    从缓存或 URL 获取图片
    """
    # 先从缓存获取
    cached_image = image_cache.get(filename)
    if cached_image:
        return cached_image
    # 若缓存中不存在且提供了 URL，则从 URL 获取
    if url:
        try:
            async with AsyncSession() as session:
                response = await session.get(url, timeout=10)
                if response.status_code == 200:
                    image_data = response.content
                    # 存入缓存
                    image_cache.put(filename, image_data)
                    return image_data
        except (CurlError, asyncio.TimeoutError) as e:
            logger.error(f"Error fetching image from URL {url}: {e}")
    return None

async def serialize_segments_with_images(segments: MessageSegment | list[MessageSegment] | Message) -> tuple[list[dict], set[str]]:
    """
    Convert MessageSegment or list of MessageSegment to a dictionary.
    将 MessageSegment 或 MessageSegment 列表转换为字典，仅转换特定类型
    """
    if isinstance(segments, MessageSegment):
        segments = [segments]
    result = []
    image_names: set[str] = set()
    for segment in segments:
        match segment.type:
            case "text":
                result.append({"type": "text", "data": segment.data})
            case "image":
                if await fetch_image(segment.data.get("file", ""), segment.data.get("url", None)):
                    result.append({"type": "image", "data": {"file": segment.data.get("file", "")}})
                    image_names.add(segment.data.get("file", ""))
                else:
                    logger.warning(f"Failed to fetch image for segment: {segment.data}")
            case _:
                # 其他类型不处理
                continue
    return result, image_names

async def serialize_message_event_with_images(event: MessageEvent) -> tuple[dict, set[str]]:
    """
    Convert MessageEvent to a dictionary.
    将 MessageEvent 转换为字典
    """

    message, image_names = await serialize_segments_with_images(event.message)

    result = {
        "event_type": event.post_type, # 事件类型
        "time": datetime.fromtimestamp(event.time).isoformat(), # 事件时间
        "message_type": event.message_type, # 消息类型
        "message_id": event.message_id, # 消息 ID
        "sender_id": event.user_id, # 发送者 ID
        "sender_nickname": event.sender.nickname or "", # 发送者昵称
        "target_id": event.target_id, # 目标 ID
        "self_id": event.self_id, # 机器人 ID
        "message": message, # 消息内容
    }

    if event.reply:
        reply_message, reply_image_names = await serialize_segments_with_images(event.reply.message)
        result["reply_to"] = { # 被回复的消息信息
            "time": datetime.fromtimestamp(event.reply.time).isoformat(),
            "message_type": event.reply.message_type,
            "message_id": event.reply.message_id,
            "sender_id": event.reply.sender.user_id,
            "sender_nickname": event.reply.sender.nickname or "",
            "message": reply_message,
        }
        image_names.update(reply_image_names)

    return result, image_names

async def call_llm_with_retry(prompts: list, max_retries: int = 3, delay: float = 1.0):
    """调用 LLM 接口，失败时重试"""
    for attempt in range(max_retries):
        try:
            response = await llm_client.chat.completions.create(
                model=conf.meme_llm_model,
                messages=prompts,
                temperature=conf.meme_llm_temperature,
                top_p=conf.meme_llm_top_p,
                timeout=conf.meme_llm_timeout,
            )
            return response.choices[0].message.content
        except FinishedException:
            raise FinishedException()
        except RateLimitError:
            logger.warning(f"LLM rate limit exceeded, attempt {attempt + 1}/{max_retries}")
        except (APIConnectionError, APITimeoutError):
            logger.warning(f"LLM connection or timeout error, attempt {attempt + 1}/{max_retries}")
        except (NotFoundError, BadRequestError, AuthenticationError, PermissionDeniedError, InternalServerError, APIStatusError) as e:
            logger.error(f"LLM API error: {e}")
            return e
        await asyncio.sleep(delay)
    return "LLM request failed after retries."

import os
from pebble import ProcessPool
from concurrent.futures import TimeoutError

async def worker_with_limits(python_code: str, globals: dict, locals: dict, timeout: int = 5, memory_limit_mb: int = 100):
    """在子进程中执行受限的 Python 代码，限制时间和内存"""
    def _(python_code: str, globals: dict, locals: dict, memory_limit_mb: int):
        from RestrictedPython import compile_restricted
        if os.name == 'posix':
            import resource
            # 设置内存限制
            memory_limit_bytes = memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        try:
            byte_code = compile_restricted(python_code, '<string>', 'exec')
            exec(byte_code, globals, locals)
            return locals.get("result", None)
        except MemoryError:
            raise MemoryError("Code execution exceeded memory limit.")
    with ProcessPool(max_workers=1) as pool:
        try:
            return await pool.schedule(_,kwargs={"python_code": python_code, "globals": globals, "locals": locals, "memory_limit_mb": memory_limit_mb}, timeout=timeout)
        except TimeoutError:
            raise TimeoutError("Code execution exceeded time limit.")

@(on_message(priority=0, block=False).handle())
async def record_message(event: MessageEvent):
    """记录每条消息到会话历史中"""
    session_history.add_event(event.target_id, event)
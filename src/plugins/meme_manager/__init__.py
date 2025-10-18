from datetime import datetime
from nonebot.adapters.onebot.v11 import MessageEvent, Message, MessageSegment
from nonebot import on_message, logger
from curl_cffi import AsyncSession

from collections import OrderedDict

class LRUCache(OrderedDict):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def get(self, key, default=None):
        """获取键值，若不存在返回 None，并将该键标记为最近使用"""
        if key not in self:
            return default
        # 将键移到末尾（最近使用）
        self.move_to_end(key)
        return self[key]

    def put(self, key, value):
        """插入键值对，若容量超限则删除最久未使用的键"""
        if key in self:
            # 若键已存在，先移到末尾（更新为最近使用）
            self.move_to_end(key)
        # 插入/更新值
        self[key] = value
        # 若容量超限，删除最前面的键（最久未使用）
        if len(self) > self.capacity:
            self.popitem(last=False)

image_cache: LRUCache[str, bytes] = LRUCache(capacity=128)  # 图片缓存，容量为128

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
        async with AsyncSession() as session:
            response = await session.get(url)
            if response.status_code == 200:
                image_data = response.content
                # 存入缓存
                image_cache.put(filename, image_data)
                return image_data
    return None

async def segments_to_dicts(segments: MessageSegment | list[MessageSegment] | Message) -> list[dict]:
    """
    Convert MessageSegment or list of MessageSegment to a dictionary.
    将 MessageSegment 或 MessageSegment 列表转换为字典，仅转换特定类型
    """
    if isinstance(segments, MessageSegment):
        segments = [segments]
    result = []
    for segment in segments:
        match segment.type:
            case "text":
                result.append({"type": "text", "data": segment.data})
            case "image":
                result.append({"type": "image", "data": {"file": segment.data.get("file", "")}})
            case _:
                # 其他类型不处理
                continue
    return result

async def message_event_to_dict(event: MessageEvent) -> dict:
    """
    Convert MessageEvent to a dictionary.
    将 MessageEvent 转换为字典
    """

    result = {
        "event_type": event.post_type, # 事件类型
        "time": datetime.fromtimestamp(event.time).isoformat(), # 事件时间
        "message_type": event.message_type, # 消息类型
        "message_id": event.message_id, # 消息 ID
        "sender_id": event.user_id, # 发送者 ID
        "sender_nickname": event.sender.nickname or "", # 发送者昵称
        "target_id": event.target_id, # 目标 ID
        "self_id": event.self_id, # 机器人 ID
        "message": segments_to_dicts(event.message), # 消息内容
    }

    if event.reply:
        result["reply_to"] = { # 被回复的消息信息
            "time": datetime.fromtimestamp(event.reply.time).isoformat(),
            "message_type": event.reply.message_type,
            "message_id": event.reply.message_id,
            "sender_id": event.reply.sender.user_id,
            "sender_nickname": event.reply.sender.nickname or "",
            "message": segments_to_dicts(event.reply.message),
        }

    return result

meme_manager_event = on_message(priority=5)

@meme_manager_event.handle()
async def handle_meme_manager_event(event: MessageEvent):
    """
    Handle meme manager related events.
    处理 meme 管理器相关事件
    """
    event_dict = message_event_to_dict(event)
    logger.debug(f"Meme Manager Event: {event_dict}")
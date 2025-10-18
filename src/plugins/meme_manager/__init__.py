from datetime import datetime
from nonebot.adapters.onebot.v11 import MessageEvent, Message, MessageSegment
from nonebot import on_message, logger

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
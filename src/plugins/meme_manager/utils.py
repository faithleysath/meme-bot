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
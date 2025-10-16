from pydantic import BaseModel


class Config(BaseModel):
    """Plugin Config Here"""
    meme_listen_user_id: int | None = None # 插件监听的用户ID（默认为 None，表示监听第一个连接的机器人，也就是监听自身）
    meme_self_sent_timeout: int = 2 # 插件忽略自身消息的时间窗口，单位为秒
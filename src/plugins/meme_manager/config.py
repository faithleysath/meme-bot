from pydantic import BaseModel


class Config(BaseModel):
    """Plugin Config Here"""
    meme_listen_user_id: int | None = None # 插件监听的用户ID（默认为 None，表示监听第一个连接的机器人，也就是监听自身）
    meme_self_sent_timeout: int = 2 # 插件忽略自身消息的时间窗口，单位为秒
    meme_llm_base_url: str = "https://api.openai.com/v1" # LLM模型的基础URL
    meme_llm_api_key: str = "" # LLM模型的API Key
    meme_llm_model: str = "google/gemini-2.5-pro" # LLM模型名称
    meme_llm_temperature: float = 0.7 # LLM模型的温度参数
    meme_llm_top_p: float = 1.0 # LLM模型的Top-p参数
    meme_llm_timeout: int = 15 # LLM模型请求的超时时间，单位为秒
    meme_max_history_messages: int = 20 # 用于生成上下文的最大历史消息数量
    meme_images_cache_capacity: int = 128 # 图片缓存的最大容量
    meme_sandbox_code_timeout: int = 5 # 沙箱代码执行的超时时间，单位为秒
    meme_sandbox_code_memory_limit_mb: int = 500 # 沙箱代码执行的内存限制，单位为MB
from nonebot import get_plugin_config, on_message, logger
from nonebot.adapters.onebot.v11 import MessageEvent
from nonebot.plugin import PluginMetadata

from .config import Config

__plugin_meta__ = PluginMetadata(
    name="meme-manager",
    description="",
    usage="",
    config=Config,
)

config = get_plugin_config(Config)

meme_manager = on_message(priority=5, block=False)

@meme_manager.handle()
async def handle_meme_manager(event: MessageEvent):
    logger.debug(f"Received message: {event.message}")
    # Add your meme management logic here
    pass
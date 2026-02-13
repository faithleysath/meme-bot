import asyncio
import logging

from napcat import NapCatClient

from meme_bot.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # 输出到文件
        logging.StreamHandler(),  # 输出到控制台
    ],
)


async def main():
    config = get_config()
    async for event in NapCatClient(config.ws_url, config.token):
        logging.info(f"Received event: {event}")
        # 在这里处理事件，例如回复消息、执行命令等


asyncio.run(main())

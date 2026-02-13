import asyncio
from napcat import NapCatClient
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"), # 输出到文件
        logging.StreamHandler()         # 输出到控制台
    ]
)


async def main():
    async for event in NapCatClient()


asyncio.run(main())

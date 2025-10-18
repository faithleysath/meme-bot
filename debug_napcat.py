import asyncio
import websockets

# 定义服务器的主机和端口
HOST = '198.18.0.1'
PORT = 8080

async def handler(websocket):
    """
    处理 WebSocket 连接的函数。
    会持续监听来自客户端的消息并打印。
    """
    print(f"客户端 {websocket.remote_address} 已连接。")
    try:
        # async for 循环会一直等待并接收来自客户端的消息
        async for message in websocket:
            print(f"收到来自 {websocket.remote_address} 的消息: {message}")
            # 如果需要，你可以在这里向客户端回发一条消息
            # await websocket.send(f"服务器收到了你的消息: {message}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"客户端 {websocket.remote_address} 已断开连接: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

async def main():
    """
    服务器主函数。
    """
    # 启动 WebSocket 服务器
    async with websockets.serve(handler, HOST, PORT):
        print(f"WebSocket 服务器正在监听 ws://{HOST}:{PORT}")
        # 服务器将一直运行，直到你手动停止它 (例如按 Ctrl+C)
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("服务器已手动关闭。")
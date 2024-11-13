import asyncio
import websockets
import json
import pandas as pd
from datetime import datetime

# 处理每个连接
async def handle_connection(websocket, path):
    print(f"新的客户端连接：{websocket.remote_address}")
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get('type') == 'recorded_data':
                recorded_data = data.get('data', [])
                if recorded_data:
                    # 将数据转换为 DataFrame
                    df = pd.DataFrame(recorded_data)

                    # 添加时间戳作为文件名的一部分
                    filename_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"recorded_data_{filename_timestamp}.xlsx"

                    # 确保 'timestamp' 列为 datetime 类型
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                    # 保存为 Excel 文件
                    df.to_excel(filename, index=False)
                    print(f"录制数据已保存为 {filename}")
            else:
                print("收到未知类型的数据：", data)
    except websockets.exceptions.ConnectionClosed:
        print("客户端连接已关闭。")
    except Exception as e:
        print(f"发生错误：{e}")

# 启动WebSocket服务器
async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket服务器已启动，监听ws://localhost:8765")
        await asyncio.Future()  # 运行直到手动停止

if __name__ == "__main__":
    asyncio.run(main())
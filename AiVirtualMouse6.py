import asyncio
import websockets
import json
from pynput.mouse import Controller, Button
from pynput.keyboard import Controller as KeyboardController, Key
import numpy as np
import time
from screeninfo import get_monitors  # 用于获取多显示器信息

# 配置变量
frameR = 20  # 窗口缩减
smoothening = 10  # 平滑系数，用于鼠标移动

# 初始化鼠标控制器
mouse = Controller()
keyboard = KeyboardController()

# 获取显示器信息
monitors = get_monitors()

# 假设你想控制第一个显示器（主显示器），你可以根据需要调整索引
monitor_index = 1
monitor = monitors[monitor_index]

# 获取显示器的宽度和高度
wScr, hScr = monitor.width, monitor.height

# 用于存储食指和拇指之间的前一帧距离
prevDistance = 0

# 0:握拳 1:食指移动鼠标 2:食指和中指拖动 3:食指和拇指缩放
action_flag = 0

last_zoom_time = 0  # 记录上次缩放的时间
zoom_delay = 0.05  # 延迟时间（秒）

plocX, plocY = 0, 0  # 上一帧鼠标位置
clocX, clocY = 0, 0  # 当前帧鼠标位置

async def process_landmarks(websocket):
    global plocX, plocY, clocX, clocY, prevDistance, action_flag, last_zoom_time
    
    async for message in websocket:
        try:
            # 解析从 WebSocket 接收到的 JSON 数据
            data = json.loads(message)
            landmarks = data.get('landmarks', [])

            if len(landmarks) == 0:
                if action_flag == 2:
                    mouse.release(Button.left)
                action_flag = 0
            else:
                # 检测哪些手指抬起
                fingers = get_fingers_up(landmarks)

                # 没有手指伸出
                if fingers == [0, 0, 0, 0, 0]:
                    action_flag = 0
                    mouse.release(Button.left)

                # 只有食指伸出
                elif fingers == [0, 1, 0, 0, 0]:
                    if action_flag == 2:
                        mouse.release(Button.left)
                    action_flag = 1
                    forefinger_x, forefinger_y = get_landmark_position(landmarks, 8)  # 食指指尖
                    x3 = np.interp(forefinger_x, (frameR, wScr - frameR), (0, wScr))
                    y3 = np.interp(forefinger_y, (frameR, hScr - frameR), (0, hScr))

                    # 平滑移动
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening

                    # 限制鼠标位置在指定显示器的范围内
                    clocX = np.clip(clocX, 0, wScr)
                    clocY = np.clip(clocY, 0, hScr)

                    # 设置鼠标位置
                    mouse.position = (monitor.x + wScr - clocX, monitor.y + clocY)
                    plocX, plocY = clocX, clocY

                # 食指和中指同时伸出
                elif fingers == [0, 1, 1, 0, 0]:
                    if action_flag != 2:
                        mouse.press(Button.left)
                        action_flag = 2
                    length = find_distance(landmarks, 8, 12)
                    forefinger_x, forefinger_y = get_landmark_position(landmarks, 8)  # 食指指尖
                    # 控制鼠标移动
                    x3 = np.interp(forefinger_x, (frameR, wScr - frameR), (0, wScr))
                    y3 = np.interp(forefinger_y, (frameR, hScr - frameR), (0, hScr))
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening

                    # 限制鼠标位置在指定显示器的范围内
                    clocX = np.clip(clocX, 0, wScr)
                    clocY = np.clip(clocY, 0, hScr)

                    mouse.position = (monitor.x + wScr - clocX, monitor.y + clocY)
                    plocX, plocY = clocX, clocY

                # 食指和拇指同时抬起
                elif fingers == [1, 1, 0, 0, 0]:
                    action_flag = 3
                    length = find_distance(landmarks, 4, 8)

                    # 比较当前距离和之前的距离
                    if prevDistance == 0:
                        prevDistance = length

                    current_time = time.time()
                    if current_time - last_zoom_time > zoom_delay:  # 检查延迟
                        if length > prevDistance + 5:  # 增大
                            mouse.scroll(0, 2)
                        elif length < prevDistance - 5:  # 减小
                            mouse.scroll(0, -2)

                        last_zoom_time = current_time  # 更新最后缩放时间

                    prevDistance = length
                else:
                    prevDistance = 0
        except json.JSONDecodeError:
            print("接收到无效的JSON数据")

def get_landmark_position(landmarks, index):
    """从 landmarks 列表中获取指定关键点的 x, y 坐标."""
    return landmarks[index]["x"] * wScr, landmarks[index]["y"] * hScr

def find_distance(landmarks, p1, p2):
    """计算两个手指之间的距离."""
    x1, y1 = get_landmark_position(landmarks, p1)
    x2, y2 = get_landmark_position(landmarks, p2)
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return length

def get_fingers_up(landmarks):
    """检测哪些手指抬起."""
    fingers = []
    # 检测大拇指
    if landmarks[4]["x"] > landmarks[3]["x"]:
        fingers.append(1)
    else:
        fingers.append(0)

    # 检测其他手指
    finger_tip_ids = [8, 12, 16, 20]
    for tip_id in finger_tip_ids:
        if landmarks[tip_id]["y"] < landmarks[tip_id - 2]["y"]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

async def main():
    async with websockets.serve(process_landmarks, "localhost", 8765):
        await asyncio.Future()  # 保持服务器运行

if __name__ == "__main__":
    asyncio.run(main())
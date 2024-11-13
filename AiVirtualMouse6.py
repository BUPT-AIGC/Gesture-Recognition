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
monitor_index = 0
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
            # [{'id': 0, 'x': 0.21106281876564026, 'y': 0.6906967163085938, 'z': 1.864898990788788e-07}, {'id': 1, 'x': 0.2679467797279358, 'y': 0.6191461086273193, 'z': -0.015972217544913292}, {'id': 2, 'x': 0.3069320023059845, 'y': 0.5354823470115662, 'z': -0.03601836785674095}, {'id': 3, 'x': 0.3289414644241333, 'y': 0.47037482261657715, 'z': -0.060581792145967484}, {'id': 4, 'x': 0.3104434013366699, 'y': 0.4346066117286682, 'z': -0.08329997211694717}, {'id': 5, 'x': 0.20613731443881989, 'y': 0.4162310063838959, 'z': -0.020058132708072662}, {'id': 6, 'x': 0.20588070154190063, 'y': 0.31829342246055603, 'z': -0.05353628471493721}, {'id': 7, 'x': 0.2108362764120102, 'y': 0.24940361082553864, 'z': -0.07433730363845825}, {'id': 8, 'x': 0.21060961484909058, 'y': 0.19137197732925415, 'z': -0.0884762555360794}, {'id': 9, 'x': 0.1686714142560959, 'y': 0.4340590238571167, 'z': -0.034772224724292755}, {'id': 10, 'x': 0.23444925248622894, 'y': 0.37317806482315063, 'z': -0.08723011612892151}, {'id': 11, 'x': 0.27842026948928833, 'y': 0.44422000646591187, 'z': -0.09949985146522522}, {'id': 12, 'x': 0.28530609607696533, 'y': 0.4957854747772217, 'z': -0.09542591124773026}, {'id': 13, 'x': 0.1478165090084076, 'y': 0.4687500596046448, 'z': -0.05219428241252899}, {'id': 14, 'x': 0.23448556661605835, 'y': 0.44986194372177124, 'z': -0.09834955632686615}, {'id': 15, 'x': 0.2616789937019348, 'y': 0.5218505263328552, 'z': -0.0879969671368599}, {'id': 16, 'x': 0.2539047300815582, 'y': 0.5599760413169861, 'z': -0.06882551312446594}, {'id': 17, 'x': 0.13913360238075256, 'y': 0.5140267014503479, 'z': -0.07208467274904251}, {'id': 18, 'x': 0.2136615812778473, 'y': 0.5006354451179504, 'z': -0.0969117060303688}, {'id': 19, 'x': 0.23015275597572327, 'y': 0.5495054125785828, 'z': -0.08603189885616302}, {'id': 20, 'x': 0.21580693125724792, 'y': 0.5777723789215088, 'z': -0.07122195512056351}]

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
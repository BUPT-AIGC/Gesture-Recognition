import cv2
import HandTrackingModule as htm
from pynput.mouse import Controller, Button
from pynput.keyboard import Controller as KeyboardController, Key
import numpy as np
import time

# 配置变量
frameR = 20  # 窗口缩减
smoothening = 10  # 平滑系数，用于鼠标移动

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 获取摄像头宽高
wCam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hCam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(wCam, hCam)

# 用于计算帧率和鼠标位置的变量
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# 初始化手部检测器和鼠标控制器
detector = htm.handDetector(maxHands=1)
mouse = Controller()
keyboard = KeyboardController()

# 屏幕分辨率
wScr, hScr = 2560, 1440

# 用于存储食指和拇指之间的前一帧距离
prevDistance = 0

# 0:握拳 1:食指移动鼠标 2:食指和中指拖动 3:食指和拇指缩放
action_flag = 0

last_zoom_time = 0  # 记录上次缩放的时间
zoom_delay = 0.05  # 延迟时间（秒）

while True:
    # 从摄像头读取帧
    success, img = cap.read()

    # 检测手部
    img = detector.findHands(img)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (0, 255, 0), 2)

    # 获取手部关键点位置
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) == 0:
        if action_flag == 2:
            mouse.release(Button.left)
        action_flag = 0
    else:
        # 检测哪些手指抬起
        fingers = detector.fingersUp()

        # 没有手指伸出
        if fingers == [0, 0, 0, 0, 0]:
            action_flag = 0
            mouse.release(Button.left)

        # 只有食指伸出
        elif fingers == [0, 1, 0, 0, 0]:
            if action_flag == 2:
                mouse.release(Button.left)
            action_flag = 1
            forefinger_x, forefinger_y = lmList[8][1:]  # 食指指尖
            x3 = np.interp(forefinger_x, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(forefinger_y, (frameR, hCam - frameR), (0, hScr))

            # 平滑移动
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 设置鼠标位置
            mouse.position = (wScr - clocX, clocY)
            cv2.circle(img, (forefinger_x, forefinger_y), 5, (255, 0, 0), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 食指和中指同时伸出
        elif fingers == [0, 1, 1, 0, 0]:
            if action_flag != 2:
                mouse.press(Button.left)
                action_flag = 2
            length, img, pointInfo = detector.findDistance(8, 12, img)
            cv2.circle(img, (pointInfo[4], pointInfo[5]), 5, (0, 255, 0), cv2.FILLED)
            forefinger_x, forefinger_y = lmList[8][1:]  # 食指指尖
            # 控制鼠标移动
            x3 = np.interp(forefinger_x, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(forefinger_y, (frameR, hCam - frameR), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            mouse.position = (wScr - clocX, clocY)
            plocX, plocY = clocX, clocY

            # 食指和拇指同时抬起
        elif fingers == [1, 1, 0, 0, 0]:
            action_flag = 3
            length, img, _ = detector.findDistance(4, 8, img)

            # 比较当前距离和之前的距离
            if prevDistance == 0:
                prevDistance = length

            current_time = time.time()
            if current_time - last_zoom_time > zoom_delay:  # 检查延迟
                if length > prevDistance + 5:  # 增大
                    keyboard.press(Key.ctrl)
                    mouse.scroll(0, 2)
                    keyboard.release(Key.ctrl)
                elif length < prevDistance - 5:  # 减小
                    keyboard.press(Key.ctrl)
                    mouse.scroll(0, -2)
                    keyboard.release(Key.ctrl)

                last_zoom_time = current_time  # 更新最后缩放时间

            prevDistance = length
        else:
            prevDistance = 0

    # 计算并显示帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    # print(fps)
    pTime = cTime
    cv2.putText(img, f'fps:{int(fps)}', (15, 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

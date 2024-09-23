import cv2
import HandTrackingModule as htm
from pynput.mouse import Controller, Button
from pynput.keyboard import Controller as KeyboardController, Key
import numpy as np
import time

# 配置变量
frameR = 0  # 窗口缩减
smoothening = 10  # 平滑系数，用于鼠标移动
dragging = False  # 用于记录拖动状态

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
detector = htm.handDetector()
mouse = Controller()
keyboard = KeyboardController()

# 屏幕分辨率（替换为你的）
wScr, hScr = 2160, 1440

# 用于存储食指和拇指之间的前一帧距离
prevDistance = 0

while True:
    # 从摄像头读取帧
    success, img = cap.read()

    # 检测手部
    img = detector.findHands(img)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (0, 255, 0), 2)

    # 获取手部关键点位置
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # 获取食指和中指指尖坐标
        x1, y1 = lmList[8][1:]  # 食指指尖
        x2, y2 = lmList[12][1:]  # 中指指尖
        x_thumb, y_thumb = lmList[4][1:]  # 大拇指指尖

        # 检测哪些手指抬起
        fingers = detector.fingersUp()

        # 仅食指伸出，控制鼠标移动
        if fingers == [0, 1, 0, 0, 0]:  # 仅食指伸出
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 平滑移动
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 设置鼠标位置
            mouse.position = (wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 仅食指和中指同时伸出，控制鼠标左键拖动
        elif fingers == [0, 1, 1, 0, 0]:  # 仅食指和中指伸出
            # 控制鼠标左键拖动
            length, img, pointInfo = detector.findDistance(8, 12, img)
            cx, cy = pointInfo[4], pointInfo[5]  # 使用两指尖的中点控制

            x3 = np.interp(cx, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(cy, (frameR, hCam - frameR), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            mouse.position = (wScr - clocX, clocY)
            plocX, plocY = clocX, clocY

            if not dragging:  # 如果不是拖动状态，按下鼠标左键
                mouse.press(Button.left)
                dragging = True

        else:
            if dragging:  # 如果拖动状态结束，松开鼠标左键
                mouse.release(Button.left)
                dragging = False

        # 食指和大拇指同时抬起，控制缩放
        if fingers[1] and fingers[0]:
            length, img, _ = detector.findDistance(4, 8, img)

            # 比较当前距离和之前的距离
            if prevDistance == 0:
                prevDistance = length

            if length > prevDistance + 5:  # 增大
                keyboard.press(Key.ctrl)
                mouse.scroll(0, 1)
                keyboard.release(Key.ctrl)

            elif length < prevDistance - 5:  # 减小
                keyboard.press(Key.ctrl)
                mouse.scroll(0, -1)
                keyboard.release(Key.ctrl)

            prevDistance = length
        else:
            prevDistance = 0

    # 计算并显示帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'fps:{int(fps)}', (15, 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
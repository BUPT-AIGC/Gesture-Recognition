# Gesture-Recognition
这个仓库是做手势识别
1、进入到hand-landmark-detection子文件夹，开启http服务：python -m http.server
这会对手部做关键点检测，并将检测到的关键点坐标通过WebSocket传给后端
2、运行AiVirtualMouse5.py：python AiVirtualMouse5.py
这会通过WebSocket接收检测到的关键点坐标，通过逻辑运算，实现鼠标控制

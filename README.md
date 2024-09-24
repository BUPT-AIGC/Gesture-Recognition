# Gesture-Recognition

这个仓库是做手势识别。

## 使用步骤

1. 进入到 `hand-landmark-detection` 子文件夹，启动NodeJS：

   ```bash
   npm start
   ```

   这会对手部做关键点检测，并将检测到的关键点坐标通过 WebSocket 传给后端。

2. 运行 `AiVirtualMouse6.py`：

   ```bash
   python AiVirtualMouse6.py
   ```

   这会通过 WebSocket 接收检测到的关键点坐标，通过逻辑运算，实现鼠标控制。

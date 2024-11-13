// 版权声明
/*
  Copyright 2023 The MediaPipe Authors.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

// 引入 MediaPipe HandLandmarker 和 FilesetResolver
import {
  HandLandmarker,
  FilesetResolver
} from "/node_modules/@mediapipe/tasks-vision/vision_bundle.js";

// 建立WebSocket连接
const socket = new WebSocket('ws://localhost:8765');

// 连接打开时的回调
socket.addEventListener('open', function (event) {
  console.log('WebSocket连接已打开。');
});

// 连接关闭时的回调
socket.addEventListener('close', function (event) {
  console.log('WebSocket连接已关闭。');
});

// 连接错误时的回调
socket.addEventListener('error', function (event) {
  console.error('WebSocket错误:', event);
});

// 获取演示部分元素
const demosSection = document.getElementById("demos");

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;

// FPS 相关变量
let fpsDisplay = document.getElementById("fps");
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

// 录制相关变量
let isRecording = false;
let recordedData = [];

// 新增：录制帧率控制变量
const RECORD_INTERVAL = 100; // 每100毫秒记录一次（10 FPS）
let lastRecordTime = 0;
let currentFrameId = 0; // 当前帧ID
const MAX_FRAMES = 20; // 最大录制帧数

// 获取录制按钮元素
const startRecordButton = document.getElementById("startRecordButton");
const stopRecordButton = document.getElementById("stopRecordButton");

// 获取进度条元素
const recordProgress = document.getElementById("recordProgress");
const progressText = document.getElementById("progressText");

// 添加按钮事件监听器
startRecordButton.addEventListener("click", startRecording);
stopRecordButton.addEventListener("click", stopRecording);

// 创建 HandLandmarker 实例
const createHandLandmarker = async () => {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "/node_modules/@mediapipe/tasks-vision/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `/models/hand_landmarker.task`,
        delegate: "GPU"
      },
      runningMode: runningMode,
      numHands: 1
    });
    demosSection.classList.remove("invisible");
  } catch (error) {
    console.error("HandLandmarker 创建失败:", error);
    alert("手部关键点检测模型加载失败，请刷新页面重试。");
  }
};
createHandLandmarker();

// 获取实时检测的坐标显示元素
const liveLandmarksDisplay = document.getElementById("liveLandmarks");

// 检查浏览器是否支持摄像头访问
const hasGetUserMedia = () => !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

// 如果支持，添加按钮事件监听器
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("您的浏览器不支持 getUserMedia() 方法");
  alert("抱歉，您的浏览器不支持摄像头访问。");
}

// 启用摄像头并开始检测
function enableCam(event) {
  if (!handLandmarker) {
    console.log("请等待 HandLandmarker 加载完成后再启用摄像头。");
    return;
  }

  if (webcamRunning) {
    webcamRunning = false;
    enableWebcamButton.innerText = "启用摄像头";
    // 停止摄像头
    const stream = video.srcObject;
    if (stream) {
      const tracks = stream.getTracks();
      tracks.forEach(track => track.stop());
    }
    video.srcObject = null;
    // 清空实时坐标显示
    if (liveLandmarksDisplay) {
      liveLandmarksDisplay.innerHTML = "";
    }
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "禁用检测";

    // 摄像头配置
    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 }
      }
    };

    // 激活摄像头流
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      video.srcObject = stream;
      video.addEventListener("loadeddata", predictWebcam);
    }).catch((err) => {
      console.error("摄像头访问失败:", err);
      alert("无法访问摄像头，请检查您的设备设置。");
    });
  }
}

let lastVideoTime = -1;
let results;

// 开始录制函数
function startRecording() {
  if (!handLandmarker) {
    console.log("请等待 HandLandmarker 加载完成后再开始录制。");
    return;
  }

  isRecording = true;
  recordedData = []; // 清空之前的数据
  lastRecordTime = 0; // 初始化上一次记录时间
  currentFrameId = 0; // 重置帧ID

  // 初始化进度条
  recordProgress.value = 0;
  progressText.innerText = `0 / ${MAX_FRAMES}`;

  console.log("录制已开始。");

  startRecordButton.disabled = true;
  stopRecordButton.disabled = false;
}

// 停止录制函数
function stopRecording() {
  if (!isRecording) return;

  isRecording = false;
  console.log("录制已停止。");

  startRecordButton.disabled = false;
  stopRecordButton.disabled = true;

  // 重置进度条
  recordProgress.value = 0;
  progressText.innerText = `0 / ${MAX_FRAMES}`;

  // 发送录制的数据到后端
  if (recordedData.length > 0 && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'recorded_data', data: recordedData }));
    console.log("录制数据已发送到服务器。");
  } else {
    console.warn("没有录制数据或WebSocket未连接。");
  }
}

// 预测摄像头图像
async function predictWebcam() {
  canvasElement.style.width = video.videoWidth + 'px';
  canvasElement.style.height = video.videoHeight + 'px';
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;

  // 设置运行模式为 VIDEO
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await handLandmarker.setOptions({ runningMode: "VIDEO" });
  }

  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    try {
      results = handLandmarker.detectForVideo(video, startTimeMs);
    } catch (error) {
      console.error("手部关键点检测失败:", error);
      return;
    }
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // 绘制检测结果
  if (results.landmarks && results.landmarks.length > 0) {
    for (const landmarks of results.landmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 5
      });
      drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 2 });
    }
  }

  canvasCtx.restore();

  // 计算并更新FPS
  frameCount++;
  const currentTime = performance.now();
  const deltaTime = currentTime - lastFrameTime;

  if (deltaTime >= 1000) { // 每秒更新一次
    fps = frameCount;
    frameCount = 0;
    lastFrameTime = currentTime;
    if (fpsDisplay) {
      fpsDisplay.innerText = `FPS: ${fps}`;
    }
  }

  // 显示实时关键点坐标
  if (results.landmarks && results.landmarks.length > 0 && liveLandmarksDisplay) {
    // 清空之前的内容
    liveLandmarksDisplay.innerHTML = "<h3>检测到的手部关键点坐标</h3>";
    results.landmarks.forEach((landmarks, index) => {
      const title = `手 ${index + 1} 的关键点坐标`;
      const table = createEmptyTable(liveLandmarksDisplay, title);
      displayLandmarkCoordinates(landmarks, table);
    });
  }

  // 发送手部关键点数据到Python WebSocket服务器（保持原有功能）
  if (results.landmarks && results.landmarks.length > 0 && socket.readyState === WebSocket.OPEN) {
    const landmarksData = results.landmarks.map(landmarks => {
      return landmarks.map((lm, lmIndex) => {
        return {
          id: lmIndex,  // 使用索引作为关键点ID
          x: lm.x,
          y: lm.y,
          z: lm.z
        };
      });
    });

    // 将所有手的关键点发送过去（可根据需要修改）
    const dataToSend = {
      landmarks: landmarksData  // 如果需要只发送第一只手，请使用 landmarksData[0]
    };

    socket.send(JSON.stringify(dataToSend));
  }

  // **修改后的录制逻辑：仅在检测到手部关键点时记录**
  if (isRecording && results.landmarks && results.landmarks.length > 0) {
    const currentTimeMs = Date.now();

    // 检查是否已经超过录制间隔时间
    if (currentTimeMs - lastRecordTime >= RECORD_INTERVAL) {
      lastRecordTime = currentTimeMs; // 更新上一次记录时间

      // 为当前帧的所有关键点添加相同的 frame_id
      results.landmarks.forEach((landmarks, handIndex) => {
        landmarks.forEach((lm, lmIndex) => {
          recordedData.push({
            frame_id: currentFrameId, // 添加 frame_id
            timestamp: currentTimeMs,
            hand_id: handIndex + 1, // 从1开始编号
            landmark_id: lmIndex,
            x: lm.x,
            y: lm.y,
            z: lm.z
          });
        });
      });

      console.log(`Recorded frame_id ${currentFrameId} at ${new Date(currentTimeMs).toISOString()}`);
      currentFrameId++; // 递增帧ID

      // 更新进度条
      if (recordProgress) {
        recordProgress.value = currentFrameId;
      }
      if (progressText) {
        progressText.innerText = `${currentFrameId} / ${MAX_FRAMES}`;
      }

      // 检查是否达到最大帧数
      if (currentFrameId >= MAX_FRAMES) {
        console.log(`已达到最大帧数 ${MAX_FRAMES}，自动停止录制。`);
        stopRecording();
      }
    }
  }

  // 如果摄像头正在运行，继续请求下一帧
  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

/********************************************************************
 // 辅助函数：创建空的表格
 ********************************************************************/
function createEmptyTable(container, title) {
  const heading = document.createElement("h3");
  heading.innerText = title;
  container.appendChild(heading);

  const table = document.createElement("table");
  const headerRow = document.createElement("tr");

  const idHeader = document.createElement("th");
  idHeader.innerText = "关键点 ID";
  const xHeader = document.createElement("th");
  xHeader.innerText = "X";
  const yHeader = document.createElement("th");
  yHeader.innerText = "Y";
  const zHeader = document.createElement("th");
  zHeader.innerText = "Z";

  headerRow.appendChild(idHeader);
  headerRow.appendChild(xHeader);
  headerRow.appendChild(yHeader);
  headerRow.appendChild(zHeader);
  table.appendChild(headerRow);

  container.appendChild(table);
  return table;
}

/********************************************************************
 // 辅助函数：显示关键点坐标
 ********************************************************************/
function displayLandmarkCoordinates(landmarks, table) {
  landmarks.forEach((landmark, index) => {
    const row = document.createElement("tr");

    const idCell = document.createElement("td");
    idCell.innerText = index;

    const xCell = document.createElement("td");
    xCell.innerText = parseFloat(landmark.x).toFixed(3);

    const yCell = document.createElement("td");
    yCell.innerText = parseFloat(landmark.y).toFixed(3);

    const zCell = document.createElement("td");
    zCell.innerText = parseFloat(landmark.z).toFixed(3);

    row.appendChild(idCell);
    row.appendChild(xCell);
    row.appendChild(yCell);
    row.appendChild(zCell);

    table.appendChild(row);
  });
}

// 获取摄像头视频和画布元素
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
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
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// 建立WebSocket连接
const socket = new WebSocket("ws://localhost:8765");

socket.addEventListener("open", () => {
  console.log("WebSocket连接已打开。");
});

socket.addEventListener("close", () => {
  console.log("WebSocket连接已关闭。");
});

socket.addEventListener("error", (event) => {
  console.error("WebSocket错误:", event);
});

const SEND_INTERVAL = 1000 / 5; // 200 FPS
let canSend = true;

// 获取演示部分元素
const demosSection = document.getElementById("demos");
const fpsDisplay = document.getElementById("fps");
const liveLandmarksDisplay = document.getElementById("liveLandmarks");
const lockStatusElement = document.getElementById("lockStatus");

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;

// FPS 相关变量
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

// 锁定状态相关变量
let lockTimeout = null;
let lockStartTime = null;
let isEffectActive = false;
let lockedHandIndex = null;
let particles = [];
const maxParticles = 100;

// 粒子类定义
class Particle {
  constructor(x, y, radius, color) {
    this.x = x;
    this.y = y;
    this.radius = radius;
    this.color = color;
    this.velocity = {
      x: (Math.random() - 0.5) * 2,
      y: (Math.random() - 0.5) * 2,
    };
    this.alpha = 1; // 透明度
    this.life = 100; // 粒子的寿命
  }

  draw(ctx) {
    ctx.save();
    ctx.globalAlpha = this.alpha;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2, false);
    ctx.fillStyle = this.color;
    ctx.fill();
    ctx.restore();
  }

  update() {
    this.x += this.velocity.x;
    this.y += this.velocity.y;
    this.alpha -= 0.01; // 粒子逐渐消失
    this.life--;
  }
}

// 创建粒子效果
function createParticles(x, y) {
  for (let i = 0; i < 10; i++) {
    const radius = Math.random() * 3 + 1; // 粒子大小
    const color = `rgba(255, ${Math.random() * 255}, 0, 1)`; // 粒子颜色
    particles.push(new Particle(x, y, radius, color));
  }
}

// 更新并绘制粒子
function updateParticles(ctx) {
  particles = particles.filter((particle) => particle.life > 0); // 移除已消失的粒子
  particles.forEach((particle) => {
    particle.update();
    particle.draw(ctx);
  });
}

// 创建 HandLandmarker 实例
const createHandLandmarker = async () => {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        delegate: "GPU",
      },
      runningMode: runningMode,
      numHands: 2,
    });
    demosSection.classList.remove("invisible");
  } catch (error) {
    console.error("HandLandmarker 创建失败:", error);
    alert("手部关键点检测模型加载失败，请刷新页面重试。");
  }
};

createHandLandmarker();

// 检查浏览器是否支持摄像头访问
const hasGetUserMedia = () =>
  !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

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
    stopWebcam();
  } else {
    startWebcam();
  }
}

// 重置锁定状态
function resetLockStatus() {
  lockedHandIndex = null;
  lockStartTime = null;
  isEffectActive = false;
  particles = [];
  lockStatusElement.innerText = "未锁定任何手";
  clearTimeout(lockTimeout);
}

// 开启摄像头
function startWebcam() {
  webcamRunning = true;
  enableWebcamButton.innerText = "禁用检测";
  const constraints = {
    video: { width: { ideal: 640 }, height: { ideal: 480 } },
  };

  navigator.mediaDevices
    .getUserMedia(constraints)
    .then((stream) => {
      video.srcObject = stream;
      video.addEventListener("loadeddata", predictWebcam);
    })
    .catch((err) => {
      console.error("摄像头访问失败:", err);
      alert("无法访问摄像头，请检查您的设备设置。");
    });
}

// 关闭摄像头
function stopWebcam() {
  webcamRunning = false;
  enableWebcamButton.innerText = "启用摄像头";
  const stream = video.srcObject;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  video.srcObject = null;
  resetLockStatus();
}

// 定义左上角选择区域
const selectionRegion = {
  x: 0,
  y: 0,
  width: 0.25,
  height: 0.25,
};

// 判断手的关键点是否在选择区域
function isHandInSelectionRegion(landmarks, canvasElement) {
  return landmarks.some((landmark) => {
    const x = landmark.x * canvasElement.width;
    const y = landmark.y * canvasElement.height;

    const regionX = selectionRegion.x * canvasElement.width;
    const regionY = selectionRegion.y * canvasElement.height;
    const regionWidth = selectionRegion.width * canvasElement.width;
    const regionHeight = selectionRegion.height * canvasElement.height;

    return (
      x >= regionX &&
      x <= regionX + regionWidth &&
      y >= regionY &&
      y <= regionY + regionHeight
    );
  });
}

// 锁定手的逻辑
function lockHandIfInSelection(landmarksArray, canvasElement) {
  if (landmarksArray.length === 0) {
    resetLockStatus();
    return;
  }

  if (lockedHandIndex !== null) return;

  for (let i = 0; i < landmarksArray.length; i++) {
    const landmarks = landmarksArray[i];
    if (isHandInSelectionRegion(landmarks, canvasElement)) {
      if (!lockStartTime) {
        lockStartTime = performance.now();
        lockTimeout = setTimeout(() => {
          lockedHandIndex = i;
          console.log(`锁定手 ${i}`);
          lockStatusElement.innerText = `已锁定手 ${i + 1}`;
          isEffectActive = false;
        }, 2000); // 2秒锁定时间
      }
      isEffectActive = true;
      return;
    }
  }

  resetLockStatus();
}

// 绘制锁定效果
function drawLockingEffect(canvasCtx, canvasElement) {
  const currentTime = performance.now();
  const elapsedTime = (currentTime - lockStartTime) / 2000;
  const alpha = Math.sin(elapsedTime * Math.PI);
  const radius = 30 + 20 * alpha;

  canvasCtx.strokeStyle = `rgba(255, 0, 0, ${alpha})`;
  canvasCtx.lineWidth = 5;

  const centerX =
    selectionRegion.x * canvasElement.width +
    (selectionRegion.width * canvasElement.width) / 2;
  const centerY =
    selectionRegion.y * canvasElement.height +
    (selectionRegion.height * canvasElement.height) / 2;

  canvasCtx.beginPath();
  canvasCtx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
  canvasCtx.stroke();
}

// 绘制虚线选择区域
function drawSelectionRegion(canvasCtx, canvasElement) {
  const regionX = selectionRegion.x * canvasElement.width;
  const regionY = selectionRegion.y * canvasElement.height;
  const regionWidth = selectionRegion.width * canvasElement.width;
  const regionHeight = selectionRegion.height * canvasElement.height;

  canvasCtx.strokeStyle = "#FF0000"; // 红色虚线
  canvasCtx.lineWidth = 2;
  canvasCtx.setLineDash([5, 5]); // 设置虚线样式
  canvasCtx.strokeRect(regionX, regionY, regionWidth, regionHeight);
  canvasCtx.setLineDash([]); // 重置线条样式
}

let lastVideoTime = -1;
let results;

// 预测摄像头图像
async function predictWebcam() {
  if (!webcamRunning || !video.srcObject) return;

  canvasElement.style.width = video.videoWidth + "px";
  canvasElement.style.height = video.videoHeight + "px";
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;

  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await handLandmarker.setOptions({ runningMode: "VIDEO" });
  }

  const startTimeMs = performance.now();
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

  // 绘制选择区域
  drawSelectionRegion(canvasCtx, canvasElement);

  // 检测并锁定手
  if (results.landmarks && results.landmarks.length > 0) {
    lockHandIfInSelection(results.landmarks, canvasElement);

    if (isEffectActive && lockStartTime) {
      drawLockingEffect(canvasCtx, canvasElement);
      const centerX =
        selectionRegion.x * canvasElement.width +
        (selectionRegion.width * canvasElement.width) / 2;
      const centerY =
        selectionRegion.y * canvasElement.height +
        (selectionRegion.height * canvasElement.height) / 2;
      createParticles(centerX, centerY);
    }

    updateParticles(canvasCtx);

    if (lockedHandIndex !== null && results.landmarks[lockedHandIndex]) {
      const lockedHandLandmarks = results.landmarks[lockedHandIndex];
      if (Array.isArray(lockedHandLandmarks)) {
        drawConnectors(canvasCtx, lockedHandLandmarks, HAND_CONNECTIONS, {
          color: "#00FF00",
          lineWidth: 5,
        });
        drawLandmarks(canvasCtx, lockedHandLandmarks, {
          color: "#FF0000",
          lineWidth: 2,
        });

        if (socket.readyState === WebSocket.OPEN) {
          const landmarksData = lockedHandLandmarks.map((lm, index) => ({
            id: index,
            x: lm.x,
            y: lm.y,
            z: lm.z,
          }));
          socket.send(JSON.stringify({ landmarks: landmarksData }));
        }
      }
    }
  } else {
    resetLockStatus();
  }

  canvasCtx.restore();

  // 计算并更新FPS
  frameCount++;
  const currentTime = performance.now();
  const deltaTime = currentTime - lastFrameTime;

  if (deltaTime >= 1000) {
    fps = frameCount;
    frameCount = 0;
    lastFrameTime = currentTime;
    if (fpsDisplay) {
      fpsDisplay.innerText = `FPS: ${fps}`;
    }
  }

  // 显示实时关键点坐标
  if (results.landmarks && liveLandmarksDisplay) {
    liveLandmarksDisplay.innerHTML = "<h3>检测到的手部关键点坐标</h3>";
    results.landmarks.forEach((landmarks, index) => {
      const title = `手 ${index + 1} 的关键点坐标`;
      const table = createEmptyTable(liveLandmarksDisplay, title);
      displayLandmarkCoordinates(landmarks, table);
    });
  }

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// 辅助函数：创建空的表格
function createEmptyTable(container, title) {
  const heading = document.createElement("h3");
  heading.innerText = title;
  container.appendChild(heading);

  const table = document.createElement("table");
  const headerRow = document.createElement("tr");

  ["关键点 ID", "X", "Y", "Z"].forEach((headerText) => {
    const header = document.createElement("th");
    header.innerText = headerText;
    headerRow.appendChild(header);
  });

  table.appendChild(headerRow);
  container.appendChild(table);
  return table;
}

// 辅助函数：显示关键点坐标
function displayLandmarkCoordinates(landmarks, table) {
  landmarks.forEach((landmark, index) => {
    const row = document.createElement("tr");

    ["id", "x", "y", "z"].forEach((key) => {
      const cell = document.createElement("td");
      cell.innerText = key === "id" ? index : parseFloat(landmark[key]).toFixed(3);
      row.appendChild(cell);
    });

    table.appendChild(row);
  });
}

// 获取摄像头视频和画布元素
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
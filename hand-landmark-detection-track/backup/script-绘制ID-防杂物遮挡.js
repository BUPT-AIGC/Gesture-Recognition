// 引入 MediaPipe HandLandmarker 和 FilesetResolver
import {
  HandLandmarker,
  FilesetResolver
} from "/js/mediapipe/vision_bundle.js";

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
let lockedHandLastPosition = null; // 新增：记录锁定手的最后位置
let lastDetectedLockTime = 0; // 新增：记录最后检测到锁定手的时间
const LOCK_TIMEOUT = 2000; // 新增：锁定超时时间，单位为毫秒

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
      "/js/mediapipe/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `/models/hand_landmarker.task`,
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
  lockedHandLastPosition = null; // 新增
  lastDetectedLockTime = 0; // 新增
}

// 启用摄像头并开始检测
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

// 定义选择区域（右上角）
const selectionRegion = {
  x: 0,      // 相对于画布宽度的比例
  y: 0,      // 相对于画布高度的比例
  width: 0.25,  // 相对于画布宽度的比例
  height: 0.25  // 相对于画布高度的比例
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
  if (lockedHandIndex !== null) {
    // 已经锁定，不再尝试重新锁定
    return;
  }

  if (landmarksArray.length === 0) {
    return;
  }

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

          // 记录锁定手的最后位置（腕部）
          const wristLandmark = landmarks[0];
          if (wristLandmark) {
            lockedHandLastPosition = {
              x: wristLandmark.x * canvasElement.width,
              y: wristLandmark.y * canvasElement.height,
            };
            lastDetectedLockTime = performance.now();
          }
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
  if (!lockStartTime) return;

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

  if (results.landmarks && results.landmarks.length > 0) {
    if (lockedHandIndex === null) {
      lockHandIfInSelection(results.landmarks, canvasElement);
    }

    if (lockedHandIndex !== null) {
      // 尝试找到与锁定手相近的当前手
      let found = false;
      for (let i = 0; i < results.landmarks.length; i++) {
        const landmarks = results.landmarks[i];
        const wristLandmark = landmarks[0];
        if (wristLandmark) {
          const wristX = wristLandmark.x * canvasElement.width;
          const wristY = wristLandmark.y * canvasElement.height;

          const distance = Math.hypot(
            wristX - lockedHandLastPosition.x,
            wristY - lockedHandLastPosition.y
          );

          // 设定一个距离阈值，例如50像素
          if (distance < 50) {
            // 更新锁定手的最后位置和检测时间
            lockedHandLastPosition = { x: wristX, y: wristY };
            lastDetectedLockTime = performance.now();
            lockedHandIndex = i;
            found = true;

            // // 获取当前时间，用于将 performance.now() 转换为可读时间
            // const currentTime = new Date(Date.now() - (performance.now() - lastDetectedLockTime));

            // // 格式化当前时间为可阅读的字符串
            // const readableTime = currentTime.toLocaleString();

            // // 输出锁定手的最后位置、检测时间和锁定的手索引
            // console.log('lockedHandLastPosition:', lockedHandLastPosition);
            // console.log('lastDetectedLockTime (readable):', readableTime);
            // console.log('lockedHandIndex:', lockedHandIndex);

            break;
          }
        }
      }

      if (!found) {
        // 检查是否超出锁定超时时间
        const currentTime = performance.now();
        if (currentTime - lastDetectedLockTime > LOCK_TIMEOUT) {
          resetLockStatus();
        }
      }
    }

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

    // 遍历所有检测到的手
    results.landmarks.forEach((landmarks, index) => {
      // 获取腕部（关键点0）的位置信息
      const wristLandmark = landmarks[0];
      if (wristLandmark) {
        // 将归一化坐标转换为像素坐标
        const wristX = wristLandmark.x * canvasElement.width;
        const wristY = wristLandmark.y * canvasElement.height;

        // 手动设置 isFlipped 为 true，因为 CSS 已经镜像了 Canvas
        const isFlipped = true;

        // 设置文本样式
        canvasCtx.font = "20px Arial";
        canvasCtx.fillStyle = "blue";
        canvasCtx.strokeStyle = "white";
        canvasCtx.lineWidth = 2;
        const idText = `ID: ${index + 1}`;
        const textWidth = canvasCtx.measureText(idText).width;

        // 绘制 ID 文字
        if (isFlipped) {
          // 保存当前上下文状态
          canvasCtx.save();

          // 水平翻转上下文以抵消 CSS 镜像
          canvasCtx.scale(-1, 1);

          // 设置文本居中对齐
          canvasCtx.textAlign = "center";

          // 绘制文本（先描边再填充）
          canvasCtx.strokeText(idText, -wristX, wristY - 10);
          canvasCtx.fillText(idText, -wristX, wristY - 10);

          // 恢复上下文到原始状态
          canvasCtx.restore();
        } else {
          // 如果没有翻转，正常绘制文本
          const drawX = wristX;
          canvasCtx.textAlign = "center";
          canvasCtx.strokeText(idText, drawX, wristY - 10);
          canvasCtx.fillText(idText, drawX, wristY - 10);
        }
      }

      // 如果这是锁定的手，继续绘制关键点等
      if (lockedHandIndex === index) {
        // 绘制手的连接线
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
          color: "#00FF00",
          lineWidth: 5,
        });
        // 绘制手的关键点
        drawLandmarks(canvasCtx, landmarks, {
          color: "#FF0000",
          lineWidth: 2,
        });

        // 发送被锁定手的关键点数据 via WebSocket
        if (socket.readyState === WebSocket.OPEN) {
          const landmarksData = landmarks.map((lm, idx) => ({
            id: idx,
            x: lm.x,
            y: lm.y,
            z: lm.z,
          }));
          socket.send(JSON.stringify({ landmarks: landmarksData }));
        }
      }
    });
  } else {
    // 当没有检测到手时，不立即重置锁定状态
    if (lockedHandIndex !== null) {
      const currentTime = performance.now();
      if (currentTime - lastDetectedLockTime > LOCK_TIMEOUT) {
        resetLockStatus();
      }
    } else {
      resetLockStatus();
    }
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

// 获取摄像头视频和画布元素
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

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
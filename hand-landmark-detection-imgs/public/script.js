import {
    HandLandmarker,
    FilesetResolver
  } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
  
  const demosSection = document.getElementById("demos");
  
  let handLandmarker = undefined;
  let runningMode = "IMAGE";
  let videoRunning = false;
  
  // 创建手部关键点标记器
  const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        delegate: "GPU"
      },
      runningMode: runningMode,
      numHands: 2
    });
    demosSection.classList.remove("invisible");
  };
  createHandLandmarker();
  
  /********************************************************************
  // 图像检测逻辑
  ********************************************************************/
  
  const imageContainers = document.getElementsByClassName("detectOnClick");
  
  // 给每个图片加上点击事件监听器
  for (let i = 0; i < imageContainers.length; i++) {
    imageContainers[i].children[0].addEventListener("click", handleImageClick);
  }
  
  async function handleImageClick(event) {
    if (!handLandmarker) {
      console.log("Wait for handLandmarker to load before clicking!");
      return;
    }
  
    if (runningMode === "VIDEO") {
      runningMode = "IMAGE";
      await handLandmarker.setOptions({ runningMode: "IMAGE" });
    }
  
    // 移除之前绘制的所有标记
    const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
    for (let i = allCanvas.length - 1; i >= 0; i--) {
      allCanvas[i].parentNode.removeChild(allCanvas[i]);
    }
  
    // 进行检测并获取结果
    const handLandmarkerResult = handLandmarker.detect(event.target);
    const canvas = document.createElement("canvas");
    canvas.setAttribute("class", "canvas");
    canvas.setAttribute("width", event.target.naturalWidth + "px");
    canvas.setAttribute("height", event.target.naturalHeight + "px");
    canvas.style = `left: 0px; top: 0px; width: ${event.target.width}px; height: ${event.target.height}px;`;
  
    event.target.parentNode.appendChild(canvas);
    const cxt = canvas.getContext("2d");
  
    // 绘制检测到的手部关键点
    for (const landmarks of handLandmarkerResult.landmarks) {
      drawConnectors(cxt, landmarks, HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
      drawLandmarks(cxt, landmarks, { color: "#FF0000", lineWidth: 1 });
    }
  }
  
  /********************************************************************
  // 视频检测逻辑
  ********************************************************************/
  
  const video = document.getElementById("video");
  const inferenceButton = document.getElementById("inferenceButton");
  const canvasElement = document.getElementById("output_canvas");
  const canvasCtx = canvasElement.getContext("2d");
  
  inferenceButton.addEventListener("click", runVideoInference);
  
  async function runVideoInference() {
    if (!handLandmarker) {
      console.log("Wait for handLandmarker to load before running inference!");
      return;
    }
  
    if (runningMode === "IMAGE") {
      runningMode = "VIDEO";
      await handLandmarker.setOptions({ runningMode: "VIDEO" });
    }
  
    if (videoRunning) {
      console.log("Inference is already running!");
      return;
    }
  
    videoRunning = true;
  
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
  
    // 获取每一帧并进行推理
    video.addEventListener("play", async () => {
      while (!video.paused && !video.ended) {
        const startTimeMs = performance.now();
  
        const results = await handLandmarker.detectForVideo(video, startTimeMs);
  
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);
  
        if (results.landmarks) {
          for (const landmarks of results.landmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
              color: "#00FF00",
              lineWidth: 5
            });
            drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 2 });
          }
        }
        canvasCtx.restore();
  
        // 下一帧
        await new Promise((requestAnimationFrame) => setTimeout(requestAnimationFrame, 1000 / 30)); // 大约每秒30帧
      }
      videoRunning = false;
    });
  
    // 播放视频开始推理
    video.play();
  }
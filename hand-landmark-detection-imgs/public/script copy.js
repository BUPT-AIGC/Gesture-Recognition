import {
    HandLandmarker,
    FilesetResolver
  } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
  
  const demosSection = document.getElementById("demos");
  
  let handLandmarker = undefined;
  let runningMode = "IMAGE";
  
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
  
  // 处理图片点击事件
  const imageContainers = document.getElementsByClassName("detectOnClick");
  
  // 给每个图片加上点击事件监听器
  for (let i = 0; i < imageContainers.length; i++) {
    imageContainers[i].children[0].addEventListener("click", handleClick);
  }
  
  async function handleClick(event) {
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
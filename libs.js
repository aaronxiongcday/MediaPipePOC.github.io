// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//      http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {
  FaceDetector,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");

let faceDetector;
let runningMode = "IMAGE";

// Initialize the object detector
const initializefaceDetector = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  faceDetector = await FaceDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
      delegate: "GPU"
    },
    runningMode: runningMode
  });
  demosSection.classList.remove("invisible");
};
initializefaceDetector();


/********************************************************************
 // Demo 2: Continuously grab image from webcam stream and detect it.
 ********************************************************************/

let video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
let enableWebcamButton;
const instruction = document.getElementById("errorMessage");
let captureButton;
const capturedImage = document.getElementById('captured-image');

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// Keep a reference of all the child elements we create
// so we can remove them easilly on each render.
var children = [];

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
  captureButton = document.getElementById("captureButton");
  captureButton.addEventListener("click", captureImage);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}


function captureImage(){
	// Create a canvas element to capture the image
	const canvas = document.createElement('canvas');
	canvas.width = video.videoWidth;
	canvas.height = video.videoHeight;
	const ctx = canvas.getContext('2d');

	// Draw the current frame from the video onto the canvas
	ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

	// Convert the canvas content to a data URL (base64)
	const dataURL = canvas.toDataURL('image/png');

	// Display the captured image in the image element
	capturedImage.src = dataURL;
}

// Enable the live webcam view and start detection.
async function enableCam(event) {
  if (!faceDetector) {
    alert("Face Detector is still loading. Please try again..");
    return;
  }

  // Hide the button.
  enableWebcamButton.classList.add("removed");

  // getUsermedia parameters
  const constraints = {
    video: true
  };

  // Activate the webcam stream.
  navigator.mediaDevices
    .getUserMedia(constraints)
    .then(function (stream) {
    const track = stream.getVideoTracks()[0];
    const settings = track.getSettings();
    //console.log("Camera specifications:", JSON.stringify(settings));
    console.log("Camera specifications:", settings);
      video.srcObject = stream;
      video.addEventListener("loadeddata", predictWebcam);
    })
    .catch((err) => {
      console.error(err);
    });
}

let lastVideoTime = -1;
async function predictWebcam() {
  // if image mode is initialized, create a new classifier with video runningMode
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await faceDetector.setOptions({ runningMode: "VIDEO" });
  }
  let startTimeMs = performance.now();

  // Detect faces using detectForVideo
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const detections = faceDetector.detectForVideo(video, startTimeMs)
      .detections;
    displayVideoDetections(detections);
  }

  // Call this function again to keep predicting when the browser is ready
  window.requestAnimationFrame(predictWebcam);
}

function displayVideoDetections(detections) {
  
  //console.log("Face Detections:", detections);
  captureButton.disabled = false;
  instruction.classList.remove("displayed");
  if(detections.length > 1){
    instruction.innerText = "Multiple people are detected.";
    instruction.classList.add("displayed");
    captureButton.disabled = true;
  }
  
  // Remove any highlighting from previous frame.
  for (let child of children) {
    liveView.removeChild(child);
  }
  children.splice(0);

  // Iterate through predictions and draw them to the live view
  for (let detection of detections) {
    /*const leftEye = detection.keypoints[0];
    const rightEye = detection.keypoints[1];

    const distance = Math.sqrt(
      Math.pow(rightEye.x - leftEye.x, 2) + Math.pow(rightEye.y - leftEye.y, 2)
    );

    console.log("Distance between eyes:", distance);
    */
    
    let score = Math.round(parseFloat(detection.categories[0].score) * 100);
    if(score < 70){
      instruction.innerText = "Face detection score is less than 70";
      instruction.classList.add("displayed");
      captureButton.disabled = true;
    }
    else if(detection.boundingBox.width < 130){//height and width
      instruction.innerText = "You are standing too far away. Please step forward. BoundingBox height/width: " + detection.boundingBox.width;
      instruction.classList.add("displayed");
      captureButton.disabled = true;
    }
    
    const p = document.createElement("p");
    p.innerText =
      "Quality: " +
      Math.round(parseFloat(detection.categories[0].score) * 100) +
      "% .";
    p.style =
      "left: " +
      (video.offsetWidth -
        detection.boundingBox.width -
        detection.boundingBox.originX) +
      "px;" +
      "top: " +
      (detection.boundingBox.originY - 30) +
      "px; " +
      "width: " +
      (detection.boundingBox.width - 10) +
      "px;";

    const highlighter = document.createElement("div");
    highlighter.setAttribute("class", "highlighter");
    highlighter.style =
      "left: " +
      (video.offsetWidth -
        detection.boundingBox.width -
        detection.boundingBox.originX) +
      "px;" +
      "top: " +
      detection.boundingBox.originY +
      "px;" +
      "width: " +
      (detection.boundingBox.width - 10) +
      "px;" +
      "height: " +
      detection.boundingBox.height +
      "px;";

    liveView.appendChild(highlighter);
    liveView.appendChild(p);

    // Store drawn objects in memory so they are queued to delete at next call
    children.push(highlighter);
    children.push(p);
    for (let keypoint of detection.keypoints.slice(0, 2)) {
      const keypointEl = document.createElement("spam");
      keypointEl.className = "key-point";
      keypointEl.style.top = `${keypoint.y * video.offsetHeight - 3}px`;
      keypointEl.style.left = `${
        video.offsetWidth - keypoint.x * video.offsetWidth - 3
      }px`;
      liveView.appendChild(keypointEl);
      children.push(keypointEl);
    }
  }
}

"use client";

import React, { useRef, useEffect, useState } from "react";

const LivenessDetection: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [result, setResult] = useState<string>("");
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);

  useEffect(() => {
    const worker = new Worker(new URL("../worker/worker-main.worker.js", import.meta.url), { type: 'module' });

    worker.onmessage = (e) => {
      if (e.data.type === "modelLoaded") {
        setModelLoaded(true); // Update state when model is loaded
        console.log("Model loaded successfully");
      } else if (e.data.type === "inferenceResult") {
        console.log("Inference result received:", e.data.result);
        const livenessScore = e.data.result;
        setResult(livenessScore > 0.5 ? "Live" : "Spoof");
      } else if (e.data.type === 'error') {
        console.error('Worker error:', e.data.message);
      }
    };

    worker.postMessage({ type: "loadModel", modelPath: '/model.onnx' });

    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Failed to start video stream", err);
      }
    };

    const captureFrame = () => {
      if (videoRef.current) {
        const canvas = document.createElement("canvas");
        const videoWidth = videoRef.current.videoWidth;
        const videoHeight = videoRef.current.videoHeight;

        if (videoWidth === 0 || videoHeight === 0) {
          console.warn('Video dimensions are zero.');
          return null;
        }

        canvas.width = videoWidth;
        canvas.height = videoHeight;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
          return ctx.getImageData(0, 0, canvas.width, canvas.height);
        }
      }
      return null;
    };

    const processFrame = (imageData: ImageData) => {
      if (!modelLoaded) {
        console.warn("Model not loaded yet.");
        return;
      }

      const inputTensor = preprocessImageData(imageData);
      console.log("Sending frame to worker for inference", inputTensor);
      worker.postMessage({ type: "runInference", inputTensor });
    };

    const intervalId = setInterval(() => {
      const frame = captureFrame();
      if (frame) {
        processFrame(frame);
      }
    }, 1000);

    startVideo();

    return () => {
      clearInterval(intervalId);
      worker.terminate();
    };
  }, [modelLoaded]);

  const preprocessImageData = (imageData: ImageData) => {
    const inputArray = new Float32Array(imageData.data);
    return inputArray;
  };

  return (
    <div>
      <video ref={videoRef} autoPlay playsInline />
      {result && <div>Liveness Result: {result}</div>}
    </div>
  );
};

export default LivenessDetection;

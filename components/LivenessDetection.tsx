"use client";

import React, { useRef, useEffect, useState } from "react";

const LivenessDetection: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [result, setResult] = useState<string>("");
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);

  useEffect(() => {
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
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
          return ctx.getImageData(0, 0, canvas.width, canvas.height);
        }
      }
      return null;
    };

    const processFrame = (imageData: ImageData) => {
      const inputTensor = new Float32Array(imageData.data);
      worker.postMessage({ type: "runInference", inputTensor });
    };

    const worker = new Worker(new URL("../worker/worker-main.worker.js", import.meta.url));

    worker.onmessage = (e) => {
      if (e.data.type === "modelLoaded") {
        setModelLoaded(true);
      }
      if (e.data.type === "inferenceResult") {
        const livenessScore = e.data.result;
        const result = livenessScore > 0.5 ? "Live" : "Spoof";
        setResult(result);

        if (result === "Spoof") {
          stopVideo();
        }
      }
      if (e.data.type === "error") {
        console.error("Worker error:", e.data.message);
      }
    };

    worker.postMessage({ type: "loadModel", modelPath: "/model.onnx" });

    const intervalId = setInterval(() => {
      if (modelLoaded) {
        const frame = captureFrame();
        if (frame) {
          processFrame(frame);
        }
      }
    }, 1000);  // Capture and process every 1 second

    startVideo();

    return () => clearInterval(intervalId);
  }, [modelLoaded]);

  const stopVideo = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      const tracks = stream.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  return (
    <div>
      <video ref={videoRef} autoPlay playsInline />
      {result && <div>Liveness Result: {result}</div>}
    </div>
  );
};

export default LivenessDetection;

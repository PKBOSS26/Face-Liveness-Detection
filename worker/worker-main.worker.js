import * as onnx from 'onnxjs';

let session = null;

onmessage = async (event) => {
  const { type, modelPath, inputTensor } = event.data;

  try {
    if (type === 'loadModel') {
      session = new onnx.InferenceSession();
      await session.loadModel(modelPath);
      postMessage({ type: 'modelLoaded' });
    }

    if (type === 'runInference' && session) {
      // Adjust input shape based on your specific model's requirements
      const input = new onnx.Tensor(inputTensor, 'float32', [1, 3, 224, 224]); // Modify shape accordingly
      const output = await session.run({ input });

      // Extracting the first output (customize based on model)
      const result = output.values().next().value;

      // Send result back to the main thread
      postMessage({ type: 'inferenceResult', result });
    }
  } catch (error) {
    postMessage({ type: 'error', message: error.message });
  }
};

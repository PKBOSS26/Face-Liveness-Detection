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
      const input = new onnx.Tensor(inputTensor, 'float32', [1, 3, 224, 224]); // Adjust shape as needed
      const output = await session.run({ input });
      const result = output.values().next().value.data[0]; // Adjust based on output shape

      postMessage({ type: 'inferenceResult', result });
    }
  } catch (error) {
    postMessage({ type: 'error', message: error.message });
  }
};

import onnx

# Load the ONNX model
model_path = "iris_model.onnx"
try:
    onnx_model = onnx.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Failed to load model from {model_path}: {e}")
    exit(1)

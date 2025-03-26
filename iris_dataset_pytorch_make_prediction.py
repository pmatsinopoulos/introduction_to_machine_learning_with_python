import json
from sklearn.datasets import load_iris
import onnxruntime as ort
import numpy as np

iris_dataset = load_iris()

# Load the ONNX model
model_path = "iris_model.onnx"
try:
    session = ort.InferenceSession(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Failed to load model from {model_path}: {e}")
    exit(1)

# Make a prediction for new data
X_new = np.array([[5.8, 2.7, 5, 2.4]], dtype=np.float32)
input_name = session.get_inputs()[0].name # Get the input name for the model
output_name = session.get_outputs()[0].name # Get the output name for the model
prediction = session.run([output_name], {input_name: X_new})[0]
predicted_class = np.argmax(prediction, axis=1)
print(f'Prediction: {predicted_class.item()}')
print(f'Predicted target name: {iris_dataset["target_names"][predicted_class.item()]}')

# save the data as new_data.json
new_data = {
    "input_shapes": [[len(X_new[0])]],
    "input_data": X_new.tolist(),
    "output_data": [[predicted_class.item()]],
    "public_output_idxs": [[0,0]]
}

# Save the ezkl-compatible data to a JSON file
with open("new_data.json", "w") as json_file:
    json.dump(new_data, json_file, indent=4)

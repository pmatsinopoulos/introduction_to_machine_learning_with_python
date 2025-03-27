import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import json

# Load the iris dataset
iris_dataset = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define a simple neural network model
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # Input features: 4, Hidden layer: 10
        self.fc2 = nn.Linear(10, 3)  # Hidden layer: 10, Output classes: 3

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model, define the loss function and the optimizer
model = IrisNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
model.train()
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Export the model in ONNX format
dummy_input = torch.randn(1, 4)
torch.onnx.export(model, dummy_input, 'iris_model.onnx')

# Save the training data
np.savez('iris_data.npz', X_train=X_train.numpy(), y_train=y_train.numpy())

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = torch.argmax(y_pred, axis=1)
    accuracy = (y_pred_classes == y_test).float().mean()
    print(f'Test set accuracy: {accuracy:.2f}')

# Use the test dataset as a set to calibrate the ezkl settings
ezkl_data = {
    "input_shapes": [[len(X_test[0])]],  # Shape of each input sample
    "input_data": X_test.tolist(),  # Convert X_test tensor to list
    "output_data": [[y] for y in y_test.tolist()],  # Convert y_test tensor to list and wrap each value in a list
    "public_output_idxs": [[i, i] for i in range(len(X_test))]  # Map each input index to its corresponding output index
}

# Save the ezkl-compatible data to a JSON file
with open("calibration_data.json", "w") as json_file:
    json.dump(ezkl_data, json_file, indent=4)


# call the file iris_dataset_pytorch_make_prediction.py
# to make a prediction using this trained data model.

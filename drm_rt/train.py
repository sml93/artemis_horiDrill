import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import GripLSTM

# Load dataset
data = np.load("grip_data1.npz")
X = data["data"]  # shape (samples, timesteps, features)
y = data["labels"]  # shape (samples,)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Train-test split
train_size = int(0.8 * len(X_tensor))
train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
test_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model, loss, optimizer
input_size = X.shape[2]
model = GripLSTM(input_size=input_size, hidden_size=64, num_layers=1, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    total = 0
    correct = 0
    # for inputs, labels in train_loader:
    #     outputs = model(inputs)
    #     loss = criterion(outputs, labels)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     _, predicted = torch.max(outputs, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()

    # acc = correct / total
    # print(f"Epoch {epoch+1}/{EPOCHS}, Accuracy: {acc:.4f}")

    confidences_epoch = []
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = torch.softmax(outputs, dim=1)
        conf_scores, predicted = torch.max(probs, 1)
        confidences_epoch.extend(conf_scores.tolist())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_conf = np.mean(confidences_epoch)
    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Accuracy: {acc:.4f}, Avg Confidence: {avg_conf:.4f}")

# # Save the trained model
# torch.save(model.state_dict(), "grip_lstm_model.pth")
# print("Model saved to grip_lstm_pth")


# # Evaluate on test set
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Test Accuracy: {correct / total:.4f}")


# Evaluate on test set with confidence scoring
model.eval()
correct = 0
total = 0
confidences = []
confident_correct = 0
confidence_threshold = 0.8  # You can adjust this threshold

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)  # Get probabilities
        conf_scores, predicted = torch.max(probs, dim=1)  # Confidence + prediction

        # Store confidence scores
        confidences.extend(conf_scores.tolist())

        # Total accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Confident predictions accuracy
        mask = conf_scores >= confidence_threshold
        confident_correct += ((predicted == labels) & mask).sum().item()

print(f"Test Accuracy: {correct / total:.4f}")
print(f"Average Confidence: {np.mean(confidences):.4f}")
print(f"Confident Predictions (>{confidence_threshold*100:.0f}% confidence): {confident_correct}")


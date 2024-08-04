import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
import sys
import pandas as pd
import random
import numpy as np
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available. Using GPU for computation.")
else:
    device = torch.device('cpu')
    print("No GPU available. Using CPU for computation.")


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 
X_train = X_train / 255.0
test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
num_classes = 10

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.long)
test_tensor = torch.tensor(test, dtype=torch.float32)

# Create datasets
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(test_tensor)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Define model
class Convolutional(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

model = Convolutional(input_shape=1, hidden_units=10, output_shape=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.03)

def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device=device):
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        X = X.permute(0, 3, 1, 2)  # Change shape to [batch_size, channels, height, width]
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader, model, loss_fn, accuracy_fn, device=device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 2:  # Ensure that the batch contains both inputs and labels
                X, y = batch
                X, y = X.to(device), y.to(device)
                X = X.permute(0, 3, 1, 2)  # Change shape to [batch_size, channels, height, width]
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                test_loss += loss.item()
                test_acc += accuracy_fn(y, y_pred.argmax(dim=1))
            else:  # The batch only contains inputs (for the test set)
                X = batch[0].to(device)
                X = X.permute(0, 3, 1, 2)  # Change shape to [batch_size, channels, height, width]
                y_pred = model(X)
                # No loss or accuracy to compute for the test set
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")

def accuracy_fn(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc

epochs = 30

for epoch in range(epochs):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn, device=device)
    test_step(data_loader=test_dataloader, model=model, loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)

model.eval()
predictions = []
image_numbers = []

# Assuming you have a batch size of 32
batch_size = 32
current_image_number = 1

with torch.no_grad():
    for X in test_dataloader:
        X = X[0].to(device)
        X = X.permute(0, 3, 1, 2)  # Change shape to [batch_size, channels, height, width]
        y_pred = model(X)
        predicted_labels = y_pred.argmax(dim=1).cpu().numpy()
        predictions.extend(predicted_labels)
        num_images = len(predicted_labels)
        image_numbers.extend(range(current_image_number, current_image_number + num_images))
        
        # Update the current image number
        current_image_number += num_images

predictions_df = pd.DataFrame({
    'ImageId': image_numbers,
    'Label': predictions
})

predictions_df.to_csv('test_predictions.csv', index=False)
print("Predictions saved to test_predictions.csv")
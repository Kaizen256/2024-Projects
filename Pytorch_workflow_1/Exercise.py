import torch
from torch import nn
import matplotlib.pyplot as plt

device = "cuda"

weight = 0.3
bias = 0.9

start = 0
end = 10
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

class LinearReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
model = LinearReg()
model.to(device)
print(model.state_dict())

loss_fn = nn.L1Loss()
optim = torch.optim.SGD(params=model.parameters(), lr=0.001)

epochs = 1500

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

train_losses = []
test_losses = []
predictions = []

for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optim.zero_grad()
    loss.backward()
    optim.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    predictions.append(y_pred.detach().cpu().numpy())

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

plt.figure(figsize=(10, 5))
plt.plot(X_train.cpu().numpy(), y_train.cpu().numpy(), 'bo', label='Actual')
plt.plot(X_train.cpu().numpy(), predictions[-1], 'r-', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Predicted vs Actual')
plt.legend()
plt.show()

from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_exercise.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)
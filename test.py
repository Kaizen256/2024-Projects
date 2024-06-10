import torch
import torch.nn as nn

# Example input data (batch of 3 samples, each with 4 features)
x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [9.0, 10.0, 11.0, 12.0]])

# Define a fully connected layer with 4 input features and 2 output features
fc = nn.Linear(in_features=4, out_features=2)

# Perform the linear transformation (matrix multiplication + bias addition)
y = fc(x)

print("Input:", x)
print("Output:", y)
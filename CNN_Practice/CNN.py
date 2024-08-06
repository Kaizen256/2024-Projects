import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available. Using GPU for computation.")
else:
    device = torch.device('cpu')
    print("No GPU available. Using CPU for computation.")

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

class_names = train_data.classes
train_dataloader = DataLoader(train_data,
                              batch_size=32,
                              shuffle=True)

test_dataloader = DataLoader(test_data,
                              batch_size=32,
                              shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dataloader))
#shape = [32, 1, 28, 28]

flatten_model = nn.Flatten()
x = train_features_batch[0] #shape = [1, 28, 28]
output = flatten_model(x) #shape = [1, 784]

class Convolutional(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape:int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
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
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

model = Convolutional(input_shape=1,
                       hidden_units=10,
                         output_shape=10).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.03)

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0,0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode(): 
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1)
            )

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def save_checkpoint(state, filename="MNIST_Checkpoint2.pth.tar"):
    print("Checkpoint")
    torch.save(state, filename)

def load_checkpoint(filepath):
    print("Loading Checkpoint")
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

Load_model = True
prediction_T= True
epochs = 30

if Load_model:
    load_checkpoint("MNIST_Checkpoint2.pth.tar")

if prediction_T:
    def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
        pred_probs = []
        model.eval()
        with torch.inference_mode():
            for sample in data:
                sample = torch.unsqueeze(sample, dim=0).to(device)
                pred_logit = model(sample)
                pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
                pred_probs.append(pred_prob.cpu())
        return torch.stack(pred_probs)
    import random
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(test_data), k=25):
        test_samples.append(sample)
        test_labels.append(label)
    pred_probs= make_predictions(model=model, 
                             data=test_samples)
    pred_classes = pred_probs.argmax(dim=1)
    plt.figure(figsize=(15, 15))
    nrows = 5
    ncols = 5
    for i, sample in enumerate(test_samples):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(sample.squeeze(), cmap="gray")
        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[test_labels[i]] 
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g") # green text if correct
        else:
            plt.title(title_text, fontsize=10, c="r") # red text if wrong
    plt.axis(False)
    plt.tight_layout(pad=1.5)
    plt.show()
    sys.exit()

for epoch in range(epochs):
    if epoch % 10 == 0:
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    test_step(data_loader=test_dataloader,
        model=model,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )
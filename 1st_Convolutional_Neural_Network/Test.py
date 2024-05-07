import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from CNN import Convolutional
from torchvision import datasets, transforms

device = "cuda"
model_path = "models/CNN02.pth"
model = torch.load(model_path)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)
class_names = test_data.classes
test_dataloader = DataLoader(test_data,
                              batch_size=32,
                              shuffle=False)

def make_predictions(model, data):
    pred_probs = []
    with torch.no_grad():
        for sample in data:
            sample = sample.to(device) if torch.cuda.is_available() else sample
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit, dim=1)
            pred_probs.append(pred_prob.cpu())
    return torch.cat(pred_probs, dim=0)

pred_probs = make_predictions(model, test_dataloader)
pred_classes = pred_probs.argmax(dim=1)

import random
random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs= make_predictions(model=model, 
                             data=test_samples)

pred_classes = pred_probs.argmax(dim=1)

plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, (sample, label) in enumerate(test_data):
    if i >= 9:
        break
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(sample.squeeze(), cmap="gray")
    pred_label = class_names[pred_classes[i]]
    truth_label = class_names[label]
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g")  # green text if correct
    else:
        plt.title(title_text, fontsize=10, c="r")  # red text if wrong
    plt.axis(False)
plt.show()
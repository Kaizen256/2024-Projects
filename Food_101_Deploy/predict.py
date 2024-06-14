import requests
import random
import os
import torchvision
import torch
from going_modular.going_modular.predictions import pred_and_plot_image
from pathlib import Path
from torch import nn
import matplotlib.pyplot as plt
import random

device = torch.device("cpu")

image_path = Path("data/food-101/images/MI")
image_files = [os.path.join(image_path, filename) for filename in os.listdir(image_path) if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
random_image_paths = random.sample(image_files, 10)

class_names_path = Path("data/food-101/meta/classes.txt")

with open(class_names_path, 'r') as file:
    class_names = [line.strip() for line in file]

weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
model = torchvision.models.efficientnet_b2(weights=weights).to(device)
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features=1408, out_features=101),
).to(device)
from pathlib import Path
effnetb2_food101_model_path = "effnetb2_food101.pth" 
model_path = Path("models", effnetb2_food101_model_path)
checkpoint = torch.load(model_path, map_location=device)

if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
for i in random_image_paths:
    pred_and_plot_image(model=model,
                        class_names=class_names,
                        image_path=i)
    plt.show()

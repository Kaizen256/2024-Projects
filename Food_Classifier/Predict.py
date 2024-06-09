import requests
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

device = "cpu"

image_path = Path("data/food-101/images/MI")
image_files = [os.path.join(image_path, filename) for filename in os.listdir(image_path) if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
random_image_paths = random.sample(image_files, 10)

class_names_path = Path("data/food-101/meta/classes.txt")

with open(class_names_path, 'r') as file:
    class_names = [line.strip() for line in file]

weights = torchvision.models.ViT_B_16_Weights.DEFAULT
model = torchvision.models.vit_b_16(weights=weights).to(device)
for param in model.parameters():
        param.requires_grad = False

model.heads = nn.Linear(in_features=768, out_features=101).to(device)

def load_checkpoint(checkpoint):
        print("Loading Checkpoint")
        model.load_state_dict(checkpoint['state_dict'])
load_checkpoint(torch.load("CLASSIFIERViT.pth.tar"))

for i in random_image_paths:
    pred_and_plot_image(model=model,
                        class_names=class_names,
                        image_path=i)
    plt.show()

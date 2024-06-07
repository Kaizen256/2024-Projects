import torchvision.transforms as transforms
from torchvision.datasets import Food101
from torch.utils.data import DataLoader
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
import mlflow
import mlflow.pytorch
import os
from torchinfo import summary
import zipfile
from pathlib import Path
import requests
from typing import Dict, List
from tqdm.auto import tqdm

weights = torchvision.models.ViT_B_16_Weights.DEFAULT
model = torchvision.models.vit_b_16(weights=weights)
print(summary(model=model, 
            input_size=(32, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    ))
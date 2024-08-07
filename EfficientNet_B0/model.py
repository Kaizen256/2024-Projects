import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary
from going_modular.going_modular import data_setup, engine
import os
import zipfile
from pathlib import Path
import requests

def main():
    Load_model = False #LOAD MODEL OR CREATE A NEW ONE
    prediction_shit = False # MAKE PREDICTIONS OR NAH
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

    train_dir = image_path / "train"
    test_dir = image_path / "test"
    """""
    # Create a transforms pipeline manually
    manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # Reshape all images to 224x224
        transforms.ToTensor(), # Turn image values to between 0 & 1 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # A mean of [0.485, 0.456, 0.406] (across each colour channel)
                            std=[0.229, 0.224, 0.225]) # A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
    ])
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                test_dir=test_dir,
                                                                                transform=manual_transforms,
                                                                                batch_size=32)
    """"" #Manual Transforming, use if you want not really necessary

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    auto_transforms = weights.transforms()

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                test_dir=test_dir,
                                                                                transform=auto_transforms,
                                                                                batch_size=64)

    """
    print(summary(model=model, 
            input_size=(32, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    ))
    """

    for param in model.features.parameters():
        param.requires_grad = False

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    output_shape = len(class_names)

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=output_shape,
                        bias=True)).to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def save_checkpoint(state, filename="CNN.pth.tar"):
        print("Checkpoint")
        torch.save(state, filename)

    def load_checkpoint(checkpoint):
        print("Loading Checkpoint")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if Load_model:
        load_checkpoint(torch.load("CNN.pth.tar"))

    results = engine.train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=4,
                        device=device)

    print("Training Accuracy:", results['train_acc'])
    print("Testing Accuracy:", results['test_acc'])
    checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

if __name__ == '__main__':
    main()
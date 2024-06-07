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
from going_modular.going_modular.engine import train_step, test_step
from going_modular.going_modular import data_setup, engine
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("No GPU available. Using CPU for computation.")

def main():
    mlflow.set_experiment("Food101 ViT")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = Food101(root='data',
                        split='train',
                        transform=transform,
                        target_transform=None,
                        download=True)

    test_data = Food101(root='data',
                        split='test',
                        transform=transform,
                        target_transform=None,
                        download=True)

    BS = 128

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                shuffle=True,
                                                batch_size=BS,
                                                num_workers=os.cpu_count(),
                                                pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                shuffle=False,
                                                batch_size=BS,
                                                num_workers=os.cpu_count(),
                                                pin_memory=True)

    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights).to(device)
    """
    print(summary(model=model, 
            input_size=(32, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    ))
    """

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model.heads = nn.Linear(in_features=768, out_features=101).to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    def save_checkpoint(state, filename="CLASSIFIERViT.pth.tar"):
        print("Checkpoint")
        torch.save(state, filename)

    def load_checkpoint(checkpoint):
        print("Loading Checkpoint")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    Load_model=False

    if Load_model:
        load_checkpoint(torch.load("CLASSIFIER.pth.tar"))

    def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = engine.train_step(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device)
            
            test_loss, test_acc = engine.test_step(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device)

            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_acc", test_acc, step=epoch)
            
            checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)

        return results

    epochs = 10

    for epochs in [epochs]:
        if mlflow.active_run():
            mlflow.end_run()

        train(model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader, 
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            device=device)
                    
        mlflow.pytorch.log_model(model, "model")
        print("-"*50 + "\n")

if __name__ == "__main__":
    main()
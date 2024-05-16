import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torchinfo import summary
from going_modular.going_modular import data_setup, engine
import os
import zipfile
from pathlib import Path
import requests

def main():
    device = "cuda"

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    def download_data(source: str, 
                    destination: str,
                    remove_source: bool = True) -> Path:
        data_path = Path("data/")
        image_path = data_path / destination
        if image_path.is_dir():
            print(f"[INFO] {image_path} directory exists, skipping download.")
        else:
            print(f"[INFO] Did not find {image_path} directory, creating one...")
            image_path.mkdir(parents=True, exist_ok=True)

            target_file = Path(source).name
            with open(data_path / target_file, "wb") as f:
                request = requests.get(source)
                print(f"[INFO] Downloading {target_file} from {source}...")
                f.write(request.content)

            with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
                print(f"[INFO] Unzipping {target_file} data...") 
                zip_ref.extractall(image_path)

            if remove_source:
                os.remove(data_path / target_file)
        
        return image_path

    image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                            destination="pizza_steak_sushi")

    train_dir = image_path / "train"
    test_dir = image_path / "test"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=manual_transforms,
        batch_size=32
    )

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = torch.nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280,
                out_features=3,
                bias=True)).to(device)

    #print(summary(model,
    #        input_size=(32, 3, 224, 224),
    #        verbose=0,
    #        col_names=["input_size", "output_size", "num_params", "trainable"],
    #        col_width=20,
    #        row_settings=["var_names"]))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    from typing import Dict, List
    from tqdm.auto import tqdm

    from going_modular.going_modular.engine import train_step, test_step

    def train(model: torch.nn.Module, 
            train_dataloader: torch.utils.data.DataLoader, 
            test_dataloader: torch.utils.data.DataLoader, 
            optimizer: torch.optim.Optimizer,
            loss_fn: torch.nn.Module,
            epochs: int,
            device: torch.device) -> Dict[str, List]:
        
        results = {"train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []
        }

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
            test_loss, test_acc = test_step(model=model,
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

            writer.add_scalars(main_tag="Loss", 
                            tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                            global_step=epoch)

            writer.add_scalars(main_tag="Accuracy", 
                            tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                            global_step=epoch)
            
            writer.add_graph(model=model, 
                            input_to_model=torch.randn(32, 3, 224, 224).to(device))

        writer.close()
        return results

#    results = train(model=model,
#                    train_dataloader=train_dataloader,
#                    test_dataloader=test_dataloader,
#                    optimizer=optimizer,
#                    loss_fn=loss_fn,
#                    epochs=5,
#                    device=device)

        print(results)

if __name__ == '__main__':
    main()
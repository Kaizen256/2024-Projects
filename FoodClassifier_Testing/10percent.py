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
import mlflow
import mlflow.pytorch
from typing import Dict, List
from tqdm.auto import tqdm
from going_modular.going_modular.engine import train_step, test_step
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("No GPU available. Using CPU for computation.")

def download_data(source: str, destination: str, remove_source: bool = True) -> Path:
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

def main():
    mlflow.set_experiment("EfficientNet Model to use")
    data_10_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                                        destination="pizza_steak_sushi")

    data_20_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                        destination="pizza_steak_sushi_20_percent")

    train_dir_10_percent = data_10_percent_path / "train"
    train_dir_20_percent = data_20_percent_path / "train"

    test_dir = data_10_percent_path / "test"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    simple_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize 
    ])

    BATCH_SIZE = 32

    # Create 10% training and test DataLoaders
    train_dataloader_10_percent, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir_10_percent,
        test_dir=test_dir, 
        transform=simple_transform,
        batch_size=BATCH_SIZE
    )

    # Create 20% training and test data DataLoders
    train_dataloader_20_percent, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir_20_percent,
        test_dir=test_dir,
        transform=simple_transform,
        batch_size=BATCH_SIZE
    )

    #effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    #effnetb2 = torchvision.models.efficientnet_b2(effnetb2_weights)
    #print(summary(effnetb2, input_size=(32, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"],
    #              col_width=20,
    #              row_settings=["var_names"]))

    OUT_FEATURES = len(class_names)

    def create_effnetb0():
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        model = torchvision.models.efficientnet_b0(weights=weights).to(device)

        # 2. Freeze the base model layers
        for param in model.features.parameters():
            param.requires_grad = False

        # 3. Set the seeds
        torch.manual_seed(42)

        # 4. Change the classifier head
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=OUT_FEATURES)
        ).to(device)

        # 5. Give the model a name
        model.name = "effnetb0"
        print(f"[INFO] Created new {model.name} model.")
        return model

    # Create an EffNetB2 feature extractor
    def create_effnetb2():
        # 1. Get the base model with pretrained weights and send to target device
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=weights).to(device)

        # 2. Freeze the base model layers
        for param in model.features.parameters():
            param.requires_grad = False

        # 3. Set the seeds
        torch.manual_seed(42)

        # 4. Change the classifier head
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1408, out_features=OUT_FEATURES)
        ).to(device)

        # 5. Give the model a name
        model.name = "effnetb2"
        print(f"[INFO] Created new {model.name} model.")
        return model

    num_epochs = 10

    models = ["effnetb0", "effnetb2"]
    model_name = "effnetb2"
    dataloader_name = "data_20_percent"

    train_dataloaders = {"data_10_percent": train_dataloader_10_percent,
                            "data_20_percent": train_dataloader_20_percent}

    experiment_number = 0
    from Classifier import train
    for epochs in [num_epochs]:
        experiment_number += 1
        print(f"[INFO] Experiment number: {experiment_number}")
        print(f"[INFO] Model: {model_name}")
        print(f"[INFO] DataLoader: {dataloader_name}")
        print(f"[INFO] Number of epochs: {epochs}")

        if model_name == "effnetb0":
            model = create_effnetb0()
        else:
            model = create_effnetb2()

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

        if mlflow.active_run():
            mlflow.end_run()

        run_name = f"{model_name}_{dataloader_name}_{epochs}_epochs"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("dataloader_name", dataloader_name)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", BATCH_SIZE)

        train(model=model,
            train_dataloader=train_dataloader_20_percent,
            test_dataloader=test_dataloader, 
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            device=device)
                    
        mlflow.pytorch.log_model(model, "model")

        from going_modular.going_modular.utils import save_model

        save_filepath = f"07_{model_name}_{dataloader_name}_{epochs}_epochs.pth"
        save_model(model=model,
                    target_dir="models",
                    model_name=save_filepath)
        print("-"*50 + "\n")

if __name__ == "__main__":
    main()

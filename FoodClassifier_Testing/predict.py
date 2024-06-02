import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from going_modular.going_modular import data_setup, engine
import os
import zipfile
from pathlib import Path
import requests
import mlflow
import mlflow.pytorch

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
    data_20_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                        destination="pizza_steak_sushi_20_percent")
    train_dir_20_percent = data_20_percent_path / "train"

    test_dir = data_20_percent_path / "test"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    simple_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize 
    ])

    BATCH_SIZE = 32
    train_dataloader_20_percent, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir_20_percent,
        test_dir=test_dir,
        transform=simple_transform,
        batch_size=BATCH_SIZE
    )

    OUT_FEATURES = len(class_names)

    def create_effnetb2():
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=weights).to(device)
        for param in model.features.parameters():
            param.requires_grad = False
        torch.manual_seed(42)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1408, out_features=OUT_FEATURES)
        ).to(device)
        model.name = "effnetb2"
        print(f"[INFO] Created new {model.name} model.")
        return model

    model_path = "models/07_effnetb2_data_20_percent_10_epochs.pth"
    model = create_effnetb2()
    model.load_state_dict(torch.load(model_path))
    from going_modular.going_modular.predictions import pred_and_plot_image

    import random
    num_images_to_plot = 10
    test_image_path_list = list(Path(data_20_percent_path / "test").glob("*/*.jpg"))
    test_image_path_sample = random.sample(population=test_image_path_list,
                                        k=num_images_to_plot)
    for image_path in test_image_path_sample:
        pred_and_plot_image(model=model,
                            image_path=image_path,
                            class_names=class_names,
                            image_size=(224, 224))
    plt.show()

if __name__ == "__main__":
    main()

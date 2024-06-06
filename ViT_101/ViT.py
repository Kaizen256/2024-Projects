import torch
import torchvision
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary
from going_modular.going_modular import data_setup, engine
from helper_functions import download_data, set_seeds, plot_loss_curves

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                            destination="pizza_steak_sushi")

    train_dir = image_path / "train"
    test_dir = image_path / "test"

    IMG_SIZE = 224

    manual_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    BS = 64

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=manual_transforms,
        batch_size=BS
    )

    image_batch, label_batch = next(iter(train_dataloader))
    image, label = image_batch[0], label_batch[0]
    plt.imshow(image.permute(1, 2, 0))
    plt.axis(False)
    plt.show()


if __name__ == "__main__":
    main()
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

    #image_batch, label_batch = next(iter(train_dataloader))
    #image, label = image_batch[0], label_batch[0]
    #plt.imshow(image.permute(1, 2, 0))
    #plt.axis(False)
    #plt.show()
    height = 224
    width = 224
    CC=3
    patch_size=16 #Found default for ViTB
    patches=196 # (224*224) / 16^2

    embedding_layer_input_shape = (height, width, CC)
    embedding_layer_output_shape = (patches, patch_size**2 * CC) #196 by 768 The 768 is embedding dimesionality so the amount of options hence 3 CC per pixel with 256 pixels


    conv2d = nn.Conv2d(in_channels=3,
                       out_channels=768,
                       kernel_size=patch_size,
                       stride=patch_size,
                       padding=0)
    
    class PatchEmbedding(nn.Module):
        def __init__(self,
                     in_channels:int=3,
                     patch_size:int=16,
                     embedding_dim:int=768):
            super().__init__()
            self.patcher = nn.Conv2d(in_channels=in_channels,
                                     out_channels=embedding_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size,
                                     padding=0)
            
            self.flatten = nn.Flatten(start_dim=2,
                                      end_dim=3)
        
        def forward(self, x):
            image_resolution = x.shape[-1]
            assert image_resolution % patch_size == 0
            x_patched = self.patcher(x)
            x_flattened = self.flatten(x_patched)
            return x_flattened.permute(0, 2, 1)

#    random_input_image = (1, 3, 224, 224)
#    random_input_image_error = (1, 3, 250, 250)

#    print(summary(PatchEmbedding(),
#             input_size=random_input_image, # try swapping this for "random_input_image_error"
#             col_names=["input_size", "output_size", "num_params", "trainable"],
#             col_width=20,
#             row_settings=["var_names"]))

if __name__ == "__main__":
    main()
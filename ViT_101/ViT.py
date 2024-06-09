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

    patchify = PatchEmbedding(in_channels=3,
                          patch_size=16,
                          embedding_dim=768)
    patch_embedded_image = patchify(image.unsqueeze(0))
    batch_size = patch_embedded_image.shape[0]
    embedding_dimension = patch_embedded_image.shape[-1]

    class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension),
                               requires_grad=True)
    
    PEICE = torch.cat((class_token, patch_embedded_image),
                      dim=1)

    number_of_patches = int((height * width) / patch_size**2)

    embedding_dimension = PEICE.shape[2]
    position_embedding = nn.Parameter(torch.ones(1,
                                                 number_of_patches+1,
                                                 embedding_dimension),
                                                 requires_grad=True)
    
    patch_and_position_embedding = PEICE + position_embedding

    class MultiheadSelfAttentionBlock(nn.Module):
        def __init__(self,
                     embedding_dim:int=768,
                     num_heads:int=12,
                     attn_dropout:float=0):
            super().__init__()
            self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
            self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                        num_heads=num_heads,
                                                        dropout=attn_dropout,
                                                        batch_first=True)
        
        def forward(self, x):
            x = self.layer_norm(x)
            attn_output, _ = self.multihead_attn(query=x,
                                                 key=x,
                                                 value=x,
                                                 need_weights=False)
            return attn_output
        
    class MLPBlock(nn.Module):
        def __init__(self,
                     embedding_dim:int=768,
                     mlp_size:int=3072,
                     dropout:float=0.1):
            super().__init__()
            self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
            self.mlp = nn.Sequential(
                nn.Linear(in_features=embedding_dim,
                          out_features=mlp_size),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=mlp_size,
                          out_features=embedding_dim),
                nn.Dropout(p=dropout)
            )

        def forward(self, x):
            x = self.layer_norm(x)
            x = self.mlp(x)
            return x
        
    class TransformerEncoderBlock(nn.Module):
        def __init__(self,
                    embedding_dim:int=768,
                    num_heads:int=12,
                    mlp_size:int=3072,
                    mlp_dropout:float=0.1,
                    attn_dropout:float=0):
            super().__init__()

            # 3. Create MSA block (equation 2)
            self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                        num_heads=num_heads,
                                                        attn_dropout=attn_dropout)

            self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                    mlp_size=mlp_size,
                                    dropout=mlp_dropout)

        def forward(self, x):
            x =  self.msa_block(x) + x
            x = self.mlp_block(x) + x

            return x
#orrrrr:
    
    torch_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768, # Hidden size D from Table 1 for ViT-Base
                                                             nhead=12, # Heads from Table 1 for ViT-Base
                                                             dim_feedforward=3072, # MLP size from Table 1 for ViT-Base
                                                             dropout=0.1, # Amount of dropout for dense layers from Table 3 for ViT-Base
                                                             activation="gelu", # GELU non-linear activation
                                                             batch_first=True, # Do our batches come first?
                                                             norm_first=True) # Normalize first or after MSA/MLP layers?

    class ViT(nn.Module):
        def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000): # Default for ImageNet but can customize this
            super().__init__()
            assert img_size % patch_size == 0
            self.num_patches = (img_size * img_size) // patch_size**2
            self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                                requires_grad=True)
            self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                                   requires_grad=True)
            self.embedding_dropout = nn.Dropout(p=embedding_dropout)
            self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                                  patch_size=patch_size,
                                                  embedding_dim=embedding_dim)
            self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
            self.classifier = nn.Sequential(
                nn.LayerNorm(normalized_shape=embedding_dim),
                nn.Linear(in_features=embedding_dim,
                          out_features=num_classes)
            )
        
        def forward(self, x):
            batch_size = x.shape[0]
            class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

            x = self.patch_embedding(x)
            x = torch.cat((class_token, x), dim=1)
            x = self.position_embedding + x
            x = self.embedding_dropout(x)
            x = self.transformer_encoder(x)
            x = self.classifier(x[:, 0])
            return x
        
    optimizer = torch.optim.Adam(params=ViT.parameters(),
                             lr=3e-3, # Base LR from Table 3 for ViT-* ImageNet-1k
                             betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                             weight_decay=0.3) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

    # Setup the loss function for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()

    # Set the seeds
    set_seeds()

    # Train the model and save the training results to a dictionary
    results = engine.train(model=ViT,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=10,
                        device=device)


if __name__ == "__main__":
    main()
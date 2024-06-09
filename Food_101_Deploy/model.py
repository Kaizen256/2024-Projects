import torch
import torchvision
from torch import nn
from torchvision import datasets
from pathlib import Path
import os
from going_modular.going_modular import engine

device = "cuda"

def create_effnetb2_model(num_classes:int=3, 
                          seed:int=42):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )
    return model, transforms

effnetb2_food101, effnetb2_transforms = create_effnetb2_model(num_classes=101)
food101_train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.TrivialAugmentWide(),
    effnetb2_transforms,
])

data_dir = Path("data")

train_data = datasets.Food101(root=data_dir,
                              split="train",
                              transform=food101_train_transforms,
                              download=True)

test_data = datasets.Food101(root=data_dir,
                             split="test",
                             transform=effnetb2_transforms,
                             download=True)

food101_class_names = train_data.classes

BS = 32
NW = 2 if os.cpu_count() <= 4 else 4

train_dataloader = torch.utils.data.DataLoader(train_data,
                                            batch_size=BS,
                                            shuffle=True,
                                            num_workers=NW)
test_dataloader = torch.utils.data.DataLoader(train_data,
                                            batch_size=BS,
                                            shuffle=False,
                                            num_workers=NW)

optimizer = torch.optim.Adam(params=effnetb2_food101.parameters(),
                             lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

effnetb2_food101_results = engine.train(model=effnetb2_food101,
                                        train_dataloader=train_dataloader,
                                        test_dataloader=test_dataloader,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        epochs=5,
                                        device=device)

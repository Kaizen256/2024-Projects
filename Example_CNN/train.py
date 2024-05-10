import os
import torch
import data_setup, engine, model_builder, utils
import matplotlib.pyplot as plt

from torchvision import transforms

def main():
  # Setup hyperparameters
  NUM_EPOCHS = 5
  BATCH_SIZE = 32
  HIDDEN_UNITS = 10
  LEARNING_RATE = 0.001

  # Setup directories
  train_dir = "data/pizza_steak_sushi/train"
  test_dir = "data/pizza_steak_sushi/test"

  # Setup target device
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Create transforms
  data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
  ])

  # Create DataLoaders with help from data_setup.py
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
      train_dir=train_dir,
      test_dir=test_dir,
      transform=data_transform,
      batch_size=BATCH_SIZE
  )

  # Create model with help from model_builder.py
  model = model_builder.TinyVGG(
      input_shape=3,
      hidden_units=HIDDEN_UNITS,
      output_shape=len(class_names)
  ).to(device)

  # Set loss and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                              lr=LEARNING_RATE)

  # Start training with help from engine.py
  engine.train(model=model,
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              loss_fn=loss_fn,
              optimizer=optimizer,
              epochs=NUM_EPOCHS,
              device=device)

  # Save the model with help from utils.py
  utils.save_model(model=model,
                  target_dir="models",
                  model_name="05_going_modular_script_mode_tinyvgg_model.pth")
  
  visualize_predictions(model, test_dataloader, class_names, device)

def visualize_predictions(model, test_dataloader, class_names, device):
    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Plot images with predicted labels
            plt.figure(figsize=(10, 10))
            for i in range(len(images)):
                plt.subplot(1, len(images), i + 1)
                plt.imshow(images[i].permute(1, 2, 0).cpu())
                plt.title(f"Predicted: {class_names[predicted[i]]}\nActual: {class_names[labels[i]]}")
                plt.axis('off')
            plt.show()
            break

if __name__ == '__main__':
    main()
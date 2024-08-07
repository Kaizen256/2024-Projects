# Pizza, Steak, Sushi Classification with EfficientNet-B0

This project is one of many I followed through a pytorch tutorial. It demonstrates how to classify images of pizzas, steaks, and sushi using the EfficientNet-B0 model in PyTorch. The model is trained on a custom dataset and can be configured to either load a pre-trained model or create a new one from scratch. Additionally, it includes functionality for saving and loading model checkpoints.

Architecture:
 EfficientNet-B0 with pre-trained weights.
The feature extractor layers are frozen, and only the classifier head is trained.
Classifier head: Dropout layer followed by a Linear layer with output units corresponding to the number of classes (3 in this case).
Loss Function: CrossEntropyLoss.
Optimizer: Adam with a learning rate of 0.001.
Epochs: 4
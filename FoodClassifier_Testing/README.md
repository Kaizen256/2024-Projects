# EfficientNetB0 for Food101 Dataset


This was more of a test for EfficientNetB2 and ViT_B_16. It was modified from a deep learning course to fit my testing. Model was trained on only 20% of the data to save time. Logged in mlflow for testing I used pretrained weights and froze the parameters to adjust the amount of output features. Model was saved in the models folder.


Helper functions were taken from a deep learning course.


Epochs = 15
Optimizer = Adam
Loss = CrossEntropyLoss
lr = 0.0008
Batch size = 32

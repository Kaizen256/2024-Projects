# ViT_B_16 for Food101 Dataset


For this project, I used a pretrained ViT model on the Food101 dataset. This model achieved ~80% accuracy and it was confident in its predictions, unlike the EfficientNetB2. The huge downside was the size of the model. 300 MB is far too large for a model and I am unable to commit it to github. I froze the parameters to adjust the amount of output features. going_modular was taken from a deep learning course.


Epochs = 10
Optimizer = Adam
Loss = CrossEntropyLoss
lr = 0.03
Batch size = 128


I coded a couple scripts to reduce the number of images I committed. I also added a script that visualizes predictions.

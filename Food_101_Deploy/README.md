# EfficientNetB2 for Food101 Dataset


For this project, I was trying to create a model that was small enough to be able to commit to Git. My previous model was trained using the torchvision model ViT_B_16. While that model had a really high accuracy due to millions of more parameters, it also was 300 MB which is an unrealistic model size. For this model I wanted to compact the model while maintaining its accuracy. I settled for EfficientNetB2 from ImageNet. After the model was trained it achieved ~65% accuracy which isn't bad considering the model was only 27 MB. I used pretrained weights and froze the parameters to adjust the amount of output features. Model was saved in the models folder. Helper functions was taken from a deep learning course.


Epochs = 10
Optimizer = Adam
Loss = CrossEntropyLoss
lr = 0.001
Batch size = 32


I coded a couple scripts to reduce the number of images I committed. I also added a script that visualizes predictions.

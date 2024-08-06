# CS231 Classification with Pytorch

Dataset taken from stanfords CS231. This project demonstrates how to create and classify a spiral dataset using a simple neural network in PyTorch. The model is trained to classify data points into three different classes.

Uses a simple neural network with three linear layers and ReLU activations.The output layer has three units corresponding to the three classes. Uses CrossEntropyLoss as the loss function and Adam as the optimizer. Includes a function to plot the decision boundary of the trained model.
Visualizes the decision boundary at regular intervals during training and at the end of training for both the training and test datasets. Trained on 150 epochs and achieves 100% accuracy on test data.
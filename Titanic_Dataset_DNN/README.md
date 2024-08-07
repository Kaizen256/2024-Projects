# Titanic Dataset with a DNN


This was my attempt of creating a really advanced neural network for a simple dataset. I wanted to see if 'more advanced' = 'better'. This model achieved an accuracy of 78.47% which isn't bad. But a model that just used decision trees would probably have a higher accuracy. Using bigger models usually only works for bigger datsets, and they must suit the model. Using a huge model on the titanic dataset was overkill.


# Each layer summarized:


Input Layer:
self.fc1: Fully connected layer (linear transformation) from the input features to 128 neurons.
self.bn1: Batch normalization layer for 128 features.


Hidden Layer 1:
self.fc2: Fully connected layer from 128 neurons to 64 neurons.
self.bn2: Batch normalization layer for 64 features.


Hidden Layer 2:
self.fc3: Fully connected layer from 64 neurons to 32 neurons.
self.bn3: Batch normalization layer for 32 features.


Hidden Layer 3:
self.fc4: Fully connected layer from 32 neurons to 16 neurons.
self.bn4: Batch normalization layer for 16 features.


Output Layer:
self.fc5: Fully connected layer from 16 neurons to 1 output neuron (for binary classification).


Activation Functions:
self.relu: ReLU activation function.


Regularization:
self.dropout: Dropout layer with a dropout probability of 0.5.


Output Activation:
self.sigmoid: Sigmoid activation function for the output layer.


Forward Pass (forward Method):


Input to Hidden Layer 1:
Apply self.fc1 (linear transformation), then self.bn1 (batch normalization), followed by self.relu (activation function).
Apply self.dropout (dropout).


Hidden Layer 1 to Hidden Layer 2:
Apply self.fc2 (linear transformation), then self.bn2 (batch normalization), followed by self.relu (activation function).
Apply self.dropout (dropout).


Hidden Layer 2 to Hidden Layer 3:
Apply self.fc3 (linear transformation), then self.bn3 (batch normalization), followed by self.relu (activation function).
Apply self.dropout (dropout).


Hidden Layer 3 to Output Layer:
Apply self.fc4 (linear transformation), then self.bn4 (batch normalization), followed by self.relu (activation function).
Apply self.dropout (dropout).


Output Layer:
Apply self.fc5 (linear transformation), followed by self.sigmoid (activation function).


# Parameters


Epochs: 1000
Loss: BCELoss
Optimizer: Adam
lr = 0.001
Scheduler: ReduceLROnPlateau


import os 
import scipy
import numpy as np
import pandas as pd
import IPython
import tensorflow as tf
import keras 
import seaborn as sns
import warnings as w
import sklearn.metrics as Metric_tools
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt 


np.random.seed(1)
w.filterwarnings('ignore')

train_file = pd.read_csv('train.csv')
test_file = pd.read_csv('test.csv')

train_file_norm = train_file.iloc[:, 1:] / 255.0
test_file_norm = test_file / 255.0

disc_train = train_file_norm.describe().T
disc_test = test_file_norm.describe().T
num_examples_train = train_file.shape[0]
num_examples_test = test_file.shape[0]
n_h = 32
n_w = 32
n_c = 3

Train_input_images = np.zeros((num_examples_train, n_h, n_w, n_c))
Test_input_images = np.zeros((num_examples_test, n_h, n_w, n_c))

for example in range(num_examples_train):
    Train_input_images[example,:28,:28,0] = train_file.iloc[example, 1:].values.reshape(28,28)
    Train_input_images[example,:28,:28,1] = train_file.iloc[example, 1:].values.reshape(28,28)
    Train_input_images[example,:28,:28,2] = train_file.iloc[example, 1:].values.reshape(28,28)
    
for example in range(num_examples_test):
    Test_input_images[example,:28,:28,0] = test_file.iloc[example, :].values.reshape(28,28)
    Test_input_images[example,:28,:28,1] = test_file.iloc[example, :].values.reshape(28,28)
    Test_input_images[example,:28,:28,2] = test_file.iloc[example, :].values.reshape(28,28)

for example in range(num_examples_train):
    Train_input_images[example] = cv2.resize(Train_input_images[example], (n_h, n_w))
    
for example in range(num_examples_test):
    Test_input_images[example] = cv2.resize(Test_input_images[example], (n_h, n_w))

Train_labels = np.array(train_file.iloc[:, 0])

from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=27,
    width_shift_range=0.3,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.2,
    horizontal_flip=False)

validation_datagen = ImageDataGenerator()

pretrained_model = keras.applications.resnet50.ResNet50(input_shape=(n_h, n_w, n_c),
                                                        include_top=False, weights='imagenet')

model = keras.Sequential([
    pretrained_model,
    keras.layers.Flatten(),
    keras.layers.Dense(units=60, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax')
])

Optimizer = 'RMSprop'

model.compile(optimizer=Optimizer, 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

train_images, dev_images, train_labels, dev_labels = train_test_split(Train_input_images, 
                                                                      Train_labels,
                                                                      test_size=0.1, train_size=0.9,
                                                                      shuffle=True,
                                                                      random_state=44)
test_images = Test_input_images

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.999999):
            print("Stop training!")
            self.model.stop_training = True

callbacks = myCallback()

EPOCHS = 12
batch_size = 212

history = model.fit(
    train_datagen.flow(train_images, train_labels, batch_size=batch_size),
    steps_per_epoch=train_images.shape[0] // batch_size, 
    epochs=EPOCHS,   
    validation_data=validation_datagen.flow(dev_images, dev_labels, batch_size=batch_size),
    validation_steps=dev_images.shape[0] // batch_size,
    callbacks=[callbacks]
)

# Make predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Prepare submission file
submission = pd.DataFrame({'ImageId': np.arange(1, len(predicted_labels) + 1), 'Label': predicted_labels})
submission.to_csv('digit_submission.csv', index=False)
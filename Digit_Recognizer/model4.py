import tensorflow as tf
import keras

from keras._tf_keras.keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras._tf_keras.keras.callbacks import ModelCheckpoint
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.vstack((X_train, X_test))
y_train = np.concatenate([y_train, y_test])
X_train = X_train.reshape(-1, 28, 28, 1)
print(X_train.shape, y_train.shape)

train = pd.read_csv('train.csv').values
y_val = train[:,0].astype('int32')
X_val = train[:,1:].astype('float32')
X_val = X_val.reshape(-1,28,28,1)
print(X_val.shape, y_val.shape)

X_test = pd.read_csv('test.csv').values.astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1)

X_train = X_train.astype('float32')/255
X_val = X_val.astype('float32')/255
X_test = X_test.astype('float32')/255 

y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=192, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, verbose=1,
                              patience=2, min_lr=0.00000001)
hist = model.fit(X_train, y_train, batch_size=100, epochs=25,
          validation_data=(X_val, y_val), callbacks=[reduce_lr],
          verbose=1, shuffle=True)

predictions = model.predict(X_test, verbose=2)
testY = np.argmax(predictions, axis=1)

# Create submission file
sub = pd.read_csv('sample_submission.csv')
sub['Label'] = testY
sub.to_csv('submission.csv', index=False)
#referenced by https://github.com/aymericdamien/TensorFlow-Examples/
#using tensorflow to build the neural network for chess training
from __future__ import division, print_function, absolute_import
import sys
import tensorflow as tf
import numpy as np
import sklearn.preprocessing
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

#load data from numpy
with np.load("trainData/trainData1k.npz") as data:
  features = data["X"]
  labels = data["y"]

(features, labels) = shuffle(features, labels, random_state=0)

#one hot encode labels
features = np.reshape(features, (features.shape[0], 8,8,5))
label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(max(labels)+1))
labels = label_binarizer.transform(labels)

trainSize = len(features)
trainPer = 0.9
assert features.shape[0] == labels.shape[0]

N_EPHOS = 50
BATCH_SIZE = 150

#create Model
model = Sequential()
#add model layers
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(8,8,5)))
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(32, kernel_size=3, padding='same', strides=2, activation='relu'))

model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=2, padding='same', strides=2,       activation='relu'))

model.add(Conv2D(64, kernel_size=2, padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=2, padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=2, padding='same',strides=2, 		activation='relu'))

model.add(Conv2D(128, kernel_size=1, activation='relu'))
model.add(Conv2D(128, kernel_size=1, activation='relu'))
model.add(Conv2D(128, kernel_size=1, activation='relu'))

model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
     optimizer='adam',
     metrics=['accuracy'])

print(model.summary())
model.fit(features, labels, epochs=N_EPHOS, batch_size=BATCH_SIZE)
scores = model.evaluate(features, labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
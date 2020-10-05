import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import urllib.request
from random import randint


#load data: ant, bee, bird, cat, dog, dolphin, dragon, horse, rabbit, swan
#10 classes with animals
""" this function loads the data from the data folder and preprocesses them
    before training. Returns a train-test split dataset """
def load_data(path, test_ratio = 0.2, itemsPerClass = 4000):

    #initialize variables
    x = np.empty([0, 784])
    y = np.empty([0])
    classes = [] #referring to the ten classes in the data folder

    #load each file in the data folder
    files = [x for x in os.listdir(path)]

    for i, file in enumerate(files):
        data = np.load(path + "/" + file)
        data = data[0: itemsPerClass, :]
        labels = np.full(data.shape[0], i)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(file))
        classes.append(class_name)

    data = None
    labels = None

    #randomize the dataset
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    #separate dataset into training and testing
    test_size = int(x.shape[0] / 100 * (test_ratio * 100))

    x_test = x[0:test_size, :]
    y_test = y[0:test_size]

    x_train = x[test_size:x.shape[0], :]
    y_train = y[test_size:y.shape[0]]

    # Reshape and normalize, image size is 28x28
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    x_train /= 255.0
    x_test /= 255.0

    # Convert class vectors to class matrices
    num_classes = len(classes)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, classes

x_train, y_train, x_test, y_test, classes = load_data("data")

#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#(160000, 28, 28, 1) (160000, 10) (40000, 28, 28, 1) (40000, 10)

# Show some random data
# idx = randint(0, len(x_train))
# plt.imshow(x_train[idx], cmap="gray_r")
# plt.title(classes[int((np.where(y_train[idx] == 1)[0]))])

# ------------- Define model ---------------
# We use convolutional neural network (CNN) to train on the images

model = keras.Sequential()
model.add(keras.layers.Convolution2D(16, (3, 3), padding="same",
                                        input_shape=x_train.shape[1:], activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Convolution2D(32, (3, 3), padding="same", activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Convolution2D(64, (3, 3), padding="same", activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
#model.add(keras.layers.Dropout(0.1))
#model.add(keras.layers.Dense(64, activation="relu"))
#model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["top_k_categorical_accuracy"])
#print(model.summary())

# Train model
history = model.fit(x = x_train, y = y_train, validation_split = 0.1, verbose=2, epochs=5)

# pd.DataFrame(history.history).plot(figsize=(8,5))
# plt.grid(True)

# Test model
pred = model.predict(x_test)
score = model.evaluate(x_test, y_test, verbose=0)
#print('Test accuracy: {:0.2f}%'.format(score[1] * 100))
# Test accuracy: 96.88% - pretty goooood

# ---------- Errors -------------------------------------
# Get predictions and true labels
y_pred = np.argmax(pred, axis = 1)
y = np.argmax(y_test, axis = 1)

# Get the indices of the bad labels
bad_pred = np.argwhere((y == y_pred) == False).squeeze()

#Plot
# fig, axes = plt.subplots(3, 3, figsize=(9, 9))
#
# for i, ax in enumerate(axes.flatten()):
#     idx = np.random.choice(bad_pred)
#     x_show = np.squeeze(x_test[idx])
#     ax.imshow(x_show, cmap="gray_r", interpolation="nearest")
#     ax.set_title(f"True label: {classes[y[idx]]}, \nPrediction: {classes[y_pred[idx]]}")
#     ax.axis("off")

# --------------------------------------------------------

# Print top 5 predictions for a object
idx = randint(0, len(x_test))
img = x_test[idx]
plt.imshow(img.squeeze(), cmap="gray_r")
pred1 = model.predict(np.expand_dims(img, axis=0))[0]
ind = (-pred1).argsort()[:5]
latex = [classes[x] for x in ind]
plt.title(latex)
plt.show()

# ---------------------------------------------------------

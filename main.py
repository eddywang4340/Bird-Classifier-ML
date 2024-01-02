import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import model_from_json

data = []
labels = []

# creating dataset for canada goose
canada_goose = os.listdir("ML - Bird Classifier/canada_goose")
for goose in canada_goose:
    imag = cv2.imread("ML - Bird Classifier/canada_goose/"+goose)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)

# creating dataset for blue jay
blue_jay = os.listdir("ML - Bird Classifier/blue_jay")
for jay in blue_jay:
    imag = cv2.imread("ML - Bird Classifier/blue_jay/"+jay)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)

# creating dataset for northern cardinal
northern_cardinal = os.listdir("ML - Bird Classifier/northern_cardinal")
for cardinal in northern_cardinal:
    imag = cv2.imread("ML - Bird Classifier/northern_cardinal/"+cardinal)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(2)

# creating dataset for pigeon
pigeon = os.listdir("ML - Bird Classifier/pigeon")
for pigeon in pigeon:
    imag = cv2.imread("ML - Bird Classifier/pigeon/"+pigeon)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(3)

# creating dataset for loon
loons = os.listdir("ML - Bird Classifier/loon")
for loon in loons:
    imag = cv2.imread("ML - Bird Classifier/loon/"+loon)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(4)

# creating dataset for seagulls
seagulls = os.listdir("ML - Bird Classifier/seagull")
for gull in seagulls:
    imag = cv2.imread("ML - Bird Classifier/seagull/"+gull)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(5)

# creating dataset for red-tailed hawk
red_tailed_hawks = os.listdir("ML - Bird Classifier/red_tailed_hawk")
for hawk in red_tailed_hawks:
    imag = cv2.imread("ML - Bird Classifier/red_tailed_hawk/"+hawk)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(6)

# creating dataset for great blue heron
great_blue_heron = os.listdir("ML - Bird Classifier/great_blue_heron")
for heron in great_blue_heron:
    imag = cv2.imread("ML - Bird Classifier/great_blue_heron/"+heron)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(7)

# converting and saving dataset into numpy array
birds = np.array(data)
labels = np.array(labels)
np.save("birds",birds)
np.save("labels",labels)

# loading dataset
birds = np.load("birds.npy")
labels = np.load("labels.npy")

# shuffling dataset
s = np.arange(birds.shape[0])
np.random.shuffle(s)
birds = birds[s]
labels = labels[s]

num_species = len(np.unique(labels))
data_length = len(birds)

# creating train and test datasets
x_train = birds[(int)(0.1*data_length):]
x_test = birds[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

train_length = len(x_train)
test_length = len(x_test)

# creating train and test label dataset
y_train = labels[(int)(0.1*data_length):]
y_test = labels[:(int)(0.1*data_length)]

# one-hot encoding label dataset
y_train = keras.utils.to_categorical(y_train, num_species)
y_test = keras.utils.to_categorical(y_test, num_species)

# building classification model
# adding increasing filter sizes helps with adding more depth to images
model= tf.keras.models.Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(8, activation="softmax"))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=50, epochs=100, verbose=1)

# evaluating the model with the test dataset
score = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test accuracy:', score[1])

# saving model into JSON file 
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")


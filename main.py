import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf

from tensorflow import keras

data = []
labels = []

canada_goose = os.listdir("canada_goose")
for goose in canada_goose:
    imag=cv2.imread("canada_goose/"+goose)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)

blue_jay = os.listdir("blue_jay")
for jay in blue_jay:
    imag=cv2.imread("blue_jay/"+blue_jay)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)

northern_cardinal = os.listdir("northern_cardinal")
for cardinal in northern_cardinal:
    imag=cv2.imread("northern_cardinal/"+northern_cardinal)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(2)

pigeon = os.listdir("pigeon")
for pigeon in pigeon:
    imag=cv2.imread("pigeon/"+pigeon)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(3)

birds=np.array(data)
labels=np.array(labels)
np.save("birds",birds)
np.save("labels",labels)

birds=np.load("birds.npy")
labels=np.load("labels.npy")

s=np.arange(birds.shape[0])
np.random.shuffle(s)
birds=birds[s]
labels=labels[s]

num_classes=len(np.unique(labels))
data_length=len(birds)

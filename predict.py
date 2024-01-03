from keras.models import model_from_json
import cv2
from PIL import Image
import numpy as np

# load the model from the json file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

# predicting on a single image
def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)
def get_bird_name(label):
    if label == 0:
        return "canada goose"
    if label == 1:
        return "blue jay"
    if label == 2:
        return "northern cardinal"
    if label == 3:
        return "pigeon"
    if label == 4:
        return "loon"
    if label == 5:
        return "seagull"
    if label == 6:
        return "red-tailed hawk"
    if label == 7:
        return "great blue heron"
def predict_bird(file):
    print("Predicting .................................")
    ar = convert_to_array(file)
    ar = ar/255
    label = 1
    a = []
    a.append(ar)
    a = np.array(a)
    score = loaded_model.predict(a, verbose=1)
    label_index = np.argmax(score)
    acc = np.max(score)
    animal = get_bird_name(label_index)
    print("The predicted bird is a "+animal+" with accuracy = "+str(acc))

predict_bird("blue_heron_test1.jpeg")


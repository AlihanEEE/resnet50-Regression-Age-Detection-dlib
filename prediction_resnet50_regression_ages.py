import tensorflow as tf
from glob import glob
import cv2
import numpy as np

import matplotlib.pyplot as plt

modepath = "training_2/trainable_false__optimizer_adam__activation_sigmoid.hdf5"
# Load the model from the .hd5f file
models = tf.keras.models.load_model(modepath)
# --------------------------------------------------------

folders = glob("facedetect\\face_age\\*")

class_names=[]


for name in folders:
        number_string = name.split('\\')[-1].split('.')[0].lstrip('0')
        class_names.append(int(number_string)/110)
print(class_names)

def preprocess(img):
        resized = cv2.resize(img, (224, 224))
        n_resized = resized / 255.0
        n_resized = np.expand_dims(n_resized, axis=0)
        return n_resized

imgpath = glob("age_test_images\\*.jpg")
               
for imge in imgpath:
        # Load the image to classify
        org_image = cv2.imread(imge)

        image = preprocess(org_image.copy())
        
    
        # Predict the class of the image
        predictions = models.predict(image)
        
        # Get the class with the highest probability
        bird_class = predictions.argmax()
        print(bird_class)

        cv2.imshow(str(bird_class), org_image)
        cv2.waitKey(0)
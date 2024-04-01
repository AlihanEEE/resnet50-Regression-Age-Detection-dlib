import numpy as np
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import keras
from CustomDataGen import CustomDataGen
import tensorflow_addons as tfa
import datetime


train_image_size = (224, 224)
X,y = [], []


# folders = glob("facedetect\\*")
print("Choose the data set:")
print('face_age: 1')
print('UTKFace: 2')
dataset = input('Your choice: ')
dataset= int(dataset)
while True:
    if dataset == 1: 
        datasetname="face_age"
        folders = glob("facedetect\\face_age\\*")
        for files in folders:
            for imagepath in glob(files+"\\*.png"):
                X.append(imagepath)
                number_string = files.split('\\')[-1].split('.')[0].lstrip('0')
                number = float(number_string)/110
                y.append(number)
        break    

    elif dataset == 2:
        datasetname="UTKFace"
        for imagepath in glob("UTKFace\\*.jpg"):
            X.append(imagepath)
            number_string = imagepath.split('_')[0].split('\\')[-1]
            number = float(number_string)/110
            y.append(number)
            
        break

    else:
        print("invalid choice. Please Choose the data set:")
        print(f'face_age: 1')
        print(f'UTKFace: 2')
        dataset = input('Your choice: ')
        dataset= int(dataset)
        continue
        

X,y= np.array(X), np.array(y)
print(f'X shape:{X.shape}  y shape:{y.shape}')


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .75)

print(f'X_train shape:{X_train.shape} - X_test shape:{X_test.shape} - y_train shape:{y_train.shape} - y_test shape:{y_test.shape}' )




traingen = CustomDataGen(X_train, y_train, img_size=train_image_size, augmentation=2, shuffle=True)

valgen = CustomDataGen(X_test, y_test, img_size=train_image_size, shuffle=False)



# Define input tensor
input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))

# Load ResNet50 model with pre-trained weights and without the fully connected layers
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

for layer in base_model.layers:
    layer.trainable = True

# Add new layers for the output
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
predictions = tf.keras.layers.Dense(1, activation='relu')(x)


# Define the full model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)


# Freeze the layers of the base model to use pre-trained weights


early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=6, verbose=0, mode='auto'
)

model_name = input("Model name: ") + "_" + datasetname
filepath = "training_2/" +model_name+".hdf5"

mc_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch'
)

tlrp_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', mode='auto'
)

# trainable_False__optimizer_adam-lr=0.000001__activation_relu
# Compile the model

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001), loss='mse', metrics=['mse', 'mae'])


log_dir = "logs/fit/" + model_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#  tensorboard --logdir=C:\Users\Aliha\Desktop\stajj\logs\fit

model.fit(traingen, validation_data=valgen, epochs=10, callbacks=[early_stop_cb, mc_cb, tlrp_cb,tensorboard_callback])


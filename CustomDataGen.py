import tensorflow as tf
import numpy as np
import random
import cv2
import albumentations as A
from sklearn.preprocessing import MinMaxScaler


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, XAll, yAll,
                 img_size=(224, 224),augmentation = 1,
                 shuffle=True):

        self.XAll = XAll
        self.yAll = yAll
        self.img_size = img_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ], p=0.4),
            # A.OneOf([
            #     A.HorizontalFlip(p=0.3),
            #     A.VerticalFlip(p=0.3)  
            # ], p=0.3),
            # A.ShiftScaleRotate(shift_limit=[0.0, 0.0], scale_limit=[0.0, 0.0], rotate_limit=179, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            # A.augmentations.geometric.rotate.Rotate(p=0.8, limit=179, border_mode=cv2.BORDER_CONSTANT),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3)
            ])

        self.n = len(self.XAll)

    def preprocess(self, img):
        resized = cv2.resize(img, self.img_size)
        n_resized = resized / 255.0
        return n_resized

    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.XAll, self.yAll)) # burada data ve labelları birbirine bağlıyor çünkü random shuffle yaptığımızda datalar ve labellar birbirinden ayrılmamalı
            random.shuffle(temp) # random shuffle
            self.XAll, self.yAll = zip(*temp) # shuffle ladığımız zipi açıyoruz
            self.XAll, self.yAll = list(self.XAll), list(self.yAll) # listeye çeviriyor

    def __getitem__(self, index):
        X_All = []
        y_All = []

        X = cv2.imread(self.XAll[index]) # indexe göre datanın directorysini okuyor ve x e kaydediyor
        y = self.yAll[index] # okunan datanın indexinde onun label'ı var onu y ye kaydediyor

        if self.augmentation == 1:
            X = self.preprocess(X)
            X_All.append(X)
            y_All.append(y)
        else:  
            X2 = self.preprocess(X)
            X_All.append(X2)
            y_All.append(y)
            for a in range(self.augmentation):
                X1 = self.transform(image=X) 
                new_img = X1["image"] 
                X_All.append(self.preprocess(new_img)) 
                y_All.append(y)



        X_All = np.array(X_All) 
        y_All = np.array(y_All) 

        return X_All, y_All # data ve label'ı eğitime döndürüyor

    def __len__(self):
        return self.n

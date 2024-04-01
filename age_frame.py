import cv2
import dlib
from time import time
import matplotlib.pyplot as plt
import numpy as np
from age_test import pred


#capture

cap = cv2.VideoCapture(0) # buraya girilen 0 default kameramızı belirtir
#laptopta harici bir kamera takıyor isek 1
#masaüstünde harici kamera takılı ise 0 yazmak zorundayız

counter = 0
acc_age = []

def preprocess(img):
            resized = cv2.resize(img, (224, 224))
            n_resized = resized / 255.0
            n_resized = np.expand_dims(n_resized, axis=0)
            return n_resized

hog_face_detector = dlib.get_frontal_face_detector()
temp_text =""


def hogDetectFaces(image, hog_face_detector,counter, display = True):
    global temp_text
    preprocess(image)

    print(image.shape)

    height, width, _ = image.shape

    output_image = image.copy()

    imgGRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    results = hog_face_detector(imgGRAY, 0)
    
    text_x, text_y = 0, 0

    for bbox in results:    
        
        x1 = bbox.left()
        y1 = bbox.top()
        x2 = bbox.right()
        y2 = bbox.bottom()


        output_image = cv2.rectangle(output_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=width//200)
        
        # cv2.imshow("rectangle image",output_image)
        cropped_image = output_image[y1:y2, x1:x2]
        # cv2.imshow("cropped image",cropped_image)

        cropped_image = preprocess(cropped_image)
        
        age = pred(cropped_image)

        acc_age.append(age)

        if counter % 5 == 0:
            avg_age=sum(acc_age)/len(acc_age)     
            cv2.putText(output_image, str(avg_age), (x2+5, y1-5), fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 1, color = (0,0,255), thickness = 1)
            acc_age.clear
            temp_text = str(avg_age)

        else: 
            cv2.putText(output_image, temp_text, (x2+5, y1-5), fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 1, color = (0,0,255), thickness = 1)
            continue
            
        
    return output_image


while True:
    
    ret, frame = cap.read() # video kameramızdan gelen resimleri frame e ve gelip gelmediğini return e aktaracak
    i=0
    avg_age=0
    
    frame = hogDetectFaces(frame,hog_face_detector,counter)
    counter+=1
    cv2.imshow("output image",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break

    

cap.release() # capture ı serbest bırakalım
cv2.destroyAllWindows()
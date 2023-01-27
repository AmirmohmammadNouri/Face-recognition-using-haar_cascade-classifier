import os
from PIL import Image
import numpy as np
import cv2
import pickle
Base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(Base_dir,'images')
#print(Base_dir)
#print(image_dir)
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
current_id  = 0
label_ids = {}
x_train = []
y_labels = []
for root , dirs , files in os.walk(image_dir):
    for file in files :
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root , file)
            label = os.path.basename(os.path.dirname(path)).replace(' ','-').lower()
            #print(path)
            if  not label in label_ids :
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #print(label_ids)
            #y_labels.append(label)
            #x_train.append(path)
            pil_image = Image.open(path).convert("L")
            size = (550,550)
            final_image = pil_image.resize(size,Image.LANCZOS)
            image_array = np.array(final_image,"uint8")
            #print(image_gray)
            faces = face_cascade.detectMultiScale(image_array , scaleFactor=1.1 , minNeighbors=5)

            for (x,y,w,h ) in faces :
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)
with open ('labels.pickle','wb') as f :
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save('trainer.yml')
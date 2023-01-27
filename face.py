import  cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

labels = {'person_name':1}
with open ('labels.pickle','rb') as f :
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while (True):

    ret,frame = cap.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray , scaleFactor=1.2 , minNeighbors=5)
    for (x,y,w,h) in faces :
        print(x,y,w,h)
        roi_gray = gray[y:y+h , x:x+w]
        roi_color = frame[y:y+h , x:x+w]

        id_ , conf = recognizer.predict(roi_gray)
        if conf >= 65 :# and conf<= 85:
            print(id_)
            print(labels[id_])
            color = (255,255,255)
            name = labels[id_]
            stroke = 2
            cv2.putText(frame,name , (x,y),cv2.FONT_HERSHEY_SIMPLEX,1,color,stroke)
        img_item = "my_image.png"
        cv2.imwrite(img_item , roi_gray)
        width = x+w
        height = y+h
        Rectangle_color = (255,0,0)
        thickness = 5
        cv2.rectangle(frame,(x,y),(width,height),Rectangle_color,thickness)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes :
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey+eh), (0,0,255), 1)
        smile = smile_cascade.detectMultiScale(roi_gray)
        #for (sx, sy, sw, sh) in smile:
            #cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), thickness)

    cv2.imshow('frame',frame)
    #cv2.imshow("grayscaled ",gray)
    if cv2.waitKey(10) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
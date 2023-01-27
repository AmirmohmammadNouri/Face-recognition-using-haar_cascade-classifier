
# face recognition using haar cascade classifier

## What are Haar Cascades?

Haar cascade is an algorithm that can detect objects in images, irrespective of their scale in image and location.

![App Screenshot](https://cdn-images-1.medium.com/max/1750/1*3itGCmU4Q2INsIaQ0x5Hsw.png)

This algorithm is not so complex and can run in real-time. We can train a haar-cascade detector to detect various objects like cars, bikes, buildings, fruits, etc.

Haar cascade uses the cascading window, and it tries to compute features in every window and classify whether it could be an object.
Sample haar features traverse in window-sized across the picture to compute and match features.

it works as a classifier. It classifies positive data points → that are part of our detected object and negative data points → that don’t contain our object.

## Pre-trained Haar Cascades

- Human face detection
- Eye detection
- Nose / Mouth detection
- Vehicle detection

![App Screenshot](https://cdn-images-1.medium.com/max/1750/1*cBpyXGq_I9wIkkCDLeyyOg.png)

Haar cascades are XML files that can be used in OpenCV to detect specified objects.

## Implementing Haar-cascades in OpenCV

If you find your target object haar-cascade available in the pre-trained repository provided by OpenCV, you need to download the pre-trained XML file.

### Installing OpenCV in Python

```bash
!pip install opencv-python
#---OR ---
!pip install opencv-contrib-python
```
### Loading Haar Cascade in OpenCV
```bash
face_detector=cv2.CascadeClassifier(‘haarcascade_frontalface_default.xml’)
eye_dectector = cv2.CascadeClassifier(‘haarcascade_eye.xml’)
```
### Get Results


```bash
results = face_detector.detectMultiScale(gray_img, scaleFactor=1.05,minNeighbors=5,minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
```


### Object Detection in Real-time

We will be using OpenCV video cam feed input to take images in real-time (video)

```bash
import cv2
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_dectector = cv2.CascadeClassifier('haarcascade_eye.xml')
# reading the input image now
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,1.1, 4 )
    for (x,y, w, h) in faces:
    cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  3)
    roi_gray = gray[y:y+h,x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    eyes = eye_dectector.detectMultiScale(roi_gray)
    for (ex,ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 5)
    cv2.imshow("window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```
```bash
frame.release()
```

/home/omid/Desktop/result.png


#### Haar cascade paper : [Documentation](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj3kduvhuj8AhVg_7sIHWFnCpsQFnoECA8QAQ&url=https%3A%2F%2Fwww.cs.cmu.edu%2F~efros%2Fcourses%2FLBMV07%2FPapers%2Fviola-cvpr-01.pdf&usg=AOvVaw27yCB2tUSGu6jhcPRte6HS)

#### opencv docsumentations :https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

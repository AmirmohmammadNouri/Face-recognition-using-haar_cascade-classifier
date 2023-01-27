
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






### Haar cascade paper :
[Documentation](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj3kduvhuj8AhVg_7sIHWFnCpsQFnoECA8QAQ&url=https%3A%2F%2Fwww.cs.cmu.edu%2F~efros%2Fcourses%2FLBMV07%2FPapers%2Fviola-cvpr-01.pdf&usg=AOvVaw27yCB2tUSGu6jhcPRte6HS)


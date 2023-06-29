import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

face_classifier = cv2.CascadeClassifier('./Harcascade/haarcascade_frontalface_default.xml')
classifier = load_model('./Models/model_v_47.hdf5')
class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load the image
image_path = r'C:\Users\SubZero\Documents\GitHub\Face_And_Emotion_Detection\Test_Images\e4.jpeg'
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)
allfaces = []
rects = []
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    allfaces.append(roi_gray)
    rects.append((x, w, y, h))

i = 0
for face in allfaces:
    roi = face.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    preds = classifier.predict(roi)[0]
    label = class_labels[preds.argmax()]
    label_position = (rects[i][0] + int((rects[i][1]/2)), abs(rects[i][2] - 10))
    i += 1
    cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Emotion Detector", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

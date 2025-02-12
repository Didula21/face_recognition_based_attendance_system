from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import numpy as np
import os


video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl','rb') as f:
        LABELS=pickle.load(f)  

with open('data/faces_data.pkl','rb') as f:
        FACES=pickle.load(f)

knn=KNeighborsClassifier(n_neighbors=5) 
knn.fit(FACES,LABELS)


while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert to grayscale and store it in a variable
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        crop_img =frame[y:y+h ,x:x+w, :]
        resized_img = cv2.resize(crop_img,(50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.putText(frame,str(output[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Face Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
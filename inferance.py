import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model('final_model.keras')

# Initialize the face detector
face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start the video capture
source = cv2.VideoCapture(0)  

# Dictionary for labeling predictions
labels_dict = {0: 'NO  MASK', 1: 'MASK'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

while True:
    ret, img = source.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)  
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)
        
        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)

    if key == ord('q'):  
        break

source.release()
cv2.destroyAllWindows()


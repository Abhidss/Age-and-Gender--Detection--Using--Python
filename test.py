import cv2
import numpy as np

# Paths to model files
age_model = 'age_net.caffemodel'
age_proto = 'age_deploy.prototxt'
gender_model = 'gender_net.caffemodel'
gender_proto = 'gender_deploy.prototxt'

# Load pre-trained models
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# Age and gender labels
age_labels = ['(0-5)', '(6-10)', '(11-15)', '(16-20)', '(21-25)','(26-30)', '(31-35)', '(36-40)', '(41-100)']
gender_labels = ['Male', 'Female']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = frame[y:y + h, x:x + w]

        # Prepare input for deep learning models
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_pred = gender_net.forward()
        gender = gender_labels[gender_pred[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_pred = age_net.forward()
        age = age_labels[age_pred[0].argmax()]

        # Draw bounding box and label
        label = f'{gender}, {age}'
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Age and Gender Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
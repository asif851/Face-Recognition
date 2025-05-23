import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

# Load model and class labels
model = load_model("model/face_model.h5")
with open("model/class_indices.json", "r") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

def detect_face_and_predict(image_path):
    img = cv2.imread(image_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return "No face detected", 0.0

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]

        # Resize for MobileNetV2
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        pred = model.predict(face_img)
        predicted_class = np.argmax(pred[0])
        confidence = pred[0][predicted_class]

        if confidence < 0.4:
            return "Not Recognized", confidence

        return labels[predicted_class], confidence

    return "Face not recognized", 0.0

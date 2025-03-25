from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import numpy as np
import pymongo
from datetime import datetime
from PIL import Image

app = Flask(__name__)

# MongoDB Setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["FaceRecognitionDB"]
users_collection = db["Users"]
attendance_collection = db["Attendance"]

# Face Recognition Setup
DATASET_PATH = "dataset"
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

TRAINED_MODEL_PATH = "trainer.yml"
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")  # Load the form

    """Register a new user by capturing their face"""
    user_name = request.form["name"].strip()
    user_id = request.form["user_id"].strip()

    if not user_name or not user_id:
        return "❌ Name and ID cannot be empty."

    # Check if user already exists
    if users_collection.find_one({"user_id": user_id}):
        return "⚠️ User ID already exists!"

    cap = cv2.VideoCapture(0)  # Open laptop camera
    if not cap.isOpened():
        return "❌ Camera error! Please check if another app is using it."

    user_folder = os.path.join(DATASET_PATH, f"{user_id}_{user_name}")
    os.makedirs(user_folder, exist_ok=True)

    count = 0
    while count < 30:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return "❌ Camera error. Check your laptop webcam."

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            img_path = os.path.join(user_folder, f"{count}.jpg")
            cv2.imwrite(img_path, face)
            count += 1

        if count >= 30:
            break

    cap.release()
    users_collection.insert_one({
        "user_id": user_id,
        "name": user_name,
        "registration_date": datetime.now()
    })

    return "✅ User Registered Successfully!"

    

@app.route("/train")
def train_model():
    """Train the face recognition model"""
    faces, labels = [], []
    for folder in os.listdir(DATASET_PATH):
        user_id = folder.split("_")[0]
        folder_path = os.path.join(DATASET_PATH, folder)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            gray_img = Image.open(img_path).convert("L")
            faces.append(np.array(gray_img, "uint8"))
            labels.append(int(user_id))

    if faces:
        recognizer.train(faces, np.array(labels))
        recognizer.save(TRAINED_MODEL_PATH)

    return redirect(url_for("home"))

@app.route("/recognize")
def recognize_faces():
    """Recognize faces and mark attendance"""
    if not os.path.exists(TRAINED_MODEL_PATH):
        return "❌ Train the model first!"

    recognizer.read(TRAINED_MODEL_PATH)
    cap = cv2.VideoCapture(0)  # Open laptop camera

    if not cap.isOpened():
        return "❌ Camera error! Please check if another app is using it."

    recognized_users = set()  # Prevents duplicate entries for the same session

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_resized = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            label, confidence = recognizer.predict(face_resized)

            if confidence < 80:  # Lower confidence threshold for better accuracy
                user = users_collection.find_one({"user_id": str(label)})
                if user:
                    user_name = user["name"]
                    user_id = user["user_id"]

                    if user_id not in recognized_users:
                        recognized_users.add(user_id)  # Prevents multiple entries

                        attendance_collection.insert_one({
                            "user_id": user_id,
                            "name": user_name,
                            "entry_time": datetime.now(),
                            "exit_time": None
                        })

                    print(f"✅ Recognized: {user_name} (Confidence: {confidence:.2f})")

                    color = (0, 255, 0)  # Green for recognized face
                    text = f"{user_name} ({confidence:.2f})"
                else:
                    color = (0, 0, 255)  # Red for unknown face
                    text = "Unknown"
            else:
                color = (0, 0, 255)
                text = "Unknown"

            # Draw rectangle around face and label it
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show the camera feed
        cv2.imshow("Face Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for("home"))

@app.route("/attendance")
def attendance():
    records = list(attendance_collection.find())
    return render_template("attendance.html", records=records)

if __name__ == "__main__":
    app.run(debug=True)

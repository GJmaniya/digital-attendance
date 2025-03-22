import cv2
import os
import numpy as np
import pymongo
from datetime import datetime
from PIL import Image

# MongoDB Connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["FaceRecognitionDB"]
users_collection = db["Users"]

# Create directories if they don't exist
DATASET_PATH = "dataset"
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

TRAINED_MODEL_PATH = "trainer.yml"

# Initialize OpenCV Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Use iPhone camera via DroidCam (replace <IP> with your iPhone's IP)
DROIDCAM_URL = "http://192.168.1.6:5000/video"
cap = cv2.VideoCapture(DROIDCAM_URL)

# Set camera properties for high FPS
cap.set(cv2.CAP_PROP_FPS, 180)  # Set FPS to 60
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPEG codec
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Reduce resolution for higher FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def register_user():
    """Register a new user and capture their face images."""
    user_name = input("Enter your name: ").strip()
    user_id = input("Enter your unique ID: ").strip()

    if not user_name or not user_id:
        print("❌ Invalid input. Name and ID cannot be empty.")
        return

    # Check if user already exists
    existing_user = users_collection.find_one({"user_id": user_id})
    if existing_user:
        print("⚠️ User ID already exists! Please use a unique ID.")
        return

    user_folder = os.path.join(DATASET_PATH, f"{user_id}_{user_name}")
    os.makedirs(user_folder, exist_ok=True)

    print(f"📸 Capturing images for {user_name}. Look at the camera...")

    count = 0
    while count < 30:  # Capture 30 images
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image. Check your camera.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (200, 200))  # Standardize image size
            img_path = os.path.join(user_folder, f"{count}.jpg")
            cv2.imwrite(img_path, face_resized)
            count += 1
            print(f"✅ Image {count}/30 captured.")

        if count >= 30:
            break

    # Save user details in MongoDB
    users_collection.insert_one({
        "user_id": user_id,
        "name": user_name,
        "registration_date": datetime.now()
    })

    print(f"✅ User {user_name} registered successfully!")


def train_model():
    """Train the face recognition model using LBPH."""
    faces = []
    labels = []

    print("\n🔍 Training Model...")

    for folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder)

        # Ensure folder name follows the expected format
        parts = folder.split("_")
        if len(parts) < 2:
            print(f"❌ Skipping invalid folder name: {folder}")
            continue

        user_id = parts[0]  # Extract user ID

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                gray_img = Image.open(img_path).convert("L")  # Convert to grayscale
                img_array = np.array(gray_img, "uint8")

                # 🚀 Debugging statement to verify labels
                print(f"📂 Training on {img_name} with label {user_id}")

                faces.append(img_array)
                labels.append(int(user_id))
            except Exception as e:
                print(f"⚠️ Skipping corrupt image {img_name}: {e}")

    # Debugging - Check training data
    print(f"✅ Total faces trained: {len(faces)}, Unique users: {len(set(labels))}")

    if len(faces) == 0:
        print("❌ No faces found for training.")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINED_MODEL_PATH)
    print(f"✅ Model trained successfully and saved as '{TRAINED_MODEL_PATH}'")


def recognize_faces():
    """Recognize faces using the trained model."""
    if not os.path.exists(TRAINED_MODEL_PATH):
        print("❌ No trained model found. Train the model first!")
        return

    recognizer.read(TRAINED_MODEL_PATH)
    print("🎭 Face recognition started at 60 FPS. Press 'q' to quit.")

    frame_count = 0  # To process alternate frames

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Process face detection every 2 frames to reduce CPU load
        if frame_count % 2 == 0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (200, 200))
                label, confidence = recognizer.predict(face_resized)

                print(f"🔍 Detected: ID {label}, Confidence: {confidence:.2f}%")  # Debugging

                if confidence < 50:
                    user = users_collection.find_one({"user_id": str(label)})
                    user_name = user["name"] if user else "Unknown"
                else:
                    user_name = "Unknown"

                # Display the result
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{user_name} ({confidence:.2f}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        frame_count += 1

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("📴 Face recognition stopped.")
    cv2.destroyAllWindows()

def main():
    """Main menu for the face recognition system."""
    while True:
        print("\n🎭 Face Recognition System")
        print("1️⃣ Register User")
        print("2️⃣ Train Model")
        print("3️⃣ Recognize Faces")
        print("4️⃣ Exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            register_user()
        elif choice == "2":
            train_model()
        elif choice == "3":
            recognize_faces()
        elif choice == "4":
            print("📴 Exiting...")
            break
        else:
            print("❌ Invalid option! Please select a valid number.")


if __name__ == "__main__":
    main()
    

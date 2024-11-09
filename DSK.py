import torch
import cv2
import numpy as np
from PIL import Image
import pygame  # For alarm sound
import pyttsx3  # For text-to-speech
from facenet_pytorch import MTCNN, InceptionResnetV1
import mediapipe as mp
import tkinter as tk  # For pop-up message

# Initialize pygame for alarm sound
pygame.mixer.init()
pygame.mixer.music.load(r"C:\Users\ACER\Downloads\loud-emergency-alarm-54635.mp3")  # Ensure this file exists
pygame.mixer.music.set_volume(0.8)  # Set alarm volume to 80% q

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('volume', 1.0)  # Set volume to maximum (1.0)

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)
# Load pre-trained InceptionResnetV1 model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize MediaPipe for face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Constants for drowsiness detection
EAR_THRESHOLD = 0.25  # Eye aspect ratio threshold
CONSECUTIVE_FRAMES = 10  # Consecutive frames to trigger alarm
frame_count = 0  # Initialize frame count
alarm_on = False  # To keep track of alarm state

# Function to get face embeddingsq
def get_embedding(image):
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return None
    faces = mtcnn(image)
    embeddings = model(faces)
    return embeddings

# Function to register a face
def register_face(image_path, name, known_faces):
    image = Image.open(image_path)
    embedding = get_embedding(image)
    if embedding is not None:
        known_faces[name] = embedding[0]
        print(f"Registered {name}'s face.")
    else:
        print(f"No face detected in the image for {name}.")

# Function to verify a face
def verify_face(unknown_embedding, known_faces):
    for name, known_embedding in known_faces.items():
        distance = torch.norm(known_embedding - unknown_embedding[0]).item()
        print(f"Distance to {name}: {distance}")

        threshold = 0.6
        if distance < threshold:
            return name  
    return None  

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance
    B = np.linalg.norm(eye[2] - eye[4])  # Vertical distance
    C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Function to play alarm sound
def play_alarm():
    global alarm_on
    if not alarm_on:
        print("Playing alarm sound...")  # Debugging output
        pygame.mixer.music.play(-1)  # Play the alarm in a loop
        alarm_on = True
        speak_message("Warning! You are drowsy. Please take a break.")

# Function to stop alarm sound
def stop_alarm():
    global alarm_on
    if alarm_on:
        print("Stopping alarm sound...")  # Debugging output
        pygame.mixer.music.stop()  # Stop the alarm sound
        alarm_on = False
        speak_message("You are driving well. Stay alert!")

# Function to speak messages
def speak_message(message):
    pygame.mixer.music.stop()  # Stop alarm before speaking
    engine.say(message)
    engine.runAndWait()
    if alarm_on:  # Restart alarm if it's still on
        pygame.mixer.music.play(-1)

# Function to display a pop-up message indicating the driver is back to normal
def driver_normal_popup():
    root = tk.Tk()
    root.title("Driver Status")
    label = tk.Label(root, text="Driver is back to normal!", padx=20, pady=10)
    label.pack()
    ok_button = tk.Button(root, text="OK", command=root.destroy, padx=20, pady=5)
    ok_button.pack()
    root.mainloop()

# Initialize known faces dictionary
known_faces = {}
# Register faces
register_face(r"C:\Users\ACER\Downloads\lfw-funneled\lfw_funneled\Peter_Struck\Peter_Struck_0001.jpg", "Peter_Struck", known_faces)
register_face(r"C:\Users\ACER\Downloads\lfw-funneled\lfw_funneled\Will_Smith\Will_Smith_0001.jpg", "Will_Smith", known_faces)
register_face(r"C:\Users\ACER\Downloads\photo_6305232580504568782_y.jpg", "Tejas Alte", known_faces)
register_face(r"C:\Users\ACER\Downloads\photo_6305232580504568781_y.jpg", "Nahid Ansari", known_faces)

# Start video capture
cap = cv2.VideoCapture(0)  # Change to 1 if it doesn't work

# Check if the camera opened correctly
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()                      

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read the frame from the camera.")
        break

    # Convert frame to RGB for face detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    unknown_embedding = get_embedding(pil_image)

    # Drowsiness detection
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmarks for eyes
            left_eye_indices = [33, 160, 158, 133, 153, 144]  # Indices for the left eye
            right_eye_indices = [362, 385, 387, 263, 373, 380]  # Indices for the right eye

            left_eye = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y] for i in left_eye_indices])
            right_eye = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y] for i in right_eye_indices])

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Check if EAR is below the threshold
            if ear < EAR_THRESHOLD:
                frame_count += 1
                if frame_count >= CONSECUTIVE_FRAMES:
                    play_alarm()  # Play alarm if drowsy for consecutive frames
                    cv2.putText(frame, "Drowsy!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                if alarm_on:
                    stop_alarm()  # Stop alarm when no drowsiness is detected
                    driver_normal_popup()  # Show popup when driver is back to normal
                frame_count = 0  # Reset count if not drowsy

    # Face verification
    if unknown_embedding is not None:
        recognized_name = verify_face(unknown_embedding, known_faces)
        if recognized_name:
            cv2.putText(frame, f"Recognized: {recognized_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not Recognized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Driver Monitoring System', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
pygame.quit()
  

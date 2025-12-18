import cv2
import time
import pandas as pd
import os
from utils import blink_utils  # Make sure you have __init__.py inside utils folder
import mediapipe as mp

# Ask for user input
user_id = input("Enter user name or ID: ")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Setup Camera
cap = cv2.VideoCapture(0)

# Calibrate EAR for this user
EAR_THRESHOLD = blink_utils.calibrate_threshold(cap, face_mesh)

# Save EAR threshold for future use
blink_utils.save_threshold(user_id, EAR_THRESHOLD)

# Reset camera after calibration
cap.release()
time.sleep(1)
cap = cv2.VideoCapture(0)

# Variables for blink tracking
blink_id = 0
blink_data = []
blink_start = None
blink_active = False

print("Starting Blink Detection. Press ESC to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = blink_utils.get_ear(landmarks, blink_utils.LEFT_EYE_IDX, w, h)
        right_ear = blink_utils.get_ear(landmarks, blink_utils.RIGHT_EYE_IDX, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if avg_ear < EAR_THRESHOLD and not blink_active:
            blink_start = time.time()
            blink_active = True

        elif avg_ear >= EAR_THRESHOLD and blink_active:
            blink_duration = time.time() - blink_start
            blink_id += 1
            duration_ms = round(blink_duration * 1000)
            blink_data.append([user_id, blink_id, duration_ms, round(avg_ear, 3)])
            print(f"Blink {blink_id}: Duration {duration_ms} ms - EAR: {round(avg_ear,3)}")
            blink_active = False

    cv2.imshow("Blink Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Save blink data
df = pd.DataFrame(blink_data, columns=["User_ID", "Blink_ID", "Duration_ms", "EAR"])
blink_utils.save_blink_data(df)

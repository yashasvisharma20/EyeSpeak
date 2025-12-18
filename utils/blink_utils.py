import numpy as np
import cv2
import time
import os
import pandas as pd
import mediapipe as mp
import json
import joblib

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_ear(landmarks, eye_indices, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    a = euclidean_distance(points[1], points[5])
    b = euclidean_distance(points[2], points[4])
    c = euclidean_distance(points[0], points[3])
    return round((a + b) / (2.0 * c), 3)

def calibrate_threshold(cap, face_mesh, duration=5):
    print("Calibration started. Keep your eyes open for 5 seconds...")
    ear_list = []
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = get_ear(landmarks, LEFT_EYE_IDX, w, h)
            right_ear = get_ear(landmarks, RIGHT_EYE_IDX, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_list.append(avg_ear)

        cv2.putText(frame, "Calibrating... Keep eyes open", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    if ear_list:
        mean_ear = np.mean(ear_list)
        threshold = round(mean_ear * 0.75, 3)
        print(f"Calibration complete. Mean EAR: {round(mean_ear, 3)} | EAR Threshold: {threshold}")
        return threshold
    else:
        print("Calibration failed. Using default EAR threshold.")
        return 0.22

def save_blink_data(df, file_path="blink_dataset.csv"):
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)
    print(f"Blink data saved to {file_path}")

def save_user_history(df, file_path="user_blink_history.csv"):
    """
    Saves user-specific blink history for the application log.
    """
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)
    print(f"User blink history saved to {file_path}")

def save_threshold(user_id, threshold, file_path="thresholds.json"):
    thresholds = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            thresholds = json.load(f)
    thresholds[user_id] = threshold
    with open(file_path, "w") as f:
        json.dump(thresholds, f)

def load_threshold(user_id, file_path="thresholds.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            thresholds = json.load(f)
        return thresholds.get(user_id, 0.22)
    return 0.22

morse_dict = {
    '....': 'Help',
    '..-': 'Urgent',
    '---': 'Pain',
    '-.': 'Nurse',
    '..': 'I\'m okay',
    '-': 'Yes',
    '.': 'No',
    '/-': 'Clear',
    '//': 'Speak',
    '/': ' ',
    '': ''
}

def load_ml_model(model_path='blink_model.pkl'):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    print(f"Warning: Model not found at {model_path}. Please train the model first.")
    return None

def load_label_encoder(encoder_path='label_encoder.pkl'):
    if os.path.exists(encoder_path):
        le = joblib.load(encoder_path)
        return le
    print(f"Warning: Label encoder not found at {encoder_path}. Cannot predict labels.")
    return None

def predict_blink(model, le, blink_duration, ear):
    if model and le:
        features = np.array([[blink_duration, ear]])
        prediction = model.predict(features)
        label = le.inverse_transform(prediction)
        return label[0]
    return '.' if blink_duration < 400 else '-'

__all__ = [
    'mp_face_mesh', 'calibrate_threshold',
    'get_ear', 'LEFT_EYE_IDX', 'RIGHT_EYE_IDX',
    'save_blink_data', 'save_user_history', 'morse_dict',
    'save_threshold', 'load_threshold',
    'load_ml_model', 'load_label_encoder', 'predict_blink'
]
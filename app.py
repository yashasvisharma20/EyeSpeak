from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import time
import numpy as np
from utils import blink_utils
import threading
import pandas as pd
import joblib
import uuid
import csv

app = Flask(__name__)

# Global variables and thread lock
translated_text = ''
morse_code = ''
blink_active = False
blink_start = None
last_blink_time = time.time()
PAUSE_DURATION = 2.0  # seconds
EAR_THRESHOLD = 0.22  # default fallback
streaming = False
current_user = "default"
face_mesh = None
last_blink_symbol = ''
frame_data = b''
frame_lock = threading.Lock()
video_thread = None
is_collecting_data = False # New flag for data collection

# Morse Dictionary (from blink_utils)
morse_dict = blink_utils.morse_dict

# Load ML model and label encoder once
blink_model = blink_utils.load_ml_model()
label_encoder = blink_utils.load_label_encoder()

def video_processing_thread():
    global translated_text, morse_code, blink_active, blink_start, last_blink_time, streaming, face_mesh, last_blink_symbol, frame_data, frame_lock, is_collecting_data

    cap = cv2.VideoCapture(0)
    if face_mesh is None:
        face_mesh = blink_utils.mp_face_mesh.mp_face_mesh.FaceMesh(refine_landmarks=True)

    # Prepare a separate file for training data if collection is enabled
    training_file = 'blink_training_data.csv'
    if is_collecting_data:
        # Create a new file or append to an existing one with headers
        with open(training_file, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write header only if the file is new
            if f.tell() == 0:
                writer.writerow(['User_ID', 'Timestamp', 'Duration_ms', 'EAR'])
        print(f"Starting data collection for training. Data will be saved to {training_file}")

    while streaming:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = blink_utils.get_ear(landmarks, blink_utils.LEFT_EYE_IDX, w, h)
            right_ear = blink_utils.get_ear(landmarks, blink_utils.RIGHT_EYE_IDX, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD and not blink_active:
                blink_start = time.time()
                blink_active = True
            elif avg_ear >= EAR_THRESHOLD and blink_active:
                blink_duration = (time.time() - blink_start) * 1000
                blink_active = False

                symbol = blink_utils.predict_blink(blink_model, label_encoder, blink_duration, avg_ear)

                morse_code += symbol
                last_blink_symbol = symbol
                last_blink_time = time.time()

                # Always save to user history
                blink_utils.save_user_history(
                    pd.DataFrame([[current_user, int(time.time()), round(blink_duration, 2), round(avg_ear, 3)]],
                                 columns=["User_ID", "Blink_ID", "Duration_ms", "EAR"])
                )

                # Save to training data if flag is set
                if is_collecting_data:
                    with open(training_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([current_user, time.time(), round(blink_duration, 2), round(avg_ear, 3)])
        
        if morse_code and time.time() - last_blink_time > PAUSE_DURATION:
            translated_text += morse_dict.get(morse_code.strip(), '?')
            morse_code = ''

        with frame_lock:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()

    cap.release()

@app.route('/')
def index():
    return render_template('index.html', morse_dict=morse_dict)

@app.route('/start/<user_id>')
def start(user_id):
    global streaming, current_user, video_thread
    if not streaming:
        current_user = user_id
        streaming = True
        video_thread = threading.Thread(target=video_processing_thread)
        video_thread.daemon = True
        video_thread.start()
    return Response(stream_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop():
    global streaming
    streaming = False
    return "Stopped"

def stream_frames():
    global frame_data, frame_lock
    while streaming:
        with frame_lock:
            if frame_data:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.03)

@app.route('/calibrate')
def do_calibration():
    global EAR_THRESHOLD, face_mesh
    if face_mesh is None:
        face_mesh = blink_utils.mp_face_mesh.FaceMesh(refine_landmarks=True)

    new_user_id = str(uuid.uuid4()).split('-')[0]
    cap = cv2.VideoCapture(0)
    EAR_THRESHOLD = blink_utils.calibrate_threshold(cap, face_mesh)
    blink_utils.save_threshold(new_user_id, EAR_THRESHOLD)
    return jsonify({"threshold": EAR_THRESHOLD, "user_id": new_user_id})

@app.route('/text')
def get_text():
    global translated_text, morse_code

    if morse_code == '/-':
        translated_text = ''
        morse_code = ''
        return jsonify({"text": "Cleared"})
    if morse_code == '//':
        text_to_speak = translated_text
        translated_text = ''
        morse_code = ''
        return jsonify({"text": "Speaking...", "speak_now": text_to_speak})

    return jsonify({"text": translated_text})

@app.route('/blink-status')
def blink_status():
    global last_blink_symbol
    symbol = last_blink_symbol
    last_blink_symbol = ''
    return jsonify({"symbol": symbol})

@app.route('/clear')
def clear():
    global translated_text, morse_code
    translated_text = ''
    morse_code = ''
    return jsonify({"status": "cleared"})

# NEW: Route to start data collection
@app.route('/start_data_collection')
def start_data_collection():
    global is_collecting_data, current_user, streaming, video_thread
    is_collecting_data = True
    if not streaming:
        current_user = "data_collector_" + str(uuid.uuid4()).split('-')[0]
        streaming = True
        video_thread = threading.Thread(target=video_processing_thread)
        video_thread.daemon = True
        video_thread.start()
    return jsonify({"status": "Data collection started."})

# NEW: Route to stop data collection
@app.route('/stop_data_collection')
def stop_data_collection():
    global is_collecting_data
    is_collecting_data = False
    return jsonify({"status": "Data collection stopped."})


@app.route('/dashboard')
def dashboard():
    try:
        df = pd.read_csv('user_blink_history.csv')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Create an empty DataFrame with the correct headers if the file is missing or empty
        df = pd.DataFrame(columns=["User_ID", "Blink_ID", "Duration_ms", "EAR"])
    
    # Check if 'User_ID' column exists, otherwise create it as an empty series
    if 'User_ID' in df.columns:
        users = df['User_ID'].unique().tolist()
    else:
        users = []
    
    return render_template('dashboard.html', users=users)

# NEW: API endpoint for fetching specific user data
@app.route('/dashboard_data/<user_id>')
def dashboard_data(user_id):
    try:
        df = pd.read_csv('user_blink_history.csv')
        if user_id != 'all':
            df = df[df['User_ID'] == user_id]
    except Exception:
        return jsonify([])

    # Convert dataframe to a list of dictionaries for JSON response
    data = df.to_dict(orient='records')
    return jsonify(data)

@app.route('/download')
def download():
    return send_file("user_blink_history.csv", as_attachment=True)

@app.route('/explore')
def explore():
    return render_template('explore.html')

if __name__ == '__main__':
    print("Initializing Flask app...")
    try:
        print("Loading machine learning model...")
        blink_model = blink_utils.load_ml_model()
        label_encoder = blink_utils.load_label_encoder()
        if blink_model and label_encoder:
            print("ML model and label encoder loaded successfully.")
        else:
            print("WARNING: ML model files not found. The app will fall back to using hardcoded thresholds.")
    except Exception as e:
        print(f"ERROR: Failed to load ML model files. Please check if `blink_model.pkl` and `label_encoder.pkl` exist. Error: {e}")

    print("\nStarting Flask App on http://127.0.0.1:5000 ...")
    app.run(debug=True, threaded=True, port=5000, use_reloader=False)
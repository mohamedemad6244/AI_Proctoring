# ========================= IMPORTS =========================
import os
from datetime import datetime
import cv2
import numpy as np
import mediapipe as mp
import tensorflow_hub as hub
from ultralytics import YOLO
import soundfile as sf
import io
import requests

# ========================= CONFIGURATION & CONSTANTS =========================

IMAGE_DIR = "cheating_evidence"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Cheating thresholds
YAW_THRESHOLD = 0.22
PITCH_THRESHOLD = 0.18

# Audio thresholds
AUDIO_WHISPERING_CONF = 0.35
AUDIO_TALK_CONF = 0.45
AUDIO_MEDIA_CONF = 0.50

# Audio classes
WHISPERING = {"Whispering"}
MULTI_TALK = {"Conversation", "Chatter", "Speech"}
VIDEO_AUDIO = {"Music", "Singing", "Narration", "Television", "Radio"}

# ========================= MODEL INITIALIZATION =========================


YOLO_LOCAL_PATH = "yolo11l.pt"

YOLO_GITHUB_URL = "https://github.com/mohamedemad6244/AI_Proctoring/releases/download/yoloV.1/yolo11l.pt"
# محاولة تحميل النموذج
try:
    if os.path.exists(YOLO_LOCAL_PATH):
        print("Loading YOLO model from local file...")
        yolo_model = YOLO(YOLO_LOCAL_PATH)
    else:
        raise FileNotFoundError(f"{YOLO_LOCAL_PATH} not found")
except Exception as e:
    print(f"Local model failed: {e}")
    print("Downloading YOLO model from GitHub Release...")
    r = requests.get(YOLO_GITHUB_URL, stream=True)
    with open(YOLO_LOCAL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("YOLO model downloaded, loading now...")
    try:
        yolo_model = YOLO(YOLO_LOCAL_PATH)
    except Exception as e2:
        print(f"Failed to load YOLO model even after download: {e2}")
        yolo_model = None

# YAMNet Model (Audio)
try:
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = yamnet_model.class_map_path().numpy()
    audio_class_names = open(class_map_path, 'r').read().splitlines()
except Exception as e:
    print(f"Error loading YAMNet model: {e}")
    yamnet_model = None

# Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6
)

# ========================= CORE HELPERS =========================

def get_head_ratios(landmarks):
    """Return head yaw and pitch ratios from landmarks."""
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose = landmarks[1]
    eye_width = abs(right_eye.x - left_eye.x) + 1e-6
    dx = (nose.x - (left_eye.x + right_eye.x)/2) / eye_width
    dy = (nose.y - (left_eye.y + right_eye.y)/2) / eye_width
    return dx, dy

# ========================= API PROCESS FUNCTIONS =========================

def process_frame_for_api(frame):
    """
    Process a single frame (image) for cheating detection.
    Returns a dict suitable for JSON response.
    """
    frame_copy = frame.copy()
    cheating_alert = False
    alert_type = None
    details = ""

    # Mediapipe FaceMesh check
    rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        dx, dy = get_head_ratios(landmarks)
        if abs(dx) > YAW_THRESHOLD or abs(dy) > PITCH_THRESHOLD:
            cheating_alert = True
            alert_type = "EYE/HEAD"
            details = f"HeadYaw={dx:.2f}, Pitch={dy:.2f}"

    # YOLO Object Detection check
    if yolo_model:
        yolo_results = yolo_model.predict(frame_copy, conf=0.50, classes=[0,67,73], verbose=False)
        for r in yolo_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = yolo_model.names[cls_id]
                if cls_name != "person":
                    cheating_alert = True
                    alert_type = "YOLO"
                    details = cls_name

    # Save frame if cheating detected
    screenshot_path = None
    if cheating_alert:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S_%f")
        img_filename = f"{alert_type}{timestamp}.jpg" if alert_type else f"frame{timestamp}.jpg"
        screenshot_path = os.path.join(IMAGE_DIR, img_filename)
        cv2.imwrite(screenshot_path, frame_copy)

    return {
        "cheating": cheating_alert,
        "type": alert_type,
        "details": details,
        "image_path": screenshot_path
    }

def process_audio_for_api(audio_bytes):
    """
    Process audio bytes for cheating detection.
    Returns a dict suitable for JSON response.
    """
    if not yamnet_model:
        return {
            "cheating": False,
            "label": None,
            "confidence": 0.0
        }

    audio, sr = sf.read(io.BytesIO(audio_bytes))
    scores, embeddings, spectrogram = yamnet_model(audio)
    scores_np = scores.numpy().mean(axis=0)
    top_idx = scores_np.argmax()
    label = audio_class_names[top_idx]
    confidence = float(scores_np[top_idx])
    cheating_detected = False
    if label in WHISPERING.union(MULTI_TALK, VIDEO_AUDIO) and confidence > AUDIO_MEDIA_CONF:
        cheating_detected = True

    return {
        "cheating": cheating_detected,
        "label": label,
        "confidence": confidence
    }

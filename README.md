# AI_Proctoring

This project provides a Flask-based API for AI-powered exam proctoring. It uses:

- **Mediapipe FaceMesh** to monitor head and eye movement.
- **YOLO Object Detection** to detect unauthorized objects (e.g., phones, books).
- **YAMNet Audio Classification** to detect whispering, conversation, or media audio.

### Features

- Real-time image and audio analysis.
- Returns JSON response for frontend integration.
- Can generate alerts and save frames for cheating evidence.
- Compatible with deployment on cloud platforms like Railway.

### Usage

- `POST /upload_frame` → send an image frame, get JSON detection.
- `POST /upload_audio` → send audio file, get JSON detection.

### Requirements

- Python 3.10+
- OpenCV, Mediapipe, Ultralytics YOLO, TensorFlow Hub, SoundFile

### Deployment

- Designed to deploy easily on cloud platforms (e.g., Railway) with API endpoints accessible online.

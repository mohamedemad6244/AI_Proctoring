from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

from proctor_system import process_frame_for_api, process_audio_for_api

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "AI Proctoring Flask API running"})

@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    if "frame" not in request.files:
        return jsonify({"error": "No frame uploaded"}), 400
    
    file = request.files["frame"]
    contents = file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = process_frame_for_api(frame)
    return jsonify(result)

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400
    
    file = request.files["audio"]
    audio_bytes = file.read()

    result = process_audio_for_api(audio_bytes)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)


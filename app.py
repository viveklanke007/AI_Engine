from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import traceback
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv() # Load environment variables from .env file

app = Flask(__name__)
CORS(app)

# Initialize face detection using OpenCV's Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "Attendance System AI Engine",
        "status": "online",
        "message": "Welcome to the AI Face Recognition Engine API"
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "ai-engine",
        "version": "1.1.0",
        "face_detection": "OpenCV Haar Cascade",
        "feature_extraction": "ORB-1024"
    })

@app.route('/detect-face', methods=['POST'])
def detect_face():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
        image = Image.open(BytesIO(image_data))
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            image_area = gray.shape[0] * gray.shape[1]
            face_area = w * h
            confidence = min(0.99, (face_area / image_area) * 10)
            return jsonify({
                "face_detected": True,
                "confidence": float(confidence),
                "num_faces": len(faces),
                "face_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
            })
        else:
            return jsonify({"face_detected": False, "confidence": 0.0, "num_faces": 0})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/extract-embedding', methods=['POST'])
def extract_embedding():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
        image = Image.open(BytesIO(image_data))
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return jsonify({"success": False, "error": "No face detected"}), 400
        
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (128, 128))
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(face_resized, None)
        
        if descriptors is None:
            embedding = cv2.calcHist([face_resized], [0], None, [256], [0, 256]).flatten().tolist()
        else:
            embedding = descriptors.flatten().tolist()
            
        # Standardize to 1024
        if len(embedding) < 1024:
            embedding.extend([0] * (1024 - len(embedding)))
        else:
            embedding = embedding[:1024]
        
        return jsonify({
            "success": True, 
            "embedding": embedding,
            "embedding_size": len(embedding),
            "num_keypoints": len(keypoints) if keypoints else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/compare-faces', methods=['POST'])
def compare_faces():
    try:
        data = request.get_json()
        if not data or 'embedding1' not in data or 'embedding2' not in data:
            return jsonify({"error": "Two embeddings required"}), 400
        
        # Super-Robust Normalization
        def force_1024(emb):
            if emb is None:
                return np.zeros(1024)
            arr = np.array(emb).flatten()
            if arr.size == 0:
                return np.zeros(1024)
            if arr.size < 1024:
                return np.pad(arr, (0, 1024 - arr.size), 'constant')
            return arr[:1024]

        emb1 = force_1024(data['embedding1'])
        emb2 = force_1024(data['embedding2'])
        
        # Debug Output
        logger.info(f"Vector Shapes - Stored: {emb1.shape}, Live: {emb2.shape}")
        
        # Critical Alignment Check
        if emb1.shape != emb2.shape:
            # This should NEVER happen now with force_1024, but adding as a safety net
            return jsonify({"error": f"Internal Alignment Failure: {emb1.shape} vs {emb2.shape}"}), 500

        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        
        if n1 == 0 or n2 == 0:
            similarity = 0.0
        else:
            similarity = np.dot(emb1, emb2) / (n1 * n2)
            
        threshold = 0.60
        is_match = bool(similarity > threshold)
        
        return jsonify({
            "similarity": float(similarity),
            "match": is_match,
            "threshold": threshold,
            "confidence": float(similarity * 100)
        })
    except Exception as e:
        logger.error(f"Comparison Crash: {traceback.format_exc()}")
        return jsonify({"error": f"AI Comparison Failed: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"AI Engine starting on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)

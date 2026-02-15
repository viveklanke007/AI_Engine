# AI Engine - Face Recognition

This service provides face detection and recognition capabilities using OpenCV and ORB descriptors.

## 🚀 Features
- **Face Detection**: Fast Haar Cascade detection.
- **Feature Extraction**: ORB descriptors (1024D vectors).
- **Comparison**: Cosine similarity matching.
- **Production Ready**: Optimized Docker image, Gunicorn server, and comprehensive logging.

## 📡 API Endpoints

### 1. Root / Health Check
- `GET /`: Basic service info.
- `GET /health`: Detailed status and versioning.

### 2. Detect Face
- `POST /detect-face`: Returns if a face is present and its coordinates.
- **Input**: `{ "image": "base64_string" }`

### 3. Extract Embedding
- `POST /extract-embedding`: Returns a 1024D feature vector for a detected face.
- **Input**: `{ "image": "base64_string" }`

### 4. Compare Faces
- `POST /compare-faces`: Compares two embeddings and returns similarity.
- **Input**: `{ "embedding1": [], "embedding2": [] }`

## 🛠️ Deployment on Railway

1. **Dockerfile**: The project includes a production-ready `Dockerfile`.
2. **Environment Variables**:
   - `PORT`: Set by Railway (default 5001).
3. **Dependencies**: automatically installed via `requirements.txt`.

## 🔬 Technical Details
- **Face Detection**: OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`).
- **Embedding**: ORB descriptors flattened and padded/truncated to 1024 dimensions.
- **Matching Algorithm**: Cosine Similarity.
- **Threshold**: default 0.60.

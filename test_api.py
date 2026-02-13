# Test AI Engine

import requests
import base64
import json
import os
from dotenv import load_dotenv

load_dotenv()



# Test 1: Health Check
print("🧪 Test 1: Health Check")

BASE_URL = os.getenv('AI_ENGINE_URL', 'http://localhost:5001')
print(f"Testing against: {BASE_URL}")

try:
    response = requests.get(f'{BASE_URL}/health')
    print(f"✅ Status: {response.status_code}")
    print(f"📄 Response: {response.json()}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*50 + "\n")

# Test 2: Face Detection (you'll need to add an actual image)
print("🧪 Test 2: Face Detection")
print("ℹ️  To test face detection, you need to:")
print("   1. Take a photo with your webcam")
print("   2. Convert it to base64")
print("   3. Send it to /detect-face endpoint")
print("\n   Example code:")
print(f"""
   import cv2
   import base64
   
   # Capture from webcam
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   cap.release()
   
   # Convert to base64
   _, buffer = cv2.imencode('.jpg', frame)
   img_base64 = base64.b64encode(buffer).decode('utf-8')
   
   # Send to API
   response = requests.post(f'{BASE_URL}/detect-face', 
                           json={'image': img_base64})
   print(response.json())
""")

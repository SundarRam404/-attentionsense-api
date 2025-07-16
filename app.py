# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ NEW
import cv2
import numpy as np
import base64
from attention_core import analyze_attention

app = Flask(__name__)
CORS(app)  # ✅ ENABLE CROSS-ORIGIN REQUESTS

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        image_data = data.get('image')
        print("Got image data:", image_data[:50] if image_data else "None")
        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        # Convert base64 to numpy image
        img_bytes = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Analyze attention
        result = analyze_attention(frame)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
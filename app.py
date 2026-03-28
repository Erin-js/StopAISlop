from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io, base64, traceback
from model import predict_with_gradcam   # ← ganti dari predict

app = Flask(__name__)
CORS(app)

MAX_SIZE_MB = 5

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        if "image" in request.files:
            file = request.files["image"]
            image = Image.open(file.stream).convert("RGB")

        elif request.json and "image_base64" in request.json:
            data = request.json["image_base64"]
            if "," in data:
                data = data.split(",")[1]
            image_bytes = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        else:
            return jsonify({"error": "No image provided"}), 400

        result = predict_with_gradcam(image)   # ← ganti dari predict
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    import torch
    return jsonify({
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
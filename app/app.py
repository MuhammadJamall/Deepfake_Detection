
from flask import Flask, render_template, request, jsonify
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models import create_model

app = Flask(__name__)

# Load models at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = PROJECT_ROOT / "results" / "models"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def load_models():
    print("Loading models...")
    efficientnet = create_model("efficientnet_b4", num_classes=6, pretrained=False)
    xception = create_model("xception", num_classes=6, pretrained=False)

    efficientnet_path = MODEL_DIR / "efficientnet_b4_best.pth"
    xception_path = MODEL_DIR / "xception_best.pth"

    efficientnet.load_state_dict(torch.load(efficientnet_path, map_location=device))
    xception.load_state_dict(torch.load(xception_path, map_location=device))

    efficientnet = efficientnet.to(device).eval()
    xception = xception.to(device).eval()
    print("Models loaded.")
    return efficientnet, xception


def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
    return image_tensor.to(device)


efficientnet, xception = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_tensor = preprocess_image(file.read())
    except UnidentifiedImageError:
        return jsonify({'error': 'Invalid image file'}), 400

    with torch.no_grad():
        eff_probs = torch.softmax(efficientnet(image_tensor), dim=1)
        xep_probs = torch.softmax(xception(image_tensor), dim=1)

    # Ensemble (average)
    ensemble_probs = (eff_probs + xep_probs) / 2.0
    final_pred = int(torch.argmax(ensemble_probs, dim=1).item())
    final_conf = float(ensemble_probs[0, final_pred].item() * 100)

    result = {
        'final_label': 'REAL' if final_pred == 0 else 'FAKE',
        'confidence': f'{final_conf:.2f}%',
        'efficientnet_confidence': f'{eff_probs[0, final_pred].item() * 100:.2f}%',
        'xception_confidence': f'{xep_probs[0, final_pred].item() * 100:.2f}%'
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
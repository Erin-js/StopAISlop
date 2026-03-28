import torch
import os
import timm
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import base64
import io

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\ACER\OneDrive\Dokumen\IRIS_2026\AI_DETECTOR\models\best_xception_tiny_genimage.pth"
CLASS_NAMES = ["Real (Nature)", "AI-Generated"]

def build_model():
    model = timm.create_model("legacy_xception", pretrained=False, num_classes=2)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan di: {os.path.abspath(MODEL_PATH)}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"✅ Model loaded | epoch {checkpoint.get('epoch','?')} | val F1: {checkpoint.get('best_val_f1',0):.4f}")
    return model

model = build_model()

TRANSFORM = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def predict(image: Image.Image) -> dict:
    img_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx  = int(np.argmax(probs))
    real_prob = float(probs[0])
    ai_prob   = float(probs[1])

    return {
        "label": CLASS_NAMES[pred_idx],
        "is_ai": pred_idx == 1,
        "confidence": round(float(probs[pred_idx]) * 100, 2),
        "ai_probability": round(ai_prob * 100, 2),
        "real_probability": round(real_prob * 100, 2),
        "confidence_level": _get_confidence_level(float(probs[pred_idx]))
    }

def predict_with_gradcam(image: Image.Image) -> dict:
    """Predict + hasilkan Grad-CAM heatmap asli dari model."""

    # ── Simpan ukuran asli untuk overlay ──────────────────────────
    orig_w, orig_h = image.size
    img_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # ── Hook untuk tangkap aktivasi & gradien ─────────────────────
    # Target layer: conv4 = layer konvolusi terakhir sebelum global pooling
    activations = {}
    gradients   = {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    # Daftarkan hook ke layer conv4 (exit flow terakhir Xception)
    target_layer = model.conv4
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # ── Forward pass (dengan gradient) ────────────────────────────
    model.zero_grad()
    logits = model(img_tensor)
    probs  = torch.softmax(logits, dim=1).squeeze()

    pred_idx  = int(torch.argmax(probs).item())
    real_prob = float(probs[0].item())
    ai_prob   = float(probs[1].item())

    # ── Backward pass terhadap kelas prediksi ─────────────────────
    score = logits[0, pred_idx]
    score.backward()

    # ── Hapus hook setelah selesai ─────────────────────────────────
    fwd_handle.remove()
    bwd_handle.remove()

    # ── Hitung Grad-CAM ───────────────────────────────────────────
    grads = gradients["value"].squeeze().cpu().detach().numpy()  # [C, H, W]
    acts  = activations["value"].squeeze().cpu().detach().numpy() # [C, H, W]

    # Bobot = rata-rata global gradient tiap channel
    weights = grads.mean(axis=(1, 2))                            # [C]

    # Gabungkan feature maps dengan bobot
    cam = np.zeros(acts.shape[1:], dtype=np.float32)             # [H, W]
    for i, w in enumerate(weights):
        cam += w * acts[i]

    # ReLU — hanya area positif yang relevan
    cam = np.maximum(cam, 0)

    # Normalize ke 0-255
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = (cam * 255).astype(np.uint8)

    # ── Resize ke ukuran gambar asli & buat heatmap ───────────────
    cam_resized  = cv2.resize(cam, (orig_w, orig_h))
    heatmap_bgr  = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap_rgb  = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # ── Overlay heatmap di atas gambar asli ───────────────────────
    orig_np   = np.array(image.resize((orig_w, orig_h))).astype(np.uint8)
    overlay   = cv2.addWeighted(orig_np, 0.5, heatmap_rgb, 0.5, 0)

    # ── Encode ke base64 ──────────────────────────────────────────
    overlay_pil = Image.fromarray(overlay)
    buffer      = io.BytesIO()
    overlay_pil.save(buffer, format="PNG")
    heatmap_b64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    return {
        "label": CLASS_NAMES[pred_idx],
        "is_ai": pred_idx == 1,
        "confidence": round(float(probs[pred_idx].item()) * 100, 2),
        "ai_probability": round(ai_prob * 100, 2),
        "real_probability": round(real_prob * 100, 2),
        "confidence_level": _get_confidence_level(float(probs[pred_idx].item())),
        "gradcam": heatmap_b64   # ← base64 PNG siap ditampilkan di <img>
    }

def _get_confidence_level(prob: float) -> str:
    if prob >= 0.90: return "High"
    if prob >= 0.70: return "Medium"
    return "Low"
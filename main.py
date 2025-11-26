# app.py
import io
import base64
import time
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

MODEL_PATH = "emotion_model.joblib"   # ensure exists
EMOTIONS = ["angry","happy","sad","neutral","fear","calm"]

app = FastAPI(title="Audio Emotion Detector API")

# Allow local frontend; adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend origin when deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print("WARNING: model not loaded:", e)

def extract_features_from_array(audio: np.ndarray, sr: int):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, -1)

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return b64

def waveform_image(audio, sr):
    fig, ax = plt.subplots(figsize=(8,2.5), dpi=100)
    ax.plot(np.linspace(0, len(audio)/sr, num=len(audio)), audio, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.set_facecolor("#14161c")
    fig.patch.set_facecolor("#14161c")
    for spine in ax.spines.values():
        spine.set_color("#2b2b2b")
    ax.tick_params(colors="#9b9b9b")
    ax.title.set_color("#ffffff")
    return plot_to_base64(fig)

def spectrogram_image(audio, sr):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(8,2.5), dpi=100)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.set_title("Mel-spectrogram (dB)")
    ax.set_facecolor("#14161c")
    fig.patch.set_facecolor("#14161c")
    ax.title.set_color("#ffffff")
    ax.tick_params(colors="#9b9b9b")
    return plot_to_base64(fig)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    start = time.time()

    # read file bytes and convert to numpy audio
    data = await file.read()
    try:
        audio_np, sr = sf.read(io.BytesIO(data))
    except Exception:
        # fallback: try librosa
        audio_np, sr = librosa.load(io.BytesIO(data), sr=None)

    # if stereo, convert to mono
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)

    # feature extraction & predict
    features = extract_features_from_array(audio_np, sr)
    probs = None
    try:
        probs = model.predict_proba(features)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
    except Exception:
        pred_idx = int(model.predict(features)[0])
        confidence = None

    predicted_label = EMOTIONS[pred_idx] if pred_idx < len(EMOTIONS) else str(pred_idx)

    # generate images (base64)
    wave_b64 = waveform_image(audio_np, sr)
    spec_b64 = spectrogram_image(audio_np, sr)

    elapsed = time.time() - start

    return {
        "label": predicted_label,
        "confidence": confidence,
        "processing_time": round(elapsed, 3),
        "waveform_png_b64": wave_b64,
        "spectrogram_png_b64": spec_b64,
    }

# run with: uvicorn app:app --host 0.0.0.0 --port 8000

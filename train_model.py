import os
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset"
EMOTIONS = ["Angry","Happy","Sad","Neutral","Fearful","Suprised","Disgusted"]

X, y = [], []

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=22050)

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=25)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])
    return np.mean(features.T, axis=0)



for emotion in EMOTIONS:
    emotion_path = os.path.join(DATASET_PATH, emotion)
    for file in os.listdir(emotion_path):
        if file.endswith(".wav"):
            file_path = os.path.join(emotion_path, file)
            features = extract_mfcc(file_path)
            X.append(features)
            y.append(emotion)

X = np.array(X)
y = np.array(y)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

#  features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================model
model = RandomForestClassifier(                                    # svM changed to forest
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",     # balacen aadded
    random_state=42,
    n_jobs=-1
)
   

model.fit(X_scaled, y_encoded)


joblib.dump(model, "emotion_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(encoder, "label_encoder.joblib")

print('\n')
print("Model, scaler, and encoder saved successfully")
print('\n ----DONE --\n')
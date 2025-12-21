<img width="1919" height="1079" alt="Screenshot 2025-11-01 172912" src="https://github.com/user-attachments/assets/6e4ae254-7971-4e2e-8c81-8e91195f4b75" /># Audio Emotion Detector

A web-based application that detects human emotions from audio input using machine learning.

## Project Overview
This project allows users to upload or record audio and predicts the underlying emotion. It also visualizes the audio waveform and mel-spectrogram for better interpretation.

## Features
- Upload audio files or record live audio
- Emotion classification from speech
- Waveform visualization
- Mel-spectrogram visualization
- Confidence score and processing time display

## Tech Stack
- Python
- FastAPI
- Scikit-learn
- Librosa
- NumPy
- HTML, CSS, JavaScript
- Chart.js

## Project Structure
- `main.py` – FastAPI backend for emotion prediction
- `index.html` – Dashboard UI
- `cover.html` – Landing page UI
- `requirements.txt` – Python dependencies

## Setup Instructions
```
pip install -r requirements.txt
```

## Run Backend
```
python main.py
```

## Run Frontend 
```
Open cover.html or index.html in any browser
```
## Screenshots (Output)
Dashboard 
```
<img width="1915" height="909" alt="Screenshot 2025-12-21 103219" src="https://github.com/user-attachments/assets/ae04447c-c871-4694-b9f4-e9fd67f26a59" />
<img width="1903" height="937" alt="Screenshot 2025-12-21 103144" src="https://github.com/user-attachments/assets/dce350cd-bbb9-4aab-b46e-b29b59ae5f09" />
```
Waveform & Visualizations of results

```
<img width="1900" height="908" alt="Screenshot 2025-12-21 103253" src="https://github.com/user-attachments/assets/e9a8aa69-660e-4c67-ac99-978c993be94e" />
```


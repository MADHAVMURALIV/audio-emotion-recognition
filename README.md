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

# UI and Outputs

```
<img width="1900" height="908" alt="spectral overview1" src="https://github.com/user-attachments/assets/95a7a596-6c44-4319-9ce5-d8a5b9f132d5" />
<img width="1900" height="902" alt="Screenshot 2025-12-22 132627" src="https://github.com/user-attachments/assets/631ba796-e4fa-4d94-893b-0b7611fab698" />
<img width="1919" height="912" alt="Screenshot 2025-12-22 132615" src="https://github.com/user-attachments/assets/88525a9b-5dfc-4962-915f-e733bb5140bd" />
<img width="1919" height="908" alt="Screenshot 2025-12-22 132432" src="https://github.com/user-attachments/assets/83e4d258-2eec-498c-a18f-40a20fdd6f19" />
<img width="1919" height="903" alt="Screenshot 2025-12-22 132339" src="https://github.com/user-attachments/assets/51f6687b-8cc5-4b73-ad1b-1d32df2ea0f2" />
<img width="1915" height="909" alt="initial_dashboard" src="https://github.com/user-attachments/assets/8c43898a-b3bd-4d8c-9348-2940f4d677cb" />
<img width="1903" height="937" alt="coverpage" src="https://github.com/user-attachments/assets/cf08f97d-a4de-4bd9-b087-333d316e6acc" />
<img width="1919" height="978" alt="Screenshot 2025-12-22 132319" src="https://github.com/user-attachments/assets/0ed61c33-a378-46c4-a4da-61db3e10414e" />

```

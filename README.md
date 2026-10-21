

## 🧠 TRANSFORMER — Deep Learning Application

This project uses the Whisper Transformer deep learning model
to convert spoken audio into written text.

### 🚀 Launch TRANSFORMER Application

## 🧠 TRANSFORMER

[🚀 CLICK HERE TO LAUNCH TRANSFORMER](https://aispeechtotext-nb2hfcqjxw2xcv7iykubty.streamlit.app/)

### How to Use
1. Click the TRANSFORMER application link.
2. Upload an audio recording (MP3, WAV or M4A).
3. Click "Transcribe audio".
4. View and download the transcription.




# 🎙️ AI Speech-to-Text — Deep Learning Streamlit App

A web application that turns audio recordings into editable text using a pretrained Whisper speech recognition model via [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Features
- Upload MP3, WAV, M4A, OGG, or FLAC recordings (up to 20 MB for this demo).
- Transcribe in multiple languages, or automatically detect the language.
- Switch between `tiny` and `base` models for different speed/quality trade-offs.
- Download the transcription as a `.txt` file.
- Runs on CPU without an API key.

## Deploy to Streamlit Community Cloud
1. Create a **new** GitHub repository (e.g. `AI_Speech_to_Text`). Do not overwrite your image-classifier repo.
2. Upload **the contents of this folder** to the repository root: `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and `.streamlit/config.toml`.
3. Visit https://share.streamlit.io and choose **Create app** → **Yup, I have an app**.
4. Pick your new repository, branch `main`, main file path `app.py`.
5. In **Advanced settings**, choose Python **3.12** if available, then **Deploy**.
6. Once deployed, upload a short, clear audio recording and press **Transcribe audio**.

**Note:** The first transcription downloads the model from the internet; cold starts can be slower. CPU-based inference can take longer on large files. Free-tier resource limits apply. If installation or inference fails, view the app's **Manage app → Logs** for details.

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Privacy
Audio uploads are sent to the Streamlit hosting server for processing. Temporary audio files are deleted after transcription, but avoid uploading confidential audio to a public demo.

## Model
Uses pretrained Whisper model weights with `faster-whisper` (CTranslate2) on CPU in INT8 mode. No model training is included in this demonstration.

"""Speech-to-text demo using a pretrained Whisper model (CPU-friendly)."""
from pathlib import Path
import os
import tempfile

import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="AI Speech-to-Text", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
.stApp {background: #0b1220; color: #f1f5f9;}
.block-container {max-width: 980px; padding-top: 2rem;}
.banner {padding: 2rem; border-radius: 20px; background: linear-gradient(120deg,#182e52,#393070); margin-bottom: 1.5rem;}
.banner h1 {color:white; margin:0 0 .6rem 0;}
.banner p {color:#d4d8ed; margin:0;}
[data-testid="stFileUploader"] {border: 1px solid #56617a; border-radius: 12px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="banner"><h1>🎙️ AI Speech-to-Text</h1><p>Turn spoken audio into editable text using a deep learning speech recognition model.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("About this project")
    st.write("Powered by the open-source Whisper model through faster-whisper. Audio is processed on the app server.")
    model_size = st.selectbox("Model size", ["tiny", "base"], index=0,
        help="Tiny starts faster; base can be more accurate but uses more resources.")
    language_choice = st.selectbox("Language", ["Auto-detect", "English", "Spanish", "French", "German", "Hindi", "Arabic"], index=0)
    language_map = {"Auto-detect": None, "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Hindi": "hi", "Arabic": "ar"}
    st.caption("First use downloads the pretrained model. No API key required.")
    st.warning("Avoid uploading confidential recordings, as audio is processed on the hosting server.")

@st.cache_resource(show_spinner=False)
def load_model(size: str):
    return WhisperModel(size, device="cpu", compute_type="int8")

uploaded = st.file_uploader("Upload an audio recording (MP3, WAV, M4A, OGG, FLAC)", type=["mp3", "wav", "m4a", "ogg", "flac"])
if uploaded is not None:
    st.audio(uploaded)
    st.caption(f"Selected: {uploaded.name} — {uploaded.size / (1024 * 1024):.2f} MB")
    if uploaded.size > 20 * 1024 * 1024:
        st.error("Please upload a recording smaller than 20 MB for this demo.")
    elif st.button("✨ Transcribe audio", type="primary", use_container_width=True):
        suffix = Path(uploaded.name).suffix.lower()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            with st.spinner("Loading the model and transcribing. First run may need a model download..."):
                model = load_model(model_size)
                segments, info = model.transcribe(
                    tmp_path,
                    language=language_map[language_choice],
                    beam_size=5,
                    vad_filter=True,
                )
                # The segments iterator triggers actual model inference.
                text = " ".join(segment.text.strip() for segment in segments).strip()
            if text:
                st.success("Transcription complete")
                st.text_area("Recognized speech", text, height=220)
                st.download_button("⬇️ Download transcript (.txt)", data=text.encode("utf-8"), file_name="transcript.txt", mime="text/plain")
                st.caption(f"Detected language: {info.language} • Detection probability: {info.language_probability:.0%} (estimate)")
            else:
                st.warning("No speech was detected. Try a clearer recording with audible speech.")
        except Exception as exc:
            st.error("Transcription did not finish. Check that the audio is valid and see your deployment logs.")
            with st.expander("Technical details"):
                st.code(str(exc))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
else:
    st.info("Upload an audio file, then click Transcribe audio to begin.")

st.divider()
st.caption("Whisper speech recognition • Streamlit")

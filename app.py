import streamlit as st
import os
from utils import transcribe_with_speechrecognition, transcribe_with_wav2vec
from audio_utils import convert_to_wav

st.set_page_config(page_title="Speech Recognizer", layout="centered")

# Header
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50; font-size: 3em;'>
        🎤 Speech Recognizer
    </h1>
    <p style='text-align: center; font-size: 1.1em; color: gray;'>
        Transcribe your audio into text using AI-powered models
    </p>
    <hr style='margin-top: 10px;'>
    """,
    unsafe_allow_html=True
)

# Sidebar settings
with st.sidebar:
    st.header("⚙ Settings")
    model_choice = st.radio("Select Model", ["SpeechRecognition", "Wav2Vec2"])
    audio_refinement = st.checkbox("Normalize Audio")

# Step 1: Upload audio
st.subheader("Step 1: Provide Audio")
uploaded_audio = st.file_uploader("📂 Upload audio file (MP3/WAV/OGG)", type=['wav', 'mp3', 'ogg'])

# Convert uploaded audio
audio_file_path = None
if uploaded_audio:
    with st.spinner("Processing uploaded audio..."):
        audio_file_path = convert_to_wav(uploaded_audio)

# Step 2: Transcription
transcribed_text = ""
if st.button("📄 Transcribe Audio"):
    if not audio_file_path:
        st.warning("Please upload an audio file before transcribing.")
    else:
        with st.spinner("Transcribing..."):
            try:
                if model_choice == "SpeechRecognition":
                    transcribed_text = transcribe_with_speechrecognition(audio_file_path)
                else:
                    transcribed_text = transcribe_with_wav2vec(audio_file_path)
                st.success("Transcription complete!")
            except Exception as e:
                st.error(f"Transcription failed: {e}")

# Display transcript
if transcribed_text:
    st.text_area("📝 Transcript", transcribed_text, height=200)
    st.download_button("📥 Download Transcript", transcribed_text, file_name="transcript.txt")

# Reset session
if st.button("🔁 Clear/Reset"):
    st.session_state.clear()
    st.success("Session reset. Please reload the page.")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray;'>
        Built with ❤️ by <strong>CodTech Intern</strong>
    </p>
    """,
    unsafe_allow_html=True
)

from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = which("ffmpeg")  # 👈 Add this line

def convert_to_wav(uploaded_file):
    audio = AudioSegment.from_file(uploaded_file)
    wav_path = f"temp_audio.wav"
    audio.export(wav_path, format="wav")
    return wav_path

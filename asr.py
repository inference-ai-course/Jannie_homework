import whisper
import tempfile
import os  # <-- Added import

# Load Whisper model once
asr_model = whisper.load_model("small")

def transcribe_audio(audio_bytes):
    """
    Converts audio bytes to text using Whisper ASR.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        temp_path = f.name

    result = asr_model.transcribe(temp_path)
    os.remove(temp_path)  # Now works because os is imported
    return result["text"]

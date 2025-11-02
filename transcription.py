import os
import yt_dlp
import whisper
import subprocess
import jsonlines

# 1. Download audio from YouTube and convert to MP3
def download_audio(video_url, output_dir="talks"):
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        video_id = info.get("id")
        downloaded_ext = info.get("ext", "webm")
        downloaded_path = os.path.join(output_dir, f"{video_id}.{downloaded_ext}")

    # Convert to MP3
    mp3_path = os.path.join(output_dir, f"{video_id}.mp3")
    if not os.path.exists(mp3_path):
        print(f"Converting {downloaded_path} to MP3...")
        subprocess.run([
            "ffmpeg", "-y", "-i", downloaded_path,
            "-vn", "-acodec", "libmp3lame", "-q:a", "2", mp3_path
        ], check=True)
    else:
        print(f"MP3 already exists: {mp3_path}")

    return mp3_path, video_id

# 2. Transcribe audio with Whisper
def transcribe_audio(audio_path, model_name="medium"):
    model = whisper.load_model(model_name)
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path)
    return result["segments"]

# 3. Save JSONL
def save_jsonl(data, output_file="talks_transcripts.jsonl"):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with jsonlines.open(output_file, mode="a") as writer:
        writer.write(data)
    print(f"Saved to {output_file}")

# 4. Main process
if __name__ == "__main__":
    video_urls = [
        "https://www.youtube.com/watch?v=DYu_bGbZiiQ",
        "https://www.youtube.com/watch?v=mU9VYcQWSOc",
        "https://www.youtube.com/watch?v=gO8N3L_aERg",
        "https://www.youtube.com/watch?v=5pvL2YiFJsQ",
        "https://www.youtube.com/watch?v=0IwdBEZmbfQ",
        "https://www.youtube.com/watch?v=FpGkLzGl1CI",
        "https://www.youtube.com/watch?v=lkbr5qnYSUU",
        "https://www.youtube.com/watch?v=KzjNc5A25_A",
        "https://www.youtube.com/watch?v=9T_CQedIIAc",
        "https://www.youtube.com/watch?v=ZjxmEb5n7oA"
    ]

    for url in video_urls:
        print(f"\nProcessing video: {url}")
        audio_path, vid_id = download_audio(url)

        print("Transcribing with Whisper...")
        segments = transcribe_audio(audio_path)

        save_jsonl({
            "id": vid_id,
            "type": "audio_transcript",
            "segments": segments
        })

        print(f"Finished {vid_id}\n")

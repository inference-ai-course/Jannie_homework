import os
import sys
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uuid
from datetime import datetime
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
import subprocess
import tempfile
import functools
import math

# Create FastAPI app
app = FastAPI(title="Voice Assistant API", debug=True)

# Serve static folder for generated responses
if not os.path.exists("generated_responses"):
    os.makedirs("generated_responses")
    print("✓ Created generated_responses directory")

app.mount("/generated_responses", StaticFiles(directory="generated_responses"), name="generated_responses")

# ==================== IMPORT ACTUAL MODULES ====================

# Add CosyVoice paths
sys.path.insert(0, "/mnt/d/inference/assignment 3/CosyVoice/third_party/MatchaTTS")
sys.path.append("/mnt/d/inference/assignment 3/CosyVoice/third_party/MatchaTTS/cosyvoice")
sys.path.append("/mnt/d/inference/assignment 3/CosyVoice/third_party/MatchaTTS/matcha")

# Import actual modules with proper error handling
try:
    from asr import transcribe_audio
    print("✓ ASR module imported successfully")
except ImportError as e:
    print(f"✗ ASR module import failed: {e}")
    # Fallback function
    def transcribe_audio(audio_bytes):
        return "Fallback: Audio transcription not available"

try:
    from llm import generate_response
    print("✓ LLM module imported successfully")
except ImportError as e:
    print(f"✗ LLM module import failed: {e}")
    # Fallback function
    def generate_response(user_text, conversation_history):
        return f"Fallback response to: {user_text}"

try:
    from tts import synthesize_speech
    print("✓ TTS module imported successfully")
except ImportError as e:
    print(f"✗ TTS module import failed: {e}")
    # Fallback function - FIXED to generate actual sound
    def synthesize_speech(text, cosyvoice_model, voice_character):
        print(f"Fallback TTS for: {text} with voice: {voice_character}")
        
        # Create a proper WAV file with actual sound
        import wave
        import struct
        
        sample_rate = 22050
        # Calculate duration based on text length (rough estimate)
        duration = max(2, min(10, len(text) * 0.1))  # 2-10 seconds based on text length
        
        with wave.open("generated_responses/response.wav", 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            
            # Generate a more interesting sound - sine wave with varying frequency
            for i in range(int(duration * sample_rate)):
                # Varying frequency to make it sound more natural
                base_freq = 220 + (i % 1000) * 0.1  # Varying base frequency
                freq_variation = math.sin(i * 0.01) * 50  # Add some variation
                frequency = base_freq + freq_variation
                
                # Generate sine wave
                sample = math.sin(2.0 * math.pi * frequency * i / sample_rate)
                
                # Add some amplitude variation
                amplitude = 0.3 + 0.1 * math.sin(i * 0.05)
                
                # Convert to 16-bit integer
                value = int(32767.0 * amplitude * sample)
                data = struct.pack('<h', value)
                wav_file.writeframesraw(data)
        
        print(f"✅ Generated fallback audio: {duration:.1f} seconds")
        return True

# Initialize CosyVoice
cosyvoice = None
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    print("✓ CosyVoice2 imported successfully")
    
    COSYVOICE_MODEL_PATH = "/mnt/d/inference/assignment 3/CosyVoice/third_party/MatchaTTS/pretrained_models/CosyVoice2-0.5B"
    if os.path.exists(COSYVOICE_MODEL_PATH):
        cosyvoice = CosyVoice2(
            COSYVOICE_MODEL_PATH,
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False
        )
        print("✓ CosyVoice2 initialized successfully")
    else:
        print(f"✗ CosyVoice model path not found: {COSYVOICE_MODEL_PATH}")
except ImportError as e:
    print(f"✗ CosyVoice module not found: {e}")
except Exception as e:
    print(f"✗ CosyVoice initialization failed: {e}")

# ==================== INITIALIZE COMPONENTS ====================

# Initialize components
executor = ThreadPoolExecutor(max_workers=4)
conversation_history = []
MAX_CONVERSATIONS = 5

def convert_webm_to_wav(webm_bytes):
    """Convert WebM audio to WAV format"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
            webm_file.write(webm_bytes)
            webm_path = webm_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
            wav_path = wav_file.name
        
        cmd = [
            'ffmpeg', '-i', webm_path, 
            '-acodec', 'pcm_s16le', 
            '-ac', '1', 
            '-ar', '16000',
            wav_path,
            '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            with open(wav_path, 'rb') as f:
                wav_bytes = f.read()
            
            os.unlink(webm_path)
            os.unlink(wav_path)
            return wav_bytes
        else:
            print(f"FFmpeg conversion failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Error converting WebM to WAV: {e}")
        return None

def create_proper_test_audio():
    """Create a proper test audio file with actual sound"""
    import wave
    import struct
    
    test_filename = "test_tone.wav"
    test_path = os.path.join("generated_responses", test_filename)
    
    sample_rate = 22050
    duration = 3  # seconds
    
    with wave.open(test_path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        # Generate a proper sine wave tone
        frequency = 440  # A4 note
        for i in range(int(duration * sample_rate)):
            # Generate sine wave
            sample = math.sin(2.0 * math.pi * frequency * i / sample_rate)
            # Convert to 16-bit integer with proper amplitude
            value = int(32767.0 * 0.5 * sample)
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)
    
    return test_filename

# ==================== API ENDPOINTS ====================

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...), voice_character: str = Form("pyttsx3-male")):
    try:
        print(f"🔊 Chat endpoint called - Voice: {voice_character}")
        
        conversation_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Read and process audio file
        audio_bytes = await file.read()
        print(f"📁 Received audio file: {len(audio_bytes)} bytes")
        
        # Convert if WebM
        if file.filename.endswith('.webm') or file.content_type == 'audio/webm':
            print("🔄 Converting WebM to WAV...")
            converted_bytes = convert_webm_to_wav(audio_bytes)
            if converted_bytes:
                audio_bytes = converted_bytes
                print("✅ WebM converted to WAV successfully")
            else:
                return {"success": False, "error": "Failed to convert WebM audio"}
        
        # Save user audio
        original_filename = f"user_audio_{conversation_id}.wav"
        original_filepath = os.path.join("generated_responses", original_filename)
        
        with open(original_filepath, "wb") as f:
            f.write(audio_bytes)
        print(f"💾 Saved user audio to: {original_filepath}")
        
        # Process with actual ASR
        print("🎤 Starting ASR transcription...")
        user_text = await asyncio.get_event_loop().run_in_executor(executor, transcribe_audio, audio_bytes)
        print(f"📝 Transcribed text: {user_text}")
        
        # Process with actual LLM
        print("🤖 Starting LLM response generation...")
        bot_text = await asyncio.get_event_loop().run_in_executor(executor, generate_response, user_text, conversation_history)
        print(f"💬 Generated response: {bot_text}")
        
        # Generate TTS audio filename
        audio_filename = f"response_{conversation_id}.wav"
        audio_path = os.path.join("generated_responses", audio_filename)
        
        # Process with actual TTS
        print(f"🗣️ Starting TTS synthesis with voice: {voice_character}")
        tts_success = await asyncio.get_event_loop().run_in_executor(executor, synthesize_speech, bot_text, cosyvoice, voice_character)
        
        if not tts_success:
            return {"success": False, "error": "TTS synthesis failed"}
        
        # Check if TTS generated a file and rename it
        if os.path.exists("generated_responses/response.wav"):
            # Verify the file has content
            file_size = os.path.getsize("generated_responses/response.wav")
            if file_size > 1000:  # At least 1KB
                os.rename("generated_responses/response.wav", audio_path)
                print(f"✅ TTS audio saved to: {audio_path} ({file_size} bytes)")
            else:
                print(f"⚠️ TTS output file too small: {file_size} bytes")
                return {"success": False, "error": "TTS generated empty audio file"}
        else:
            print("❌ TTS output file not found")
            return {"success": False, "error": "TTS output file not found"}
        
        # Add to conversation history
        conversation_entry = {
            "id": conversation_id,
            "timestamp": timestamp,
            "voice_character": voice_character,
            "user_audio": original_filename,
            "user_text": user_text,
            "assistant_audio": audio_filename,
            "assistant_text": bot_text
        }
        
        conversation_history.append(conversation_entry)
        if len(conversation_history) > MAX_CONVERSATIONS:
            conversation_history.pop(0)
        
        print(f"✅ Added conversation: {conversation_id} (Total: {len(conversation_history)})")
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "voice_character": voice_character,
            "user_text": user_text,
            "assistant_text": bot_text,
            "user_audio_url": f"/generated_responses/{original_filename}",
            "assistant_audio_url": f"/generated_responses/{audio_filename}"
        }
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/conversations")
async def get_conversations():
    return {
        "conversations": conversation_history,
        "total": len(conversation_history),
        "max_conversations": MAX_CONVERSATIONS
    }

@app.post("/clear")
async def clear_conversation():
    global conversation_history
    old_count = len(conversation_history)
    conversation_history = []
    return {
        "success": True, 
        "message": "Conversation history cleared",
        "cleared_conversations": old_count
    }

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join("generated_responses", filename)
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"🎵 Serving audio: {filename} ({file_size} bytes)")
        return FileResponse(file_path, media_type="audio/wav")
    else:
        print(f"❌ Audio file not found: {filename}")
        return {"error": "Audio file not found"}

@app.get("/test-audio-playback")
async def test_audio_playback():
    try:
        test_filename = create_proper_test_audio()
        test_path = os.path.join("generated_responses", test_filename)
        file_size = os.path.getsize(test_path)
        
        print(f"✅ Test audio created: {test_filename} ({file_size} bytes)")
        return {
            "success": True,
            "message": "Test audio created with proper sound",
            "filename": test_filename,
            "url": f"/audio/{test_filename}",
            "size": file_size
        }
    except Exception as e:
        print(f"❌ Test audio creation failed: {e}")
        return {"success": False, "error": str(e)}

# ==================== ORIGINAL VOICE ASSISTANT WEB INTERFACE ====================

@app.get("/", response_class=HTMLResponse)
async def voice_assistant_interface():
    # Available voices
    available_voices = {
        "pyttsx3-male": "Male Voice (pyttsx3)",
        "pyttsx3-female": "Female Voice (pyttsx3)",
    }

    # Add system voices from your TTS engine
    try:
        import pyttsx3
        engine = pyttsx3.init()
        system_voices = engine.getProperty('voices')
        
        for i, voice in enumerate(system_voices):
            voice_name = voice.name
            # Create a safe ID for the voice
            voice_id = f"system-voice-{i}"
            available_voices[voice_id] = f"{voice_name} (System)"
        
        print(f"✅ Added {len(system_voices)} system voices to available options")
        
    except Exception as e:
        print(f"⚠️ Could not load system voices: {e}")

    # Add CosyVoice voices if available
    if cosyvoice:
        female_voice_path = "/mnt/d/inference/assignment 3/generated_responses/female_voice.wav"
        male_voice_path = "/mnt/d/inference/assignment 3/generated_responses/male_voice.wav"
        
        if os.path.exists(female_voice_path):
            available_voices["cosyvoice-female"] = "Female Voice (CosyVoice)"
        if os.path.exists(male_voice_path):
            available_voices["cosyvoice-male"] = "Male Voice (CosyVoice)"

    # Generate voice options for HTML
    voice_options = ""
    if available_voices:
        first_voice_id = next(iter(available_voices))
        for voice_id, voice_desc in available_voices.items():
            selected = "selected" if voice_id == first_voice_id else ""
            voice_options += f'<option value="{voice_id}" {selected}>{voice_desc}</option>'
    else:
        voice_options = '<option value="none">No voices available</option>'
        print("⚠️ No voices available in available_voices dictionary")

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Voice Assistant</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .upload-section {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            border: 2px dashed #dee2e6;
        }}
        .upload-section h3 {{
            color: #495057;
            margin-bottom: 15px;
        }}
        .form-group {{
            margin-bottom: 15px;
        }}
        .form-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #495057;
        }}
        .file-input, .voice-select {{
            width: 100%;
            padding: 15px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            background: white;
            font-size: 1em;
        }}
        .btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }}
        .btn:disabled {{
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
        }}
        .btn-clear {{
            background: #dc3545;
        }}
        .btn-clear:hover {{
            background: #c82333;
        }}
        .btn-test {{
            background: linear-gradient(135deg, #fd7e14 0%, #e8590c 100%);
        }}
        .btn-test:hover {{
            background: linear-gradient(135deg, #e8590c 0%, #d9480f 100%);
        }}
        .status {{
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            text-align: center;
            font-weight: 600;
        }}
        .status.success {{
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .status.error {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .status.processing {{
            background: #cce7ff;
            color: #004085;
            border: 1px solid #b3d7ff;
        }}
        .conversation-history {{
            margin-top: 30px;
        }}
        .conversation-history h3 {{
            color: #495057;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
        }}
        .conversation-item {{
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .conversation-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #f1f3f4;
        }}
        .conversation-id {{
            background: #667eea;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: 600;
        }}
        .conversation-time {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .voice-badge {{
            background: #ff6b6b;
            color: white;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 0.7em;
            margin-left: 10px;
        }}
        .message {{
            margin-bottom: 15px;
        }}
        .message.user {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #2196f3;
        }}
        .message.assistant {{
            background: #f3e5f5;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #9c27b0;
        }}
        .message-label {{
            font-weight: 600;
            margin-bottom: 5px;
            color: #495057;
        }}
        .audio-section {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-top: 10px;
            flex-wrap: wrap;
        }}
        audio {{
            flex: 1;
            min-width: 250px;
            border-radius: 25px;
        }}
        .download-btn {{
            background: #28a745;
            color: white;
            padding: 8px 15px;
            border-radius: 5px;
            text-decoration: none;
            font-size: 0.9em;
            transition: background 0.3s ease;
        }}
        .download-btn:hover {{
            background: #218838;
        }}
        .empty-state {{
            text-align: center;
            color: #6c757d;
            padding: 40px;
            font-style: italic;
        }}
        .controls {{
            display: flex;
            gap: 10px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .controls .btn {{
            width: auto;
            flex: 1;
            min-width: 200px;
        }}
        .input-method-buttons {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }}
        .input-method-buttons .btn {{
            flex: 1;
        }}
        .recording-controls {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .recording-timer {{
            margin-top: 10px;
            font-size: 1.2em;
            font-weight: bold;
        }}
        .audio-error {{
            color: #dc3545;
            font-size: 0.8em;
            margin-top: 5px;
            padding: 5px;
            background: #ffe6e6;
            border-radius: 3px;
            border: 1px solid #f5c6cb;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎙️ Voice Assistant</h1>
            <p>Upload audio or record live - Get AI voice response</p>
        </div>
        
        <div class="content">
            <div class="upload-section">
                <h3>🎵 Choose Input Method & Voice</h3>
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="voiceSelect">🗣️ Voice Character:</label>
                        <select id="voiceSelect" name="voice_character" class="voice-select">
                            {voice_options}
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>🎤 Input Method:</label>
                        <div class="input-method-buttons">
                            <button type="button" class="btn" id="recordBtn">
                                🎤 Record Live Audio
                            </button>
                            <button type="button" class="btn" id="uploadBtn">
                                📁 Upload Audio File
                            </button>
                        </div>
                    </div>
                    
                    <!-- File Upload -->
                    <div class="form-group" id="fileUploadGroup">
                        <label for="fileInput">📤 Upload Audio File:</label>
                        <input type="file" id="fileInput" name="file" accept="audio/*" class="file-input">
                    </div>
                    
                    <!-- Live Recording -->
                    <div class="form-group" id="recordingGroup" style="display: none;">
                        <label>🎙️ Live Recording:</label>
                        <div class="recording-controls">
                            <div id="recordingStatus" style="margin-bottom: 15px;">
                                Click record to start
                            </div>
                            <div style="display: flex; gap: 10px; justify-content: center;">
                                <button type="button" class="btn" id="startRecordBtn">
                                    ⏺️ Start Recording
                                </button>
                                <button type="button" class="btn" id="stopRecordBtn" disabled>
                                    ⏹️ Stop Recording
                                </button>
                            </div>
                            <div id="recordingTimer" class="recording-timer">
                                00:00
                            </div>
                            <audio id="recordedAudio" controls style="margin-top: 15px; width: 100%; display: none;"></audio>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn" id="submitBtn" disabled>
                        🚀 Send Audio & Get Response
                    </button>
                </form>
                <div id="status"></div>
            </div>

            <div class="conversation-history">
                <h3>💬 Conversation History (Last 5)</h3>
                <div id="conversationList"></div>
            </div>

            <div class="controls">
                <button class="btn btn-clear" onclick="clearHistory()">🗑️ Clear All History</button>
                <button class="btn btn-test" onclick="testAudioPlayback()">🎵 Test Audio Playback</button>
                <button class="btn" onclick="showDebugInfo()">🐛 Debug Info</button>
           </div>
        </div>
    </div>

    <script>
        let conversations = [];
        let mediaRecorder = null;
        let audioChunks = [];
        let recordingTimer = null;
        let recordingStartTime = null;
        let currentInputMethod = 'upload';

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            loadConversations();
            setupEventListeners();
        }});

        function setupEventListeners() {{
            // Input method toggle buttons
            document.getElementById('uploadBtn').addEventListener('click', () => {{
                currentInputMethod = 'upload';
                document.getElementById('fileUploadGroup').style.display = 'block';
                document.getElementById('recordingGroup').style.display = 'none';
                document.getElementById('submitBtn').disabled = true;
                resetRecording();
            }});

            document.getElementById('recordBtn').addEventListener('click', () => {{
                currentInputMethod = 'record';
                document.getElementById('fileUploadGroup').style.display = 'none';
                document.getElementById('recordingGroup').style.display = 'block';
                document.getElementById('submitBtn').disabled = true;
            }});

            // Recording control buttons
            document.getElementById('startRecordBtn').addEventListener('click', startRecording);
            document.getElementById('stopRecordBtn').addEventListener('click', stopRecording);

            // File input change listener
            document.getElementById('fileInput').addEventListener('change', function() {{
                document.getElementById('submitBtn').disabled = !this.files.length;
            }});

            // Form submission
            document.getElementById('uploadForm').addEventListener('submit', handleFormSubmit);
        }}

        async function startRecording() {{
            try {{
                const stream = await navigator.mediaDevices.getUserMedia({{ 
                    audio: {{
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }} 
                }});
                
                mediaRecorder = new MediaRecorder(stream, {{
                    mimeType: 'audio/webm;codecs=opus'
                }});
                
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {{
                    if (event.data.size > 0) {{
                        audioChunks.push(event.data);
                    }}
                }};
                
                mediaRecorder.onstop = () => {{
                    const audioBlob = new Blob(audioChunks, {{ type: 'audio/webm' }});
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audioElement = document.getElementById('recordedAudio');
                    
                    audioElement.src = audioUrl;
                    audioElement.style.display = 'block';
                    document.getElementById('submitBtn').disabled = false;
                    
                    stream.getTracks().forEach(track => track.stop());
                }};
                
                mediaRecorder.start();
                recordingStartTime = Date.now();
                
                document.getElementById('startRecordBtn').disabled = true;
                document.getElementById('stopRecordBtn').disabled = false;
                document.getElementById('recordingStatus').textContent = '🔴 Recording...';
                document.getElementById('recordingStatus').style.color = '#dc3545';
                
                startRecordingTimer();
                
            }} catch (error) {{
                showStatus('❌ Microphone access denied or not available', 'error');
            }}
        }}

        function stopRecording() {{
            if (mediaRecorder && mediaRecorder.state === 'recording') {{
                mediaRecorder.stop();
                document.getElementById('startRecordBtn').disabled = false;
                document.getElementById('stopRecordBtn').disabled = true;
                document.getElementById('recordingStatus').textContent = '✅ Recording complete';
                document.getElementById('recordingStatus').style.color = '#28a745';
                stopRecordingTimer();
            }}
        }}

        function startRecordingTimer() {{
            recordingTimer = setInterval(() => {{
                const elapsed = Date.now() - recordingStartTime;
                const seconds = Math.floor(elapsed / 1000);
                const minutes = Math.floor(seconds / 60);
                const displaySeconds = seconds % 60;
                document.getElementById('recordingTimer').textContent = 
                    `${{minutes.toString().padStart(2, '0')}}:${{displaySeconds.toString().padStart(2, '0')}}`;
            }}, 1000);
        }}

        function stopRecordingTimer() {{
            if (recordingTimer) {{
                clearInterval(recordingTimer);
                recordingTimer = null;
            }}
        }}

        function resetRecording() {{
            audioChunks = [];
            const recordedAudio = document.getElementById('recordedAudio');
            if (recordedAudio) {{
                recordedAudio.style.display = 'none';
            }}
            document.getElementById('recordingTimer').textContent = '00:00';
            document.getElementById('recordingStatus').textContent = 'Click record to start';
            document.getElementById('recordingStatus').style.color = '#495057';
        }}

        async function handleFormSubmit(e) {{
            e.preventDefault();
            
            const voiceSelect = document.getElementById('voiceSelect');
            let audioBlob = null;
            
            if (currentInputMethod === 'upload') {{
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                if (!file) {{
                    showStatus('Please select an audio file', 'error');
                    return;
                }}
                audioBlob = file;
            }} else if (currentInputMethod === 'record') {{
                if (audioChunks.length === 0) {{
                    showStatus('Please record some audio first', 'error');
                    return;
                }}
                audioBlob = new Blob(audioChunks, {{ type: 'audio/webm' }});
            }}
            
            const submitBtn = document.getElementById('submitBtn');
            submitBtn.disabled = true;
            submitBtn.textContent = '⏳ Processing...';
            showStatus('Processing your audio...', 'processing');

            const formData = new FormData();
            formData.append('file', audioBlob, currentInputMethod === 'record' ? 'recording.webm' : 'audio.wav');
            formData.append('voice_character', voiceSelect.value);

            try {{
                const response = await fetch('/chat/', {{
                    method: 'POST',
                    body: formData
                }});

                const result = await response.json();
                console.log('Server response:', result);

                if (result.success) {{
                    showStatus('✅ Response generated successfully!', 'success');
                    
                    // Add to conversation history
                    const newConversation = {{
                        id: result.conversation_id,
                        timestamp: new Date().toLocaleTimeString(),
                        voice_character: result.voice_character,
                        user_audio: result.user_audio_url.split('/').pop(),
                        user_text: result.user_text,
                        assistant_audio: result.assistant_audio_url.split('/').pop(),
                        assistant_text: result.assistant_text
                    }};
                    
                    conversations.unshift(newConversation);
                    if (conversations.length > 5) {{
                        conversations = conversations.slice(0, 5);
                    }}
                    
                    renderConversations();
                    
                }} else {{
                    showStatus('❌ Error: ' + result.error, 'error');
                }}
            }} catch (error) {{
                showStatus('❌ Network error: ' + error.message, 'error');
            }} finally {{
                submitBtn.disabled = false;
                submitBtn.textContent = '🚀 Send Audio & Get Response';
                document.getElementById('uploadForm').reset();
                resetRecording();
                currentInputMethod = 'upload';
                document.getElementById('fileUploadGroup').style.display = 'block';
                document.getElementById('recordingGroup').style.display = 'none';
                document.getElementById('submitBtn').disabled = true;
            }}
        }}

        async function loadConversations() {{
            try {{
                const response = await fetch('/conversations');
                const data = await response.json();
                conversations = data.conversations || [];
                renderConversations();
            }} catch (error) {{
                console.error('Error loading conversations:', error);
            }}
        }}

        function renderConversations() {{
            const container = document.getElementById('conversationList');
            
            if (conversations.length === 0) {{
                container.innerHTML = '<div class="empty-state">No conversations yet. Upload an audio file to start chatting!</div>';
                return;
            }}

            container.innerHTML = conversations.map(conv => {{
                return `
                <div class="conversation-item">
                    <div class="conversation-header">
                        <div>
                            <span class="conversation-id">#${{conv.id}}</span>
                            <span class="voice-badge">${{conv.voice_character}}</span>
                        </div>
                        <span class="conversation-time">${{conv.timestamp}}</span>
                    </div>
                    
                    <div class="message user">
                        <div class="message-label">🎤 You said:</div>
                        <div>${{conv.user_text}}</div>
                        <div class="audio-section">
                            <audio controls onerror="handleAudioError(this, '${{conv.user_audio}}')">
                                <source src="/audio/${{conv.user_audio}}" type="audio/wav">
                            </audio>
                            <a href="/audio/${{conv.user_audio}}" download class="download-btn">
                                📥 Download
                            </a>
                        </div>
                    </div>
                    
                    <div class="message assistant">
                        <div class="message-label">🤖 Assistant replied:</div>
                        <div>${{conv.assistant_text}}</div>
                        <div class="audio-section">
                            <audio controls onerror="handleAudioError(this, '${{conv.assistant_audio}}')">
                                <source src="/audio/${{conv.assistant_audio}}" type="audio/wav">
                            </audio>
                            <a href="/audio/${{conv.assistant_audio}}" download class="download-btn">
                                📥 Download
                            </a>
                        </div>
                    </div>
                </div>
                `;
            }}).join('');
        }}

        function handleAudioError(audioElement, filename) {{
            console.error('Audio error for:', filename, audioElement.error);
            const errorDiv = document.createElement('div');
            errorDiv.className = 'audio-error';
            errorDiv.textContent = `Cannot play ${{filename}}. File may be corrupted or missing.`;
            audioElement.parentNode.appendChild(errorDiv);
        }}

        async function clearHistory() {{
            if (!confirm('Are you sure you want to clear all conversation history?')) {{
                return;
            }}

            try {{
                const response = await fetch('/clear', {{ method: 'POST' }});
                const result = await response.json();
                
                if (result.success) {{
                    showStatus('✅ History cleared successfully!', 'success');
                    conversations = [];
                    renderConversations();
                }} else {{
                    showStatus('❌ Clear failed: ' + result.message, 'error');
                }}
            }} catch (error) {{
                showStatus('❌ Error: ' + error.message, 'error');
            }}
        }}

        async function testAudioPlayback() {{
            try {{
                showStatus('Creating test audio...', 'processing');
                const response = await fetch('/test-audio-playback');
                const result = await response.json();
                
                if (result.success) {{
                    showStatus('✅ Test audio created!', 'success');
                    // Play the test audio
                    const audio = new Audio(result.url);
                    audio.play().catch(e => console.error('Audio play failed:', e));
                }} else {{
                    showStatus('❌ Test audio failed: ' + result.error, 'error');
                }}
            }} catch (error) {{
                showStatus('❌ Error creating test audio: ' + error.message, 'error');
            }}
        }}

        function showDebugInfo() {{
            alert('Debug information:\\n\\n' +
                  'Available endpoints:\\n' +
                  '- POST /chat/ - Send audio and get response\\n' +
                  '- GET /conversations - Get conversation history\\n' +
                  '- POST /clear - Clear history\\n' +
                  '- GET /audio/{{filename}} - Get audio file\\n' +
                  '- GET /test-audio-playback - Test audio playback');
        }}

        function showStatus(message, type) {{
            const statusDiv = document.getElementById('status');
            statusDiv.innerHTML = message;
            statusDiv.className = 'status ' + type;
            
            if (type !== 'processing') {{
                setTimeout(() => {{
                    statusDiv.innerHTML = '';
                    statusDiv.className = 'status';
                }}, 5000);
            }}
        }}
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

# ==================== SERVER STARTUP ====================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Voice Assistant Server...")
    print("🌐 Web interface available at: http://127.0.0.1:8000/")
    print("📋 Available endpoints:")
    print("   - GET  /                    - Web interface")
    print("   - POST /chat/               - Send audio and get response")
    print("   - GET  /conversations       - Get conversation history")
    print("   - POST /clear               - Clear history")
    print("   - GET  /audio/{filename}    - Get audio file")
    print("   - GET  /test-audio-playback - Test audio playback")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
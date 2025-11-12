import os
import pyttsx3
import warnings
warnings.filterwarnings("ignore")

def synthesize_speech(text, cosyvoice_model=None, voice_character="pyttsx3-male"):
    """
    Synthesize speech using either CosyVoice or pyttsx3
    """
    print(f"🎯 Starting TTS synthesis with voice: {voice_character}")
    
    try:
        # Use CosyVoice for cosyvoice character options
        if cosyvoice_model is not None and voice_character.startswith("cosyvoice-"):
            print(f"🔊 Attempting CosyVoice synthesis for: {voice_character}")
            cosy_success = _synthesize_cosyvoice(text, cosyvoice_model, voice_character)
            if cosy_success:
                return True
            else:
                print(f"❌ CosyVoice failed, falling back to pyttsx3 for: {voice_character}")
                # Fall back to pyttsx3 but maintain the intended voice type
                if "female" in voice_character:
                    return _synthesize_pyttsx3(text, "pyttsx3-female")
                else:
                    return _synthesize_pyttsx3(text, "pyttsx3-male")
        
        # Use pyttsx3 for pyttsx3 character options
        elif voice_character.startswith("pyttsx3-"):
            print(f"🔊 Using pyttsx3 synthesis for: {voice_character}")
            return _synthesize_pyttsx3(text, voice_character)
        
        else:
            print(f"⚠️ Unknown voice character: {voice_character}, using pyttsx3-male")
            return _synthesize_pyttsx3(text, "pyttsx3-male")
            
    except Exception as e:
        print(f"❌ Error in speech synthesis: {e}")
        # Fallback to basic pyttsx3
        return _synthesize_pyttsx3(text, "pyttsx3-male")

def _synthesize_cosyvoice(text, cosyvoice_model, voice_character):
    """Synthesize speech using CosyVoice with specific character voices"""
    try:
        print(f"🎯 CosyVoice synthesis for: {voice_character}")
        
        # Define voice file paths - CORRECT LINUX/WSL PATHS
        # Convert Windows path to WSL path
        female_voice_path = "/mnt/d/inference/assignment 3/generated_responses/female_voice.wav"
        male_voice_path = "/mnt/d/inference/assignment 3/generated_responses/male_voice.wav"
        prompt_path = os.path.join("generated_responses", "prompt.wav")
        
        # Alternative paths in case the above don't work
        alt_female_path = "/mnt/d/inference/assignment%203/generated_responses/female_voice.wav"
        alt_male_path = "/mnt/d/inference/assignment%203/generated_responses/male_voice.wav"
        
        print(f"🔍 Checking voice files...")
        print(f"   Female path: {female_voice_path}")
        print(f"   Female exists: {os.path.exists(female_voice_path)}")
        print(f"   Male path: {male_voice_path}")
        print(f"   Male exists: {os.path.exists(male_voice_path)}")
        print(f"   Prompt path: {prompt_path}")
        print(f"   Prompt exists: {os.path.exists(prompt_path)}")
        
        # Select the appropriate voice file
        voice_file = None
        if voice_character == "cosyvoice-female":
            if os.path.exists(female_voice_path):
                voice_file = female_voice_path
                print(f"✅ Using CosyVoice female character: {female_voice_path}")
            elif os.path.exists(alt_female_path):
                voice_file = alt_female_path
                print(f"✅ Using alternative female path: {alt_female_path}")
            else:
                print(f"❌ Female voice file not found at any path")
                return False
                
        elif voice_character == "cosyvoice-male":
            if os.path.exists(male_voice_path):
                voice_file = male_voice_path
                print(f"✅ Using CosyVoice male character: {male_voice_path}")
            elif os.path.exists(alt_male_path):
                voice_file = alt_male_path
                print(f"✅ Using alternative male path: {alt_male_path}")
            else:
                print(f"❌ Male voice file not found at any path")
                return False
                
        elif voice_character == "cosyvoice-custom" and os.path.exists(prompt_path):
            voice_file = prompt_path
            print(f"✅ Using CosyVoice custom character: {prompt_path}")
        else:
            print(f"❌ Voice file not found for {voice_character}")
            return False
        
        # CosyVoice inference
        output_path = os.path.join("generated_responses", "response.wav")
        
        # Debug: Check what methods are available
        methods = [method for method in dir(cosyvoice_model) if not method.startswith('_')]
        print(f"🔍 Available CosyVoice methods: {methods}")
        
        try:
            # Try different method names for CosyVoice
            if hasattr(cosyvoice_model, 'infer_to_file'):
                print("🎯 Using infer_to_file method")
                cosyvoice_model.infer_to_file(
                    text=text,
                    prompt_audio=voice_file,
                    output_file=output_path,
                    language="en",
                    speed=1.0
                )
            elif hasattr(cosyvoice_model, 'synthesize_to_file'):
                print("🎯 Using synthesize_to_file method")
                cosyvoice_model.synthesize_to_file(
                    text=text,
                    prompt_audio=voice_file,
                    output_path=output_path
                )
            elif hasattr(cosyvoice_model, 'generate_to_file'):
                print("🎯 Using generate_to_file method")
                cosyvoice_model.generate_to_file(
                    text=text,
                    reference_audio=voice_file,
                    output_path=output_path
                )
            else:
                print("❌ No known synthesis method found in CosyVoice")
                # Try to call the model directly as a function
                try:
                    print("🎯 Trying to call CosyVoice model directly")
                    result = cosyvoice_model(
                        text=text,
                        prompt_audio=voice_file,
                        output_file=output_path
                    )
                    print(f"✅ Direct call result: {result}")
                except Exception as direct_error:
                    print(f"❌ Direct call failed: {direct_error}")
                    return False
                
        except Exception as cosy_error:
            print(f"❌ CosyVoice synthesis failed: {cosy_error}")
            return False
        
        # Check if file was created successfully
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                print(f"✅ CosyVoice speech synthesized successfully: {output_path} ({file_size} bytes)")
                return True
            else:
                print("❌ CosyVoice synthesis failed: Output file is empty")
                return False
        else:
            print("❌ CosyVoice synthesis failed: Output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Error in CosyVoice synthesis: {e}")
        return False

def _synthesize_pyttsx3(text, voice_character):
    """Synthesize speech using pyttsx3 with proper voice selection"""
    try:
        print(f"🎯 Initializing pyttsx3 for voice: {voice_character}")
        
        # Initialize engine
        engine = pyttsx3.init()
        
        # Configure engine properties
        engine.setProperty('rate', 180)  # Speech rate
        engine.setProperty('volume', 0.9)  # Volume level
        
        # Get available voices
        voices = engine.getProperty('voices')
        print(f"🔍 Found {len(voices)} available system voices")
        
        # List all available voices for debugging
        for i, voice in enumerate(voices):
            gender = "Female" if any(keyword in voice.name.lower() for keyword in ['female', 'woman', 'zira']) else "Male"
            print(f"   {i}: {voice.name} - {gender}")
        
        target_voice = None
        
        if voice_character == "pyttsx3-female":
            print("🎯 Looking for female voice...")
            # Try to find a female voice
            female_voices = [
                voice for voice in voices 
                if any(keyword in voice.name.lower() for keyword in [
                    'female', 'woman', 'zira', 'karen', 'kate', 'veena', 
                    'tessa', 'samantha', 'victoria', 'moira'
                ])
            ]
            
            if female_voices:
                target_voice = female_voices[0]
                print(f"✅ Using female voice: {target_voice.name}")
            else:
                print("⚠️ No female voice found, trying alternative selection...")
                # On many systems, the second voice is often female
                if len(voices) > 1:
                    target_voice = voices[1]
                    print(f"🔄 Using alternative voice (index 1): {target_voice.name}")
                else:
                    target_voice = voices[0]
                    print(f"⚠️ Only one voice available, using: {target_voice.name}")
                    
        else:  # pyttsx3-male or default
            print("🎯 Looking for male voice...")
            # Try to find a male voice
            male_voices = [
                voice for voice in voices 
                if any(keyword in voice.name.lower() for keyword in [
                    'male', 'man', 'david', 'mark', 'alex', 'bruce',
                    'fred', 'alex', 'lee', 'rishabh'
                ])
            ]
            
            if male_voices:
                target_voice = male_voices[0]
                print(f"✅ Using male voice: {target_voice.name}")
            else:
                print("⚠️ No male voice found, using first available voice")
                target_voice = voices[0]
                print(f"🔄 Using default voice: {target_voice.name}")
        
        # Set the selected voice
        if target_voice:
            engine.setProperty('voice', target_voice.id)
            print(f"🎯 Voice set to: {target_voice.name}")
        else:
            print("❌ No voice selected, using system default")
        
        # Save to file
        output_path = os.path.join("generated_responses", "response.wav")
        
        try:
            print("🔄 Starting pyttsx3 synthesis...")
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            print("✅ pyttsx3 synthesis completed")
        except Exception as synth_error:
            print(f"❌ Error during pyttsx3 synthesis: {synth_error}")
            return False
        finally:
            try:
                engine.stop()
            except:
                pass  # Ignore stop errors
        
        # Check if file was created successfully
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                print(f"✅ pyttsx3 speech synthesized successfully: {output_path} ({file_size} bytes)")
                return True
            else:
                print("❌ pyttsx3 synthesis failed: Output file is empty")
                return False
        else:
            print("❌ pyttsx3 synthesis failed: Output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Error in pyttsx3 synthesis: {e}")
        return False

def list_available_voices():
    """List all available system voices for debugging"""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        print("\n=== Available System Voices ===")
        for i, voice in enumerate(voices):
            gender = "Female" if any(keyword in voice.name.lower() for keyword in [
                'female', 'woman', 'zira', 'karen', 'kate', 'veena', 
                'tessa', 'samantha', 'victoria', 'moira'
            ]) else "Male"
            print(f"{i}: {voice.name} | {gender} | {voice.id}")
        engine.stop()
        return voices
    except Exception as e:
        print(f"Error listing voices: {e}")
        return []
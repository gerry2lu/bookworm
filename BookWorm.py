import whisper
import sounddevice as sd
import numpy as np
import requests
import json
import subprocess
import tempfile
import wave
import os
import threading
import time
from pathlib import Path
import queue
import collections
import webrtcvad
import librosa
from scipy import signal
import re

class VoiceAI:
    def __init__(self):
        print("Booting BookWorm...")
        
        # Audio settings
        self.channels = 1
        self.sample_rate = 16000  # Standard for voice processing
        self.chunk_duration = 0.03  # 30ms chunks for VAD
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Voice Activity Detection with higher sensitivity
        print("Initializing Voice Activity Detection...")
        self.vad = webrtcvad.Vad(1)  # Reduced from 2 to 1 for higher sensitivity
        
        # Wake word detection settings
        self.wake_words = ["computer", "bookworm", "hey computer"]
        self.wake_word_threshold = 0.6
        self.listening_buffer = collections.deque(maxlen=int(2.0 / self.chunk_duration))  # 2 second buffer
        
        # Load Whisper model - using tiny for speed
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("tiny.en")
        
        # Ollama settings
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "gemma2:2b"
        
        # System prompt - optimized for concise but helpful responses
        self.system_prompt = """You are BookWorm, a friendly AI reading companion for children. Give helpful responses about books and reading in 1-2 sentences. Do not give formatting as your output will be voiced. Be enthusiastic and encouraging. If you don't know something specific, suggest related books or reading activities."""
        
        # TTS settings
        self.piper_model_path = "/home/gerrylu/voice-ai/piper-models/en_GB-semaine-medium.onnx"
        
        # Audio queues and control
        self.audio_queue = queue.Queue()
        self.should_stop = False
        self.is_processing = False
        
        print("BookWorm initialized successfully!")
        self.preload_ollama()
    
    def preload_ollama(self):
        """Pre-warm the Ollama model with longer timeout for Pi"""
        print("Pre-warming Ollama model...")
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": "Hi",
                "stream": False,
                "options": {"num_predict": 1}
            }
            # Much longer timeout for Raspberry Pi
            requests.post(self.ollama_url, json=payload, timeout=60)
            print("✅ Ollama model ready!")
        except Exception as e:
            print(f"⚠️ Could not pre-warm Ollama: {e}")
            print("   Model will load on first use (may be slow)")
    
    def audio_callback(self, indata, frames, time, status):
        """Optimized audio callback"""
        if not self.should_stop and not self.is_processing:
            # Convert to int16 for VAD
            audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
            self.audio_queue.put(audio_int16)
    
    def has_voice_activity(self, audio_chunk):
        """Check if audio chunk contains voice activity"""
        try:
            # VAD expects 16kHz and specific frame sizes (10, 20, or 30ms)
            if len(audio_chunk) == self.chunk_size:
                return self.vad.is_speech(audio_chunk.tobytes(), self.sample_rate)
            return False
        except:
            return False
    
    def simple_wake_word_detection(self, audio_buffer):
        """More sensitive wake word detection"""
        try:
            # Combine all audio chunks
            audio_data = np.concatenate(list(audio_buffer))
            
            # Lower energy threshold for better sensitivity
            energy = np.sum(audio_data.astype(np.float32) ** 2) / len(audio_data)
            if energy < 500:  # Reduced from 1000 for better sensitivity
                return False
            
            # Quick transcription of the buffer
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                try:
                    with wave.open(temp_audio.name, 'wb') as wav_file:
                        wav_file.setnchannels(self.channels)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(self.sample_rate)
                        wav_file.writeframes(audio_data.tobytes())
                    
                    # More sensitive transcription settings
                    result = self.whisper_model.transcribe(
                        temp_audio.name,
                        fp16=False,
                        language="en",
                        task="transcribe",
                        condition_on_previous_text=False,
                        no_speech_threshold=0.4,  # Reduced from 0.6 for better sensitivity
                        logprob_threshold=-1.0,   # More lenient
                        compression_ratio_threshold=2.4  # More lenient
                    )
                    
                    text = result["text"].lower().strip()
                    
                    # More flexible wake word matching
                    for wake_word in self.wake_words:
                        # Also check for partial matches
                        if wake_word in text or any(word in wake_word for word in text.split()):
                            print(f"✅ Wake word detected: '{wake_word}' in '{text}'")
                            return True
                    
                    return False
                    
                finally:
                    try:
                        os.unlink(temp_audio.name)
                    except:
                        pass
                        
        except Exception as e:
            print(f"Wake word detection error: {e}")
            return False
    
    def record_command_audio(self, duration=5):
        """Record audio for command - 5 seconds"""
        print("🎤 Listening for your question...")
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Record with correct duration
        try:
            print("   (Speak now - I'm listening for 5 seconds...)")
            recording = sd.rec(
                int(self.sample_rate * duration),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32
            )
            sd.wait()  # Wait for recording to complete
            print("   Recording complete!")
            return recording[:, 0]
        except Exception as e:
            print(f"Recording error: {e}")
            return np.array([])
    
    def speech_to_text_fast(self, audio_data):
        """Optimized speech-to-text"""
        if len(audio_data) == 0:
            return ""
        
        print("🔄 Processing...")
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                # Write audio file
                with wave.open(temp_audio.name, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.sample_rate)
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
                
                # Fast transcription
                result = self.whisper_model.transcribe(
                    temp_audio.name,
                    fp16=False,
                    language="en",
                    task="transcribe",
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    temperature=0  # More deterministic
                )
                
                text = result["text"].strip()
                print(f"📝 You said: '{text}'")
                return text
                
        except Exception as e:
            print(f"❌ Speech recognition error: {e}")
            return ""
        finally:
            try:
                os.unlink(temp_audio.name)
            except:
                pass
    
    def query_ollama_fast(self, text):
        """Better Ollama querying with improved responses"""
        print("🤖 Thinking...")
        
        try:
            # Test if Ollama is responding first with longer timeout
            test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if test_response.status_code != 200:
                print("❌ Ollama service not responding")
                return "I can't think right now. Ollama isn't working!"
            
            # Better prompt structure for more complete responses
            prompt = f"""System: {self.system_prompt}

Child asked: "{text}"

BookWorm responds:"""
            
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": 60,  # Allow longer responses
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "stop": ["\nChild:", "\nSystem:", "Child asked:"],
                    "num_ctx": 1024,  # Larger context for better understanding
                }
            }
            
            # Longer timeout for Pi - first request loads model
            response = requests.post(self.ollama_url, json=payload, stream=True, timeout=120)
            response.raise_for_status()
            
            full_response = ""
            start_time = time.time()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            full_response += chunk["response"]
                        if chunk.get("done", False):
                            break
                        
                        # More generous timeout for better responses
                        if time.time() - start_time > 60:
                            print("⚠️ Response taking long, but keeping what we have")
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            # Clean up response but keep it more complete
            ai_response = full_response.strip()
            
            # Remove any system artifacts but keep the response substantial
            if "BookWorm responds:" in ai_response:
                ai_response = ai_response.split("BookWorm responds:")[-1].strip()
            if "BookWorm:" in ai_response:
                ai_response = ai_response.split("BookWorm:")[-1].strip()
            
            # Limit to reasonable length for speech (but not too short)
            if len(ai_response) > 300:
                sentences = ai_response.split('. ')
                if len(sentences) > 1:
                    ai_response = '. '.join(sentences[:3]) + '.'
                else:
                    ai_response = ai_response[:300] + "..."
            
            if not ai_response or len(ai_response.strip()) < 10:
                ai_response = "That's an interesting question! I'd love to help you explore more books about that topic."
                
            print(f"🤖 Response: {ai_response}")
            return ai_response
            
        except requests.exceptions.Timeout:
            print("❌ Ollama timeout - Pi needs more time for first load")
            return "I'm thinking slowly. Let me try to help - what kind of books do you like to read?"
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama")
            return "I can't connect to my brain right now! But I bet you can find some great books to read while I get better."
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return "Sorry, I'm having trouble thinking! Tell me, what's your favorite book?"
    
    def text_to_speech_optimized(self, text):
        """Multi-method TTS with comprehensive fallbacks"""
        if not text:
            return
        
        print("🔊 Speaking...")
        
        # Method 1: Try Piper first (if working)
        if self.try_piper_tts(text):
            return
        
        # Method 2: Try simple_tts (our custom solution)
        if self.try_simple_tts(text):
            return
        
        # Method 3: Try Festival
        if self.try_festival_tts(text):
            return
        
        # Method 4: Fall back to espeak (most reliable)
        self.try_espeak_tts(text)
    
    def try_piper_tts(self, text):
        """Try Piper TTS"""
        try:
            if not os.path.exists(self.piper_model_path):
                return False
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                cmd = [
                    "piper", 
                    "--model", self.piper_model_path, 
                    "--output_file", temp_audio.name
                ]
                
                process = subprocess.run(
                    cmd,
                    input=text,
                    text=True,
                    capture_output=True,
                    timeout=100
                )
                
                if process.returncode == 0 and os.path.exists(temp_audio.name) and os.path.getsize(temp_audio.name) > 0:
                    play_result = subprocess.run(
                        ["aplay", temp_audio.name], 
                        capture_output=True, 
                        timeout=100
                    )
                    if play_result.returncode == 0:
                        print("✅ Piper TTS successful!")
                        return True
                        
        except Exception as e:
            print(f"🔄 Piper failed: {e}")
        finally:
            try:
                if 'temp_audio' in locals() and os.path.exists(temp_audio.name):
                    os.unlink(temp_audio.name)
            except:
                pass
        return False
    
    def try_simple_tts(self, text):
        """Try our simple TTS alternative"""
        try:
            if not os.path.exists("/usr/local/bin/simple_tts"):
                return False
                
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                process = subprocess.run(
                    ["simple_tts", "--model", "dummy", "--output_file", temp_audio.name],
                    input=text,
                    text=True,
                    capture_output=True,
                    timeout=100
                )
                
                if process.returncode == 0 and os.path.exists(temp_audio.name) and os.path.getsize(temp_audio.name) > 0:
                    play_result = subprocess.run(
                        ["aplay", temp_audio.name], 
                        capture_output=True, 
                        timeout=100
                    )
                    if play_result.returncode == 0:
                        print("✅ Simple TTS successful!")
                        return True
                        
        except Exception as e:
            print(f"🔄 Simple TTS failed: {e}")
        finally:
            try:
                if 'temp_audio' in locals() and os.path.exists(temp_audio.name):
                    os.unlink(temp_audio.name)
            except:
                pass
        return False
    
    def try_festival_tts(self, text):
        """Try Festival TTS"""
        try:
            if not subprocess.run(["which", "festival"], capture_output=True).returncode == 0:
                return False
                
            result = subprocess.run([
                "festival", "--tts"
            ], input=text, text=True, capture_output=True, timeout=10)
            
            if result.returncode == 0:
                print("✅ Festival TTS successful!")
                return True
                
        except Exception as e:
            print(f"🔄 Festival failed: {e}")
        return False
    
    def try_espeak_tts(self, text):
        """Try espeak TTS (most reliable fallback)"""
        try:
            result = subprocess.run([
                "espeak", "-s", "150", "-v", "en+f3", text
            ], capture_output=True, timeout=10)
            
            if result.returncode == 0:
                print("✅ Espeak TTS successful!")
                return True
            else:
                print(f"❌ Espeak failed: {result.stderr.decode()}")
                
        except Exception as e:
            print(f"❌ Espeak error: {e}")
        
        print("❌ All TTS methods failed!")
        return False
    
    def startup_greeting(self):
        """Quick startup message"""
        greeting = "BookWorm ready! Say 'Computer' to ask me about books!"
        print(f"🤖 {greeting}")
        self.text_to_speech_optimized(greeting)
    
    def wake_word_listener(self):
        """Optimized wake word detection loop"""
        print("👂 Listening for 'Computer' or 'BookWorm'...")
        
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            dtype=np.float32,
            blocksize=self.chunk_size
        )
        
        stream.start()
        
        try:
            while not self.should_stop:
                try:
                    # Get audio chunk
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                    
                    # Check for voice activity first (much faster)
                    if self.has_voice_activity(audio_chunk):
                        # Add to listening buffer
                        self.listening_buffer.append(audio_chunk)
                        
                        # Check for wake word when buffer is full
                        if len(self.listening_buffer) >= self.listening_buffer.maxlen // 2:
                            if self.simple_wake_word_detection(self.listening_buffer):
                                self.is_processing = True
                                
                                # Brief pause
                                time.sleep(0.3)
                                
                                # Record and process command
                                command_audio = self.record_command_audio()
                                if len(command_audio) > 0:
                                    text = self.speech_to_text_fast(command_audio)
                                    if text and len(text.strip()) > 3:
                                        response = self.query_ollama_fast(text)
                                        self.text_to_speech_optimized(response)
                                    else:
                                        self.text_to_speech_optimized("I didn't catch that!")
                                
                                # Clear buffer and resume
                                self.listening_buffer.clear()
                                self.is_processing = False
                                print("👂 Listening for wake words...")
                    
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    self.is_processing = False
                    time.sleep(0.5)
                    
        finally:
            stream.stop()
            stream.close()
    
    def run(self):
        """Main loop"""
        print("\n🚀 Starting BookWorm...")
        
        try:
            self.startup_greeting()
            self.wake_word_listener()
            
        except KeyboardInterrupt:
            print("\n👋 BookWorm shutting down...")
        except Exception as e:
            print(f"❌ Fatal error: {e}")
        finally:
            self.should_stop = True
            print("BookWorm stopped.")

def check_dependencies():
    """Check dependencies and provide helpful messages"""
    print("🔍 Checking dependencies...")
    
    # Check Piper
    piper_model = "/home/gerrylu/voice-ai/piper-models/en_GB-semaine-medium.onnx"
    if not Path(piper_model).exists():
        print(f"⚠️ Piper model missing: {piper_model}")
        print("   Will use espeak fallback")
    else:
        print("✅ Piper model found!")
    
    # Check required Python packages
    required_packages = ['whisper', 'sounddevice', 'webrtcvad', 'librosa', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service ready!")
        else:
            print("⚠️ Ollama not responding properly")
            return False
    except requests.exceptions.RequestException:
        print("❌ Ollama not running - start with 'ollama serve'")
        return False
    
    return True

if __name__ == "__main__":
    if not check_dependencies():
        print("\n❌ Please fix dependencies before starting")
        exit(1)
    
    voice_ai = VoiceAI()
    voice_ai.run()
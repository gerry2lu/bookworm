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
import pvporcupine
import struct

class VoiceAI:
    def __init__(self):
        print("Booting BookWorm...")
        
        # Initialize Porcupine wake word detection
        print("Loading Porcupine wake word detector...")
        try:
            self.porcupine = pvporcupine.create(
                keywords=['computer'],  # Built-in keyword similar to "bookworm"
                sensitivities=[0.5]     # Adjust sensitivity (0.0-1.0)
            )
            print("✅ Porcupine wake word detector loaded!")
        except Exception as e:
            print(f"❌ Failed to load Porcupine: {e}")
            raise
        
        # Load Whisper only for speech-to-text (not wake word detection)
        print("Loading Whisper model for speech recognition...")
        self.whisper_model = whisper.load_model("tiny.en")
        
        # Ollama settings
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "gemma2:2b"  # Faster than gemma3:1b
        
        # System prompt
        self.system_prompt = """You are BookWorm, a helpful AI reading companion for children. Keep responses very short and friendly since they will be spoken aloud. Use simple words and 1-2 sentences maximum unless asked for details. Be encouraging about reading and learning."""
        
        # TTS settings
        self.piper_model_path = "/home/gerrylu/voice-ai/piper-models/en_GB-semaine-medium.onnx"
        
        # Audio settings - Porcupine requires 16kHz
        self.channels = 1
        self.sample_rate = self.porcupine.sample_rate  # Porcupine requires 16kHz
        self.frame_length = self.porcupine.frame_length  # Porcupine frame size
        
        # Audio queues and threads
        self.audio_queue = queue.Queue()
        self.should_stop = False
        
        print(f"Audio settings: {self.sample_rate}Hz, frame length: {self.frame_length}")
        print("BookWorm initialized successfully!")
        
        # Pre-warm Ollama model
        self.preload_ollama()
    
    def audio_callback(self, indata, frames, time, status):
        """Audio callback for Porcupine wake word detection"""
        if not self.should_stop:
            # Convert float32 to int16 for Porcupine
            audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
            self.audio_queue.put(audio_int16)
    
    def preload_ollama(self):
        """Pre-warm the Ollama model to reduce first response delay"""
        print("Pre-warming Ollama model...")
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": "Hello",
                "stream": False,
                "options": {"num_predict": 1}
            }
            requests.post(self.ollama_url, json=payload, timeout=10)
            print("Ollama model ready!")
        except Exception as e:
            print(f"Warning: Could not pre-warm Ollama: {e}")
    
    def detect_wake_word(self, audio_frame):
        """Efficient wake word detection using Porcupine"""
        try:
            # Porcupine expects exactly frame_length samples
            if len(audio_frame) == self.frame_length:
                keyword_index = self.porcupine.process(audio_frame)
                return keyword_index >= 0  # Returns -1 if no keyword detected
            return False
        except Exception as e:
            print(f"Wake word detection error: {e}")
            return False
    
    def record_command_audio(self, duration=5):
        """Record audio for command after wake word detected"""
        print("🎤 Listening for your question...")
        
        audio_data = []
        start_time = time.time()
        
        # Clear any old audio from queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Record fresh audio for the command
        command_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32
        )
        
        with command_stream:
            audio_data = command_stream.read(int(self.sample_rate * duration))[0]
        
        return audio_data[:, 0] if len(audio_data.shape) > 1 else audio_data
    
    def speech_to_text(self, audio_data):
        """Convert speech to text using Whisper - optimized version"""
        if len(audio_data) == 0:
            return ""
        
        print("🔄 Processing your question...")
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                with wave.open(temp_audio.name, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.sample_rate)
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
                
                # Fast transcription with optimized settings
                result = self.whisper_model.transcribe(
                    temp_audio.name, 
                    fp16=False,
                    language="en",
                    task="transcribe"
                )
                text = result["text"].strip()
                print(f"📝 You asked: '{text}'")
                return text
                
        except Exception as e:
            print(f"❌ Speech recognition error: {e}")
            return ""
        finally:
            try:
                os.unlink(temp_audio.name)
            except:
                pass
    
    def query_ollama_streaming(self, text):
        """Fast streaming response from Ollama"""
        print("🤖 Thinking...")
        
        try:
            full_prompt = f"{self.system_prompt}\n\nChild: {text}\nBookWorm:"
            
            payload = {
                "model": self.ollama_model,
                "prompt": full_prompt,
                "stream": True,  # Enable streaming for faster response
                "options": {
                    "num_predict": 50,  # Shorter responses
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "stop": ["\n\nChild:", "Child:", "\n\n"]
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, stream=True, timeout=15)
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            full_response += chunk["response"]
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            # Clean up response
            ai_response = full_response.strip()
            if len(ai_response) > 200:  # Keep it short for children
                sentences = ai_response.split('. ')
                ai_response = '. '.join(sentences[:2]) + '.'
            
            print(f"🤖 BookWorm says: {ai_response}")
            return ai_response
            
        except Exception as e:
            error_msg = "Sorry, I'm having trouble thinking right now!"
            print(f"❌ {error_msg}: {e}")
            return error_msg
    
    def text_to_speech_fast(self, text):
        """Optimized text-to-speech using Piper with espeak fallback"""
        if not text:
            return
        
        print("🔊 Speaking...")
        
        try:
            # Use Piper with your model
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
                    timeout=10,
                )

                if process.returncode != 0:
                    print(f"🔄 Piper failed with return code {process.returncode}")
                    print("Piper stdout:", process.stdout)
                    print("Piper stderr:", process.stderr)
                    raise subprocess.CalledProcessError(process.returncode, cmd)

                
                # Play the generated audio
                subprocess.run(["aplay", temp_audio.name], check=True, capture_output=True)
                
        except subprocess.CalledProcessError as e:
            print("🔄 Piper failed, using espeak fallback...")
            try:
                subprocess.run([
                    "espeak", 
                    "-s", "150",  # Speed
                    "-v", "en+f3",  # Voice
                    text
                ], check=True, capture_output=True, timeout=10)
            except Exception as fallback_error:
                print(f"❌ All TTS methods failed: {fallback_error}")
        
        except subprocess.TimeoutExpired:
            print("❌ Piper timeout, using espeak...")
            try:
                subprocess.run(["espeak", text], check=True, capture_output=True)
            except:
                print("❌ Fallback TTS also failed")
        
        except Exception as e:
            print(f"❌ TTS error: {e}")
        
        finally:
            # Clean up temp file
            try:
                if 'temp_audio' in locals():
                    os.unlink(temp_audio.name)
            except:
                pass
    
    def startup_greeting(self):
        """Greeting message on startup"""
        greeting = "Hi there! I'm BookWorm, your reading companion. Say 'Computer' whenever you want to chat about books!"
        print(f"🤖 {greeting}")
        self.text_to_speech_fast(greeting)
    
    def wake_word_listener(self):
        """Efficient wake word detection loop using Porcupine"""
        print("👂 Listening for wake word 'Computer'...")
        
        # Start audio stream with Porcupine-specific settings
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            dtype=np.float32,
            blocksize=self.frame_length
        )
        
        stream.start()
        
        try:
            while not self.should_stop:
                try:
                    # Get audio frame (int16 format for Porcupine)
                    audio_frame = self.audio_queue.get(timeout=1.0)
                    
                    # Check for wake word using Porcupine
                    if self.detect_wake_word(audio_frame):
                        print("✅ Wake word 'Computer' detected!")
                        
                        # Brief pause to let user start speaking
                        time.sleep(0.5)
                        
                        # Record command
                        command_audio = self.record_command_audio()
                        
                        # Process command
                        if len(command_audio) > 0:
                            text = self.speech_to_text(command_audio)
                            if text and len(text.strip()) > 2:  # Ignore very short utterances
                                response = self.query_ollama_streaming(text)
                                self.text_to_speech_fast(response)
                            else:
                                self.text_to_speech_fast("I didn't catch that. Try asking again!")
                        
                        print("👂 Listening for wake word 'Computer'...")
                        
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error in wake word loop: {e}")
                    time.sleep(1)  # Brief pause before retrying
                    
        finally:
            stream.stop()
            stream.close()
            self.porcupine.delete()  # Clean up Porcupine resources
    
    def run(self):
        """Main application loop"""
        print("\n🚀 BookWorm is starting up!")
        
        # Give startup greeting
        self.startup_greeting()
        
        try:
            # Start wake word detection
            self.wake_word_listener()
            
        except KeyboardInterrupt:
            print("\n👋 BookWorm is going to sleep. Goodbye!")
        except Exception as e:
            print(f"❌ Application error: {e}")
        finally:
            self.should_stop = True
            print("BookWorm shutdown complete.")

def check_dependencies():
    """Check if required files and services exist"""
    piper_model = "/home/gerrylu/voice-ai/piper-models/en_GB-semaine-medium.onnx"
    if not Path(piper_model).exists():
        print(f"⚠️ Piper model not found at {piper_model} - will use espeak for TTS")
    else:
        print("✅ Piper model found!")
    
    # Check if piper command exists
    try:
        subprocess.run(["piper", "--help"], capture_output=True, check=True)
        print("✅ Piper command available!")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Piper command not found - will use espeak for TTS")
    
    # Test Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("⚠️ Ollama service not responding - please start it with 'ollama serve'")
            return False
        print("✅ Ollama service ready!")
    except requests.exceptions.RequestException:
        print("⚠️ Cannot connect to Ollama - please start it with 'ollama serve'")
        return False
    
    return True

if __name__ == "__main__":
    if not check_dependencies():
        print("❌ Please fix dependencies before starting BookWorm")
        exit(1)
    
    # Initialize and run BookWorm
    voice_ai = VoiceAI()
    voice_ai.run()
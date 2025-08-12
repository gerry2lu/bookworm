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
import select
import sys
import termios
import tty
from pathlib import Path

class VoiceAI:
    def __init__(self):
        print("Booting BookWorm")
        self.whisper_model = whisper.load_model("tiny")

        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "gemma3:1b"

        self.system_prompt = """ You are a helpful voice assistant. Keep your responses concise and conversational since they will be spoken aloud. Aim for 1-3 sentences maximum unless specifically asked for detailed information. Be friendly and natural in your responses."""

        self.piper_model_path = str(Path.home() / "voice-ai/piper-models/en_GB-semaine-medium.onnx")

        self.sample_rate = 16000
        self.channels = 1
        self.recording = False
        self.audio_data = []


        print("Voice AI initialised successfully")
        print("Controls:")
        print(" SPACE - Hold to record, release to process")
        print(" ESC - Quit Application")
    
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio recording"""
        if self.recording:
            self.audio_data.extend(indata[:, 0])
    
    def record_audio(self):
        """Record audio until user presses enter again"""
        print("\n🎤 Recording... (press ENTER again to stop)")
        self.recording = True
        self.audio_data = []
        
        # Start audio stream
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            dtype=np.float32
        )
        
        stream.start()
        
        # Wait for enter key press to stop
        try:
            input()  # Wait for enter key
        except:
            pass
        
        stream.stop()
        stream.close()
        
        self.recording = False
        print("🛑 Recording stopped")
        
        return np.array(self.audio_data, dtype=np.float32)
    
    def speech_to_text(self, audio_data):
        """Convert speech to text using Whisper"""
        if len(audio_data) == 0:
            return ""
        
        print("🔄 Converting speech to text...")
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            # Write WAV file
            with wave.open(temp_audio.name, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                # Convert float32 to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Use Whisper to transcribe
            try:
                result = self.whisper_model.transcribe(temp_audio.name)
                text = result["text"].strip()
                print(f"📝 You said: '{text}'")
                return text
            except Exception as e:
                print(f"❌ Speech recognition error: {e}")
                return ""
            finally:
                # Clean up temp file
                os.unlink(temp_audio.name)
    
    def query_ollama(self, text):
        """Send text to Ollama and get response"""
        print("🤖 Thinking...")
        
        try:
            # Combine system prompt with user input
            full_prompt = f"{self.system_prompt}\n\nUser: {text}\nAssistant:"
            
            payload = {
                "model": self.ollama_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "num_predict": 100,  # Limit response to ~100 tokens
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["\n\nUser:", "User:", "\n\n"]  # Stop at conversation markers
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            ai_response = result.get("response", "").strip()
            
            # Additional length limiting (fallback)
            if len(ai_response) > 500:  # ~500 characters max
                sentences = ai_response.split('. ')
                if len(sentences) > 3:
                    ai_response = '. '.join(sentences[:3]) + '.'
                else:
                    ai_response = ai_response[:500] + "..."
            
            print(f"🤖 AI Response: {ai_response}")
            return ai_response
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error connecting to Ollama: {e}"
            print(f"❌ {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def text_to_speech(self, text):
        """Convert text to speech using Piper TTS"""
        if not text:
            return
        
        print("🔊 Converting text to speech...")
        
        try:
            # Create temporary files for audio output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                # Use Piper to generate speech
                cmd = [
                    "piper",
                    "--model", self.piper_model_path,
                    "--output_file", temp_audio.name
                ]
                
                # Run Piper with text input
                process = subprocess.run(
                    cmd,
                    input=text,
                    text=True,
                    capture_output=True,
                    check=True
                )
                
                # Play the generated audio
                subprocess.run(["aplay", temp_audio.name], check=True)
                print("✅ Speech playback completed")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ TTS Error: {e}")
            # Fallback to espeak
            print("🔄 Falling back to espeak...")
            try:
                subprocess.run(["espeak", text], check=True)
            except subprocess.CalledProcessError:
                print("❌ Fallback TTS also failed")
        except Exception as e:
            print(f"❌ Unexpected TTS error: {e}")
        finally:
            # Clean up temp files
            try:
                if 'temp_audio' in locals():
                    os.unlink(temp_audio.name)
            except:
                pass
    
    def run(self):
        """Main application loop"""
        print("\nVoice AI is ready!")
        print("Press ENTER to start talking...")
        
        try:
            while True:
                # Wait for user input
                user_input = input("\nPress ENTER to record (or type 'q' to quit): ").strip().lower()
                
                if user_input == 'q':
                    print("Goodbye!")
                    break
                
                # Record audio
                audio_data = self.record_audio()
                
                if len(audio_data) > 0:
                    # Speech to text
                    text = self.speech_to_text(audio_data)
                    
                    if text:
                        # Query LLM
                        response = self.query_ollama(text)
                        
                        # Text to speech
                        self.text_to_speech(response)
                    else:
                        print("❌ No speech detected, try again")
                
        except KeyboardInterrupt:
            print("\n👋 Application interrupted. Goodbye!")
        except Exception as e:
            print(f"❌ Application error: {e}")
        finally:
            self.restore_terminal()

if __name__ == "__main__":
    # Check if Piper model exists
    piper_model = Path.home() / "voice-ai/piper-models/en_GB-semaine-medium.onnx"
    if not piper_model.exists():
        print("❌ Piper model not found!")
        print("Please download the model files as described in the setup instructions.")
        exit(1)
    
    # Initialize and run the voice AI
    voice_ai = VoiceAI()
    voice_ai.run()
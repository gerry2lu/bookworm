#!/usr/bin/env python3
"""
Simplified BookWorm Reading Facilitator with RFID Integration
Based on your working setup but with RFID triggering
"""

import whisper
import sounddevice as sd
import numpy as np
import requests
import subprocess
import tempfile
import wave
import os
import time
import random

# RFID
from PiicoDev_RFID import PiicoDev_RFID
from PiicoDev_Unified import sleep_ms

class BookWormRFID:
    def __init__(self):
        print("🚀 Booting BookWorm with RFID...")
        
        # Audio settings
        self.channels = 1
        self.sample_rate = 16000
        self.recording_duration = 5.0
        
        # Initialize RFID
        print("📡 Initializing RFID...")
        try:
            self.rfid = PiicoDev_RFID()
            self.rfid_working = True
            print("✅ RFID module ready")
        except Exception as e:
            print(f"❌ RFID initialization failed: {e}")
            self.rfid_working = False
            return
        
        # Load Whisper model
        print("🔄 Loading Whisper model...")
        self.whisper_model = whisper.load_model("tiny.en")
        print("✅ Whisper ready")
        
        # Ollama settings
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "gemma2:2b"
        
        # TTS settings
        self.piper_model_path = "/home/gerrylu/voice-ai/piper-models/en_GB-semaine-medium.onnx"
        
        # System state
        self.last_rfid_time = 0
        self.debounce_time = 3.0  # 3 seconds between reads
        self.running = False
        
        # Thinking responses for low latency
        self.thinking_responses = [
            "Let me think about that...",
            "Hmm, that's interesting...", 
            "Good question! Give me a moment...",
            "I'm thinking...",
            "What a great question..."
        ]
        
        print("✅ BookWorm initialization complete!")
        self.preload_ollama()
    
    def preload_ollama(self):
        """Pre-warm Ollama"""
        print("🔄 Pre-warming Ollama...")
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": "Hi",
                "stream": False,
                "options": {"num_predict": 1}
            }
            requests.post(self.ollama_url, json=payload, timeout=30)
            print("✅ Ollama ready!")
        except Exception as e:
            print(f"⚠️ Ollama pre-warm failed: {e}")
    
    def speak_piper(self, text):
        """Speak using Piper TTS"""
        if not text:
            return False
            
        print(f"🔊 Speaking: {text}")
        
        try:
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
                    timeout=30
                )
                
                if process.returncode == 0 and os.path.exists(temp_audio.name):
                    play_result = subprocess.run(
                        ["aplay", temp_audio.name], 
                        capture_output=True, 
                        timeout=30
                    )
                    if play_result.returncode == 0:
                        print("✅ Speech successful")
                        return True
                        
        except Exception as e:
            print(f"❌ Speech failed: {e}")
        finally:
            try:
                if 'temp_audio' in locals():
                    os.unlink(temp_audio.name)
            except:
                pass
        return False
    
    def record_audio(self):
        """Record audio from microphone"""
        print(f"🎤 Recording for {self.recording_duration} seconds...")
        print("   Speak now!")
        
        try:
            recording = sd.rec(
                int(self.sample_rate * self.recording_duration),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32
            )
            sd.wait()
            print("✅ Recording complete")
            return recording.flatten()
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return np.array([])
    
    def speech_to_text(self, audio_data):
        """Convert speech to text"""
        if len(audio_data) == 0:
            return ""
        
        print("🔄 Converting speech to text...")
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.sample_rate)
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
                
                result = self.whisper_model.transcribe(
                    temp_file.name,
                    fp16=False,
                    language="en",
                    task="transcribe",
                    temperature=0
                )
                
                text = result["text"].strip()
                print(f"📝 You said: '{text}'")
                return text
                
        except Exception as e:
            print(f"❌ Speech recognition error: {e}")
            return ""
        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    def query_ollama(self, text):
        """Get response from Ollama"""
        print("🤖 Generating response...")
        
        try:
            prompt = f"""You are BookWorm, a friendly AI reading companion for children. 

Child asked: "{text}"

Give a helpful, encouraging response about books or reading in 1-2 sentences. Be enthusiastic and child-friendly. Do NOT include emojis or special characters.

BookWorm responds:"""
            
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 60,
                    "temperature": 0.8,
                    "stop": ["\n", "Child:", "System:"]
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            ai_response = result.get("response", "").strip()
            
            # Clean up response
            if "BookWorm responds:" in ai_response:
                ai_response = ai_response.split("BookWorm responds:")[-1].strip()
            
            if not ai_response or len(ai_response) < 10:
                ai_response = "That's a wonderful question! I love talking about books with you."
            
            print(f"🤖 Response: {ai_response}")
            return ai_response
            
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return "That's interesting! What's your favorite book?"
    
    def handle_conversation(self, tag_id):
        """Handle a complete conversation triggered by RFID"""
        print(f"\n📖 Starting conversation for tag: {tag_id}")
        
        # Initial greeting
        greeting = "Hello! I see you have a book. What would you like to talk about?"
        self.speak_piper(greeting)
        
        # Record user speech
        audio_data = self.record_audio()
        
        if len(audio_data) > 0:
            # Convert to text
            user_text = self.speech_to_text(audio_data)
            
            if user_text and len(user_text.strip()) > 2:
                # Give immediate thinking response
                thinking = random.choice(self.thinking_responses)
                self.speak_piper(thinking)
                
                # Generate and speak AI response
                ai_response = self.query_ollama(user_text)
                self.speak_piper(ai_response)
            else:
                self.speak_piper("I didn't hear anything clearly. Try speaking a bit louder next time!")
        else:
            self.speak_piper("I couldn't record your voice. Let's try again next time!")
        
        print("📖 Conversation complete! Waiting for next book...\n")
    
    def monitor_rfid(self):
        """Monitor RFID tags"""
        print("👂 Listening for RFID tags...")
        print("📚 Place a book with an RFID tag on the reader to start!")
        
        last_tag_id = None
        
        while self.running:
            try:
                if self.rfid.tagPresent():
                    current_time = time.time()
                    tag_id = self.rfid.readID()
                    
                    if tag_id and tag_id != last_tag_id and (current_time - self.last_rfid_time) > self.debounce_time:
                        print(f"\n📡 RFID detected: {tag_id}")
                        self.last_rfid_time = current_time
                        last_tag_id = tag_id
                        
                        # Handle the conversation
                        self.handle_conversation(tag_id)
                        
                        # Clear the last tag after a delay so the same tag can be used again
                        time.sleep(2)
                        last_tag_id = None
                
                sleep_ms(100)  # Check every 100ms
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ RFID monitoring error: {e}")
                time.sleep(1)
    
    def startup_greeting(self):
        """Initial greeting"""
        greeting = "BookWorm is ready! Place your book on the RFID reader to start our conversation."
        print(f"🤖 {greeting}")
        self.speak_piper(greeting)
    
    def run(self):
        """Main run loop"""
        if not self.rfid_working:
            print("❌ Cannot start - RFID not working")
            return
        
        try:
            self.running = True
            print("\n🚀 Starting BookWorm Reading Facilitator...")
            
            # Startup greeting
            self.startup_greeting()
            
            # Start RFID monitoring
            self.monitor_rfid()
            
        except KeyboardInterrupt:
            print("\n👋 BookWorm shutting down...")
        except Exception as e:
            print(f"❌ Fatal error: {e}")
        finally:
            self.running = False
            print("✅ BookWorm stopped")

def check_dependencies():
    """Quick dependency check"""
    print("🔍 Checking dependencies...")
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama ready")
        else:
            print("❌ Ollama not responding")
            return False
    except:
        print("❌ Ollama not running")
        return False
    
    # Check audio
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        if any(d['max_input_channels'] > 0 for d in devices):
            print("✅ Audio ready")
        else:
            print("❌ No audio input")
            return False
    except:
        print("❌ Audio system error")
        return False
    
    return True

if __name__ == "__main__":
    print("📚 BookWorm Reading Facilitator - RFID Version")
    print("=" * 50)
    
    if not check_dependencies():
        print("❌ Dependencies not ready")
        exit(1)
    
    bookworm = BookWormRFID()
    bookworm.run()
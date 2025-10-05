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
import threading
from itertools import cycle

from orchestrator import RuleOrchestrator

# RFID
from PiicoDev_RFID import PiicoDev_RFID
from PiicoDev_Unified import sleep_ms

class BookWormRFID:
    def __init__(self):
        print("🚀 Booting BookWorm with RFID...")
        
        # Audio settings
        self.channels = 1
        self.sample_rate = 16000
        self.recording_duration = 5.5
        
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

        # Orchestrator settings
        orchestrator_flag = True
        self.orchestrator_enabled = orchestrator_flag
        self.orchestrator = None
        if self.orchestrator_enabled:
            cases_path = "/home/gerrylu/voice-ai/BookWorm/cases.json"
            try:
                self.orchestrator = RuleOrchestrator(cases_path=cases_path, enabled=True)
                print(
                    f"✅ Orchestrator ready with {len(self.orchestrator.cases)} cases"
                    if self.orchestrator.cases
                    else "ℹ️ Orchestrator enabled but no cases loaded"
                )
            except Exception as exc:
                print(f"⚠️ Failed to initialize orchestrator: {exc}")
                self.orchestrator_enabled = False
        else:
            print("ℹ️ Orchestrator disabled")

        # System state
        self.last_rfid_time = 0
        self.debounce_time = 3.0  # 3 seconds between reads
        self.running = False
        self.book_title = None
        self.first_tag_id = None
        self.reader_names = ["Reader One", "Reader Two"]
        self.active_reader_index = -1  # Incremented when a new turn begins
        self.current_reader = None
        self.next_reader = None
        self.turn_counter = -1

        # Thinking responses for low latency
        self.thinking_responses = [
            "Ah, let me think...",
            "Hmm, that is a brilliant question...", 
            "Good question! Give me a moment...",
            "Ooo, nice question! Let me think...",
            "What a great question! Bookworm needs to ponder that..."
        ]

        print("✅ BookWorm initialization complete!")
        self.preload_ollama()

    def format_tag_id(self, raw_tag):
        """Normalize a raw RFID value into a readable string"""
        if raw_tag is None:
            return "UNKNOWN"

        try:
            if isinstance(raw_tag, (bytes, bytearray)):
                return ''.join(f"{b:02X}" for b in raw_tag)
            if isinstance(raw_tag, (list, tuple)):
                return ''.join(f"{int(b) & 0xFF:02X}" for b in raw_tag)
        except Exception as e:
            print(f"⚠️ Failed to format RFID tag {raw_tag}: {e}")
        return str(raw_tag)

    def log_rfid_event(self, tag_str, note):
        """Persist RFID events for debugging and evaluation"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} | {tag_str} | {note}\n"
        try:
            with open("rfid_events.log", "a", encoding="utf-8") as log_file:
                log_file.write(log_line)
        except Exception as e:
            print(f"⚠️ Failed to write RFID log: {e}")

    def await_llm_response(self, fetch_fn, fallback_text):
        """Run an LLM call in a background thread while presenting a thinking state"""
        response_holder = {"text": None, "error": None}

        def runner():
            try:
                response_holder["text"] = fetch_fn()
            except Exception as exc:
                response_holder["error"] = str(exc)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()

        thinking = random.choice(self.thinking_responses)
        self.speak_piper(thinking)

        spinner = cycle(["⏳", "⌛", "🤔", "🌀"])
        showed_spinner = False
        while thread.is_alive():
            showed_spinner = True
            print(f"\r{next(spinner)} BookWorm is thinking...", end="", flush=True)
            time.sleep(0.5)

        thread.join(timeout=0)
        if showed_spinner:
            print("\r✅ BookWorm found an answer!        ")
        print()

        if response_holder["error"]:
            print(f"⚠️ LLM fetch error: {response_holder['error']}")
        return response_holder["text"] or fallback_text
    
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
    
    def query_ollama_question(self, question_text):
        """Answer the reader's question and hand off to the next reader"""
        print("🤖 Generating response...")
        
        try:
            book_context = self.book_title or "this story"
            current_reader = self.current_reader or "the current reader"
            next_reader = self.next_reader or "their reading partner"
            turn_number = self.turn_counter + 1 if self.turn_counter >= 0 else 1

            prompt = f"""You are BookWorm, a friendly AI reading companion for children who are sharing a story together.

Current book: {book_context}
Turn number: {turn_number}
Current speaker asking a question: {current_reader}
Next reader waiting to ask their question: {next_reader}

{current_reader} asked: "{question_text}"

Your response must:
- answer {current_reader}'s question about {book_context} clearly and kindly.
- mention or connect to details from {book_context} so the story stays central.
- end with an invitational question for {next_reader}, encouraging them to share their thoughts on the story.
- stay within 1-2 sentences, using simple language without emojis or special characters.

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

    def query_ollama_comment(self, question_text, response_text):
        """Comment on the reader's answer without asking a new question"""
        print("🤖 Generating follow-up comment...")

        try:
            book_context = self.book_title or "this story"
            current_reader = self.current_reader or "the original reader"
            responder = self.next_reader or "their reading partner"
            turn_number = self.turn_counter + 1 if self.turn_counter >= 0 else 1

            prompt = f"""You are BookWorm, a friendly AI reading companion for children who are sharing a story together.

Current book: {book_context}
Turn number: {turn_number}
Original question asked by {current_reader}: "{question_text}"
{responder} responded with: "{response_text}"

Your response must:
- warmly acknowledge {responder} by name.
- reflect on what {responder} said and connect it to {book_context}.
- offer one short encouraging observation.
- do NOT ask another question or assign a new task.
- stay within 1-2 sentences, using simple language without emojis or special characters.

BookWorm comments:"""

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

            if "BookWorm comments:" in ai_response:
                ai_response = ai_response.split("BookWorm comments:")[-1].strip()

            if not ai_response or len(ai_response) < 8:
                ai_response = "I like how you noticed that part of the story."

            print(f"🤖 Follow-up: {ai_response}")
            return ai_response

        except Exception as e:
            print(f"❌ Ollama follow-up error: {e}")
            return "Thanks for telling me your idea!"
    
    def handle_conversation(self, tag_id):
        """Handle a complete conversation triggered by RFID"""
        tag_label = self.format_tag_id(tag_id)
        print(f"\n📖 Starting conversation for tag: {tag_label}")
        self.log_rfid_event(tag_label, "conversation-started")

        # Determine book and turn-taking state
        if self.book_title is None:
            self.book_title = "The Tale of Peter Rabbit"
            self.first_tag_id = tag_label
            self.turn_counter = 0
            print(f"📚 New book detected via tag {tag_label}: {self.book_title}")
            self.log_rfid_event(tag_label, "book-initialised")
        else:
            if self.turn_counter < 0:
                self.turn_counter = 0
            else:
                self.turn_counter += 1
            print(f"🔄 Advancing to turn {self.turn_counter + 1} for {self.book_title}.")
            self.log_rfid_event(tag_label, f"turn-{self.turn_counter + 1}")

            if self.first_tag_id and tag_label != self.first_tag_id:
                print(f"ℹ️ New tag variation detected ({tag_label}). Continuing with the same book context.")
                self.log_rfid_event(tag_label, "tag-variation")

        self.active_reader_index = self.turn_counter % len(self.reader_names)
        self.current_reader = self.reader_names[self.active_reader_index]
        self.next_reader = self.reader_names[(self.active_reader_index + 1) % len(self.reader_names)]

        if self.turn_counter == 0:
            greeting = (
                f"Hello fellow readers! I see you've fed me a book. It tastes like The Tale of Peter Rabbit. {self.current_reader}, do you have any questions about the story?"
            )
        else:
            greeting = (
                f"Thanks for checking in! {self.current_reader} its your turn, what question about The Tale of Peter Rabbit would you like to ask?"
            )

        self.speak_piper(greeting)
        
        # Record user speech
        audio_data = self.record_audio()

        if len(audio_data) > 0:
            # Convert to text
            question_text = self.speech_to_text(audio_data)
            
            if question_text and len(question_text.strip()) > 2:
                orchestrator_decision = None
                answer_text = None

                if self.orchestrator_enabled and self.orchestrator:
                    orchestrator_decision = self.orchestrator.process(question_text)
                    if orchestrator_decision.get("source") == "orchestrator":
                        answer_text = orchestrator_decision.get("response")
                        matched = orchestrator_decision.get("metadata", {}).get("matched_case")
                        print(f"🤖 Orchestrator matched case: {matched}")

                # Generate answer with non-blocking thinking state
                if not answer_text:
                    answer_text = self.await_llm_response(
                        lambda: self.query_ollama_question(question_text),
                        "That's a wonderful question! I love talking about books with you."
                    )
                self.speak_piper(answer_text)

                # Invite follow-up and listen
                listener_name = self.next_reader or "reading partner"
                self.speak_piper(f"I'm listening for your answer, {listener_name}! Share what you think.")
                followup_audio = self.record_audio()

                if len(followup_audio) > 0:
                    followup_text = self.speech_to_text(followup_audio)
                    if followup_text and len(followup_text.strip()) > 2:
                        comment_text = self.await_llm_response(
                            lambda: self.query_ollama_comment(question_text, followup_text),
                            "Thanks for sharing your idea about the story!"
                        )
                        self.speak_piper(comment_text)
                    else:
                        self.speak_piper("I wasn't sure what you said, but thanks for thinking about the story!")
                else:
                    self.speak_piper("I didn't catch a response, but we can talk more next time!")
            else:
                self.speak_piper("I didn't hear anything clearly. Try speaking a bit louder next time!")
        else:
            self.speak_piper("I couldn't record your voice. Let's try again next time!")

        print("📖 Conversation complete! Waiting for next book...\n")
    
    def monitor_rfid(self):
        """Monitor RFID tags"""
        print("👂 Listening for RFID tags...")
        print("📚 Place a book with an RFID tag on the reader to start!")
        
        last_tag_label = None
        
        while self.running:
            try:
                if self.rfid.tagPresent():
                    current_time = time.time()
                    try:
                        tag_id = self.rfid.readID()
                    except Exception as read_error:
                        print(f"⚠️ RFID read error: {read_error}")
                        self.log_rfid_event("READ_ERROR", str(read_error))
                        tag_id = None
                        time.sleep(0.1)
                        continue

                    tag_label = self.format_tag_id(tag_id)

                    if (
                        tag_id is not None
                        and tag_label != last_tag_label
                        and (current_time - self.last_rfid_time) > self.debounce_time
                    ):
                        print(f"\n📡 RFID detected: {tag_label}")
                        self.log_rfid_event(tag_label, "detected")
                        self.last_rfid_time = current_time
                        last_tag_label = tag_label
                        
                        # Handle the conversation
                        self.handle_conversation(tag_id)
                        
                        # Clear the last tag after a delay so the same tag can be used again
                        time.sleep(2)
                        last_tag_label = None

                sleep_ms(100)  # Check every 100ms
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ RFID monitoring error: {e}")
                time.sleep(1)
    
    def startup_greeting(self):
        """Initial greeting"""
        greeting = "Good day, readers! I'm BookWorm and my tummy is empty. Let's fill it with a story!"
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

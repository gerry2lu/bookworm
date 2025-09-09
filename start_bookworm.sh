#!/bin/bash

# BookWorm startup script
echo "Starting BookWorm AI..."
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "Date: $(date)"

# Set the correct paths
export HOME=/home/gerrylu
export VOICE_AI_DIR=/home/gerrylu/voice-ai
cd "$VOICE_AI_DIR"

# Check if virtual environment exists
if [ -d "$VOICE_AI_DIR/bookworm_env" ]; then
    echo "Activating virtual environment..."
    source "$VOICE_AI_DIR/bookworm_env/bin/activate"
    
    # Verify Python packages are available
    python -c "import whisper, sounddevice, webrtcvad; print('✅ All packages available')" || {
        echo "❌ Missing packages in virtual environment"
        echo "Installing packages..."
        pip install openai-whisper sounddevice webrtcvad librosa scipy numpy requests
    }
else
    echo "⚠️  Virtual environment not found, using system Python"
    echo "Installing packages system-wide..."
    pip3 install openai-whisper sounddevice webrtcvad librosa scipy numpy requests
fi

# Wait for Ollama to be ready
echo "Waiting for Ollama service..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready"
        break
    fi
    echo "Waiting for Ollama... ($i/30)"
    sleep 2
done

# Check audio system
echo "Checking audio system..."
if arecord -l | grep -q "card"; then
    echo "✅ Audio input devices found"
else
    echo "⚠️  No audio input devices found"
fi

if command -v espeak > /dev/null; then
    echo "✅ TTS (espeak) available"
else
    echo "⚠️  TTS not available"
fi

# Run BookWorm
echo "🚀 Launching BookWorm..."
python "$VOICE_AI_DIR/BookWorm.py"

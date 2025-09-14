#!/bin/bash

# BookWorm Reading Facilitator startup script
# Updated for RFID integration and improved architecture

echo "========================================"
echo "  BookWorm Reading Facilitator v2.0"
echo "========================================"
echo "Starting at: $(date)"
echo "User: $(whoami)"
echo "Working directory: $(pwd)"

# Set environment variables
export HOME=/home/gerrylu
export VOICE_AI_DIR=/home/gerrylu/voice-ai
export PYTHONPATH="$VOICE_AI_DIR:$PYTHONPATH"

# Change to working directory
cd "$VOICE_AI_DIR" || {
    echo "❌ Cannot access $VOICE_AI_DIR"
    exit 1
}

echo "Working directory: $(pwd)"

# Check if virtual environment exists and activate it
if [ -d "$VOICE_AI_DIR/bookworm_env" ]; then
    echo "🔄 Activating virtual environment..."
    source "$VOICE_AI_DIR/bookworm_env/bin/activate"
    echo "✅ Virtual environment activated"
    
    # Verify essential packages
    python -c "
import sys
packages = ['whisper', 'sounddevice', 'webrtcvad', 'numpy', 'requests']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'❌ {pkg}')

if missing:
    print(f'Installing missing packages: {\" \".join(missing)}')
    sys.exit(1)
else:
    print('✅ All required packages available')
" || {
        echo "Installing missing packages..."
        pip install openai-whisper sounddevice webrtcvad numpy requests scipy librosa
    }
else
    echo "⚠️ Virtual environment not found, using system Python"
    echo "🔄 Installing packages system-wide..."
    pip3 install openai-whisper sounddevice webrtcvad numpy requests scipy librosa
fi

# Check for PiicoDev RFID module
echo "🔄 Checking RFID module..."
python -c "
try:
    from PiicoDev_RFID import PiicoDev_RFID
    print('✅ PiicoDev RFID module available')
except ImportError:
    print('⚠️ PiicoDev RFID not found - will use simulation mode')
    print('   Install with: pip install piicodev')
" 

# Wait for Ollama service to be ready
echo "🔄 Waiting for Ollama service..."
OLLAMA_READY=false
for i in {1..60}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama service is ready"
        OLLAMA_READY=true
        break
    fi
    echo "   Waiting for Ollama... ($i/60)"
    sleep 2
done

if [ "$OLLAMA_READY" = false ]; then
    echo "❌ Ollama service not available after 2 minutes"
    echo "   Please start Ollama with: ollama serve"
    echo "   Then start this script again"
    exit 1
fi

# Check if required model is available
echo "🔄 Checking Ollama model..."
if curl -s http://localhost:11434/api/tags | grep -q "gemma2:2b"; then
    echo "✅ Required model (gemma2:2b) is available"
else
    echo "⚠️ Model gemma2:2b not found"
    echo "   Install with: ollama pull gemma2:2b"
    echo "   This may take several minutes..."
    ollama pull gemma2:2b || {
        echo "❌ Failed to install model"
        echo "   Please install manually: ollama pull gemma2:2b"
        exit 1
    }
fi

# Test audio system
echo "🔄 Testing audio system..."
python -c "
import sounddevice as sd
import numpy as np

try:
    # List audio devices
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    output_devices = [d for d in devices if d['max_output_channels'] > 0]
    
    if input_devices:
        print(f'✅ Found {len(input_devices)} audio input device(s)')
        default_input = sd.default.device[0]
        if default_input is not None:
            print(f'   Default input: {devices[default_input][\"name\"]}')
    else:
        print('❌ No audio input devices found')
        
    if output_devices:
        print(f'✅ Found {len(output_devices)} audio output device(s)')
        default_output = sd.default.device[1]  
        if default_output is not None:
            print(f'   Default output: {devices[default_output][\"name\"]}')
    else:
        print('❌ No audio output devices found')
        
    # Test basic recording capability
    print('🔄 Testing microphone...')
    test_recording = sd.rec(int(0.5 * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    
    if len(test_recording) > 0:
        volume = np.sqrt(np.mean(test_recording**2))
        print(f'✅ Microphone test successful (volume: {volume:.4f})')
    else:
        print('❌ Microphone test failed')
        
except Exception as e:
    print(f'❌ Audio system error: {e}')
"

# Check TTS capabilities
echo "🔄 Checking TTS capabilities..."
PIPER_MODEL="/home/gerrylu/voice-ai/piper-models/en_GB-semaine-medium.onnx"

if [ -f "$PIPER_MODEL" ]; then
    echo "✅ Piper TTS model found"
    
    # Test piper command
    if command -v piper >/dev/null 2>&1; then
        echo "✅ Piper command available"
    else
        echo "⚠️ Piper command not found in PATH"
        echo "   Please install Piper TTS"
    fi
else
    echo "⚠️ Piper model not found at: $PIPER_MODEL"
    echo "   Download from: https://github.com/rhasspy/piper"
fi

# Test audio playback
if command -v aplay >/dev/null 2>&1; then
    echo "✅ Audio playback (aplay) available"
else
    echo "⚠️ aplay not available - may affect TTS playback"
fi

# Set up GPIO permissions for RFID (if needed)
echo "🔄 Checking GPIO permissions..."
if [ -w /dev/gpiomem ]; then
    echo "✅ GPIO access available"
else
    echo "⚠️ GPIO access may be limited"
    echo "   Add user to gpio group: sudo usermod -a -G gpio $USER"
fi


# Final system check
echo ""
echo "========================================"
echo "  System Check Summary"
echo "========================================"

# Run comprehensive system check
python -c "
import sys
import requests
import sounddevice as sd
from pathlib import Path

all_good = True
print('🔍 Final system verification...')

# Check Ollama
try:
    response = requests.get('http://localhost:11434/api/tags', timeout=5)
    if response.status_code == 200:
        print('✅ Ollama service ready')
    else:
        print('❌ Ollama service not responding properly')
        all_good = False
except:
    print('❌ Cannot connect to Ollama')
    all_good = False

# Check audio
try:
    devices = sd.query_devices()
    if any(d['max_input_channels'] > 0 for d in devices):
        print('✅ Audio input ready')
    else:
        print('❌ No audio input devices')
        all_good = False
except:
    print('❌ Audio system not ready')
    all_good = False

# Check TTS
piper_model = Path('/home/gerrylu/voice-ai/piper-models/en_GB-semaine-medium.onnx')
if piper_model.exists():
    print('✅ TTS model ready')
else:
    print('⚠️ TTS model missing (will use fallbacks)')

# Check main script
main_script = Path('$VOICE_AI_DIR/BookWorm-S.py')
if main_script.exists():
    print('✅ Main script found')
else:
    print('❌ Main script missing')
    all_good = False

if all_good:
    print('')
    print('🎉 All systems ready!')
    sys.exit(0)
else:
    print('')
    print('⚠️ Some issues detected but continuing...')
    sys.exit(0)
"

# Start the application
echo ""
echo "========================================"
echo "  Launching BookWorm Reading Facilitator"
echo "========================================"

echo "🚀 Starting BookWorm..."
echo ""

# Run the main application with logging
python "$VOICE_AI_DIR/BookWorm-S.py"
# Capture exit code
EXIT_CODE=$?

echo ""
echo "========================================"
echo "  BookWorm Session Ended"
echo "========================================"
echo "Exit code: $EXIT_CODE"
echo "Session ended at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Clean shutdown"
else
    echo "⚠️ Unexpected exit (code: $EXIT_CODE)"
fi

echo "Thanks for using BookWorm Reading Facilitator!"
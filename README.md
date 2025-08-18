# BookWorm Voice AI - Raspberry Pi Setup

## Installation on Raspberry Pi 5

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Porcupine Wake Word Detection

```bash
pip install pvporcupine
```

### 3. Verify Audio Setup

Make sure your USB microphone is detected:

```bash
arecord -l
```

### 4. Test Audio Permissions

```bash
# Add user to audio group
sudo usermod -a -G audio $USER
# Reboot after adding to group
sudo reboot
```

## Usage

**Wake Word:** Say "Computer" to activate BookWorm
**Previous Wake Words:** "Hey BookWorm" no longer used (replaced for better detection)

## Performance Improvements

- **CPU Usage:** Reduced from ~80% to ~5% with Porcupine
- **Response Time:** Faster wake word detection
- **Offline:** Fully offline wake word detection
- **Memory:** Lower memory footprint

## Troubleshooting

### No Response to Wake Word

1. Check microphone permissions: `groups $USER` should include `audio`
2. Test microphone: `arecord -d 3 test.wav && aplay test.wav`
3. Check service logs: `journalctl -u your-service-name -f`

### High CPU Usage

- Old implementation used Whisper for wake word detection
- New implementation uses lightweight Porcupine
- Whisper now only used for speech-to-text after wake word

### Audio Device Issues

- USB microphones should work automatically
- If issues persist, check `dmesg | grep audio`

# Voice Conversation Demo

This project implements an end-to-end, low-latency voice conversation stack using open source building blocks. The browser streams microphone audio to a FastAPI backend over WebSockets. The backend performs streaming speech recognition with Vosk, generates a lightweight dialog response, synthesizes speech with Coqui TTS, and sends audio chunks back to the browser for immediate playback. Interruption logic is handled by cancelling any active text-to-speech stream whenever new user speech is finalized.

## Features

- **Realtime audio capture** in the browser with resampling to 16 kHz PCM frames.
- **WebSocket transport** for bidirectional streaming of audio and control messages.
- **Streaming ASR** powered by an embedded Vosk model.
- **Conversation manager** that maintains short-term history and produces deterministic replies offline.
- **Text-to-speech streaming** via Coqui TTS with incremental chunk delivery for low-latency playback.
- **Interruption handling** that cancels playback when the user starts speaking again.
- **Static frontend** served by FastAPI for a self-contained developer experience.

## Project layout

```
backend/
  main.py              # FastAPI application and WebSocket session management
  requirements.txt     # Python dependencies
  pipeline/
    conversation.py    # Conversation state machine and response generator
    recognizer.py      # Streaming Vosk recognizer with lazy model download
    tts.py             # Coqui TTS streamer with resampling
frontend/
  index.html           # Single page client UI
  styles.css           # Glassmorphism-inspired styling
  app.js               # Audio capture, transport, and playback logic
```

## Getting started

### 1. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Download the Vosk speech recognition model

The backend lazily downloads `vosk-model-small-en-us-0.15` on first run. To avoid the startup delay you can download it ahead of time:

```bash
mkdir -p backend/models
cd backend/models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
mv vosk-model-small-en-us-0.15 vosk-model
```

### 3. Run the development server

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000> in a Chromium-based browser. Click **Start Conversation** to grant microphone access. Speak into the mic and the assistant will transcribe and answer in near realtime.

> **Tip:** The first text-to-speech request downloads the VCTK model (~500 MB). Subsequent runs will reuse the cached model.

## System design overview

1. **Browser capture** – Microphone audio is sampled at 48 kHz, downsampled to 16 kHz PCM16 frames, and streamed over WebSocket.
2. **WebSocket gateway** – FastAPI handles a dedicated WebSocket session per client, forwarding audio frames to the recognizer while emitting partial/final transcripts.
3. **Streaming ASR** – Vosk consumes PCM frames incrementally, providing low-latency partial hypotheses and final transcripts.
4. **Conversation manager** – A lightweight history buffer keeps the dialog coherent without external LLM dependencies.
5. **Speech synthesis** – Coqui TTS generates speech, resampled to 16 kHz PCM16 and chunked into ~250 ms packets.
6. **Playback** – The browser converts PCM chunks back to floating point audio, upsampling to the current audio context sample rate and scheduling playback to minimize jitter.
7. **Interruption logic** – When a new final transcript is produced, any active speech playback task is cancelled and the client is instructed to flush queued audio.

## Limitations & next steps

- The deterministic conversation logic is intentionally simple; swap in a local LLM or external API for richer dialog.
- AudioWorklet integration could further reduce playback latency and jitter.
- More advanced VAD could reduce bandwidth and false positives; currently all microphone frames are streamed.
- Model downloads are large; consider packaging smaller multilingual models or pruning unused voices.

## License

MIT

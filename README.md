# VoiceConv

VoiceConv is a full-stack real-time voice conversation demo that runs entirely in the browser and on a self-hosted backend. It captures microphone audio, streams it to a FastAPI WebSocket gateway, performs speech-to-text, generates a reply with a lightweight language model, and synthesizes the answer back to audio for immediate playback in the browser.

## Features

- **Low-latency microphone capture** using the Web Audio API and an `AudioWorkletProcessor` that streams 16 kHz PCM frames over WebSocket.
- **Bidirectional audio transport** via a custom WebSocket protocol with explicit control messages for start/stop and binary audio payloads.
- **Speech intelligence pipeline** built from open components:
  - [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) for streaming transcription.
  - A local [`transformers`](https://huggingface.co/transformers/) causal language model (`distilgpt2`) that keeps a rolling conversation window.
  - [`TTS`](https://github.com/coqui-ai/TTS) (Coqui) for neural text-to-speech synthesis.
- **Interruption logic** that segments the capture on button release, resets buffers, and cancels microphone tracks so the assistant can respond immediately.
- **Frontend UI** that shows both sides of the transcript, connection status, and plays synthesized replies inline.

## Getting started

### 1. Install dependencies

VoiceConv uses Python for the backend services. Create a virtual environment and install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> The first run downloads open-source speech/LLM models (~500 MB combined). Keeping the environment warm avoids repeated downloads.

### 2. Launch the server

```bash
uvicorn server.app:app --reload
```

By default the server listens on `http://127.0.0.1:8000`. Static frontend assets are served from `/static`, and the WebSocket endpoint is exposed at `/ws/audio`.

### 3. Open the client

Navigate to `http://127.0.0.1:8000/` in a Chromium-based browser. Grant microphone access when prompted.

- Hold the **“Hold to talk”** button to stream audio to the backend.
- Release the button to send an end-of-utterance signal. The server transcribes the buffered PCM, generates a reply, and streams back synthesized speech.
- Use **Reset conversation** to clear the running dialogue state.

## Architecture overview

```
Browser (Web Audio)         FastAPI Gateway                    AI Pipeline
---------------------       ---------------------------        --------------------------
AudioContext + Worklet  ->  WebSocket /ws/audio handler  ->    Whisper STT  ->  LLM  ->  TTS
PCM Int16 frames         Control JSON + binary frames        Transcript    -> Reply -> PCM stream
Playback queue           <- Assistant response payloads    <- Synthesized PCM (16-bit)
```

The client sends 1024-sample (≈64 ms at 16 kHz) PCM chunks. The backend aggregates each talk burst, resamples to Whisper’s target rate, and then:

1. **Transcribes** the utterance with `faster-whisper`.
2. **Responds** using a conversational wrapper around `distilgpt2`.
3. **Synthesizes** the reply with Coqui TTS and streams the PCM bytes back.

The frontend queues incoming audio buffers for seamless playback while updating the transcript log.

## Customisation tips

- Swap `SpeechToText`, `ConversationalResponder`, or `SpeechSynthesizer` with alternative models by editing `server/pipeline.py`.
- Add barge-in or streaming partial transcripts by exposing intermediate results over the WebSocket protocol.
- Extend the UI by modifying files under `frontend/`—for example, to show waveform visualisation or latency metrics.

## Troubleshooting

- **Model downloads time out** – ensure outbound internet access on first run, or pre-download models into the Hugging Face cache.
- **Audio glitches** – lower the chunk size in `audioWorkletProcessor.js` or adjust browser sample rate hints.
- **High latency** – run the backend on GPU (Torch/Whisper auto-detects CUDA) and choose a smaller Whisper checkpoint such as `tiny.en`.

## License

This project is provided for evaluation purposes without a specific license.

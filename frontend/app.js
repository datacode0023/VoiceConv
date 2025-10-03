const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusEl = document.getElementById('status');
const partialEl = document.getElementById('partial');
const finalEl = document.getElementById('final');
const assistantEl = document.getElementById('assistant');

let ws = null;
let audioContext = null;
let mediaStream = null;
let processorNode = null;
let streamSource = null;
let player = null;

const WS_ENDPOINT = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws`;
const CLIENT_SAMPLE_RATE = 48000;
const TARGET_SAMPLE_RATE = 16000;

startBtn.addEventListener('click', startConversation);
stopBtn.addEventListener('click', stopConversation);

async function startConversation() {
  if (ws) {
    return;
  }

  resetTranscript();
  setStatus('Requesting microphone...');

  try {
    audioContext = new AudioContext({ sampleRate: CLIENT_SAMPLE_RATE });
    player = new StreamingPlayer(audioContext, TARGET_SAMPLE_RATE);

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        channelCount: 1,
        sampleRate: CLIENT_SAMPLE_RATE,
      },
    });

    setStatus('Connecting...');

    ws = new WebSocket(WS_ENDPOINT);
    ws.binaryType = 'arraybuffer';
    ws.onopen = onSocketOpen;
    ws.onmessage = onSocketMessage;
    ws.onclose = () => {
      setStatus('Disconnected');
      stopConversation();
    };
    ws.onerror = (err) => {
      console.error('WebSocket error', err);
      setStatus('Error');
      stopConversation();
    };
  } catch (err) {
    console.error('Failed to start conversation', err);
    setStatus('Microphone access denied');
    stopConversation();
  }
}

function onSocketOpen() {
  setStatus('Connected');
  startBtn.disabled = true;
  stopBtn.disabled = false;

  setupMediaProcessing();
}

function setupMediaProcessing() {
  const input = audioContext.createMediaStreamSource(mediaStream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const gain = audioContext.createGain();
  gain.gain.value = 0;

  processor.onaudioprocess = (event) => {
    const channelData = event.inputBuffer.getChannelData(0);
    const downsampled = downsampleBuffer(channelData, audioContext.sampleRate, TARGET_SAMPLE_RATE);
    if (downsampled && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(downsampled.buffer);
    }
  };

  input.connect(processor);
  processor.connect(gain);
  gain.connect(audioContext.destination);

  streamSource = input;
  processorNode = processor;
}

function onSocketMessage(event) {
  if (typeof event.data === 'string') {
    const message = JSON.parse(event.data);
    handleControlMessage(message);
    return;
  }

  if (!player) {
    return;
  }
  const arrayBuffer = event.data;
  player.enqueue(arrayBuffer);
}

function handleControlMessage(message) {
  switch (message.type) {
    case 'partial_transcript':
      partialEl.textContent = message.text;
      break;
    case 'final_transcript':
      partialEl.textContent = '';
      finalEl.textContent = message.text;
      break;
    case 'assistant_text':
      assistantEl.textContent = message.text;
      break;
    case 'clear_audio_queue':
      if (player) {
        player.reset();
      }
      break;
    case 'session_reset':
      resetTranscript();
      break;
    default:
      break;
  }
}

function stopConversation() {
  if (processorNode) {
    processorNode.disconnect();
    processorNode = null;
  }

  if (streamSource) {
    streamSource.disconnect();
    streamSource = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  if (ws) {
    try {
      ws.close();
    } catch (err) {
      console.error('Error closing websocket', err);
    }
    ws = null;
  }

  if (player) {
    player.reset();
    player = null;
  }

  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus('Disconnected');
}

function resetTranscript() {
  partialEl.textContent = '';
  finalEl.textContent = '';
  assistantEl.textContent = '';
}

function setStatus(text) {
  statusEl.textContent = text;
}

function downsampleBuffer(buffer, sampleRate, outSampleRate) {
  if (outSampleRate === sampleRate) {
    const float32 = new Float32Array(buffer.length);
    float32.set(buffer);
    return floatTo16BitPCM(float32);
  }
  const sampleRateRatio = sampleRate / outSampleRate;
  const newLength = Math.floor(buffer.length / sampleRateRatio);
  const result = new Int16Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.floor((offsetResult + 1) * sampleRateRatio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    const value = count > 0 ? accum / count : 0;
    result[offsetResult] = Math.max(-1, Math.min(1, value)) * 0x7fff;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }

  return result;
}

function floatTo16BitPCM(float32Array) {
  const result = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    result[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return result;
}

class StreamingPlayer {
  constructor(context, sourceSampleRate) {
    this.context = context;
    this.sourceSampleRate = sourceSampleRate;
    this.playbackTime = context.currentTime;
    this.activeSources = new Set();
  }

  enqueue(arrayBuffer) {
    const int16Data = new Int16Array(arrayBuffer);
    const floatData = this._upsample(int16Data, this.sourceSampleRate, this.context.sampleRate);
    const buffer = this.context.createBuffer(1, floatData.length, this.context.sampleRate);
    buffer.copyToChannel(floatData, 0, 0);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);

    source.onended = () => {
      this.activeSources.delete(source);
    };
    this.activeSources.add(source);

    const now = this.context.currentTime + 0.05;
    if (this.playbackTime < now) {
      this.playbackTime = now;
    }

    source.start(this.playbackTime);
    this.playbackTime += buffer.duration;
  }

  reset() {
    this.playbackTime = this.context.currentTime;
    this.activeSources.forEach((source) => {
      try {
        source.stop();
      } catch (err) {
        console.warn('Failed to stop source', err);
      }
    });
    this.activeSources.clear();
  }

  _upsample(int16Data, inRate, outRate) {
    if (inRate === outRate) {
      const direct = new Float32Array(int16Data.length);
      for (let i = 0; i < int16Data.length; i++) {
        direct[i] = int16Data[i] / 0x8000;
      }
      return direct;
    }

    const ratio = outRate / inRate;
    const newLength = Math.round(int16Data.length * ratio);
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      const index = i / ratio;
      const low = Math.floor(index);
      const high = Math.min(Math.ceil(index), int16Data.length - 1);
      const weight = index - low;
      const sample = int16Data[low] * (1 - weight) + int16Data[high] * weight;
      result[i] = sample / 0x8000;
    }
    return result;
  }
}

window.addEventListener('beforeunload', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }
});

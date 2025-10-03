const talkBtn = document.getElementById('talk-btn');
const resetBtn = document.getElementById('reset-btn');
const statusEl = document.getElementById('status');
const logEl = document.getElementById('conversation-log');

let websocket;
let audioContext;
let workletNode;
let mediaStream;
let captureSource;
let muteNode;
let player;
let pendingSampleRate = 22050;
let talking = false;

class AudioPlayer {
  constructor(context) {
    this.context = context;
    this.queue = [];
    this.playing = false;
  }

  enqueue(int16Array, sampleRate) {
    const float = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i += 1) {
      float[i] = Math.max(-1, Math.min(1, int16Array[i] / 32768));
    }
    const buffer = this.context.createBuffer(1, float.length, sampleRate);
    buffer.copyToChannel(float, 0, 0);
    this.queue.push(buffer);
    this.#playNext();
  }

  #playNext() {
    if (this.playing || this.queue.length === 0) {
      return;
    }
    const buffer = this.queue.shift();
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    source.onended = () => {
      this.playing = false;
      this.#playNext();
    };
    this.playing = true;
    source.start();
  }
}

function addMessage(role, text) {
  if (!text) return;
  const entry = document.createElement('div');
  entry.className = `message ${role}`;
  entry.textContent = text;
  logEl.appendChild(entry);
  logEl.scrollTop = logEl.scrollHeight;
}

function updateStatus(message) {
  statusEl.textContent = message;
}

async function ensureAudioContext() {
  if (!audioContext) {
    audioContext = new AudioContext({ sampleRate: 16000 });
    await audioContext.audioWorklet.addModule('/static/audioWorkletProcessor.js');
    workletNode = new AudioWorkletNode(audioContext, 'capture-processor');
    muteNode = audioContext.createGain();
    muteNode.gain.value = 0;
    workletNode.connect(audioContext.destination);
    muteNode.connect(audioContext.destination);
    player = new AudioPlayer(audioContext);
  }
  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }
  return audioContext;
}

async function ensureWebSocket() {
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    return websocket;
  }
  websocket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/audio`);
  websocket.binaryType = 'arraybuffer';
  websocket.onopen = () => updateStatus('Connected');
  websocket.onclose = () => updateStatus('Disconnected');
  websocket.onerror = (event) => {
    console.error('WebSocket error', event);
    updateStatus('WebSocket error');
  };
  websocket.onmessage = async (event) => {
    if (typeof event.data === 'string') {
      const data = JSON.parse(event.data);
      handleServerEvent(data);
      return;
    }
    const arrayBuffer = event.data instanceof ArrayBuffer ? event.data : await event.data.arrayBuffer();
    if (!arrayBuffer.byteLength) return;
    const int16 = new Int16Array(arrayBuffer);
    player.enqueue(int16, pendingSampleRate);
  };
  return new Promise((resolve) => {
    websocket.addEventListener('open', () => resolve(websocket), { once: true });
  });
}

function handleServerEvent(data) {
  switch (data.type) {
    case 'transcript':
      if (data.text) addMessage('user', data.text);
      break;
    case 'assistant_response':
      pendingSampleRate = data.sample_rate;
      addMessage('assistant', data.text);
      break;
    case 'assistant_audio_end':
      break;
    default:
      console.warn('Unhandled server event', data);
  }
}

function floatArrayToInt16(floatArray) {
  const buffer = new ArrayBuffer(floatArray.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < floatArray.length; i += 1) {
    let sample = floatArray[i];
    sample = Math.max(-1, Math.min(1, sample));
    view.setInt16(i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return buffer;
}

async function startCapture() {
  await ensureAudioContext();
  await ensureWebSocket();
  if (talking) return;
  talking = true;
  updateStatus('Recording...');
  websocket.send(JSON.stringify({ type: 'start' }));
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: 16000,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  captureSource = audioContext.createMediaStreamSource(mediaStream);
  captureSource.connect(workletNode);
  captureSource.connect(muteNode);
  workletNode.port.onmessage = ({ data }) => {
    if (!talking) return;
    websocket.send(floatArrayToInt16(data));
  };
}

function stopCapture() {
  if (!talking) return;
  talking = false;
  updateStatus('Processing...');
  websocket.send(JSON.stringify({ type: 'stop' }));
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (captureSource) {
    captureSource.disconnect();
    captureSource = null;
  }
}

function resetConversation() {
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.send(JSON.stringify({ type: 'reset' }));
  }
  logEl.innerHTML = '';
}

function setupButtonEvents() {
  const start = () => startCapture().catch((error) => {
    console.error(error);
    updateStatus('Microphone error');
  });
  const stop = () => stopCapture();
  talkBtn.addEventListener('mousedown', start);
  talkBtn.addEventListener('touchstart', (event) => {
    event.preventDefault();
    start();
  });
  talkBtn.addEventListener('mouseup', stop);
  talkBtn.addEventListener('mouseleave', stop);
  talkBtn.addEventListener('touchend', stop);
  resetBtn.addEventListener('click', resetConversation);
}

setupButtonEvents();
updateStatus('Connecting...');
ensureWebSocket().catch((error) => {
  console.error('Failed to connect', error);
  updateStatus('Connection failed');
});

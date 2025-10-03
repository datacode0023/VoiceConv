class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunkSize = 1024;
    this.buffer = new Float32Array(this.chunkSize);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }

    const channel = input[0];
    if (!channel) {
      return true;
    }

    let index = 0;
    while (index < channel.length) {
      const remaining = this.chunkSize - this.offset;
      const copyCount = Math.min(remaining, channel.length - index);
      this.buffer.set(channel.subarray(index, index + copyCount), this.offset);
      this.offset += copyCount;
      index += copyCount;

      if (this.offset === this.chunkSize) {
        this.port.postMessage(this.buffer.slice(0));
        this.offset = 0;
      }
    }

    return true;
  }
}

registerProcessor('capture-processor', CaptureProcessor);

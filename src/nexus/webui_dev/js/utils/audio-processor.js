// audio-processor.js — AudioWorklet processor for mic capture.
// Runs in the audio rendering thread.  Accumulates 16kHz int16 PCM
// samples and posts chunks to the main thread via MessagePort.

class AudioCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.chunkSize = 512;          // samples per chunk (~32 ms @ 16 kHz)
        this.buffer = new Int16Array(this.chunkSize);
        this.pos = 0;
    }

    process(inputs) {
        const input = inputs[0]?.[0];  // first input, first channel (mono)
        if (!input) return true;

        for (let i = 0; i < input.length; i++) {
            // Clamp and convert float32 [-1, 1] → int16.
            const s = Math.max(-1, Math.min(1, input[i]));
            this.buffer[this.pos++] = s < 0 ? s * 32768 : s * 32767;

            if (this.pos >= this.chunkSize) {
                // Post a copy (avoid transferring the backing buffer).
                this.port.postMessage({ pcm: this.buffer.slice() });
                this.pos = 0;
            }
        }
        return true;
    }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);

import asyncio
from typing import AsyncGenerator, Tuple

import numpy as np
from scipy.signal import resample_poly
from TTS.api import TTS


class SpeechSynthesizer:
    def __init__(self, target_sample_rate: int = 16000) -> None:
        self.target_sample_rate = target_sample_rate
        self._tts = TTS("tts_models/en/vctk/vits", progress_bar=False, gpu=False)

    async def stream_speech(self, text: str) -> AsyncGenerator[bytes, None]:
        if not text.strip():
            return

        loop = asyncio.get_running_loop()
        waveform, sample_rate = await loop.run_in_executor(None, self._synthesize, text)
        pcm16 = self._to_int16(waveform, sample_rate)
        chunk_size = self.target_sample_rate // 4  # 250ms chunks

        for start in range(0, len(pcm16), chunk_size):
            end = min(start + chunk_size, len(pcm16))
            chunk = pcm16[start:end]
            await asyncio.sleep(0)
            yield chunk.tobytes()

    def _synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        wav = self._tts.tts(text)
        sample_rate = self._tts.synthesizer.output_sample_rate
        return wav, sample_rate

    def _to_int16(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != self.target_sample_rate:
            waveform = resample_poly(waveform, self.target_sample_rate, sample_rate)
        waveform = np.clip(waveform, -1.0, 1.0)
        return (waveform * 32767).astype(np.int16)

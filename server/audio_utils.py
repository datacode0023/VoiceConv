"""Utility helpers for manipulating audio streams."""

from __future__ import annotations

import numpy as np
from scipy import signal

TARGET_SAMPLE_RATE = 16_000


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 PCM samples to float32 in [-1, 1]."""
    if audio.dtype != np.int16:
        raise ValueError("Expected int16 input array")
    return audio.astype(np.float32) / 32768.0


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 samples in [-1, 1] to int16 PCM."""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample ``audio`` from ``source_rate`` to ``target_rate`` using polyphase filtering."""
    if source_rate == target_rate:
        return audio
    if audio.size == 0:
        return audio
    gcd = np.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return signal.resample_poly(audio, up, down).astype(audio.dtype)


def merge_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Concatenate a list of equally typed numpy arrays."""
    if not chunks:
        return np.array([], dtype=np.int16)
    return np.concatenate(chunks)

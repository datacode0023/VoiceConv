"""Core speech pipeline orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from TTS.api import TTS

from .audio_utils import (
    TARGET_SAMPLE_RATE,
    float32_to_int16,
    int16_to_float32,
    merge_chunks,
    resample,
)

LOGGER = logging.getLogger(__name__)


class SpeechToText:
    """Wrapper around a Whisper model for streaming transcription."""

    def __init__(self, model_size: str = "base.en") -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        LOGGER.info("Loading Whisper model %s on %s", model_size, device)
        self._model = WhisperModel(model_size, device=device)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""
        if sample_rate != TARGET_SAMPLE_RATE:
            audio = resample(audio, sample_rate, TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE
        LOGGER.debug("Transcribing %s samples at %s Hz", len(audio), sample_rate)
        segments, _ = self._model.transcribe(audio, beam_size=1, language="en")
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        LOGGER.debug("Transcript: %s", transcript)
        return transcript


class ConversationalResponder:
    """Lightweight conversational agent built on a small causal language model."""

    def __init__(self, model_name: str = "distilgpt2", max_history: int = 6) -> None:
        LOGGER.info("Loading responder language model %s", model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._max_history = max_history

    def _build_prompt(self, user_text: str, history: List[str]) -> str:
        history_text = "\n".join(history[-self._max_history :])
        prompt_parts = [
            "You are an upbeat assistant having a spoken conversation.",
            "Keep answers concise (max 2 sentences).",
        ]
        if history_text:
            prompt_parts.append(history_text)
        prompt_parts.append(f"User: {user_text}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    def reply(self, user_text: str, history: List[str]) -> str:
        prompt = self._build_prompt(user_text, history)
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        output_ids = self._model.generate(
            input_ids,
            max_new_tokens=60,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = generated.split("Assistant:")[-1].strip()
        if not response:
            response = "I'm here and listening."
        LOGGER.debug("Responder output: %s", response)
        return response


class SpeechSynthesizer:
    """Text-to-speech wrapper leveraging a neural synthesizer."""

    def __init__(self, model_name: str = "tts_models/en/vctk/vits") -> None:
        LOGGER.info("Loading TTS model %s", model_name)
        self._tts = TTS(model_name=model_name)
        self.sample_rate = int(
            getattr(self._tts, "sample_rate", getattr(getattr(self._tts, "synthesizer", None), "output_sample_rate", 22_050))
        )

    def synthesize(self, text: str) -> np.ndarray:
        LOGGER.debug("Synthesizing speech for: %s", text)
        audio = np.asarray(self._tts.tts(text), dtype=np.float32)
        if audio.size == 0:
            return np.zeros(1, dtype=np.float32)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        return audio


@dataclass
class ConversationPipeline:
    """Stateful pipeline that converts audio streams to synthesized responses."""

    stt: SpeechToText
    responder: ConversationalResponder
    synthesizer: SpeechSynthesizer
    capture_rate: int = TARGET_SAMPLE_RATE
    _chunks: List[np.ndarray] = field(default_factory=list)
    _history: List[str] = field(default_factory=list)

    def start_utterance(self) -> None:
        LOGGER.debug("Starting a new utterance")
        self._chunks.clear()

    def reset(self) -> None:
        LOGGER.debug("Resetting conversation state")
        self._chunks.clear()
        self._history.clear()

    def append_audio(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        chunk = np.frombuffer(pcm_bytes, dtype=np.int16)
        self._chunks.append(chunk)

    def has_audio(self) -> bool:
        return any(chunk.size for chunk in self._chunks)

    def process(self) -> Optional[dict]:
        if not self.has_audio():
            LOGGER.debug("No audio captured for processing")
            return None
        merged = merge_chunks(self._chunks)
        float_audio = int16_to_float32(merged)
        transcript = self.stt.transcribe(float_audio, self.capture_rate)
        if not transcript:
            LOGGER.debug("Transcript empty, skipping response generation")
            return None
        reply = self.responder.reply(transcript, self._history)
        self._history.append(f"User: {transcript}")
        self._history.append(f"Assistant: {reply}")
        response_audio = self.synthesizer.synthesize(reply)
        response_pcm = float32_to_int16(response_audio)
        return {
            "transcript": transcript,
            "reply": reply,
            "audio": response_pcm.tobytes(),
            "sample_rate": self.synthesizer.sample_rate,
        }

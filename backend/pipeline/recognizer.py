import json
import zipfile
from pathlib import Path
from typing import Optional, Tuple

from vosk import KaldiRecognizer, Model

VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "vosk-model"


class StreamingRecognizer:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.model = self._load_model()
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)

    def accept_audio(self, pcm_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
        """Feed audio to the recognizer.

        Returns a tuple of (final_result, partial_result).
        """

        if self.recognizer.AcceptWaveform(pcm_bytes):
            result = json.loads(self.recognizer.Result())
            return result.get("text", ""), None

        partial = json.loads(self.recognizer.PartialResult()).get("partial", "")
        return None, partial

    def reset(self) -> None:
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)

    def close(self) -> None:
        pass

    def _load_model(self) -> Model:
        model_path = ensure_vosk_model()
        return Model(str(model_path))


def ensure_vosk_model() -> Path:
    VOSK_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if VOSK_MODEL_DIR.exists():
        return VOSK_MODEL_DIR

    archive_path = VOSK_MODEL_DIR.parent / "vosk-model.zip"
    download_file(VOSK_MODEL_URL, archive_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(VOSK_MODEL_DIR.parent)

    candidates = [p for p in VOSK_MODEL_DIR.parent.glob("vosk-model-*") if p.is_dir()]
    if not candidates:
        raise RuntimeError("Failed to extract Vosk model")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    candidates[0].rename(VOSK_MODEL_DIR)
    archive_path.unlink(missing_ok=True)
    return VOSK_MODEL_DIR


def download_file(url: str, destination: Path) -> None:
    import requests

    if destination.exists():
        return

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

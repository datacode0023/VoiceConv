"""FastAPI application exposing the voice conversation pipeline over WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import ConversationPipeline, ConversationalResponder, SpeechSynthesizer, SpeechToText

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


class PipelineFactory:
    """Lazy factory that ensures heavy models are instantiated once per process."""

    def __init__(self) -> None:
        self._stt: SpeechToText | None = None
        self._responder: ConversationalResponder | None = None
        self._tts: SpeechSynthesizer | None = None
        self._lock = asyncio.Lock()

    async def get_pipeline(self) -> ConversationPipeline:
        if self._stt is None or self._responder is None or self._tts is None:
            async with self._lock:
                if self._stt is None:
                    self._stt = SpeechToText()
                if self._responder is None:
                    self._responder = ConversationalResponder()
                if self._tts is None:
                    self._tts = SpeechSynthesizer()
        return ConversationPipeline(self._stt, self._responder, self._tts)


factory = PipelineFactory()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def read_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.websocket("/ws/audio")
async def audio_gateway(websocket: WebSocket) -> None:
    await websocket.accept()
    pipeline = await factory.get_pipeline()
    pipeline.start_utterance()
    LOGGER.info("Client connected")
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                pipeline.append_audio(message["bytes"])
            elif "text" in message and message["text"] is not None:
                payload = json.loads(message["text"])
                await handle_control_message(websocket, pipeline, payload)
    except WebSocketDisconnect:
        LOGGER.info("Client disconnected")
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Unhandled error in WebSocket handler: %s", exc)
        await websocket.close(code=1011, reason=str(exc))


async def handle_control_message(websocket: WebSocket, pipeline: ConversationPipeline, payload: Dict[str, Any]) -> None:
    message_type = payload.get("type")
    if message_type == "start":
        pipeline.start_utterance()
    elif message_type == "stop":
        result = pipeline.process()
        if result is None:
            pipeline.start_utterance()
            await websocket.send_json({"type": "transcript", "text": ""})
            return
        await websocket.send_json({"type": "transcript", "text": result["transcript"]})
        await websocket.send_json(
            {
                "type": "assistant_response",
                "text": result["reply"],
                "sample_rate": result["sample_rate"],
            }
        )
        await websocket.send_bytes(result["audio"])
        await websocket.send_json({"type": "assistant_audio_end"})
        pipeline.start_utterance()
    elif message_type == "reset":
        pipeline.reset()
    else:
        LOGGER.warning("Unknown control message: %s", payload)

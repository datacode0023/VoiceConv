import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from .pipeline.conversation import ConversationManager
from .pipeline.recognizer import StreamingRecognizer
from .pipeline.tts import SpeechSynthesizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    app = FastAPI(title="Voice Conversation")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).resolve().parent.parent / "frontend"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="frontend")

        @app.get("/")
        async def serve_index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

    synthesizer = SpeechSynthesizer()
    conversation_manager = ConversationManager()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        recognizer = StreamingRecognizer()
        tts_task: Optional[asyncio.Task] = None
        playback_lock = asyncio.Lock()

        async def cancel_tts() -> None:
            nonlocal tts_task
            if tts_task and not tts_task.done():
                logger.info("Cancelling active TTS task")
                tts_task.cancel()
                try:
                    await tts_task
                except asyncio.CancelledError:
                    pass
            tts_task = None
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_text(json.dumps({"type": "clear_audio_queue"}))
                except (RuntimeError, WebSocketDisconnect):
                    pass
                except Exception:
                    logger.debug("Failed to notify client about cancelled playback", exc_info=True)

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect(message.get("code", 1000))

                if "text" in message and message["text"] is not None:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type")

                    if msg_type == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                        continue

                    if msg_type == "stop" or msg_type == "reset":
                        recognizer.reset()
                        conversation_manager.reset()
                        await cancel_tts()
                        await websocket.send_text(json.dumps({"type": "session_reset"}))
                        continue

                    continue

                data = message.get("bytes")
                if data is None:
                    continue

                final_result, partial = recognizer.accept_audio(data)
                if partial:
                    await websocket.send_text(
                        json.dumps({"type": "partial_transcript", "text": partial})
                    )

                if final_result:
                    await cancel_tts()
                    await websocket.send_text(
                        json.dumps({"type": "final_transcript", "text": final_result})
                    )
                    response_text = conversation_manager.generate_response(final_result)
                    await websocket.send_text(
                        json.dumps({"type": "assistant_text", "text": response_text})
                    )

                    async def stream_tts() -> None:
                        async for chunk in synthesizer.stream_speech(response_text):
                            async with playback_lock:
                                await websocket.send_bytes(chunk)

                    tts_task = asyncio.create_task(stream_tts())
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception:
            logger.exception("Unexpected error in websocket session")
        finally:
            await cancel_tts()
            recognizer.close()

    return app


app = create_app()

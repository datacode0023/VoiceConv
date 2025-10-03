from fastapi import FastAPI

from .pipeline.conversation import ConversationPipeline
from .pipeline.recognizer import Recognizer
from .pipeline.tts import TextToSpeech

app = FastAPI()

conversation_pipeline = ConversationPipeline()
recognizer = Recognizer()
tts = TextToSpeech()


@app.get("/")
def read_root():
    return {"message": "Voice conversation backend is running."}


@app.post("/recognize")
def recognize(audio: bytes):
    transcription = recognizer.transcribe(audio)
    return {"transcription": transcription}


@app.post("/speak")
def speak(text: str):
    audio_bytes = tts.synthesize(text)
    return {"audio": list(audio_bytes)}


@app.post("/converse")
def converse(text: str):
    response = conversation_pipeline.process(text)
    return {"response": response}

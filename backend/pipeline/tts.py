class TextToSpeech:
    def synthesize(self, text: str) -> bytes:
        return text.encode()

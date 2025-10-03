from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import List


@dataclass
class Utterance:
    role: str
    text: str
    timestamp: dt.datetime = field(default_factory=lambda: dt.datetime.utcnow())


class ConversationManager:
    """Maintains a lightweight conversational state and produces responses."""

    def __init__(self, max_history: int = 6) -> None:
        self.max_history = max_history
        self.history: List[Utterance] = []

    def reset(self) -> None:
        self.history.clear()

    def generate_response(self, user_text: str) -> str:
        user_text = user_text.strip()
        if not user_text:
            return "I'm still listening. Could you repeat that?"

        self._append("user", user_text)

        response = self._build_response(user_text)
        self._append("assistant", response)
        return response

    def _append(self, role: str, text: str) -> None:
        self.history.append(Utterance(role=role, text=text))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def _build_response(self, user_text: str) -> str:
        """A deterministic fallback response generator.

        This avoids relying on external LLM APIs so the pipeline can run offline.
        """

        if any(g in user_text.lower() for g in ["hello", "hi", "hey"]):
            return "Hello! How can I help you today?"

        if "time" in user_text.lower():
            now = dt.datetime.now().strftime("%H:%M")
            return f"It's currently {now}. What else would you like to talk about?"

        if "your name" in user_text.lower():
            return "I'm an offline demo assistant built for streaming conversations."

        if user_text.endswith("?"):
            return (
                "That's an interesting question. I'm not connected to a large language model, "
                "but I'd love to hear more details so we can reason about it together."
            )

        if len(user_text.split()) < 4:
            return "Tell me more so I can respond with something helpful."

        return (
            "I hear you. This prototype focuses on the real-time audio pipeline, "
            "so my responses are intentionally simple. Feel free to ask about how the system works."
        )

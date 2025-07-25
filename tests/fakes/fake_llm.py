from typing import Generator, List

from jesktop.llms.base import LLMChat
from jesktop.llms.schemas import LLMMessage


class FakeLLMChat(LLMChat):
    """Fake LLM chat that returns predefined responses."""

    def __init__(self, responses: List[str]) -> None:
        self.responses = responses

    def chat(self, messages: List[LLMMessage]) -> LLMMessage:
        return LLMMessage(role="assistant", content=self.responses[0])

    def chat_stream(self, messages: List[LLMMessage]) -> Generator[LLMMessage, None, None]:
        """Stream chat completions, yielding only new content chunks."""
        for response in self.responses:
            yield LLMMessage(role="assistant", content=response)
